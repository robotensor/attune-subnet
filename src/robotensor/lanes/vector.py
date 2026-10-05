"""Robotensor Vector: chain commitments in, king-of-the-hill duels out, through its orchestrator.

This module is the chain's side of the lane and nothing else. The lane engine is
`vector_orchestrator` (duels, the weights runtime, the store), run as a library on its `spec.json`;
the benchmark is the RoboTwin-Vector checkout; the model code is `vector_runtime`. What happens here:

**Intake.** Every `vector:` commitment on chain is read with the block it was made at. A new one is
looked up on the Hub (`hub.inspect`): a repository holding anything but the weights and a README is
refused; one the Hub does not show yet (still private) waits `private_window_blocks` and is then
refused; weights byte-identical to an earlier commitment's (same sha256, from the Hub's LFS
metadata) are a `duplicate` - the earliest commitment keeps them. Otherwise the commitment is
`queued`. A hotkey's new commitment supersedes its entry still waiting in the queue: the chain keeps
one commitment per hotkey.

**Duels.** The queue is served oldest commitment first; the oldest takes an empty throne by
genesis, scored on its own units. Each duel is seeded from the chain's finalized head, which must
come after the challenger's commitment (`protocol.seed`), and run by `Orchestrator.run`, which
publishes a record: the crown
moves only when the challenger beats the king by the margin and the paired sign test says it is no
accident. The queue is written to the store (`queue.json`) for the dashboard, and the orchestrator
writes the duel's own progress beside it (`running.json`).

**Champions.** The lane's champions are read back from the store: every record that crowned a
model, newest first, each mapped to the hotkey that committed it. The ones paid now - the newest
four a miner committed, 40/30/20/10 - are written to the store too (`champions.json`).

**While a duel runs.** A duel can take hours, and the chain does not stop meanwhile: `refresh` takes
new commitments in and republishes the queue from another thread (the worker's). Everything that
reads or writes the lane's state holds the lane's lock; the duel itself runs outside it.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .. import hub
from ..chain import Commitment
from ..config import VectorConfig
from ..protocol import commitment as commitment_
from ..protocol import seed as seed_
from ..protocol.weights import CHAMPION_SPLIT
from ..state import PENDING, QUEUED, State
from .base import FAILED, IDLE, WAITING, WORKED, Award, Progress

log = logging.getLogger(__name__)

LANE = "vector"
#: `champions.json`'s layout.
CHAMPIONS_SCHEMA = 1


@dataclass(frozen=True)
class Entry:
    key: str
    hotkey: str
    repo: str
    revision: str
    commit_block: int
    status: str

    @property
    def entry(self) -> str:
        return f"{self.repo}@{self.revision}"


class VectorLane:
    #: What the chain's commitments carry and `[vector]` configures.
    name = LANE

    def __init__(self, cfg: VectorConfig, state: State, *, hub_token: str | None = None) -> None:
        self.cfg = cfg
        self.state = state
        self.hub_token = hub_token
        self._engine: Any = None
        #: Held by whatever reads or writes the lane's state: `refresh` runs beside a duel.
        self._lock = threading.RLock()

    # -- the lane engine, built once --------------------------------------------------------

    @property
    def engine(self) -> Any:
        with self._lock:
            if self._engine is None:
                self._engine = self._build_engine()
            return self._engine

    def _build_engine(self) -> Any:
        # Where the orchestrator's RoboTwin driver finds the checkout it runs, and its interpreter.
        os.environ["ROBOTWIN_BENCH_ROOT"] = self.cfg.simulator_root
        os.environ["ROBOTWIN_BENCH_PYTHON"] = self.cfg.simulator_python
        from vector_orchestrator.duel.orchestrate import Orchestrator
        from vector_orchestrator.duel.weights_runtime import WeightsPolicyRuntime
        from vector_orchestrator.spec import load_spec
        from vector_orchestrator.store.writer import Store, store_lock

        spec = load_spec()
        store = Store(self.cfg.store, spec)
        if store.manifest() is None:
            with store_lock(store.root):
                store.init()
        runtime = WeightsPolicyRuntime(
            spec,
            python=self.cfg.policy_python,
            cache_root=self.cfg.cache,
            kwargs=self.cfg.policy_kwargs,
        )
        mirror = None
        if self.cfg.mirror:
            from vector_orchestrator.store.mirror import Mirror

            mirror = Mirror(store.root, self.cfg.mirror, token=os.environ.get("HF_TOKEN"))
        return Orchestrator(
            spec, store, runtime, self.cfg.run_dir, mirror=mirror, workers=self.cfg.workers
        )

    @property
    def spec(self) -> Any:
        return self.engine.spec

    @property
    def shape(self) -> hub.Shape:
        """What a submission may hold: `model.safetensors`, a README, and nothing that runs."""
        from vector_orchestrator.duel.weights_runtime import ALLOWED_FILES, MAX_REPO_BYTES

        return hub.vector_shape(ALLOWED_FILES, MAX_REPO_BYTES)

    # -- intake -----------------------------------------------------------------------------

    def intake(self, commitments: list[Commitment], block: int, *, api: Any = None) -> list[Entry]:
        """Take every new `vector:` commitment into the lane; the entries it changed."""
        with self._lock:
            return self._intake(commitments, block, api=api)

    def _intake(self, commitments: list[Commitment], block: int, *, api: Any = None) -> list[Entry]:
        from vector_orchestrator.ids import submission_key

        lane = self.state.lane(LANE)
        entries, by_weights = lane["entries"], lane["by_weights"]
        changed = []
        for c in sorted(commitments, key=lambda c: (c.block, c.hotkey)):
            try:
                sub = commitment_.parse(c.data)
            except commitment_.CommitmentError:
                continue
            if sub.lane != commitment_.VECTOR:
                continue
            key = submission_key(sub.repo, sub.revision)
            known = entries.get(key)
            if known is not None and known["status"] != PENDING:
                continue
            record = known or {
                "hotkey": c.hotkey,
                "repo": sub.repo,
                "revision": sub.revision,
                "commit_block": c.block,
                "status": PENDING,
                "reason": "",
            }
            try:
                found = hub.inspect(
                    sub.repo, sub.revision, self.shape, api=api, token=self.hub_token
                )
            except hub.NotVisible as exc:
                if block - c.block > self.cfg.private_window_blocks:
                    record.update(status="refused", reason=f"not on the Hub: {exc}")
                else:
                    record.update(status=PENDING, reason=f"waiting for the Hub: {exc}")
                entries[key] = record
                changed.append(self._entry(key, record))
                continue
            except hub.NotASubmission as exc:
                record.update(status="refused", reason=str(exc))
                entries[key] = record
                changed.append(self._entry(key, record))
                continue
            record["weights_sha256"] = found.weights_sha256
            owner = by_weights.get(found.weights_sha256)
            if owner is not None and owner != key:
                record.update(status="duplicate", reason=f"the same weights as {owner}")
            else:
                by_weights[found.weights_sha256] = key
                # The chain keeps one commitment per hotkey: its older entry still waiting is gone.
                for other_key, other in entries.items():
                    if (
                        other_key != key
                        and other["hotkey"] == c.hotkey
                        and other["status"] in (PENDING, QUEUED)
                    ):
                        other.update(status="superseded", reason=f"replaced by {key}")
                record.update(status=QUEUED, reason="")
            entries[key] = record
            changed.append(self._entry(key, record))
        if changed:
            self.state.save()
        return changed

    def queue(self) -> list[Entry]:
        """The entries waiting for a duel, oldest commitment first. The one being duelled is still
        queued until its duel ends."""
        with self._lock:
            lane = self.state.lane(LANE)
            waiting = [
                self._entry(key, record)
                for key, record in lane["entries"].items()
                if record["status"] == QUEUED
            ]
        return sorted(waiting, key=lambda e: (e.commit_block, e.hotkey))

    def refresh(self, chain: Any) -> list[Entry]:
        """Intake and the queue republished: what the worker runs beside a duel, so a commitment
        made while one runs is in the queue the dashboard shows within the minute rather than
        after the duel. The entries that changed."""
        changed = self.intake(chain.commitments(), chain.block())
        self.publish_queue(self.queue())
        return changed

    # -- duels ------------------------------------------------------------------------------

    def ready(self) -> bool:
        """Whether the store has a king: a genesis has been published."""
        return bool((self.engine.store.head() or {}).get("king"))

    def duel(self, entry: Entry, chain: Any) -> dict[str, Any]:
        """Duel `entry` against the king, seeded from the finalized head, which must come after its
        commitment. Raises `seed.NotYet` when finality has not passed the commitment yet, and the orchestrator's
        `HarnessUnavailable`/`DuelFailed` when the harness cannot run it (the entry stays queued)."""
        from vector_orchestrator.duel.orchestrate import CrownMoved, DuelRequest
        from vector_orchestrator.ids import SubmissionRef
        from vector_orchestrator.store.writer import store_lock

        head = self.engine.store.head() or {}
        king = SubmissionRef.from_dict(head.get("king"))
        challenger = SubmissionRef.resolved(entry.repo, entry.revision)
        with self._lock:
            record = self.state.lane(LANE)["entries"][entry.key]
            if king is not None and king.key == challenger.key:
                record.update(status="duelled", reason="already holds the crown")
                self.state.save()
                return {"status": "skipped", "reason": "already holds the crown"}
            finalized, finalized_hash = chain.finalized()
            block = seed_.seed_block(finalized, entry.commit_block)
            seed = seed_.Seed(block, seed_.normalize_hash(finalized_hash))
            # A duel resumed after a restart keeps its request (the same seed and block).
            request = record.get("request")
            if request and request.get("king") == (king.as_dict() if king else None):
                req = DuelRequest.from_dict(request)
            else:
                req = DuelRequest(
                    challenger,
                    king,
                    block=self._claim_block(),
                    seed_block=seed.block,
                    seed_block_hash=seed.block_hash,
                )
                record["request"] = req.as_dict(self.spec)
                self.state.save()
        try:
            with store_lock(self.engine.store.root):
                result = self.engine.run(req)
        except CrownMoved:
            with self._lock:
                record.pop("request", None)
                self.state.save()
            raise
        doc = result.as_dict()
        outcome = (doc.get("record") or {}) if result.published else {}
        if result.published and (king is None or outcome.get("dethroned")):
            status = "crowned"  # a genesis crowns its entrant; a duel, a challenger that won
        elif result.published:
            status = "duelled"
        else:
            status = result.status  # refused or void
        with self._lock:
            record.update(status=status, reason=result.reason, event_id=result.event_id)
            record.pop("request", None)
            self.state.save()
        return doc

    def _claim_block(self) -> int:
        """The next number in the lane's own duel sequence (the orchestrator's `block`)."""
        with self._lock:
            lane = self.state.lane(LANE)
            head = int((self.engine.store.head() or {}).get("block") or 0)
            lane["block_counter"] = max(int(lane["block_counter"]), head) + 1
            self.state.save()
            return int(lane["block_counter"])

    # -- the validator's view ----------------------------------------------------------------

    def step(self, chain: Any) -> Progress:
        """One duel: the oldest entry against the king, or on an empty throne a genesis (`duel`
        with no king). Never raises the orchestrator's."""
        from vector_orchestrator.duel.orchestrate import CrownMoved, DuelFailed

        queue = self.queue()
        self.publish_queue(queue)
        self.publish_champions()
        if not queue:
            return Progress(LANE, IDLE, "nothing queued")
        entry = queue[0]
        log.info(
            "duel: %s from %s (committed at %s)", entry.entry, entry.hotkey, entry.commit_block
        )
        try:
            result = self.duel(entry, chain)
        except seed_.NotYet as exc:
            return Progress(LANE, WAITING, str(exc))
        except CrownMoved as exc:
            # The king changed under us; the entry is still queued and duels the new one next.
            return Progress(LANE, WAITING, f"the crown moved, duelling again: {exc}")
        except DuelFailed as exc:
            log.error("duel of %s failed (it stays queued): %s", entry.entry, exc)
            return Progress(LANE, FAILED, str(exc))
        record = result.get("record") or {}
        detail = (
            f"{result['reason']}; dethroned={record.get('dethroned')} "
            f"king={record.get('king_score')} challenger={record.get('challenger_score')}"
        )
        self.publish_queue(self.queue())
        self.publish_champions()
        return Progress(LANE, WORKED, detail, record)

    def publish_queue(self, queue: list[Entry]) -> None:
        """The waiting entries into the store's `queue.json`, as the dashboard shows them; pushed
        to the mirror only when they changed."""
        from vector_orchestrator.ids import submission_key
        from vector_orchestrator.store.writer import read_json

        snapshot = {
            "entries": [
                {
                    "key": submission_key(e.repo, e.revision),
                    "repo": e.repo,
                    "revision": e.revision,
                    "hotkey": e.hotkey,
                    "block": e.commit_block,
                }
                for e in queue
            ]
        }
        with self._lock:
            store = self.engine.store
            if read_json(store.queue_path()) == snapshot:
                return
            store.write_queue(snapshot)
        self.engine.push_touched()

    def publish_champions(self) -> None:
        """The champions paid now into the store's `champions.json`; pushed only when changed."""
        from vector_orchestrator.store.writer import read_json

        doc = {
            "schema": CHAMPIONS_SCHEMA,
            "lane_share": self.cfg.share,
            "split": list(CHAMPION_SPLIT),
            "champions": self.paid(),
        }
        with self._lock:
            store = self.engine.store
            if read_json(store.champions_path()) == doc:
                return
            store.write_champions(doc)
        self.engine.push_touched()

    def award(self) -> Award:
        """The champion pool of the subnet spec: the four newest crowned models, 40/30/20/10."""
        return Award(self.champions())

    def snapshot(self) -> dict[str, Any]:
        """Where the competition stands: the king, the champions, the queue and every entry."""
        head = self.engine.store.head() or {}
        return {
            "king": head.get("king"),
            "champions": self.champions(),
            "queue": [e.__dict__ for e in self.queue()],
            "entries": self.state.lane(LANE)["entries"],
        }

    # -- champions --------------------------------------------------------------------------

    def champions(self) -> list[str | None]:
        """Every model this lane crowned, newest first, as the hotkey that committed it; None for
        one no entry of this validator's committed. Read from the store's index."""
        return [hotkey for _, hotkey, _ in self._crowned()]

    def paid(self) -> list[dict[str, Any]]:
        """The champions `protocol.weights` pays now, newest first: the newest crowned models a
        miner committed, one place each, with the part of the lane's share each place takes (the
        filled places share it all, in the split's proportions). A hotkey since deregistered still
        holds its place here; the weights burn its part."""
        held = [(model, hotkey, record) for model, hotkey, record in self._crowned() if hotkey]
        held = held[: len(CHAMPION_SPLIT)]
        parts = CHAMPION_SPLIT[: len(held)]
        total = sum(parts)
        return [
            {
                "place": place,
                **model,
                "hotkey": hotkey,
                "share": part / total,
                "event_id": record.get("event_id"),
                "crowned_at": record.get("finished_at"),
            }
            for place, ((model, hotkey, record), part) in enumerate(
                zip(held, parts, strict=True), start=1
            )
        ]

    def _crowned(self) -> list[tuple[dict[str, Any], str | None, dict[str, Any]]]:
        """Every crowning in the index, newest first: the model, its hotkey (None for one no entry
        of this validator's committed) and the record that crowned it."""
        from vector_orchestrator.ids import submission_key

        with self._lock:
            entries = dict(self.state.lane(LANE)["entries"])
        crowned = []
        for record in self.engine.store.iter_index():
            if record.get("kind") == "genesis":
                king = record.get("king")  # a genesis names what it crowns in its king slot
            elif record.get("dethroned"):
                king = record.get("challenger")
            else:
                continue
            if not isinstance(king, dict):
                crowned.append(({}, None, record))
                continue
            key = king.get("key") or submission_key(king["repo"], king["revision"])
            model = {"key": key, "repo": king.get("repo"), "revision": king.get("revision")}
            crowned.append((model, (entries.get(key) or {}).get("hotkey"), record))
        return list(reversed(crowned))

    @staticmethod
    def _entry(key: str, record: dict[str, Any]) -> Entry:
        return Entry(
            key=key,
            hotkey=str(record["hotkey"]),
            repo=str(record["repo"]),
            revision=str(record["revision"]),
            commit_block=int(record["commit_block"]),
            status=str(record["status"]),
        )


def store_path(cfg: VectorConfig) -> Path:
    return cfg.store
