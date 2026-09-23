"""Robotensor Vector: chain commitments in, king-of-the-hill duels out, through its orchestrator.

This module is the chain's side of the lane and nothing else. The lane engine is
`vector_orchestrator` (duels, the weights runtime, the signed store), run as a library on the
`vector_l1` contract (`specs/vector_l1.json` in the orchestrator); the benchmark is the RoboTwin-Vector fork's
plugin; the model code is `vector_runtime`. What happens here:

**Intake.** Every `vector:` commitment on chain is read with the block it was made at. A new one is
looked up on the Hub (`hub.inspect`): a repository holding anything but the weights and a README is
refused; one the Hub does not show yet (still private) waits `private_window_blocks` and is then
refused; weights byte-identical to an earlier commitment's (same sha256, from the Hub's LFS
metadata) are a `duplicate` - the earliest commitment keeps them. Otherwise the commitment is
`queued`. A hotkey's new commitment supersedes its entry still waiting in the queue: the chain keeps
one commitment per hotkey.

**Duels.** The queue is served oldest commitment first. Before the first duel the track's declared
baseline (`robotensor/vector-base`) takes the empty throne by genesis. Each
duel is seeded from a block after the challenger's commitment (`protocol.seed`) and run by
`Orchestrator.run`, which publishes a signed record: the crown moves only when the challenger beats
the king by the margin and the paired sign test says it is no accident.

**Champions.** The lane's champions are read back from the signed store: every record that crowned a
model, newest first, each mapped to the hotkey that committed it (the baseline maps to none).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .. import hub
from ..chain import Commitment
from ..config import VectorConfig
from ..protocol import commitment as commitment_
from ..protocol import seed as seed_
from ..state import PENDING, QUEUED, State

log = logging.getLogger(__name__)

LANE = "vector"
TRACK = "vector_l1"


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
    def __init__(self, cfg: VectorConfig, state: State, *, hub_token: str | None = None) -> None:
        self.cfg = cfg
        self.state = state
        self.hub_token = hub_token
        self._engine: Any = None

    # -- the lane engine, built once --------------------------------------------------------

    @property
    def engine(self) -> Any:
        if self._engine is None:
            self._engine = self._build_engine()
        return self._engine

    def _build_engine(self) -> Any:
        # The benchmark plugin builds its commands for this interpreter.
        os.environ["ROBOTWIN_BENCH_PYTHON"] = self.cfg.simulator_python
        from vector_orchestrator.canon import Signer
        from vector_orchestrator.duel.orchestrate import Orchestrator
        from vector_orchestrator.duel.weights_runtime import WeightsPolicyRuntime
        from vector_orchestrator.spec import load_spec
        from vector_orchestrator.store.writer import Store, store_lock

        spec = load_spec(self.cfg.spec)
        if TRACK not in spec.tracks:
            raise ValueError(f"{self.cfg.spec} declares no track {TRACK}")
        self.cfg.key.parent.mkdir(parents=True, exist_ok=True)
        if self.cfg.key.exists():
            signer = Signer.from_file(self.cfg.key)
        else:
            signer = Signer.generate()
            signer.save(self.cfg.key)
            log.warning("generated the store's signing key at %s", self.cfg.key)
        store = Store(self.cfg.store, spec, signer)
        manifest = store.manifest()
        if manifest is None:
            with store_lock(store.root):
                store.init(signer.verify_key_hex)
        elif manifest.get("validator_key") != signer.verify_key_hex:
            raise ValueError(f"{self.cfg.store} is signed by another key than {self.cfg.key}")
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
    def allowed(self) -> frozenset[str]:
        return frozenset(self.spec.model["allowed_files"])

    # -- intake -----------------------------------------------------------------------------

    def intake(self, commitments: list[Commitment], block: int, *, api: Any = None) -> list[Entry]:
        """Take every new `vector:` commitment into the lane; the entries it changed."""
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
                    sub.repo, sub.revision, self.allowed, api=api, token=self.hub_token
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
        """The entries waiting for a duel, oldest commitment first."""
        lane = self.state.lane(LANE)
        waiting = [
            self._entry(key, record)
            for key, record in lane["entries"].items()
            if record["status"] == QUEUED
        ]
        return sorted(waiting, key=lambda e: (e.commit_block, e.hotkey))

    # -- duels ------------------------------------------------------------------------------

    def ready(self) -> bool:
        """Whether the store has a king: the baseline's genesis has been published."""
        return bool((self.engine.store.head(TRACK) or {}).get("king"))

    def genesis(self, chain: Any) -> dict[str, Any]:
        """Crown the declared baseline on the empty throne, seeded from the chain's head."""
        from vector_orchestrator.duel.orchestrate import DuelRequest
        from vector_orchestrator.store.writer import store_lock

        baseline = self.spec.baseline(TRACK) or {}
        revision = baseline.get("revision") or self.cfg.genesis_revision
        if not baseline.get("repo") or not revision:
            raise ValueError("the spec declares no baseline commit and the config names none")
        ref = self.engine.runtime.resolve(str(baseline["repo"]), str(revision))
        block = chain.block() - seed_.FINALITY
        seed = seed_.Seed(block, seed_.normalize_hash(chain.block_hash(block)))
        req = DuelRequest(
            TRACK,
            ref,
            None,
            baseline.get("size"),
            block=self._claim_block(),
            seed_block=seed.block,
            seed_block_hash=seed.block_hash,
        )
        with store_lock(self.engine.store.root):
            result = self.engine.run(req)
        return result.as_dict()

    def duel(self, entry: Entry, chain: Any, size: str | None = None) -> dict[str, Any]:
        """Duel `entry` against the king, seeded from a block after its commitment. Raises
        `seed.NotYet` when the chain has not moved far enough, and the orchestrator's
        `HarnessUnavailable`/`DuelFailed` when the harness cannot run it (the entry stays queued)."""
        from vector_orchestrator.duel.orchestrate import CrownMoved, DuelRequest
        from vector_orchestrator.ids import SubmissionRef
        from vector_orchestrator.store.writer import store_lock

        head = self.engine.store.head(TRACK) or {}
        king = SubmissionRef.from_dict(head.get("king"))
        challenger = SubmissionRef.resolved(entry.repo, entry.revision)
        lane = self.state.lane(LANE)
        record = lane["entries"][entry.key]
        if king is not None and king.key == challenger.key:
            record.update(status="duelled", reason="already holds the crown")
            self.state.save()
            return {"status": "skipped", "reason": "already holds the crown"}
        block = seed_.seed_block(chain.block(), entry.commit_block)
        seed = seed_.Seed(block, seed_.normalize_hash(chain.block_hash(block)))
        # A duel resumed after a restart keeps its request (the same seed and block).
        request = record.get("request")
        if request and request.get("king") == (king.as_dict() if king else None):
            req = DuelRequest.from_dict(request)
        else:
            req = DuelRequest(
                TRACK,
                challenger,
                king,
                size or self.cfg.duel_size,
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
            record.pop("request", None)
            self.state.save()
            raise
        doc = result.as_dict()
        outcome = (doc.get("record") or {}) if result.published else {}
        if result.published and outcome.get("dethroned"):
            status = "crowned"
        elif result.published:
            status = "duelled"
        else:
            status = result.status  # refused or void
        record.update(status=status, reason=result.reason, event_id=result.event_id)
        record.pop("request", None)
        self.state.save()
        return doc

    def _claim_block(self) -> int:
        """The next number in the lane's own duel sequence (the orchestrator's `block`)."""
        lane = self.state.lane(LANE)
        head = int((self.engine.store.head(TRACK) or {}).get("block") or 0)
        lane["block_counter"] = max(int(lane["block_counter"]), head) + 1
        self.state.save()
        return int(lane["block_counter"])

    # -- champions --------------------------------------------------------------------------

    def champions(self) -> list[str | None]:
        """Every model this lane crowned, newest first, as the hotkey that committed it; None for
        the genesis baseline. Read from the signed store's index."""
        from vector_orchestrator.ids import submission_key

        entries = self.state.lane(LANE)["entries"]
        crowned = []
        for record in self.engine.store.iter_index(TRACK):
            if record.get("kind") == "genesis":
                crowned.append(None)
            elif record.get("dethroned") and isinstance(record.get("new_king"), dict):
                king = record["new_king"]
                key = king.get("key") or submission_key(king["repo"], king["revision"])
                crowned.append((entries.get(key) or {}).get("hotkey"))
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
