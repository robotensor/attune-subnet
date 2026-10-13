"""Vector: chain commitments in, king-of-the-hill duels out, through its orchestrator.

This module is the chain's side of the lane and nothing else. The lane engine is
`vector_orchestrator` (duels, the weights runtime, the store), run as a library on its `spec.json`;
the benchmark is the RoboTwin-Vector checkout; the model code is `vector_runtime`. What happens here:

**Intake.** Every `vector:` commitment on chain is read with the block it was made at: a repository,
a revision and the sha256 of its weights (`protocol.commitment`). A new one is `queued` at once:
nothing is asked of the Hub until its turn comes. One made before the configured `start_block` is
not read at all: it is left unrecorded, so it neither enters nor spends its hotkey's submission,
and the hotkey enters by committing again from that block on.

**One submission per hotkey.** A hotkey's first commitment to reach the queue is its only one: from
then on, whatever it commits is `refused` - while that entry waits, during its duel, and after it,
whatever became of it. Queued is spent, as one reveal is one shot.

**Duels.** The queue is served oldest commitment first. Each duel is seeded from the chain's
finalized head, which must come after the challenger's commitment (`protocol.seed`). Then, and only
then, the challenger is looked up on the Hub (`hub.inspect`):

- a repository the Hub does not show (deleted, private, a revision it does not have) is `missing`:
  it leaves the queue and the next entry duels instead. A Hub that does not answer is the same as a
  repository that is not there: nothing waits on it;
- a repository holding anything but the weights and a README, or weights that do not hash to the
  committed digest, is `refused`; weights byte-identical to an earlier entry's are a `duplicate`;
The oldest admitted entry takes an empty throne by genesis, scored on its own units; otherwise it
duels the king through `Orchestrator.run`, which publishes a record: the crown moves only when the
challenger's average success rate beats the king's by the margin. The queue is written to the store
(`queue.json`) for the dashboard, and the orchestrator writes the duel's own progress beside it
(`running.json`).

**Champions whose repository is gone.** The king and every champion paid now are looked up on the
Hub about once a minute (`watch`, from the step and from the refresh thread, so it goes on while a
duel runs), walking the crowned models newest first until the pool is full of models still there.
One the Hub says it does not show (deleted, private, the revision gone) `MISSES` times in a row is
gone; a Hub that does not answer says nothing about a repository, and counts neither way. A gone
model leaves the pool at once - the weights skip it and the next older champion moves up - and
the store records it at the next step, between duels: a king is vacated and the crown returns to
the newest earlier king still there (`Orchestrator.vacate` with a successor; with none, the throne
is empty and the next entry takes it by genesis), any other champion is retired
(`Orchestrator.retire`). Neither is undone, should the repository come back.

**Champions.** The lane's champions are read back from the store: every record that crowned a
model, newest first, each mapped to the hotkey that committed it, less the ones gone. The ones paid
now - the newest four a miner committed, 40/30/20/10 - are written to the store too
(`champions.json`).

**Another host.** The mirror is the lane's history, and a validator started on a machine of its
own takes the work up from it rather than starting over: before the engine writes anything, the
store's manifest, head and index parts are read from the mirror (never its events or clips), with
the lane's state beside them (`validator/vector.json`, which says which hotkey committed what, and
which the lane writes there for that reason). An empty store whose mirror cannot be read is not
started: it would publish a fresh history over the real one. Every step takes up first what the
mirror holds that the store does not, and a push from a store behind the mirror is refused.

**While a duel runs.** A duel can take hours, and the chain does not stop meanwhile: `refresh` takes
new commitments in and republishes the queue from another thread (the worker's). Everything that
reads or writes the lane's state holds the lane's lock; the duel itself runs outside it.
"""

from __future__ import annotations

import copy
import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .. import hub
from ..chain import Commitment
from ..config import VectorConfig
from ..protocol import commitment as commitment_
from ..protocol import seed as seed_
from ..protocol.weights import CHAMPION_SPLIT
from ..state import EMPTY, PENDING, QUEUED, State
from .base import FAILED, IDLE, WAITING, WORKED, Award, Progress

log = logging.getLogger(__name__)

LANE = "vector"
#: An entry that reached the queue: its hotkey has made its one submission.
SUBMITTED = (QUEUED, "duelled", "crowned", "void", "missing", "vacated", "retired")
#: An entry whose repository the Hub did not show at its turn, and which has left the queue.
MISSING = "missing"
#: A king dethroned because its repository was gone.
VACATED = "vacated"
#: A champion, not the king, struck off because its repository was gone.
RETIRED = "retired"
#: The kinds of record that strike a model off, naming it in their king slot.
STRUCK = ("vacate", "retire")
#: The least time between two looks at the champions on the Hub (`watch`), in seconds.
WATCH_S = 60.0
#: How many looks in a row must find a champion's repository missing before it is gone.
MISSES = 3
#: `champions.json`'s layout.
CHAMPIONS_SCHEMA = 1
#: The lane's state, as the store keeps it for another host (`Store.validator_path`).
STATE_FILE = f"validator/{LANE}.json"


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
        #: Held by the one `watch` running; the other thread's passes rather than waits.
        self._watching = threading.Lock()
        self._watched_at = -math.inf

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
        mirror = None
        if self.cfg.mirror:
            from vector_orchestrator.store.mirror import Mirror

            mirror = Mirror(store.root, self.cfg.mirror, token=os.environ.get("HF_TOKEN"))
            self._take_up(store, mirror, starting=True)
        if store.manifest() is None:
            with store_lock(store.root):
                store.init()
        runtime = WeightsPolicyRuntime(
            spec,
            python=self.cfg.policy_python,
            cache_root=self.cfg.cache,
            kwargs=self.cfg.policy_kwargs,
        )
        return Orchestrator(
            spec, store, runtime, self.cfg.run_dir, mirror=mirror, workers=self.cfg.workers
        )

    # -- the mirror's history, taken up -----------------------------------------------------

    def _take_up(self, store: Any, mirror: Any, *, starting: bool) -> None:
        """Whatever the mirror holds that the store does not, into the store, and the lane's state
        with it. Starting with an empty store, a mirror that cannot be read stops the lane;
        otherwise the store goes on as it is. Raises the orchestrator's `StoreDiverged` when the
        two hold different records."""
        from vector_orchestrator.store.mirror import MirrorUnreachable, restore

        try:
            taken = restore(store, mirror, extra=[STATE_FILE])
        except MirrorUnreachable as exc:
            if starting and store.last_record() is None:
                raise RuntimeError(
                    f"the store is empty and its mirror cannot be read ({exc}): not starting, "
                    "since it would publish a fresh history over the mirror's"
                ) from exc
            log.warning("the mirror could not be read; going on with the store as it is: %s", exc)
            return
        self._adopt(store, taken=taken)

    def _adopt(self, store: Any, *, taken: bool) -> None:
        """The state the store keeps for the lane, as the lane's own: when the store was just taken
        up from the mirror (what this host remembered is older), or when this host has none."""
        from vector_orchestrator.store.writer import read_json

        kept = read_json(store.validator_path(LANE))
        if not isinstance(kept, dict) or not isinstance(kept.get("entries"), dict):
            return
        now = self.state.peek(LANE)
        if now == kept or (now["entries"] and not taken):
            return
        with self._lock, self.state.writing(LANE) as doc:
            if doc == kept or (doc["entries"] and not taken):
                return
            doc.clear()
            doc.update(kept)
            for key, value in EMPTY.items():
                doc.setdefault(key, copy.deepcopy(value))
        log.info("took the lane's state up from the store: %d entries", len(kept["entries"]))

    def sync(self) -> None:
        """Take up what the mirror published that the store does not hold yet."""
        mirror = self.engine.mirror
        if mirror is not None:
            self._take_up(self.engine.store, mirror, starting=False)

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
        """Take every new `vector:` commitment into the lane; the entries it changed. The engine
        is built first, so a store taken up from the mirror brings its state before intake."""
        self.engine  # noqa: B018 - built for what it takes up
        with self._lock:
            return self._intake(commitments, block, api=api)

    def _intake(self, commitments: list[Commitment], block: int, *, api: Any = None) -> list[Entry]:
        from vector_orchestrator.ids import submission_key

        entries = self.state.lane(LANE)["entries"]
        changed = []
        published: dict[str, str] | None = None
        for c in sorted(commitments, key=lambda c: (c.block, c.hotkey)):
            try:
                sub = commitment_.parse(c.data)
            except commitment_.CommitmentError:
                continue
            if sub.lane != commitment_.VECTOR:
                continue
            if c.block < self.cfg.start_block:
                continue  # made before the competition started: not read, not recorded
            key = submission_key(sub.repo, sub.revision)
            # `pending` is what an older build left waiting for the Hub; it is queued like a new one.
            if key in entries and entries[key]["status"] != PENDING:
                continue
            record = {
                "hotkey": c.hotkey,
                "repo": sub.repo,
                "revision": sub.revision,
                "digest": sub.digest,
                "commit_block": c.block,
                "status": QUEUED,
                "reason": "",
                "submitted": True,
            }
            if published is None:
                published = self._published()
            spent = self._submission_of(c.hotkey, besides=key)
            if key in published:
                # The store duelled it before this lane's state knew of it: a state lost with the
                # host that kept it. It is not queued again.
                record.update(status=published[key], reason="published before this state knew it")
            elif spent is not None:
                record.update(
                    status="refused",
                    reason=f"this hotkey has made its one submission ({spent})",
                    submitted=False,
                )
            entries[key] = record
            changed.append(self._entry(key, record))
        if changed:
            self.state.save()
        return changed

    def _published(self) -> dict[str, str]:
        """What the store's index says became of each submission it names: `crowned` (a genesis,
        or a challenger that won), `duelled` (one that lost), `vacated` or `retired`, the latest of
        them."""
        from vector_orchestrator.ids import submission_key

        out: dict[str, str] = {}
        for record in self.engine.store.iter_index():
            kind = record.get("kind")
            if kind == "duel":
                ref, status = (
                    record.get("challenger"),
                    ("crowned" if record.get("dethroned") else "duelled"),
                )
            elif kind == "genesis":
                ref, status = record.get("king"), "crowned"
            elif kind == "vacate":
                ref, status = record.get("king"), VACATED
            elif kind == "retire":
                ref, status = record.get("king"), RETIRED
            else:
                continue
            if not isinstance(ref, dict):
                continue
            key = ref.get("key")
            if not key and ref.get("repo") and ref.get("revision"):
                key = submission_key(ref["repo"], ref["revision"])
            if key:
                out[key] = status
        return out

    def _submission_of(self, hotkey: str, *, besides: str) -> str | None:
        """The key of the submission `hotkey` has made, other than `besides`: an entry that reached
        the queue, whatever became of it. None while the hotkey has made none."""
        for key, record in self.state.lane(LANE)["entries"].items():
            if key == besides or record["hotkey"] != hotkey:
                continue
            if record.get("submitted") or record["status"] in SUBMITTED:
                return key
        return None

    def queue(self) -> list[Entry]:
        """The entries waiting for a duel, oldest commitment first. The one being duelled is still
        queued until its duel ends."""
        self.engine  # noqa: B018 - built first, for the state it takes up
        with self._lock:
            lane = self.state.lane(LANE)
            waiting = [
                self._entry(key, record)
                for key, record in lane["entries"].items()
                if record["status"] == QUEUED
            ]
        return sorted(waiting, key=lambda e: (e.commit_block, e.hotkey))

    def refresh(self, chain: Any) -> list[Entry]:
        """Intake, the champions watched and the queue republished: what the worker runs beside a
        duel, so a commitment made while one runs is in the queue the dashboard shows within the
        minute rather than after the duel, and a champion whose repository is gone stops being
        paid. The entries that changed."""
        changed = self.intake(chain.commitments(), chain.block())
        if self.watch():
            self.publish_champions()
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
                passed = self._admit(entry, record)
                if passed is not None:
                    return passed
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

    def _admit(self, entry: Entry, record: dict[str, Any]) -> dict[str, Any] | None:
        """Look the challenger up on the Hub at its turn: None when it may duel, else why it is
        passed over (`{"status": "skipped", ...}`), with its record updated. A Hub that does not
        answer is a repository that is not there. Called holding the lane's lock."""
        by_weights = self.state.lane(LANE)["by_weights"]
        try:
            found = hub.inspect(entry.repo, entry.revision, self.shape, token=self.hub_token)
        except hub.NotVisible as exc:  # HubUnreachable too
            record.update(status=MISSING, reason=f"not on the Hub: {exc}")
            self.state.save()
            return {"status": "skipped", "reason": record["reason"]}
        except hub.NotASubmission as exc:
            record.update(status="refused", reason=str(exc))
            self.state.save()
            return {"status": "skipped", "reason": record["reason"]}
        record["weights_sha256"] = found.weights_sha256
        digest = record.get("digest")
        if digest and digest != found.weights_sha256:
            record.update(
                status="refused",
                reason=(
                    f"the commitment names weights {digest}, but {hub.WEIGHTS_FILE} at "
                    f"{entry.revision[:12]} hashes to {found.weights_sha256}"
                ),
            )
            self.state.save()
            return {"status": "skipped", "reason": record["reason"]}
        owner = by_weights.get(found.weights_sha256)
        if owner is not None and owner != entry.key:
            record.update(status="duplicate", reason=f"the same weights as {owner}")
            self.state.save()
            return {"status": "skipped", "reason": record["reason"]}
        by_weights[found.weights_sha256] = entry.key
        record["reason"] = ""
        self.state.save()
        return None

    # -- champions whose repository is gone -------------------------------------------------

    def watch(self, *, every: float | None = None) -> bool:
        """Look the king and the champions paid now up on the Hub, at most once per `every`
        seconds (`WATCH_S`), newest first and on past the ones gone until the pool is full of models still
        there; whether one was found gone this time. One the Hub does not show `MISSES` looks in
        a row is gone, in the lane's state, and leaves the pool at once (`strike` records it in
        the store). A Hub that does not answer counts neither way. Never raises."""
        if not self._watching.acquire(blocking=False):
            return False  # the other thread is looking now
        try:
            now = time.monotonic()
            if now - self._watched_at < (WATCH_S if every is None else every):
                return False
            self._watched_at = now
            return self._watch()
        except Exception:  # noqa: BLE001 - a look that failed is the next one's to make
            log.exception("watching the champions failed")
            return False
        finally:
            self._watching.release()

    def _watch(self) -> bool:
        live = 0
        found = False
        for model, hotkey, _, gone in self._crowned():
            if gone or not model:
                continue
            try:
                hub.inspect(model["repo"], model["revision"], self.shape, token=self.hub_token)
            except hub.HubUnreachable as exc:
                # Nothing can be learnt from a Hub that does not answer: the next look goes on.
                log.info("watch: the Hub did not answer for %s: %s", model["repo"], exc)
                break
            except hub.NotVisible as exc:
                if self._missed(model["key"], f"its repository is gone: {exc}"):
                    found = True
                    continue
            except hub.NotASubmission:
                pass  # what a pinned revision holds cannot change; it was admitted once
            else:
                self._seen(model["key"])
            # Still there, or not yet counted gone: it holds its place.
            if hotkey:
                live += 1
                if live >= len(CHAMPION_SPLIT):
                    break
        return found

    def _missed(self, key: str, reason: str) -> bool:
        """One more look that found `key`'s repository missing; whether that makes it gone."""
        with self._lock:
            watched = self.state.lane(LANE).setdefault("watch", {})
            mark = watched.setdefault(key, {"misses": 0})
            mark["misses"] = int(mark.get("misses", 0)) + 1
            mark["reason"] = reason
            gone = mark["misses"] >= MISSES
            if gone:
                mark["gone"] = True
            self.state.save()
        if gone:
            log.warning("watch: %s is gone after %d misses: %s", key, MISSES, reason)
        else:
            log.info("watch: %s missed %d of %d: %s", key, mark["misses"], MISSES, reason)
        return gone

    def _seen(self, key: str) -> None:
        """A look that found `key`'s repository there: its misses so far are forgotten."""
        with self._lock:
            watched = self.state.lane(LANE).get("watch") or {}
            if key in watched and not watched[key].get("gone"):
                del watched[key]
                self.state.save()

    def _gone(self) -> dict[str, str]:
        """The models the watch found gone, with why: `{key: reason}`."""
        with self._lock:
            watched = self.state.lane(LANE).get("watch") or {}
            return {k: str(m.get("reason", "")) for k, m in watched.items() if m.get("gone")}

    def strike(self) -> list[str]:
        """Record in the store what the watch found gone and the store does not say yet: the king
        vacated, the crown returned to the newest earlier king still there, and any other
        champion retired. Run between duels, since it takes the store's lock. The keys struck."""
        from vector_orchestrator.ids import SubmissionRef
        from vector_orchestrator.store.writer import store_lock

        gone = self._gone()
        if not gone:
            return []
        struck = self._struck()
        crowned = [(model, out) for model, _, _, out in self._crowned() if model]
        pending = [m for m, _ in crowned if m["key"] in gone and m["key"] not in struck]
        if not pending:
            return []
        # Newest first: the first of these is the newest earlier king still there.
        live = [SubmissionRef.from_dict(m) for m, out in crowned if not out]
        done = []
        with store_lock(self.engine.store.root):
            for model in pending:
                ref = SubmissionRef.from_dict(model)
                reason = gone[ref.key]
                king = SubmissionRef.from_dict((self.engine.store.head() or {}).get("king"))
                if king is not None and king.key == ref.key:
                    successor = live[0] if live else None
                    self.engine.vacate(
                        king, block=self._claim_block(), reason=reason, successor=successor
                    )
                    status = VACATED
                else:
                    self.engine.retire(ref, block=self._claim_block(), reason=reason)
                    status = RETIRED
                with self._lock:
                    record = self.state.lane(LANE)["entries"].get(ref.key)
                    if record is not None:
                        record.update(status=status, reason=reason)
                        self.state.save()
                done.append(ref.key)
        return done

    def _struck(self, index: list[dict[str, Any]] | None = None) -> set[str]:
        """The models the store's index (or `index`, read from it) has vacated or retired."""
        if index is None:
            index = list(self.engine.store.iter_index())
        return {
            record["king"].get("key")
            for record in index
            if record.get("kind") in STRUCK and isinstance(record.get("king"), dict)
        }

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
        """One duel: the champions watched first and the ones gone struck (a king gone hands the
        crown back), then the oldest entry the Hub shows against the king, or on an empty throne a
        genesis (`duel` with no king). An entry whose repository is not there is passed over for
        the next. Never raises the orchestrator's."""
        from vector_orchestrator.duel.orchestrate import CrownMoved, DuelFailed
        from vector_orchestrator.store.mirror import StoreDiverged

        try:
            self.sync()
        except StoreDiverged as exc:
            return Progress(LANE, FAILED, f"the store and its mirror disagree: {exc}")
        self.watch()
        self.strike()
        self.publish_queue(self.queue())
        self.publish_champions()
        queue = self.queue()
        if not queue:
            return Progress(LANE, IDLE, "nothing queued")
        passed = []
        for entry in queue:
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
            if result.get("status") == "skipped":
                log.info("passed over %s: %s", entry.entry, result.get("reason"))
                passed.append(f"{entry.entry}: {result.get('reason')}")
                self.publish_queue(self.queue())
                continue
            record = result.get("record") or {}
            detail = (
                f"{result['reason']}; dethroned={record.get('dethroned')} "
                f"king={record.get('king_score')} challenger={record.get('challenger_score')}"
            )
            self.publish_queue(self.queue())
            self.publish_champions()
            return Progress(LANE, WORKED, detail, record)
        return Progress(LANE, IDLE, "; ".join(passed))

    def publish_queue(self, queue: list[Entry]) -> None:
        """The waiting entries into the store's `queue.json`, as the dashboard shows them, and the
        lane's state beside them (`validator/vector.json`); pushed to the mirror only when either
        changed."""
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
            changed = self._write_state()
            if read_json(store.queue_path()) != snapshot:
                store.write_queue(snapshot)
                changed = True
        if changed:
            self.engine.push_touched()

    def _write_state(self) -> bool:
        """The lane's state into the store, for a validator on another host to take up; whether
        it changed. Called holding the lane's lock."""
        from vector_orchestrator.store.writer import read_json

        store = self.engine.store
        doc = self.state.lane(LANE)
        if read_json(store.validator_path(LANE)) == doc:
            return False
        store.write_validator(LANE, doc)
        return True

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
        """Every model this lane crowned and that is not gone, newest first, as the hotkey that
        committed it; None for one no entry of this validator's committed. Read from the store's
        index and the watch."""
        return [hotkey for _, hotkey, _, gone in self._crowned() if not gone]

    def paid(self) -> list[dict[str, Any]]:
        """The champions `protocol.weights` pays now, newest first: the newest crowned models a
        miner committed, one place each, with the part of the lane's share each place takes (the
        filled places share it all, in the split's proportions). A model gone holds no place; a
        hotkey since deregistered still holds its place here, and the weights burn its part."""
        held = [crowned for crowned in self._crowned() if crowned[1] and not crowned[3]]
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
            for place, ((model, hotkey, record, _), part) in enumerate(
                zip(held, parts, strict=True), start=1
            )
        ]

    def _crowned(self) -> list[tuple[dict[str, Any], str | None, dict[str, Any], bool]]:
        """Every crowning in the index, newest first: the model, its hotkey (None for one no entry
        of this validator's committed), the record that crowned it, and whether it is gone - struck
        in the index (vacated or retired) or found gone by the watch and not struck yet."""
        from vector_orchestrator.ids import submission_key

        # The index first: building the engine takes the store and the lane's state up from the
        # mirror, and the hotkeys must be read after that.
        index = list(self.engine.store.iter_index())
        with self._lock:
            entries = dict(self.state.lane(LANE)["entries"])
        gone = set(self._gone()) | self._struck(index)
        crowned = []
        for record in index:
            if record.get("kind") in STRUCK:
                continue
            if record.get("kind") == "genesis":
                king = record.get("king")  # a genesis names what it crowns in its king slot
            elif record.get("dethroned"):
                king = record.get("challenger")
            else:
                continue
            if not isinstance(king, dict):
                crowned.append(({}, None, record, False))
                continue
            key = king.get("key") or submission_key(king["repo"], king["revision"])
            model = {"key": key, "repo": king.get("repo"), "revision": king.get("revision")}
            hotkey = (entries.get(key) or {}).get("hotkey")
            crowned.append((model, hotkey, record, key in gone))
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
