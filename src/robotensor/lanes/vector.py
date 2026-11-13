"""Vector: chain commitments in, king-of-the-hill duels out, through its orchestrator.

This module is the chain's side of the lane and nothing else. The lane engine is
`vector_orchestrator` (duels, the weights runtime, the store), run as a library on its `spec.json`;
the benchmark is the RoboTwin-Vector checkout; the model code is `vector_runtime`. What happens here:

**Intake.** Every `vector:` commitment on chain is read with the block it was made at: a repository,
a revision and the sha256 of its weights (`protocol.commitment`). A new one is `queued` at once:
nothing is asked of the Hub until its turn comes. One made before the configured `start_block` is
not read at all: it is left unrecorded, so it neither enters nor spends a submission, and the miner
enters by committing again from that block on.

**One submission per registration.** A UID, from the block it was registered at, makes one
submission: its first commitment from then on (`protocol.slots`), which spends the shot even when it
is not a Vector one. Every other is `refused` -
while that entry waits, during its duel, and after it, whatever became of it. Queued is spent, as
one reveal is one shot. A hotkey swap keeps the registration, so the new hotkey is refused too;
registering again is a new one. The chain says which commitment is the one (`chain.submissions`
finds it on an archive node in a bounded number of reads, however often a miner commits), so a
validator starting with no state and no mirror takes the same one; the entries this lane keeps
refuse a second as well.

**Duels.** The queue is served oldest commitment first. Each duel is seeded from the chain's
finalized head, which must come after the challenger's commitment (`protocol.seed`). Then, and only
then, the challenger is looked up on the Hub (`hub.inspect`):

- a repository the validator cannot download (deleted, private, gated without the validator let
  in, a revision it does not have) or a Hub that does not answer about it is asked again,
  `HUB_TRIES` times in all with at least `HUB_GAP_S` between them, and then the entry is `missing`:
  it leaves the queue and the next entry duels instead. That is the one moment a miner's repository
  must answer: once fetched, the duel runs on the copy. A try counts only while this validator's
  own token works and the Hub answers it (`hub.token_problem`): otherwise the queue waits, and
  nobody's submission is used up by what is not their doing;
- a repository holding anything but the weights and a README, or weights that do not hash to the
  committed digest, is `refused`; weights byte-identical to an earlier entry's are a `duplicate`;
The oldest admitted entry takes an empty throne by genesis, scored on its own units; otherwise it
duels the king through `Orchestrator.run`, which publishes a record: the crown moves only when the
challenger's average success rate beats the king's by the margin. The queue is written to the store
(`queue.json`) for the dashboard, and the orchestrator writes the duel's own progress beside it
(`running.json`).

**Kings kept.** Nothing asks a miner to keep a repository up once its duel ran: the four newest
crowned models - the four the subnet pays - are the validator's to serve (`kings.Kings`). Their
weights are kept under the lane's `kings/` and mirrored to the public `kings_repo`, brought level
at every step and after every crowning, and no duel starts until the reigning king's weights are
in both places and the results hold its record. A king is fetched for its duels from the orchestrator's cache, where its own duel
left it, and restored there from `kings/` on a host that has no such cache. Records of kind
`vacate` and `retire`, written by an earlier build that struck off a champion whose repository
was gone, still read: such a model holds no place.

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
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .. import hub
from .. import kings as kings_
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
#: An entry whose repository the validator could not download at its turn, out of the queue.
MISSING = "missing"
#: The kinds of record an earlier build wrote to strike a model off, naming it in their king slot,
#: and the statuses its entries took: read, never written now.
STRUCK = ("vacate", "retire")
VACATED = "vacated"
RETIRED = "retired"
#: How many times the Hub is asked for a challenger it will not give before the entry is
#: `missing`, and the least time between two of those tries (a step that comes sooner waits).
HUB_TRIES = 3
HUB_GAP_S = 90.0
#: How many times running an entry may fail before the entries behind it go first. It keeps its
#: place in the queue's tail and its submission: a harness down fails them all, and that is nobody's
#: loss; but whatever makes one entry fail for ever must not hold the others up behind it.
DUEL_TRIES = 3
#: `champions.json`'s layout.
CHAMPIONS_SCHEMA = 1


def _now() -> float:
    return time.time()


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
        #: Whether this process has seen the mirror hold the store (`_seed`).
        self._seeded = False
        self.kings = kings_.Kings(cfg.kings, cfg.kings_repo, owner=cfg.kings_owner, token=hub_token)

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

            repo = f"{self._account()}/{self.cfg.mirror}"
            mirror = Mirror(store.root, repo, token=self.hub_token)
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
            spec,
            store,
            runtime,
            self.cfg.run_dir,
            mirror=mirror,
            workers=self.cfg.workers,
            king_forfeits=False,
        )

    def _account(self) -> str:
        """The account HF_TOKEN belongs to, whose repositories the lane publishes to. The last
        answer is kept: a host with a history goes on through a Hub that does not answer."""
        kept = self.cfg.store.parent / ".hub_account"
        try:
            account = hub.account(token=self.hub_token)
        except hub.NotVisible:
            if kept.is_file():
                return kept.read_text(encoding="utf-8").strip()
            raise
        if account not in hub.PERMITTED_VALIDATORS:
            raise RuntimeError(f"HF_TOKEN belongs to {account}, which is not a permitted account")
        kept.parent.mkdir(parents=True, exist_ok=True)
        kept.write_text(account, encoding="utf-8")
        return account

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

    def _results_behind(self) -> str | None:
        """Why the results mirror does not hold all of this store, after pushing what it lacks; None
        when it does, or there is none. Only what a publish touches is pushed otherwise, so a store
        older than its mirror (an upgrade, a repository renamed) would never reach it, and a new
        pod would read an empty history."""
        from vector_orchestrator.store.mirror import local_seq

        mirror, store = self.engine.mirror, self.engine.store
        if mirror is None:
            return None

        def behind() -> bool:
            return int((mirror.head() or {}).get("seq") or 0) < local_seq(store.root)

        try:
            # A kill between a record's append and the head's write leaves the head behind for
            # good: nothing else rewrites it until a duel runs, and this would hold every duel.
            if int((store.head() or {}).get("seq") or 0) != local_seq(store.root):
                store.rebuild_head()
                log.warning("the store's head was behind its index; rebuilt")
            self.engine.push_touched()
            if behind() or (not self._seeded and mirror.read("manifest.json") is None):
                mirror.push_all("the store's history")
                log.info("the whole store went to %s, which lacked it", mirror.repo)
            self._seeded = True
            return f"{mirror.repo} lacks records this store holds" if behind() else None
        except Exception as exc:  # noqa: BLE001 - the next step tries again
            return f"the results could not be brought level: {type(exc).__name__}: {exc}"

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
            # What makes one submission: the registration it was made from, read from the chain
            # (`chain.submissions`), or the hotkey where it was read without one.
            slot = getattr(c, "slot", None)
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
            if slot:
                record["slot"] = slot
            if published is None:
                published = self._published()
            spent = self._submission_of(slot or c.hotkey, besides=key)
            if key in published:
                # The store duelled it before this lane's state knew of it: a state lost with the
                # host that kept it. It is not queued again.
                record.update(status=published[key], reason="published before this state knew it")
            elif not getattr(c, "submission", True) or spent is not None:
                whose = f"registration {slot}" if slot else "hotkey"
                record.update(
                    status="refused",
                    reason=f"this {whose} has made its one submission"
                    + (f" ({spent})" if spent else ", earlier on chain"),
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

    def _submission_of(self, who: str, *, besides: str) -> str | None:
        """The key of the submission `who` (a registration's slot, or a hotkey) has made, other
        than `besides`: an entry that reached the queue, whatever became of it. None while it has
        made none."""
        for key, record in self.state.lane(LANE)["entries"].items():
            if key == besides or (record.get("slot") or record["hotkey"]) != who:
                continue
            if record.get("submitted") or record["status"] in SUBMITTED:
                return key
        return None

    def queue(self) -> list[Entry]:
        """The entries waiting for a duel, oldest commitment first, those that failed `DUEL_TRIES`
        times behind the rest. The one being duelled is still queued until its duel ends."""
        self.engine  # noqa: B018 - built first, for the state it takes up
        with self._lock:
            lane = self.state.lane(LANE)
            waiting = [
                (int(record.get("duel_failures", 0)) >= DUEL_TRIES, self._entry(key, record))
                for key, record in lane["entries"].items()
                if record["status"] == QUEUED
            ]
        waiting.sort(key=lambda w: (w[0], w[1].commit_block, w[1].hotkey))
        return [entry for _, entry in waiting]

    def commitments(self, chain: Any) -> list[Commitment]:
        """What intake reads: the chain's Vector commitments from `start_block` on, each marked
        whether it is its registration's one submission. What the chain's past said is kept
        beside the store, to be read again from the chain if it is lost."""
        return chain.submissions(self.cfg.start_block, cache=first_commitments_path(self.cfg))

    def refresh(self, chain: Any) -> list[Entry]:
        """Intake and the queue republished: what the worker runs beside a duel, so a commitment
        made while one runs is in the queue the dashboard shows within the minute rather than
        after the duel. The entries that changed."""
        changed = self.intake(self.commitments(chain), chain.block())
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
                if (problem := hub.token_problem(token=self.hub_token)) is not None:
                    log.error("the queue waits: %s", problem)
                    return {"status": "waiting", "reason": f"not the entry's doing: {problem}"}
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
        if king is not None and not self._restore_king(king):
            # A duel on a king whose weights are not here would hand it a forfeit it did not earn.
            return {"status": "waiting", "reason": f"the king {king.key}'s weights are not here"}
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
        passed over (`{"status": "skipped", ...}`) with its record updated, or why it waits
        (`{"status": "waiting", ...}`): a Hub that does not answer is asked again, `HUB_TRIES`
        times in all and never sooner than `HUB_GAP_S` after the last try. Called holding the
        lane's lock."""
        by_weights = self.state.lane(LANE)["by_weights"]
        tried_at = float(record.get("hub_tried_at") or 0.0)
        if tried_at and (left := tried_at + HUB_GAP_S - _now()) > 0:
            return {"status": "waiting", "reason": f"{record['reason']}; again in {left:.0f}s"}
        try:
            found = hub.inspect(entry.repo, entry.revision, self.shape, token=self.hub_token)
            hub.accessible(entry.repo, token=self.hub_token)
        except hub.NotVisible as exc:
            if (problem := hub.token_problem(token=self.hub_token)) is not None:
                log.error("the queue waits, and no try is counted: %s", problem)
                return {"status": "waiting", "reason": f"not the entry's doing: {problem}"}
            tries = int(record.get("hub_tries", 0)) + 1
            record["hub_tries"] = tries
            record["hub_tried_at"] = _now()
            if tries < HUB_TRIES:
                record["reason"] = f"not downloadable ({tries} of {HUB_TRIES}): {exc}"
                self.state.save()
                return {"status": "waiting", "reason": record["reason"]}
            record.update(status=MISSING, reason=f"not downloadable after {tries} tries: {exc}")
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
        record.pop("hub_tries", None)
        record.pop("hub_tried_at", None)
        self.state.save()
        return None

    # -- the kings kept ---------------------------------------------------------------------

    def kept_models(self) -> list[dict[str, Any]]:
        """The newest `kings.KEPT` crowned models, newest first, each with the `sha256` its
        admission recorded for its weights (none for a crowning this state never saw): what
        `kings/` keeps."""
        newest = [m for m, _, _, gone in self._crowned() if m and not gone][: kings_.KEPT]
        with self._lock:
            entries = self.state.lane(LANE)["entries"]
            for model in newest:
                record = entries.get(model["key"]) or {}
                model["sha256"] = record.get("weights_sha256") or record.get("digest")
        return newest

    def keep_kings(self) -> str | None:
        """`kings/` brought to the newest crowned models, the results brought level, and the Hub
        copy brought level with them, in that order, so it never names a king the results do not.
        The weights of a king not kept yet come from the orchestrator's cache (its own duel left
        them there), else from the Hub copy, else from the miner's repository while it still
        answers; one found nowhere now is tried again next step. Why no duel may start now, or None:
        the results are behind, the reigning king is not resolved, or the Hub copy names kings and
        the results none, which a first crowning would wipe. Never raises."""
        try:
            missing = self.kings.keep(self.kept_models(), self._king_weights)
            if missing:
                log.warning("kings: weights not found yet for %s", missing)
            if (behind := self._results_behind()) is not None:
                return behind
            king = (self.engine.store.head() or {}).get("king")
            if not king:
                named = self.kings.named_on_hub()
                return f"the kings repository names {named}; the results no king" if named else None
            self.kings.publish()
            key = str(king["key"])
            if not self.kings.resolved(key):
                where = f"{king.get('repo')}@{king.get('revision')}"
                return (
                    f"the king {key} ({where}) is not both here and on {self.kings.name}. If this "
                    "does not clear by itself, put its weights on the kings repository as "
                    f"{key}/{hub.WEIGHTS_FILE}"
                )
            return None
        except Exception as exc:  # noqa: BLE001 - the next step tries again
            log.exception("keeping the kings failed")
            return f"{type(exc).__name__}: {exc}"

    def _king_weights(self, model: dict[str, Any]) -> Path | None:
        from vector_orchestrator.submissions.resolve import Resolved

        expected = model.get("sha256")

        def right(path: Path) -> bool:
            return path.is_file() and (not expected or hub.sha256_of(path) == expected)

        cache = self.engine.runtime.cache
        resolved = Resolved(repo=model["repo"], revision=model["revision"], sha=model["revision"])
        if cache.lookup(resolved) is not None:
            path = cache.checkout(model["revision"]) / hub.WEIGHTS_FILE
            if right(path):
                return path
        found = self.kings.fetch_from_hub(model["key"])
        if found is not None:
            if right(found):
                return found
            log.warning("kings: the Hub copy of %s is wrong; asking its repository", model["key"])
            found.unlink(missing_ok=True)
        return self.kings.fetch_from_miner(model["repo"], model["revision"])

    def _restore_king(self, king: Any) -> bool:
        """The king's weights into the orchestrator's cache from `kings/`, on a host that has no
        checkout of it: the duel then fetches it without asking the Hub. Whether they are there."""
        import shutil

        from vector_orchestrator.submissions.resolve import Resolved

        cache = self.engine.runtime.cache
        resolved = Resolved(repo=king.repo, revision=king.revision, sha=king.revision)
        kept = self.kings.path(king.key)
        if cache.lookup(resolved) is not None:
            return True
        if not kept.is_file():
            return False
        cache.store(resolved, lambda into: shutil.copyfile(kept, into / hub.WEIGHTS_FILE))
        log.info("kings: %s restored into the cache from kings/", king.key)
        return True

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
        """One duel: the kings kept first, then the oldest entry the validator can download
        against the king, or on an empty throne a genesis (`duel` with no king). An entry whose
        repository cannot be downloaded is passed over for the next. Never raises the
        orchestrator's."""
        from vector_orchestrator.duel.orchestrate import CrownMoved, DuelFailed
        from vector_orchestrator.store.mirror import StoreDiverged

        try:
            self.sync()
        except StoreDiverged as exc:
            return Progress(LANE, FAILED, f"the store and its mirror disagree: {exc}")
        unresolved = self.keep_kings()
        self.publish_queue(self.queue())
        self.publish_champions()
        queue = self.queue()
        if not queue:
            return Progress(LANE, IDLE, "nothing queued")
        if unresolved is not None:
            return Progress(LANE, WAITING, f"the kings are not resolved: {unresolved}")
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
                self._count_failure(entry)
                return Progress(LANE, FAILED, str(exc))
            except Exception as exc:  # noqa: BLE001 - nothing one entry sets off may hold the queue
                log.exception("duel of %s raised (it stays queued)", entry.entry)
                self._count_failure(entry)
                return Progress(LANE, FAILED, f"{entry.entry}: {type(exc).__name__}: {exc}")
            if result.get("status") == "waiting":
                return Progress(LANE, WAITING, f"{entry.entry}: {result.get('reason')}")
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
            self.keep_kings()
            self.publish_queue(self.queue())
            self.publish_champions()
            return Progress(LANE, WORKED, detail, record)
        return Progress(LANE, IDLE, "; ".join(passed))

    def _count_failure(self, entry: Entry) -> None:
        with self._lock:
            record = self.state.lane(LANE)["entries"].get(entry.key)
            if record is not None:
                record["duel_failures"] = int(record.get("duel_failures", 0)) + 1
                self.state.save()

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
            "kings": self.kings.kept(),
            "entries": self.state.lane(LANE)["entries"],
        }

    # -- champions --------------------------------------------------------------------------

    def champions(self) -> list[str | None]:
        """Every model this lane crowned and that is not gone, newest first, as the hotkey that
        committed it; None for one no entry of this validator's committed. Read from the store's
        index."""
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
        of this validator's committed), the record that crowned it, and whether an earlier build
        struck it off in the index (vacated or retired)."""
        from vector_orchestrator.ids import submission_key

        # The index first: building the engine takes the store and the lane's state up from the
        # mirror, and the hotkeys must be read after that.
        index = list(self.engine.store.iter_index())
        with self._lock:
            entries = dict(self.state.lane(LANE)["entries"])
        gone = self._struck(index)
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


def first_commitments_path(cfg: VectorConfig) -> Path:
    """Where `chain.submissions` keeps what it found of the chain's past: a cache, not a record."""
    return cfg.store.parent / "first_commitments.json"


def store_path(cfg: VectorConfig) -> Path:
    return cfg.store
