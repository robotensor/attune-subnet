"""Robotensor Horizon: chain commitments in, weekly rounds out, through its engine.

This module is the chain's side of Competition 2 and nothing else. The engine is
`horizon_competition` (the pool, the evaluation, the scoring and the signed store) and the model
runtime is `horizon_runtime_zerowam`; what lives here is the part only the chain can answer.

**Intake.** Every `horizon:` commitment is read with the block it was made at, and belongs to the
round that was open then (`protocol.schedule`) - not the one open when a validator got round to
reading it. A submission is up to 50 GB of shards, so it is judged from the Hub's metadata alone:
what the repository may hold and what makes two of them the same model come from the runtime's own
family definition, so the key intake computes is the one the runtime computes from the files.

**Rounds.** Open, freeze a pool of demonstrations, score every entrant on it, rank, close. The
schedule is the chain's; the entropy that draws the units is the organiser's secret and a block
hash together, neither alone (`protocol.schedule.entropy`).

**Staged, and honest about it.** What is here is what the chain can do with no simulator: the
prefix, the schedule, intake and status. Running a round is the next stage, and until then a
`step` says which round is open, how many entrants it holds, and that nothing scores them yet. The
lane's share stays 0 while that is true, and the config refuses to pay anything without
`serve_as`, because a model served as the validator's own user could read the answer key of the
episode it is being scored on.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .. import hub
from ..chain import Commitment
from ..config import HorizonConfig
from ..protocol import commitment as commitment_
from ..protocol import schedule as schedule_
from ..state import PENDING, QUEUED, State
from . import rounds as rounds_
from .base import FAILED, IDLE, WAITING, WORKED, Award, Progress

log = logging.getLogger(__name__)

LANE = "horizon"
#: What a round is called, from the number the chain gives it. Two validators agree on it
#: because they agree on the number, and it sorts the way it is counted. It is an address - the
#: round's directory and its records' names - and never shown to a reader, who sees "Round N";
#: the `e` is from when rounds were epochs, and stays so that a round opened then is found again.
ROUND_ID = "e{number:05d}"
#: The names a round's close record is published under, in the order they are looked for: the
#: engine's own now, then the one a round closed before it called them rounds still has. The
#: record is the same either way - it ranks in `scores.rank` - but for the names of its id and
#: settings (`round_id` where it said `epoch_id`), and nothing published is ever renamed.
CLOSE_RECORDS = ("round-{id}", "epoch-{id}")
#: What a submission holds, when the runtime cannot be asked: the layout of the family shipped
#: with it. `shape` prefers the runtime's own answer, which is the one it hashes.
DEFAULT_LAYOUT = {
    "directory": "transformer",
    "index": "diffusion_pytorch_model.safetensors.index.json",
    "single": "diffusion_pytorch_model.safetensors",
    "config": "config.json",
    "knobs": "zerowam.yaml",
    "norm_stats": ("norm_stats/robotwin.json", "norm_stats/robocasa.json"),
}


@dataclass(frozen=True)
class Entry:
    key: str
    hotkey: str
    repo: str
    revision: str
    commit_block: int
    status: str
    #: The round this commitment entered: the one open at its block.
    round: int = 0

    @property
    def entry(self) -> str:
        return f"{self.repo}@{self.revision}"


def shape_from(layout: dict[str, Any], max_bytes: int) -> hub.Shape:
    """What a repository may hold and what its content key covers, from a family's layout."""
    directory = str(layout["directory"])
    shards = f"{directory}/*.safetensors"
    return hub.Shape(
        allowed=(
            f"{directory}/*",
            "norm_stats/*.json",
            str(layout["knobs"]),
            "README.md",
            ".gitattributes",
        ),
        required=(shards,),
        hashed=(
            shards,
            f"{directory}/{layout['config']}",
            "norm_stats/*.json",
            str(layout["knobs"]),
        ),
        key="listing",
        max_bytes=max_bytes,
        lfs_only=(shards,),
    )


class HorizonLane:
    #: What the chain's commitments carry and `[lanes.horizon]` configures.
    name = LANE

    def __init__(self, cfg: HorizonConfig, state: State, *, hub_token: str | None = None) -> None:
        self.cfg = cfg
        self.state = state
        self.hub_token = hub_token
        self._shape: hub.Shape | None = None

    @property
    def schedule(self) -> schedule_.Schedule:
        return schedule_.Schedule(self.cfg.genesis_block, self.cfg.window_blocks)

    @property
    def shape(self) -> hub.Shape:
        """What a submission may hold, asked of the runtime that will serve it.

        The runtime prints the files its content hash covers (`horizon-runtime-zerowam family`),
        so intake and the runtime agree on what two submissions of the same bytes are. Without the
        runtime installed, the layout of the family it ships is used, and a mismatch would be
        caught by the runtime's own check before anything was scored.
        """
        if self._shape is None:
            self._shape = shape_from(self._layout(), self.cfg.max_repo_bytes)
        return self._shape

    def _layout(self) -> dict[str, Any]:
        try:
            from horizon_runtime_zerowam import family as family_module
        except ImportError:
            log.info("horizon: the runtime is not installed; reading the family it ships")
            return dict(DEFAULT_LAYOUT)
        family = family_module.load()
        weights = family.weights
        return {
            "directory": weights["directory"],
            "index": weights["index"],
            "single": weights["single"],
            "config": "config.json",
            "knobs": family_module.KNOBS_FILE,
            "norm_stats": [family.norm_stats_file(name) for name in sorted(family.norm_stats)],
        }

    # -- intake -----------------------------------------------------------------------------

    def intake(
        self,
        commitments: list[Commitment],
        block: int,
        *,
        api: Any = None,
        fetch: Any = None,
    ) -> list[Entry]:
        """Take every new `horizon:` commitment into the round that was open when it was made."""
        lane = self.state.lane(LANE)
        entries, by_weights = lane["entries"], lane["by_weights"]
        changed = []
        for c in sorted(commitments, key=lambda c: (c.block, c.hotkey)):
            try:
                sub = commitment_.parse(c.data)
            except commitment_.CommitmentError:
                continue
            if sub.lane != commitment_.HORIZON:
                continue
            try:
                number = self.schedule.of_commitment(c.block).number
            except schedule_.NotYet:
                continue  # committed before the first round opened: it entered nothing
            key = f"{sub.repo}@{sub.revision}"
            known = entries.get(key)
            if known is not None and known["status"] != PENDING:
                continue
            record = known or {
                "hotkey": c.hotkey,
                "repo": sub.repo,
                "revision": sub.revision,
                "commit_block": c.block,
                "epoch": number,
                "status": PENDING,
                "reason": "",
            }
            try:
                found = hub.inspect(
                    sub.repo,
                    sub.revision,
                    self.shape,
                    api=api,
                    token=self.hub_token,
                    fetch=fetch,
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
            record["weights_sha256"] = found.content_key
            owner = by_weights.get(found.content_key)
            if owner is not None and owner != key:
                record.update(status="duplicate", reason=f"the same weights as {owner}")
            else:
                by_weights[found.content_key] = key
                for other_key, other in entries.items():
                    if (
                        other_key != key
                        and other["hotkey"] == c.hotkey
                        and other.get("epoch") == number
                        and other["status"] in (PENDING, QUEUED)
                    ):
                        other.update(status="superseded", reason=f"replaced by {key}")
                record.update(status=QUEUED, reason="")
            entries[key] = record
            changed.append(self._entry(key, record))
        if changed:
            self.state.save()
        return changed

    def entrants(self, number: int) -> list[Entry]:
        """The submissions round `number` holds, oldest commitment first."""
        lane = self.state.lane(LANE)
        waiting = [
            self._entry(key, record)
            for key, record in lane["entries"].items()
            if record["status"] == QUEUED and int(record.get("epoch", -1)) == number
        ]
        return sorted(waiting, key=lambda e: (e.commit_block, e.hotkey))

    # -- the validator's view ----------------------------------------------------------------

    def step(self, chain: Any) -> Progress:
        """One verb of the round whose window has closed, or a word about the one still open.

        A round runs after its window closes and its seed block is final, so there is only ever
        one to work on: the window before this one. Each call runs a single verb, because one of
        them can take hours and the loop must be able to die between any two.
        """
        head = chain.block()
        try:
            window = self.schedule.at(head)
        except schedule_.NotYet as exc:
            return Progress(LANE, WAITING, str(exc))
        entrants = self.entrants(window.number)
        open_now = (
            f"round {window.number} is open until block {window.closes} "
            f"({window.closes - head} to go), {len(entrants)} entrants"
        )
        if window.number == 0:
            return Progress(LANE, IDLE, f"{open_now}; none has closed yet")
        running = self.schedule.window(window.number - 1)
        if not running.seed_ready(head):
            return Progress(
                LANE, WAITING, f"{open_now}; round {running.number}'s seed is not final"
            )
        engine = self.engine()
        if engine is None:
            return Progress(LANE, IDLE, f"{open_now}; no engine configured to score them")
        return self._run(engine, running, chain, open_now)

    def engine(self) -> rounds_.Engine | None:
        """How to call the engine on this host, or None when none is configured."""
        if not self.cfg.competition:
            return None
        return rounds_.Engine(
            python=self.cfg.engine_python or sys.executable,
            config=self.cfg.competition,
            store=self.cfg.store,
            keys=self.cfg.key.parent,
            rounds=self.cfg.rounds,
            models=self.cfg.models,
            runtime_python=self.cfg.runtime_python,
            serve_as=self.cfg.serve_as,
            devices=self.cfg.devices,
        )

    def _run(
        self, engine: rounds_.Engine, window: schedule_.Window, chain: Any, open_now: str
    ) -> Progress:
        """Run the next verb of `window`'s round, and note it when it finishes."""
        name = ROUND_ID.format(number=window.number)
        lane = self.state.lane(LANE)
        noted = (lane.get("epochs") or {}).get(name) or {}
        plan = rounds_.Plan(
            engine.directory(name), stage=str(noted.get("stage", "")), dry_run=self.cfg.dry_run
        )
        verb = plan.next()
        if verb is None:
            return Progress(LANE, IDLE, f"{open_now}; round {window.number} is finished")
        after = None
        if verb == "open":
            if not engine.store_ready:
                log.info("horizon: making the signed store at %s", engine.store)
                made = subprocess.run(engine.store_argv(), check=False)
                if made.returncode != 0:
                    return Progress(LANE, FAILED, f"store init exited {made.returncode}")
            self._register(engine, window)
            after = self._previous(engine, window)
        argv = engine.argv(
            verb, name, profile=self.cfg.profile, dry_run=self.cfg.dry_run, after=after
        )
        log.info("horizon: round %s: %s", name, verb)
        done = subprocess.run(argv, capture_output=False, check=False)
        if done.returncode != 0:
            return Progress(LANE, FAILED, f"round {name}: {verb} exited {done.returncode}")
        with self.state.writing(LANE) as doc:
            doc.setdefault("epochs", {}).setdefault(name, {})["stage"] = verb
        if verb == "close" and not self.cfg.dry_run:
            self._won(engine, window, name)
        return Progress(LANE, WORKED, f"round {name}: {verb}")

    @staticmethod
    def _previous(engine: rounds_.Engine, window: schedule_.Window) -> Path | None:
        """The round before `window`'s, when it was opened here: the one this round follows on.

        A round opened after it starts where it closed, as the chain's windows do. Without one - the
        first round, or one after a week this validator never opened - there is nothing to follow,
        and the round opens when it is opened.
        """
        if window.number < 1:
            return None
        directory = engine.directory(ROUND_ID.format(number=window.number - 1))
        return directory if rounds_.opened(directory) else None

    @staticmethod
    def _closed(engine: rounds_.Engine, name: str) -> dict[str, Any] | None:
        """Round `name`'s signed close record, under whichever name it was published; None, and
        said, when there is none to read or it is another round's."""
        from horizon_competition import store as store_

        failures = []
        for spelling in CLOSE_RECORDS:
            published = spelling.format(id=name)
            try:
                entry = store_.read(engine.store, published)
            except Exception as exc:  # noqa: BLE001 - a missing record is reported, not raised
                failures.append(str(exc))
                continue
            record = entry.get("record") or entry
            said = record.get("round_id", record.get("epoch_id"))
            if said is not None and str(said) != name:
                log.warning("horizon: %s is round %s's record, not %s's", published, said, name)
                return None
            return record
        log.warning(
            "horizon: round %s closed but its record cannot be read: %s", name, "; ".join(failures)
        )
        return None

    def _won(self, engine: rounds_.Engine, window: schedule_.Window, name: str) -> None:
        """Read who the closed round ranked first, and note the hotkey that committed them.

        The winner is read back from the signed record, never from what this validator thought was
        happening while the round ran: the record is the thing every other reader can check, and a
        weight vector built from anything else could not be checked at all. A dry run publishes no
        record and pays nobody, which is what makes it a rehearsal.
        """
        from horizon_competition import submissions

        record = self._closed(engine, name)
        if record is None:
            return
        ranked = (record.get("scores") or {}).get("rank") or []
        if not ranked:
            log.info("horizon: round %s ranked nobody", name)
            return
        first = str(ranked[0])
        entries = self.state.lane(LANE)["entries"]
        hotkey = next(
            (
                one.get("hotkey")
                for one in entries.values()
                if submissions.key_for(str(one["repo"]), str(one["revision"])) == first
            ),
            None,
        )
        if hotkey is None:
            log.warning("horizon: round %s was won by %s, which no commitment names", name, first)
            return
        with self.state.writing(LANE) as doc:
            winners = doc.setdefault("winners", [])
            if not any(w.get("epoch") == window.number for w in winners):
                winners.append({"epoch": window.number, "key": first, "hotkey": hotkey})
        log.info("horizon: round %s was won by %s (%s)", name, first, hotkey)

    def _register(self, engine: rounds_.Engine, window: schedule_.Window) -> None:
        """Put the round's entrants into the register the engine reads.

        The chain is this competition's gate: what a hotkey committed, and the content key intake
        worked out from the Hub, are what the engine signs into its store. Nothing here recomputes
        that hash, and nothing takes one from a sender.
        """
        from horizon_competition import submissions

        for entry in self.entrants(window.number):
            record = self.state.lane(LANE)["entries"].get(entry.key) or {}
            submissions.add(
                engine.store,
                store_dir=engine.store,
                key_dir=engine.keys,
                participant=entry.hotkey,
                repo=entry.repo,
                revision=entry.revision,
                weights_sha256=str(record.get("weights_sha256", "")),
                source="chain",
            )

    def award(self) -> Award:
        """Winner takes all, per round: the newest closed round's winner and nobody else.

        Until a round has closed there is nobody, so the lane's share burns - which is why the
        config makes a share above zero say `serve_as` out loud.
        """
        lane = self.state.lane(LANE)
        winners = lane.get("winners") or []
        return Award([w.get("hotkey") for w in reversed(winners)], keep=1, decay=0.0)

    def snapshot(self) -> dict[str, Any]:
        lane = self.state.lane(LANE)
        window = self.schedule.window(int(lane.get("epoch", 0)))
        return {
            "schedule": {
                "genesis_block": self.cfg.genesis_block,
                "window_blocks": self.cfg.window_blocks,
                "round": window.number,
                "opens": window.opens,
                "closes": window.closes,
                "seed_block": window.seed_block,
            },
            "share": self.cfg.share,
            "serve_as": self.cfg.serve_as,
            "scoring": "staged: the chain side is wired, no round is scored yet",
            "entrants": [e.__dict__ for e in self.entrants(window.number)],
            "entries": lane["entries"],
        }

    @staticmethod
    def _entry(key: str, record: dict[str, Any]) -> Entry:
        return Entry(
            key=key,
            hotkey=str(record["hotkey"]),
            repo=str(record["repo"]),
            revision=str(record["revision"]),
            commit_block=int(record["commit_block"]),
            status=str(record["status"]),
            round=int(record.get("epoch", 0)),
        )
