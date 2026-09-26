"""Driving one Horizon round, one verb at a time.

The engine does the work - open, pool, screen, shortlist, evaluate, score, close - and each of
those verbs takes minutes to hours. The validator's loop must be able to die between any two of
them and pick the round up where it was left, so nothing here keeps a plan in memory: what has
been done is read from the round's own directory, and the two verbs that leave no single file
behind are read from the lane's note of the last one that finished.

Re-running a verb is safe. The engine's drains skip what is already there, and its `open`, `pool`
and `score` refuse to overwrite what they wrote, so a note one step stale costs a re-scan and
never a wrong result. That is the whole reason the note is allowed to be an optimisation rather
than the truth.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

#: The engine's files that say a verb finished.
ROUND_JSON = "round.json"
#: What `open` left before the engine called its rounds epochs no more. A round directory is read
#: by what is in it: one holding this was opened, and is carried on, not opened again.
EPOCH_JSON = "epoch.json"
POOL_MANIFEST_JSON = "pool_manifest.json"
SHORTLIST_JSON = "shortlist.json"
SCORES_JSON = "scores.json"
DRY_RUN_SCORES_JSON = "scores-dry-run.json"

#: The order a round is run in. `screen` and `full` are the engine's two drains: the screening
#: stage every entrant runs, and the full stage the shortlist runs.
STEPS = ("open", "pool", "screen", "shortlist", "full", "score", "close")
#: What each verb leaves behind, where it leaves one file; any one of them says it ran.
MARKERS = {
    "open": (ROUND_JSON, EPOCH_JSON),
    "pool": (POOL_MANIFEST_JSON,),
    "shortlist": (SHORTLIST_JSON,),
}


@dataclass(frozen=True)
class Plan:
    """Where one round has got to, and what to run next.

    `stage` is the last verb the lane noted as finished; it decides only the two drains, which
    leave no file of their own.
    """

    directory: Path
    stage: str = ""
    #: Drains that publish no live result, a close that publishes nothing and a score that writes
    #: beside the real one: what a smoke round does, so a rehearsal can be run against a store
    #: nobody has to throw away.
    dry_run: bool = False
    steps: Sequence[str] = STEPS

    def markers(self, name: str) -> tuple[str, ...]:
        """The files `name` leaves behind; empty for a verb that leaves none of its own."""
        if name == "score":
            return (DRY_RUN_SCORES_JSON if self.dry_run else SCORES_JSON,)
        return MARKERS.get(name, ())

    def finished(self, name: str) -> bool:
        """Whether `name` has run: from what it left behind, or from the note."""
        markers = self.markers(name)
        if markers:
            return any((self.directory / marker).is_file() for marker in markers)
        if not self.stage or self.stage not in self.steps:
            return False
        return self.steps.index(name) <= self.steps.index(self.stage)

    def next(self) -> str | None:
        """The verb to run now, or None when the round is finished."""
        for name in self.steps:
            if not self.finished(name):
                return name
        return None

    def at(self) -> str:
        """Where the round is, in one phrase."""
        following = self.next()
        return "finished" if following is None else f"next: {following}"


@dataclass(frozen=True)
class Engine:
    """How to call the competition's engine on this host."""

    python: str
    config: Path
    store: Path
    keys: Path
    #: Where each round's directory is made: `<rounds>/<id>`.
    rounds: Path
    models: Path
    runtime_python: str = ""
    serve_as: str = ""
    devices: tuple[int, ...] = ()

    def directory(self, round_id: str) -> Path:
        return self.rounds / round_id

    def store_argv(self) -> list[str]:
        """`store init`: the signed store and the key that signs it, made once."""
        return [
            self.python,
            "-m",
            "horizon_competition.cli",
            "store",
            "init",
            "--store",
            str(self.store),
            "--keys",
            str(self.keys),
        ]

    @property
    def store_ready(self) -> bool:
        return (self.store / "index.json").is_file()

    def argv(
        self, verb: str, round_id: str, *, profile: str = "", dry_run: bool = False
    ) -> list[str]:
        """The command for one verb, as the engine's own CLI takes it: `round <verb> --round
        <directory>`. The engine keeps no `epoch` spelling of its commands."""
        directory = str(self.directory(round_id))
        head = [self.python, "-m", "horizon_competition.cli", "round"]
        if verb == "open":
            argv = [
                *head,
                "open",
                "--config",
                str(self.config),
                "--round",
                directory,
                "--store",
                str(self.store),
                "--keys",
                str(self.keys),
                "--register",
                str(self.store),
            ]
            if self.runtime_python:
                argv += ["--runtime-python", self.runtime_python]
            return [*argv, "--profile", profile] if profile else argv
        if verb == "pool":
            return [*head, "pool", "--round", directory]
        if verb in ("screen", "full"):
            argv = [
                *head,
                "drain",
                "--round",
                directory,
                "--models",
                str(self.models),
                "--register",
                str(self.store),
            ]
            if self.runtime_python:
                argv += ["--runtime", self.runtime_python]
            if self.serve_as:
                argv += ["--serve-as", self.serve_as]
            if self.devices:
                argv += ["--gpus", ",".join(str(d) for d in self.devices)]
            if not dry_run:
                # Each model's result is signed into the store the moment it finishes a stage,
                # rather than a week later at close: provisional, and the close record stays the
                # authority. A rehearsal publishes none, because a result is published once and
                # would take the name the real round's result needs.
                argv += ["--store", str(self.store), "--keys", str(self.keys)]
            return [*argv, "--screen-only"] if verb == "screen" else argv
        if verb == "shortlist":
            return [*head, "shortlist", "--round", directory]
        if verb == "score":
            argv = [*head, "score", "--round", directory, "--register", str(self.store)]
            return [*argv, "--dry-run"] if dry_run else argv
        if verb == "close":
            argv = [
                *head,
                "close",
                "--round",
                directory,
                "--store",
                str(self.store),
                "--keys",
                str(self.keys),
            ]
            return [*argv, "--dry-run"] if dry_run else argv
        raise ValueError(f"{verb} is not one of {', '.join(STEPS)}")
