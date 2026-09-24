"""Driving one Horizon epoch, one verb at a time.

The engine does the work - open, pool, screen, shortlist, evaluate, score, close - and each of
those verbs takes minutes to hours. The validator's loop must be able to die between any two of
them and pick the epoch up where it was left, so nothing here keeps a plan in memory: what has
been done is read from the epoch's own directory, and the two verbs that leave no single file
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
EPOCH_JSON = "epoch.json"
POOL_MANIFEST_JSON = "pool_manifest.json"
SHORTLIST_JSON = "shortlist.json"
SCORES_JSON = "scores.json"
DRY_RUN_SCORES_JSON = "scores-dry-run.json"

#: The order an epoch is run in. `screen` and `full` are the engine's two drains: the screening
#: round every entrant runs, and the full round the shortlist runs.
STEPS = ("open", "pool", "screen", "shortlist", "full", "score", "close")
#: What each verb leaves behind, where it leaves one file.
MARKERS = {
    "open": EPOCH_JSON,
    "pool": POOL_MANIFEST_JSON,
    "shortlist": SHORTLIST_JSON,
}


@dataclass(frozen=True)
class Plan:
    """Where one epoch has got to, and what to run next.

    `stage` is the last verb the lane noted as finished; it decides only the two drains, which
    leave no file of their own.
    """

    directory: Path
    stage: str = ""
    #: A close that publishes nothing and a score that writes beside the real one: what a smoke
    #: epoch does, so a rehearsal can be run against a store nobody has to throw away.
    dry_run: bool = False
    steps: Sequence[str] = STEPS

    def marker(self, name: str) -> str | None:
        if name == "score":
            return DRY_RUN_SCORES_JSON if self.dry_run else SCORES_JSON
        return MARKERS.get(name)

    def finished(self, name: str) -> bool:
        """Whether `name` has run: from what it left behind, or from the note."""
        marker = self.marker(name)
        if marker is not None:
            return (self.directory / marker).is_file()
        if not self.stage or self.stage not in self.steps:
            return False
        return self.steps.index(name) <= self.steps.index(self.stage)

    def next(self) -> str | None:
        """The verb to run now, or None when the epoch is finished."""
        for name in self.steps:
            if not self.finished(name):
                return name
        return None

    def at(self) -> str:
        """Where the epoch is, in one phrase."""
        following = self.next()
        return "finished" if following is None else f"next: {following}"


@dataclass(frozen=True)
class Engine:
    """How to call the competition's engine on this host."""

    python: str
    config: Path
    store: Path
    keys: Path
    epochs: Path
    models: Path
    runtime_python: str = ""
    serve_as: str = ""
    devices: tuple[int, ...] = ()

    def directory(self, epoch: str) -> Path:
        return self.epochs / epoch

    def argv(self, verb: str, epoch: str, *, profile: str = "", dry_run: bool = False) -> list[str]:
        """The command for one verb, as the engine's own CLI takes it."""
        directory = str(self.directory(epoch))
        head = [self.python, "-m", "horizon_competition.cli", "epoch"]
        if verb == "open":
            argv = [
                *head,
                "open",
                "--config",
                str(self.config),
                "--epoch",
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
            return [*head, "pool", "--epoch", directory]
        if verb in ("screen", "full"):
            argv = [
                *head,
                "drain",
                "--epoch",
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
            return [*argv, "--screen-only"] if verb == "screen" else argv
        if verb == "shortlist":
            return [*head, "shortlist", "--epoch", directory]
        if verb == "score":
            argv = [*head, "score", "--epoch", directory, "--register", str(self.store)]
            return [*argv, "--dry-run"] if dry_run else argv
        if verb == "close":
            argv = [
                *head,
                "close",
                "--epoch",
                directory,
                "--store",
                str(self.store),
                "--keys",
                str(self.keys),
            ]
            return [*argv, "--dry-run"] if dry_run else argv
        raise ValueError(f"{verb} is not one of {', '.join(STEPS)}")
