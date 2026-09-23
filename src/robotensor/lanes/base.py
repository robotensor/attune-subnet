"""What the validator knows about a competition, whatever its cadence.

Vector duels a challenger against the reigning king; Horizon opens an epoch, freezes a pool of
demonstrations, scores every entrant on it and closes. The validator drives both the same way -
take in what the chain says, do one piece of work, say who should be paid - and knows nothing of
duels or epochs. Everything that does is behind this shape.

Two rules make that possible:

- **One `step()` is one resumable piece of work**: a whole duel for Vector, one epoch verb for
  Horizon. Killed halfway, the next `step()` picks the same work up where the engine left it, so
  the loop never has to remember what it was doing.
- **A lane never raises its engine's exceptions.** The loop cannot catch what it cannot import,
  and a validator running one competition does not install the other's engine. A lane catches its
  own and returns a `Progress` saying what happened. `tests/test_lane.py` holds it to that.

`doctor()` and the lane's own subcommands are not here yet: they arrive with the one `robotensor`
command, and adding them to this protocol before there is anything to call would be a promise
without a caller.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..protocol.weights import CHAMPIONS

#: A step did work that moved the competition on: a duel published, an epoch verb finished.
WORKED = "worked"
#: There was nothing to do. Everything the chain has offered is settled.
IDLE = "idle"
#: The work exists but cannot start yet - a seed block that is not final, a window still open.
WAITING = "waiting"
#: The step failed for a reason that is nobody's loss: the entry stays where it was and the next
#: step tries again. A harness failure is never a result.
FAILED = "failed"
OUTCOMES = (WORKED, IDLE, WAITING, FAILED)
#: The outcomes the loop should rest after, rather than stepping straight back in.
RESTING = (IDLE, WAITING, FAILED)


@dataclass(frozen=True)
class Progress:
    """What one `step()` did, in words the loop can act on without knowing the competition."""

    lane: str
    outcome: str
    detail: str = ""
    #: What was published, when something was: the engine's own record, for logging alone.
    record: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.outcome not in OUTCOMES:
            raise ValueError(f"{self.outcome!r} is not one of {', '.join(OUTCOMES)}")

    @property
    def resting(self) -> bool:
        """Whether the loop has nothing better to do than wait before stepping again."""
        return self.outcome in RESTING

    def __str__(self) -> str:
        return f"{self.lane}: {self.outcome}{f': {self.detail}' if self.detail else ''}"


@dataclass(frozen=True)
class Award:
    """Who a lane says should be paid, newest first, and how its cadence spreads the share.

    `entries` are hotkeys; `None` is an entry that belongs to no miner (the organizer's genesis
    baseline), which is skipped rather than paid. `keep` and `decay` are the knobs of
    `protocol.weights`: five entries paid equally is Vector's pool, one entry is Horizon's
    winner-takes-all.
    """

    entries: Sequence[str | None] = field(default_factory=list)
    keep: int = CHAMPIONS
    decay: float = 1.0


@runtime_checkable
class Lane(Protocol):
    """The whole of what the validator asks of a competition."""

    #: The name the chain's commitments carry and the config's `[lanes.<name>]` table uses.
    name: str

    def intake(self, commitments: Sequence[Any], block: int, *, api: Any = None) -> list[Any]:
        """Take every commitment of this lane into it; the entries that changed."""

    def step(self, chain: Any) -> Progress:
        """Do one resumable piece of work, or say why there is none. Never raises the engine's."""

    def award(self) -> Award:
        """Who to pay, read back from what the lane published - never from `state.json`."""

    def snapshot(self) -> dict[str, Any]:
        """What `status` prints for this lane: where the competition stands, as JSON."""
