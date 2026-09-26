"""How the validator's weight vector follows from the lanes' champions.

The subnet spec: Competition 1 (Vector) takes a share of miner emissions (30%), and within it the
five most recent champion models share equally. Until the other lanes launch, what no lane claims
goes to the burn UID (configurable: `lanes` and `burn` in the validator's config).

Rules, all keyed by hotkey and mapped to UIDs only at the end, because UIDs are recycled:

- A lane's champions are the miner models it crowned, newest first, one entry per crowned model.
  The organizer's genesis baseline is not a miner's model and is never an entry. The lane's newest
  `entries` share its share, weighted by `decay` to the power of how far back they are; a hotkey
  that holds two of them gets two parts.
- An entry whose hotkey is no longer registered gives its part to the burn UID rather than to the
  other champions: a champion's reward does not grow because another one left.
- Everything not given to a champion goes to the burn UID. Weights sum to 1.

Two cadences, one formula. `decay = 1.0, entries = 5` is Vector's pool: five champions, equal
parts. `decay = 0.0, entries = 1` is Horizon's round: the winner takes all, and if that winner
deregisters the lane's whole share burns until the next round closes - a real cliff, and the
reason these are knobs rather than constants.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

#: The subnet spec's reward pool: the five most recent champions.
CHAMPIONS = 5


@dataclass(frozen=True)
class Lane:
    name: str
    #: The lane's share of the miner emissions, in [0, 1].
    share: float
    #: Its champions' hotkeys, newest first; None for a baseline entry, which is skipped.
    champions: Sequence[str | None]
    #: How many of the newest entries are paid at all.
    entries: int = CHAMPIONS
    #: What each step back is worth, as a factor: 1.0 pays them equally, 0.0 pays only the newest.
    decay: float = 1.0

    def __post_init__(self) -> None:
        if self.entries < 1:
            raise ValueError(f"{self.name}: entries must be at least 1, not {self.entries}")
        if not 0.0 <= self.decay <= 1.0:
            raise ValueError(f"{self.name}: decay must be in [0, 1], not {self.decay}")


def weight_vector(
    lanes: Sequence[Lane],
    uids: Mapping[str, int],
    burn_uid: int,
) -> dict[int, float]:
    """`{uid: weight}` summing to 1: each lane's share split over its newest entries that are
    registered hotkeys, and the rest to `burn_uid`."""
    total = sum(lane.share for lane in lanes)
    if any(lane.share < 0 for lane in lanes) or total > 1 + 1e-9:
        raise ValueError(f"lane shares must be non-negative and sum to at most 1, not {total}")
    weights: dict[int, float] = {}
    burned = 1.0 - total
    for lane in lanes:
        entries = [hotkey for hotkey in lane.champions if hotkey is not None][: lane.entries]
        if not entries:
            burned += lane.share
            continue
        # `decay ** 0` is 1 even for decay 0.0, so a decay of zero pays the newest entry and
        # nothing else, and a decay of one pays every entry the same: one formula, both cadences.
        parts = [lane.decay**step for step in range(len(entries))]
        scale = lane.share / sum(parts)
        for hotkey, part in zip(entries, parts, strict=True):
            uid = uids.get(hotkey)
            if uid is None:
                burned += part * scale
            else:
                weights[uid] = weights.get(uid, 0.0) + part * scale
    weights[burn_uid] = weights.get(burn_uid, 0.0) + burned
    norm = sum(weights.values())
    return {uid: w / norm for uid, w in sorted(weights.items()) if w > 0}
