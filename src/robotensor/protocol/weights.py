"""How the validator's weight vector follows from the lanes' champions.

The subnet spec: Competition 1 (Vector) takes a share of miner emissions (30%), and within it the
four most recent champion models are paid 40%, 30%, 20% and 10%, newest first. Until the other
lanes launch, what no lane claims goes to the burn UID (configurable: `lanes` and `burn` in the
validator's config).

Rules, all keyed by hotkey and mapped to UIDs only at the end, because UIDs are recycled:

- A lane's champions are the miner models it crowned, newest first, one entry per crowned model.
  The organizer's genesis baseline is not a miner's model and is never an entry. The lane's newest
  `len(split)` entries share its share by `split`; a hotkey that holds two of them gets both parts.
  With fewer champions than `split` has places, the places that are filled share the whole share
  in the same proportions.
- An entry whose hotkey is no longer registered gives its part to the burn UID rather than to the
  other champions: a champion's reward does not grow because another one left.
- A champion vacated as king because its repository was gone (`BURNED`) keeps its place, and its
  part goes to the burn UID: the emission it would have earned is burned, not handed on.
- Everything not given to a champion goes to the burn UID. Weights sum to 1.

Two cadences, one formula. `split = (0.4, 0.3, 0.2, 0.1)` is Vector's pool. `split = (1.0,)` is
Horizon's round: the winner takes all, and if that winner deregisters the lane's whole share burns
until the next round closes - a real cliff, and the reason the split is a knob rather than a
constant.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

#: The subnet spec's reward pool: the four most recent champions, newest first.
CHAMPION_SPLIT = (0.40, 0.30, 0.20, 0.10)
#: A champion's place whose part is burned: a king vacated because its repository was gone. Never
#: an ss58 address, so it never maps to a UID.
BURNED = "<burned>"


@dataclass(frozen=True)
class Lane:
    name: str
    #: The lane's share of the miner emissions, in [0, 1].
    share: float
    #: Its champions' hotkeys, newest first; None for a baseline entry, which is skipped, and
    #: `BURNED` for a place whose part burns.
    champions: Sequence[str | None]
    #: Each paid place's part of the lane's share, newest first; its length is how many are paid.
    split: Sequence[float] = CHAMPION_SPLIT

    def __post_init__(self) -> None:
        if not self.split or any(part <= 0 for part in self.split):
            raise ValueError(f"{self.name}: split must be non-empty and positive, not {self.split}")
        if abs(sum(self.split) - 1.0) > 1e-9:
            raise ValueError(f"{self.name}: split must sum to 1, not {sum(self.split)}")


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
        entries = [hotkey for hotkey in lane.champions if hotkey is not None][: len(lane.split)]
        if not entries:
            burned += lane.share
            continue
        parts = lane.split[: len(entries)]
        scale = lane.share / sum(parts)
        for hotkey, part in zip(entries, parts, strict=True):
            uid = None if hotkey == BURNED else uids.get(hotkey)
            if uid is None:
                burned += part * scale
            else:
                weights[uid] = weights.get(uid, 0.0) + part * scale
    weights[burn_uid] = weights.get(burn_uid, 0.0) + burned
    norm = sum(weights.values())
    return {uid: w / norm for uid, w in sorted(weights.items()) if w > 0}
