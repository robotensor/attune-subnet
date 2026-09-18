"""How the validator's weight vector follows from the lanes' champions.

The subnet spec: Competition 1 (Vector) takes a share of miner emissions (30%), and within it the
five most recent champion models share equally. Until the other lanes launch, what no lane claims
goes to the burn UID (configurable: `lanes` and `burn` in the validator's config).

Rules, all keyed by hotkey and mapped to UIDs only at the end, because UIDs are recycled:

- A lane's champions are the miner models it crowned, newest first, one entry per crowned model.
  The organizer's genesis baseline is not a miner's model and is never an entry. The five newest
  entries share the lane's share equally; a hotkey that holds two of them gets two parts.
- An entry whose hotkey is no longer registered gives its part to the burn UID rather than to the
  other champions: a champion's reward does not grow because another one left.
- Everything not given to a champion goes to the burn UID. Weights sum to 1.
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


def weight_vector(
    lanes: Sequence[Lane],
    uids: Mapping[str, int],
    burn_uid: int,
    champions: int = CHAMPIONS,
) -> dict[int, float]:
    """`{uid: weight}` summing to 1: each lane's share split over its newest `champions` entries
    that are registered hotkeys, and the rest to `burn_uid`."""
    total = sum(lane.share for lane in lanes)
    if any(lane.share < 0 for lane in lanes) or total > 1 + 1e-9:
        raise ValueError(f"lane shares must be non-negative and sum to at most 1, not {total}")
    weights: dict[int, float] = {}
    burned = 1.0 - total
    for lane in lanes:
        entries = [hotkey for hotkey in lane.champions if hotkey is not None][:champions]
        if not entries:
            burned += lane.share
            continue
        part = lane.share / len(entries)
        for hotkey in entries:
            uid = uids.get(hotkey)
            if uid is None:
                burned += part
            else:
                weights[uid] = weights.get(uid, 0.0) + part
    weights[burn_uid] = weights.get(burn_uid, 0.0) + burned
    norm = sum(weights.values())
    return {uid: w / norm for uid, w in sorted(weights.items()) if w > 0}
