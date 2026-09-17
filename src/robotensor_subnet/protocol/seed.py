"""Which chain block seeds a duel.

A duel's units are drawn from its id and a block hash (`icil_orchestrator` `DuelRequest.entropy`).
The block is the current one when the duel starts, less `finality` blocks so its hash is final, and
it must come after the challenger's commitment: then nobody - the challenger included - could know
the hash when the challenger committed, so nobody could have trained for, or picked, its units.
The hash is always read from the chain by the validator, never taken from a miner, and it is
published with the duel, so anyone can read it back and derive the units again.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Blocks behind the head a seed block sits, so its hash is final (GRANDPA finalizes within a few).
FINALITY = 3


@dataclass(frozen=True)
class Seed:
    block: int
    block_hash: str


class NotYet(RuntimeError):
    """The chain has not moved far enough past the commitment to seed its duel."""


def seed_block(current: int, committed_at: int, finality: int = FINALITY) -> int:
    """The block whose hash seeds a duel starting at `current` for a commitment made at
    `committed_at`; `NotYet` while that block would not come after the commitment."""
    block = current - finality
    if block <= committed_at:
        raise NotYet(
            f"block {current} is too close to the commitment at {committed_at}: the seed block must "
            f"come after it and be {finality} blocks behind the head"
        )
    return block


def normalize_hash(value: str) -> str:
    """A block hash as `0x` + 64 lowercase hex; `ValueError` for anything else."""
    text = str(value).strip().lower()
    if not text.startswith("0x"):
        text = "0x" + text
    if len(text) != 66 or any(c not in "0123456789abcdef" for c in text[2:]):
        raise ValueError(f"{value!r} is not a block hash")
    return text
