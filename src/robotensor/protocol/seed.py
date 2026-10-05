"""Which chain block seeds a duel.

A duel's units are drawn from its id and a block hash (`vector_orchestrator` `DuelRequest.entropy`).
The block is the chain's finalized head when the duel starts - GRANDPA has finalized it, so its
hash can never change - and it must come after the challenger's commitment: then nobody, the
challenger included, could know the hash when the challenger committed, so nobody could have
trained for, or picked, its units. The block and its hash are read together from the chain by the
validator, never taken from a miner, and published with the duel, so anyone can read the hash back
and derive the units again.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Blocks behind the head that Horizon's schedule waits before it reads a seed block
#: (`protocol.schedule`); a Vector duel reads the finalized head instead.
FINALITY = 3


@dataclass(frozen=True)
class Seed:
    block: int
    block_hash: str


class NotYet(RuntimeError):
    """The chain has not moved far enough past the commitment to seed its duel."""


def seed_block(finalized: int, committed_at: int) -> int:
    """The block whose hash seeds a duel starting when the chain's finalized head is `finalized`,
    for a commitment made at `committed_at`: the finalized head itself; `NotYet` while it does not
    come after the commitment."""
    if finalized <= committed_at:
        raise NotYet(
            f"the finalized head {finalized} is not past the commitment at {committed_at} yet: the "
            "seed block must come after it"
        )
    return finalized


def normalize_hash(value: str) -> str:
    """A block hash as `0x` + 64 lowercase hex; `ValueError` for anything else."""
    text = str(value).strip().lower()
    if not text.startswith("0x"):
        text = "0x" + text
    if len(text) != 66 or any(c not in "0123456789abcdef" for c in text[2:]):
        raise ValueError(f"{value!r} is not a block hash")
    return text
