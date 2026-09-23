"""When an epoch opens, closes, and what decides its units. All of it is the chain's.

Horizon runs in epochs: a window in which miners commit, then one frozen pool of demonstrations
every entrant is scored on. The window is counted in blocks, not days, for one reason - two
validators at the same head must compute the same epoch and the same seed block with no
coordination between them, no clock, and nobody to ask. A cadence written in a config file would
be a second source of truth, and two validators whose files disagreed would publish two different
epochs under one number.

    number     = (head - genesis_block) // window_blocks
    opens      = genesis_block + number * window_blocks
    closes     = opens + window_blocks          (the next window opens at the same block)
    seed_block = closes + seed_delay

Windows are back to back, so every commitment lands in exactly one epoch: the one open at its
block. The seed block is after the window closed, so no entrant can know what their model will be
scored on while they may still commit; `seed_delay` keeps it far enough behind the head to be
final.

**The entropy is two things, and neither is enough alone.** The organiser draws a secret `S` and
commits to its hash when the epoch opens; the units are drawn from `sha256(derivation | S |
block_hash(seed_block))`. The organiser cannot grind `S`, because the block that decides the units
will not exist for a week. A block author cannot grind the hash, because without `S` they are
grinding blind. A miner cannot predict either. What the close record publishes - `S`, the seed
block and its hash - lets anyone redraw the units; a reader with a node gets the further check
that the hash really is that block's, which is a different and stronger claim, and the two must
never be written as though they were one.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .seed import FINALITY, normalize_hash

#: How far behind the close the seed block sits, so it is final when it is read.
SEED_DELAY = FINALITY


class NotYet(RuntimeError):
    """The chain has not reached the block this needs."""


@dataclass(frozen=True)
class Window:
    """One epoch's place on the chain."""

    number: int
    opens: int
    closes: int
    seed_delay: int = SEED_DELAY

    @property
    def seed_block(self) -> int:
        """The block whose hash, with the organiser's secret, decides the units."""
        return self.closes + self.seed_delay

    def holds(self, block: int) -> bool:
        """Whether a commitment made at `block` belongs to this epoch."""
        return self.opens <= block < self.closes

    def seed_ready(self, head: int) -> bool:
        """Whether the seed block exists and is far enough behind the head to be read."""
        return head - FINALITY >= self.seed_block


@dataclass(frozen=True)
class Schedule:
    """The epochs of one competition, as the chain lays them out."""

    genesis_block: int
    window_blocks: int
    seed_delay: int = SEED_DELAY

    def __post_init__(self) -> None:
        if self.window_blocks < 1:
            raise ValueError(f"an epoch is at least one block, not {self.window_blocks}")
        if self.genesis_block < 0 or self.seed_delay < 0:
            raise ValueError("the genesis block and the seed delay are not negative")

    def number_at(self, block: int) -> int:
        """Which epoch is open at `block`; before genesis there is none."""
        if block < self.genesis_block:
            raise NotYet(f"the first epoch opens at block {self.genesis_block}, not yet at {block}")
        return (block - self.genesis_block) // self.window_blocks

    def window(self, number: int) -> Window:
        if number < 0:
            raise ValueError(f"there is no epoch {number}")
        opens = self.genesis_block + number * self.window_blocks
        return Window(number, opens, opens + self.window_blocks, self.seed_delay)

    def at(self, block: int) -> Window:
        """The epoch open at `block`."""
        return self.window(self.number_at(block))

    def of_commitment(self, block: int) -> Window:
        """The epoch a commitment made at `block` entered. The same call as `at`, named for what
        it decides: a commitment belongs to the window that was open when it was made, and to no
        other, however long the validator took to read it."""
        return self.at(block)


def entropy(derivation: str, secret: str, block_hash: str) -> str:
    """What the units are drawn from: the organiser's secret and the seed block's hash, together.

    `derivation` is the engine's own label for how it draws; it is in here so that a change to the
    drawing gives different units for the same secret and the same block, rather than quietly
    reusing a published one.
    """
    material = f"{derivation}|{secret}|{normalize_hash(block_hash)}".encode()
    return hashlib.sha256(material).hexdigest()


def commitment_of(secret: str) -> str:
    """What the open record publishes instead of the secret: its hash, so that the secret it
    reveals at close can be held to what it promised at open."""
    return hashlib.sha256(secret.encode()).hexdigest()
