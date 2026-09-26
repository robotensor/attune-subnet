"""Rounds are the chain's: two validators at the same head must agree without talking."""

import hashlib

import pytest

from robotensor.protocol.schedule import (
    NotYet,
    Schedule,
    commitment_of,
    entropy,
)

WEEK = 7 * 24 * 60 * 60 // 12  # a week of 12-second blocks


def test_windows_are_back_to_back_so_every_commitment_lands_in_one_round():
    schedule = Schedule(genesis_block=1000, window_blocks=100)

    first, second = schedule.window(0), schedule.window(1)

    assert (first.opens, first.closes) == (1000, 1100)
    assert second.opens == first.closes
    assert first.holds(1000) and first.holds(1099)
    assert not first.holds(1100) and second.holds(1100)


def test_two_validators_at_the_same_head_compute_the_same_round():
    """No coordination, no clock, nobody to ask: the head is the whole input."""
    one, other = Schedule(1000, WEEK), Schedule(1000, WEEK)
    head = 1000 + 3 * WEEK + 17

    assert one.at(head) == other.at(head)
    assert one.at(head).number == 3


def test_before_the_first_round_there_is_none():
    with pytest.raises(NotYet, match="opens at block 1000"):
        Schedule(1000, 100).at(999)


def test_the_seed_block_is_after_the_window_closed():
    """Nobody may know what they will be scored on while they can still commit."""
    window = Schedule(1000, 100).window(0)

    assert window.seed_block > window.closes
    assert not window.holds(window.seed_block)


def test_the_seed_is_read_only_once_it_is_final():
    window = Schedule(1000, 100, seed_delay=3).window(0)

    assert not window.seed_ready(window.seed_block)  # it exists, but is not settled
    assert window.seed_ready(window.seed_block + 3)


def test_a_commitment_belongs_to_the_window_that_was_open_when_it_was_made():
    """However long the validator took to read it."""
    schedule = Schedule(1000, 100)

    assert schedule.of_commitment(1150).number == 1
    assert schedule.of_commitment(1150) == schedule.window(1)


def test_the_units_come_from_the_secret_and_the_block_together():
    """The organiser cannot grind the secret - the block will not exist for a week - and a block
    author grinding the hash is grinding blind."""
    block_hash = "0x" + "ab" * 32

    with_secret = entropy("horizon-competition/units/4", "s3cret", block_hash)
    other_secret = entropy("horizon-competition/units/4", "other", block_hash)
    other_block = entropy("horizon-competition/units/4", "s3cret", "0x" + "cd" * 32)

    assert len({with_secret, other_secret, other_block}) == 3
    assert len(with_secret) == 64


def test_a_change_to_how_units_are_drawn_gives_different_units():
    block_hash = "0x" + "ab" * 32

    assert entropy("units/4", "s", block_hash) != entropy("units/5", "s", block_hash)


def test_the_hash_does_not_depend_on_how_the_block_hash_was_written():
    assert entropy("d", "s", "0x" + "AB" * 32) == entropy("d", "s", "ab" * 32)


def test_the_open_record_commits_to_the_secret_without_revealing_it():
    secret = "s3cret"

    assert commitment_of(secret) == hashlib.sha256(secret.encode()).hexdigest()
    assert secret not in commitment_of(secret)


@pytest.mark.parametrize("bad", [{"window_blocks": 0}, {"genesis_block": -1}, {"seed_delay": -1}])
def test_a_schedule_nobody_could_follow_is_refused(bad):
    with pytest.raises(ValueError):
        Schedule(**{"genesis_block": 10, "window_blocks": 100, **bad})
