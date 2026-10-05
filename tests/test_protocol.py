"""The pure protocol: commitments, seeds and the weight vector."""

import pytest

from robotensor.protocol import commitment, seed
from robotensor.protocol.weights import Lane, weight_vector

SHA = "0123456789abcdef0123456789abcdef01234567"


def test_a_commitment_round_trips():
    data = commitment.encode("robotensor/vector-update1", SHA)
    assert data == f"vector:robotensor/vector-update1@{SHA}"
    sub = commitment.parse(data)
    assert (sub.lane, sub.repo, sub.revision) == ("vector", "robotensor/vector-update1", SHA)
    assert sub.entry == f"robotensor/vector-update1@{SHA}"


@pytest.mark.parametrize(
    ("repo", "revision"),
    [
        ("robotensor/model", "main"),  # a branch can change after the block that orders the queue
        ("robotensor/model", SHA[:12]),
        ("not-a-repo", SHA),
        ("o/" + "x" * 100, SHA),  # over the chain's 128 bytes
    ],
)
def test_what_cannot_be_a_commitment_is_refused(repo, revision):
    with pytest.raises(commitment.CommitmentError):
        commitment.encode(repo, revision)


@pytest.mark.parametrize(
    "data",
    [
        "",
        "vector1:o/n@" + SHA,  # the prefix Vector used before it was named,
        "vector:o/n@main",
        "vector:o/n",
        "vector:" + SHA,
        "hello",
        "vector:o/n@" + SHA + "0",
    ],
)
def test_what_is_not_a_vector_commitment_does_not_parse(data):
    with pytest.raises(commitment.CommitmentError):
        commitment.parse(data)


def test_the_seed_block_is_the_finalized_head_once_it_is_past_the_commitment():
    assert seed.seed_block(97, 90) == 97
    with pytest.raises(seed.NotYet):
        seed.seed_block(90, 90)  # the commitment's own block is not after it
    assert seed.normalize_hash("AB" * 32) == "0x" + "ab" * 32
    with pytest.raises(ValueError):
        seed.normalize_hash("0x1234")


def test_one_champion_takes_the_lane_share_and_the_rest_burns():
    w = weight_vector([Lane("vector", 0.30, ["B", None])], {"A": 0, "B": 5}, burn_uid=0)
    assert w == pytest.approx({0: 0.70, 5: 0.30})


def test_four_champions_are_paid_40_30_20_10_and_older_ones_drop_out():
    hotkeys = ["h1", "h2", "h3", "h4", "h5"]
    uids = {h: i + 1 for i, h in enumerate(hotkeys)} | {"owner": 0}
    w = weight_vector([Lane("vector", 0.30, hotkeys)], uids, burn_uid=0)
    assert w[0] == pytest.approx(0.70)
    assert [w[uids[h]] for h in hotkeys[:4]] == pytest.approx([0.12, 0.09, 0.06, 0.03])
    assert uids["h5"] not in w


def test_fewer_champions_than_places_share_the_whole_lane_in_proportion():
    w = weight_vector([Lane("vector", 0.30, ["a", "b"])], {"a": 1, "b": 2, "o": 0}, burn_uid=0)
    assert w[1] == pytest.approx(0.30 * 4 / 7) and w[2] == pytest.approx(0.30 * 3 / 7)
    assert w[0] == pytest.approx(0.70)


def test_a_hotkey_with_two_crowned_models_gets_two_parts():
    w = weight_vector([Lane("vector", 0.30, ["a", "b", "a"])], {"a": 1, "b": 2, "o": 0}, burn_uid=0)
    # 0.4 + 0.2 and 0.3 of 0.9, over the lane's 0.30.
    assert w[1] == pytest.approx(0.20) and w[2] == pytest.approx(0.10)


def test_a_deregistered_champion_burns_its_part_rather_than_growing_the_others():
    w = weight_vector([Lane("vector", 0.30, ["a", "gone"])], {"a": 1, "o": 0}, burn_uid=0)
    assert w[1] == pytest.approx(0.30 * 4 / 7) and w[0] == pytest.approx(0.70 + 0.30 * 3 / 7)


def test_no_champion_yet_burns_everything():
    assert weight_vector([Lane("vector", 0.30, [None])], {"o": 0}, burn_uid=0) == {0: 1.0}


def test_shares_over_one_are_refused():
    with pytest.raises(ValueError):
        weight_vector([Lane("a", 0.7, []), Lane("b", 0.5, [])], {}, burn_uid=0)


def test_winner_takes_all_pays_the_newest_entry_and_nothing_older():
    """Horizon's cadence: one round, one winner. The runners-up of a closed round are champions
    of earlier ones and are not paid again."""
    lane = Lane("horizon", 0.70, ["new", "older", "oldest"], split=(1.0,))

    w = weight_vector([lane], {"new": 1, "older": 2, "oldest": 3, "o": 0}, burn_uid=0)

    assert w == pytest.approx({0: 0.30, 1: 0.70})


def test_a_winner_who_deregistered_burns_the_whole_lane_share():
    """The cliff winner-takes-all has: with one entry there is nobody to fall back to, and the
    share burns until the next round closes."""
    lane = Lane("horizon", 0.70, ["gone", "older"], split=(1.0,))

    w = weight_vector([lane], {"older": 2, "o": 0}, burn_uid=0)

    assert w == pytest.approx({0: 1.0})


def test_the_default_split_is_the_champion_pool():
    hotkeys = ["h1", "h2", "h3", "h4"]
    uids = {h: i + 1 for i, h in enumerate(hotkeys)} | {"o": 0}

    plain = weight_vector([Lane("vector", 0.30, hotkeys)], uids, burn_uid=0)
    spelled_out = weight_vector(
        [Lane("vector", 0.30, hotkeys, split=(0.4, 0.3, 0.2, 0.1))], uids, burn_uid=0
    )

    assert plain == spelled_out


@pytest.mark.parametrize("bad", [(), (0.5, 0.4), (1.2, -0.2), (1.0, 0.0)])
def test_a_split_that_is_not_a_partition_of_the_share_is_refused(bad):
    with pytest.raises(ValueError):
        Lane("vector", 0.30, ["a"], split=bad)


def test_both_competitions_commitments_are_read_from_one_storage():
    """One subnet, one commitment slot per hotkey: the prefix says which competition it entered,
    so a miner never has to say which netuid they meant."""
    vector = commitment.parse(f"vector:o/n@{SHA}")
    horizon = commitment.parse(f"horizon:o/n@{SHA}")

    assert (vector.lane, horizon.lane) == ("vector", "horizon")
    assert vector.entry == horizon.entry
    assert commitment.encode("o/n", SHA, lane="horizon") == f"horizon:o/n@{SHA}"
