"""The pure protocol: commitments, seeds and the weight vector."""

import pytest

from robotensor_subnet.protocol import commitment, seed
from robotensor_subnet.protocol.weights import Lane, weight_vector

SHA = "0123456789abcdef0123456789abcdef01234567"


def test_a_commitment_round_trips():
    data = commitment.encode("robotensor/vector-update1", SHA)
    assert data == f"vector1:robotensor/vector-update1@{SHA}"
    sub = commitment.parse(data)
    assert (sub.lane, sub.repo, sub.revision) == ("vector1", "robotensor/vector-update1", SHA)
    assert sub.entry == f"robotensor/vector-update1@{SHA}"


@pytest.mark.parametrize(
    ("repo", "revision"),
    [
        ("robotensor/vector", "main"),  # a branch can change after the block that orders the queue
        ("robotensor/vector", SHA[:12]),
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
        "zw1:o/n@" + SHA,
        "vector1:o/n@main",
        "vector1:o/n",
        "vector1:" + SHA,
        "hello",
        "vector1:o/n@" + SHA + "0",
    ],
)
def test_what_is_not_a_vector_commitment_does_not_parse(data):
    with pytest.raises(commitment.CommitmentError):
        commitment.parse(data)


def test_the_seed_block_comes_after_the_commitment_and_behind_the_head():
    assert seed.seed_block(100, 90) == 97
    with pytest.raises(seed.NotYet):
        seed.seed_block(92, 90)  # 89 is not after 90
    assert seed.normalize_hash("AB" * 32) == "0x" + "ab" * 32
    with pytest.raises(ValueError):
        seed.normalize_hash("0x1234")


def test_one_champion_takes_the_lane_share_and_the_rest_burns():
    w = weight_vector([Lane("vector", 0.30, ["B", None])], {"A": 0, "B": 5}, burn_uid=0)
    assert w == pytest.approx({0: 0.70, 5: 0.30})


def test_five_champions_share_equally_and_older_ones_drop_out():
    hotkeys = ["h1", "h2", "h3", "h4", "h5", "h6"]
    uids = {h: i + 1 for i, h in enumerate(hotkeys)} | {"owner": 0}
    w = weight_vector([Lane("vector", 0.30, hotkeys)], uids, burn_uid=0)
    assert w[0] == pytest.approx(0.70)
    assert all(w[uids[h]] == pytest.approx(0.06) for h in hotkeys[:5])
    assert uids["h6"] not in w


def test_a_hotkey_with_two_crowned_models_gets_two_parts():
    w = weight_vector([Lane("vector", 0.30, ["a", "b", "a"])], {"a": 1, "b": 2, "o": 0}, burn_uid=0)
    assert w[1] == pytest.approx(0.20) and w[2] == pytest.approx(0.10)


def test_a_deregistered_champion_burns_its_part_rather_than_growing_the_others():
    w = weight_vector([Lane("vector", 0.30, ["a", "gone"])], {"a": 1, "o": 0}, burn_uid=0)
    assert w[1] == pytest.approx(0.15) and w[0] == pytest.approx(0.85)


def test_no_champion_yet_burns_everything():
    assert weight_vector([Lane("vector", 0.30, [None])], {"o": 0}, burn_uid=0) == {0: 1.0}


def test_shares_over_one_are_refused():
    with pytest.raises(ValueError):
        weight_vector([Lane("a", 0.7, []), Lane("b", 0.5, [])], {}, burn_uid=0)
