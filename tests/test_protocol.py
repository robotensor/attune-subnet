"""The pure protocol: commitments, seeds and the weight vector."""

import pytest

from robotensor_subnet.protocol import commitment

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
