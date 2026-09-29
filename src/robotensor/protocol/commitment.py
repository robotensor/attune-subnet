"""What a miner writes on chain: one short string naming its submission.

    vector:<owner>/<name>@<40-hex commit>
    horizon:<owner>/<name>@<40-hex commit>

The first word names the competition the submission enters, so one subnet reads both competitions'
commitments from the same storage and a miner never has to say which netuid slot they meant. The
string is stored as a raw commitment, which holds at most `MAX_BYTES` bytes, so a repository id can
be at most about 80 characters. The revision is always a full commit sha: a branch would let a
miner change what was committed after its block, and the block is what orders the queue.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: A raw commitment's limit on chain (`Raw` data, as `Subtensor.set_commitment` writes it).
MAX_BYTES = 128
#: Competition 1, action-grounded in-context learning: the only one live at launch.
VECTOR = "vector"
#: Competition 2, video-prompted world-action learning. Its commitments are read and taken in;
#: what they enter does not pay yet (`config.SHARES`), and the round that scores them is
#: staged in `lanes/horizon.py`.
HORIZON = "horizon"
LANES = (VECTOR, HORIZON)

REPO_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")
SHA_RE = re.compile(r"[0-9a-f]{40}")


class CommitmentError(ValueError):
    """A commitment that is not one this subnet reads."""


@dataclass(frozen=True)
class Submission:
    """A parsed commitment: which lane, which repository, at which commit."""

    lane: str
    repo: str
    revision: str

    @property
    def entry(self) -> str:
        return f"{self.repo}@{self.revision}"


def encode(repo: str, revision: str, lane: str = VECTOR) -> str:
    """The commitment string for `repo@revision`; `CommitmentError` if it cannot be one."""
    if lane not in LANES:
        raise CommitmentError(f"unknown lane {lane!r}; this subnet reads {', '.join(LANES)}")
    if not REPO_RE.fullmatch(repo):
        raise CommitmentError(f"{repo!r} is not a Hugging Face repo id (owner/name)")
    if not SHA_RE.fullmatch(revision):
        raise CommitmentError(
            f"{revision!r} is not a full 40-hex commit sha; commit the revision the upload printed"
        )
    data = f"{lane}:{repo}@{revision}"
    if len(data.encode()) > MAX_BYTES:
        raise CommitmentError(
            f"the commitment is {len(data.encode())} bytes, over the chain's {MAX_BYTES}; use a "
            "shorter repository name"
        )
    return data


def parse(data: str) -> Submission:
    """A commitment string as a `Submission`; `CommitmentError` for anything else."""
    if not isinstance(data, str) or len(data.encode()) > MAX_BYTES:
        raise CommitmentError("not a commitment string of at most 128 bytes")
    lane, sep, rest = data.strip().partition(":")
    if not sep or lane not in LANES:
        raise CommitmentError(f"no known lane prefix in {data[:40]!r}")
    repo, at, revision = rest.rpartition("@")
    if not at or not REPO_RE.fullmatch(repo) or not SHA_RE.fullmatch(revision):
        raise CommitmentError(f"{data[:80]!r} is not {lane}:<owner>/<name>@<40-hex sha>")
    return Submission(lane=lane, repo=repo, revision=revision)
