"""What a miner writes on chain: one short string naming its submission.

    vector:<owner>/<name>@<commit>.<digest>
    horizon:<owner>/<name>@<40-hex commit>

A Vector commitment names its weights as well as where they are: `<commit>` is the repository
revision's 20 bytes and `<digest>` the sha256 of its `model.safetensors`, both in unpadded base64url
(27 and 43 characters), so that a repository id of up to 49 characters fits. The validator refuses
a commitment whose digest is not what the Hub says that file at that revision hashes to, so a
miner's commitment pins the exact weights it duels with from the block it was made at.

The first word names the competition the submission enters, so one subnet reads both competitions'
commitments from the same storage and a miner never has to say which netuid slot they meant. The
string is stored as a raw commitment, which holds at most `MAX_BYTES` bytes, so a repository id can
be at most about 80 characters. The revision is always a full commit sha: a branch would let a
miner change what was committed after its block, and the block is what orders the queue.
"""

from __future__ import annotations

import base64
import binascii
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
DIGEST_RE = re.compile(r"[0-9a-f]{64}")
#: Unpadded base64url of 20 and of 32 bytes.
B64_RE = {20: re.compile(r"[A-Za-z0-9_-]{27}"), 32: re.compile(r"[A-Za-z0-9_-]{43}")}


class CommitmentError(ValueError):
    """A commitment that is not one this subnet reads."""


@dataclass(frozen=True)
class Submission:
    """A parsed commitment: which lane, which repository, at which commit."""

    lane: str
    repo: str
    revision: str
    #: The sha256 of the weights, as 64 lowercase hex; None for a lane whose commitment has none.
    digest: str | None = None

    @property
    def entry(self) -> str:
        return f"{self.repo}@{self.revision}"


def _b64(hex_text: str) -> str:
    return base64.urlsafe_b64encode(bytes.fromhex(hex_text)).rstrip(b"=").decode("ascii")


def _unb64(text: str, size: int) -> str | None:
    """`text` as the hex of `size` bytes, or None when it is not their unpadded base64url."""
    if not B64_RE[size].fullmatch(text):
        return None
    try:
        raw = base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
    except (ValueError, binascii.Error):
        return None
    return raw.hex() if len(raw) == size and _b64(raw.hex()) == text else None


def encode(repo: str, revision: str, digest: str | None = None, lane: str = VECTOR) -> str:
    """The commitment string for `repo@revision` (and, for Vector, the weights' sha256 `digest`);
    `CommitmentError` if it cannot be one."""
    if lane not in LANES:
        raise CommitmentError(f"unknown lane {lane!r}; this subnet reads {', '.join(LANES)}")
    if not REPO_RE.fullmatch(repo):
        raise CommitmentError(f"{repo!r} is not a Hugging Face repo id (owner/name)")
    if not SHA_RE.fullmatch(revision):
        raise CommitmentError(
            f"{revision!r} is not a full 40-hex commit sha; commit the revision the upload printed"
        )
    if lane == VECTOR:
        if digest is None or not DIGEST_RE.fullmatch(digest):
            raise CommitmentError(
                f"{digest!r} is not the 64-hex sha256 of model.safetensors; `check` prints it"
            )
        data = f"{lane}:{repo}@{_b64(revision)}.{_b64(digest)}"
    elif digest is not None:
        raise CommitmentError(f"a {lane} commitment names no digest")
    else:
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
    repo, at, pinned = rest.rpartition("@")
    if not at or not REPO_RE.fullmatch(repo):
        raise CommitmentError(f"{data[:80]!r} names no <owner>/<name>@...")
    if lane != VECTOR:
        if not SHA_RE.fullmatch(pinned):
            raise CommitmentError(f"{data[:80]!r} is not {lane}:<owner>/<name>@<40-hex sha>")
        return Submission(lane=lane, repo=repo, revision=pinned)
    commit, dot, weights = pinned.partition(".")
    revision, digest = _unb64(commit, 20), _unb64(weights, 32)
    if not dot or revision is None or digest is None:
        raise CommitmentError(
            f"{data[:80]!r} is not vector:<owner>/<name>@<commit>.<digest>: a Vector commitment "
            "names the weights' sha256 too (`robotensor-miner commit --digest`)"
        )
    return Submission(lane=lane, repo=repo, revision=revision, digest=digest)
