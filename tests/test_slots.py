"""One submission per registration (`protocol.slots`), and the chain's past it reads."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from robotensor import chain as chain_
from robotensor.chain import Commitment
from robotensor.protocol import commitment, slots

DIGEST = hashlib.sha256(b"w").hexdigest()


def vec(sha="a"):
    return commitment.encode("m/x", sha * 40, DIGEST)


def run(registered, commits, *, roots=None, start=0, unknown=()):
    """`commits`: hotkey -> [(block, data)], oldest first, the newest of each what it holds now.
    The verdicts, the hotkeys searched and the reads made, in order; a hotkey in `unknown` is one
    the archive does not answer for."""
    searched, reads = [], []
    data = {(h, b): d for h, made in commits.items() for b, d in made}

    def first(h, newest, floor):
        searched.append(h)
        if h in unknown:
            return None
        return min(b for b, _ in commits[h] if b >= floor)

    def read(h, b):
        reads.append((h, b))
        return data[(h, b)]

    last = {h: made[-1][0] for h, made in commits.items() if made}
    current = {h: Commitment(h, *made[-1]) for h, made in commits.items() if made}
    out = slots.submissions(registered, roots or {}, last, current, first, read, start)
    return [(c.hotkey, c.block, c.slot, c.submission) for c in out], searched, reads


def test_a_registrations_first_commitment_is_its_submission_and_a_later_one_is_refused():
    out, _, _ = run({"hk1": (3, 5)}, {"hk1": [(10, vec()), (20, vec("b"))]})
    assert out == [("hk1", 10, "3@5", True), ("hk1", 20, "3@5", False)]


def test_a_commitment_overwritten_before_any_validator_saw_it_is_still_the_submission():
    """Only the newest is in `CommitmentOf`; the first is read from the past, and only it."""
    out, _, reads = run({"hk1": (3, 5)}, {"hk1": [(10, vec()), (11, vec("b")), (12, vec("c"))]})
    assert out == [("hk1", 10, "3@5", True), ("hk1", 12, "3@5", False)]
    assert reads == [("hk1", 10)]


def test_a_first_commitment_still_held_is_read_from_the_head():
    out, _, reads = run({"hk1": (3, 5)}, {"hk1": [(10, vec())]})
    assert out == [("hk1", 10, "3@5", True)] and reads == []


def test_a_hotkey_swapped_into_a_registration_inherits_what_it_spent():
    """hk1 committed, then swapped its UID to hk2: hk2 holds the same registration, and only the
    hotkey that held it first is searched."""
    out, searched, _ = run(
        {"hk2": (3, 5)}, {"hk1": [(10, vec())], "hk2": [(30, vec("b"))]}, roots={"hk2": "hk1"}
    )
    assert out == [("hk1", 10, "3@5", True), ("hk2", 30, "3@5", False)]
    assert searched == ["hk1"]


def test_swaps_are_followed_through_every_hotkey_of_the_lineage():
    out, _, _ = run(
        {"hk3": (3, 5)},
        {"hk1": [(10, vec())], "hk2": [(20, vec("c"))], "hk3": [(30, vec("b"))]},
        roots={"hk2": "hk1", "hk3": "hk1"},
    )
    assert out == [
        ("hk1", 10, "3@5", True),
        ("hk2", 20, "3@5", False),
        ("hk3", 30, "3@5", False),
    ]


def test_a_new_registration_is_a_new_submission():
    """Commitments made before the UID was registered (again) belong to the one before."""
    out, _, _ = run({"hk1": (3, 25)}, {"hk1": [(10, vec()), (30, vec("b"))]})
    assert out == [("hk1", 30, "3@25", True)]


def test_a_predecessor_registered_again_keeps_its_commitments_to_its_own_registration():
    out, _, _ = run(
        {"hk2": (3, 5), "hk1": (4, 40)},
        {"hk1": [(45, vec())], "hk2": [(30, vec("b"))]},
        roots={"hk2": "hk1"},
    )
    assert out == [("hk2", 30, "3@5", True), ("hk1", 45, "4@40", True)]


def test_a_first_commitment_that_is_not_a_vector_one_spends_the_shot():
    """Junk first cannot be used to push the real submission back."""
    out, _, _ = run({"hk1": (3, 5)}, {"hk1": [(12, "hello"), (14, vec())]})
    assert out == [("hk1", 14, "3@5", False)]


def test_what_came_before_the_start_block_spends_nothing():
    out, _, _ = run({"hk1": (3, 5)}, {"hk1": [(8, vec()), (14, vec("b"))]}, start=10)
    assert out == [("hk1", 14, "3@5", True)]


def test_an_unregistered_hotkey_of_no_registration_now_is_not_read():
    assert run({}, {"gone": [(10, vec())]}) == ([], [], [])


def test_a_registration_the_archive_does_not_answer_for_is_left_out_and_the_rest_go_on():
    out, _, _ = run(
        {"hk1": (3, 5), "hk2": (4, 5)},
        {"hk1": [(10, vec())], "hk2": [(11, vec("b")), (12, vec("c"))]},
        unknown={"hk2"},
    )
    assert out == [("hk1", 10, "3@5", True)]


# -- the chain's past ------------------------------------------------------------------------------


class Archive:
    """`LastCommitment` and `CommitmentOf` of one netuid as they stood at each block: `commits`
    maps a hotkey to the blocks it committed at, oldest first, each committing `at <block>`."""

    def __init__(self, commits, *, vector=False):
        self.commits = commits
        self.vector = vector
        self.reads = 0

    def query_module(self, module, name, params, block):
        self.reads += 1
        _, hotkey = params
        made = self.commits.get(hotkey, [])
        low, high = 0, len(made)
        while low < high:
            middle = (low + high) // 2
            low, high = (middle + 1, high) if made[middle] <= block else (low, middle)
        if low == 0:
            return SimpleNamespace(value=None)
        newest = made[low - 1]
        if name == "LastCommitment":
            return SimpleNamespace(value=newest)
        return SimpleNamespace(
            value={"block": newest, "data": vec() if self.vector else f"at {newest}"}
        )


def bare(archive=None):
    out = object.__new__(chain_.Chain)
    out.network, out.netuid, out._archive, out._cache = "test", 7, archive, None
    out._last_at, out._firsts, out._made = {}, {}, {}
    return out


@pytest.fixture(autouse=True)
def decoded(monkeypatch):
    import bittensor.core.chain_data.utils as utils

    monkeypatch.setattr(chain_, "ARCHIVE_GAP_S", 0.0)
    monkeypatch.setattr(utils, "decode_metadata", lambda record: record.get("data", vec()))


def test_a_first_commitment_is_found_a_few_steps_back():
    chain = bare(archive := Archive({"hk1": [10, 20, 30]}))
    assert chain.first("hk1", 30, 0) == 10 and archive.reads == 3
    assert chain.first("hk1", 30, 15) == 20
    assert chain.made("hk1", 20) == "at 20"


def test_a_hotkey_that_commits_thousands_of_times_costs_a_search_not_a_walk():
    blocks = list(range(1_000, 1_000 + 3 * 5_000, 3))
    chain = bare(archive := Archive({"hk1": blocks}))
    assert chain.first("hk1", blocks[-1], 0) == 1_000
    assert archive.reads <= chain_.STEPS_BACK + 16
    assert bare(Archive({"hk1": blocks})).first("hk1", blocks[-1], 5_000) == 5_002


def test_what_was_found_is_not_read_again():
    chain = bare(archive := Archive({"hk1": [10, 20, 30]}))
    chain.first("hk1", 30, 0)
    reads = archive.reads
    assert chain.first("hk1", 30, 0) == 10 and archive.reads == reads


def test_a_past_the_archive_cannot_give_is_a_chain_error_naming_the_override():
    class Pruned:
        def query_module(self, *a, **kw):
            raise RuntimeError("State already discarded")

    with pytest.raises(chain_.ChainError, match="SUBTENSOR_ARCHIVE"):
        bare(Pruned()).first("hk1", 30, 0)


def test_a_rate_limited_past_read_is_tried_again():
    archive, failures = Archive({"hk1": [10]}), [1, 1]
    real = archive.query_module

    def limited(*a, **kw):
        if failures:
            failures.pop()
            raise RuntimeError("Historical work rate limit exceeded")
        return real(*a, **kw)

    archive.query_module = limited
    assert bare(archive).first("hk1", 10, 0) == 10


def test_the_archive_can_be_named_in_the_environment(monkeypatch):
    assert bare().archive == "test"
    monkeypatch.setenv("SUBTENSOR_ARCHIVE", "wss://mine:443")
    assert bare().archive == "wss://mine:443"


def head(commits, registered, roots=()):
    """A head: `Uids`, `BlockAtRegistration`, `HotkeyRoot`, and each hotkey's newest commitment."""
    tables = {
        "Uids": [(hk, uid) for hk, (uid, _) in registered.items()],
        "BlockAtRegistration": [(uid, at) for uid, at in registered.values()],
        "HotkeyRoot": list(roots),
        "LastCommitment": [(hk, made[-1]) for hk, made in commits.items()],
        "CommitmentOf": [(hk, {"block": made[-1], "data": vec()}) for hk, made in commits.items()],
    }
    return SimpleNamespace(query_map=lambda module, name, params: tables[name])


def test_submissions_reads_registrations_swaps_and_the_past(tmp_path):
    commits = {"hk1": [10], "hk2": [30], "old": [2]}
    chain = bare(Archive(commits))
    chain.subtensor = head(commits, {"hk2": (3, 5), "hk9": (4, 6)}, [("hk2", "hk1")])

    out = chain.submissions(start_block=5)

    assert [(c.hotkey, c.block, c.slot, c.submission) for c in out] == [
        ("hk1", 10, "3@5", True),
        ("hk2", 30, "3@5", False),
    ]


def test_a_validator_restarted_reads_nothing_of_the_past_again(tmp_path):
    commits = {"hk1": [10, 20, 30]}
    cache = tmp_path / "first_commitments.json"
    chain = bare(Archive(commits))
    chain.subtensor = head(commits, {"hk1": (3, 5)})
    first = chain.submissions(0, cache=cache)
    assert json.loads(cache.read_text())["firsts"] == {"hk1@5": 10}

    again = bare(archive := Archive(commits))
    again.subtensor = chain.subtensor
    assert again.submissions(0, cache=cache) == first and archive.reads == 0


def test_a_lost_or_broken_cache_only_costs_reads(tmp_path):
    commits = {"hk1": [10, 20, 30]}
    cache = tmp_path / "first_commitments.json"
    cache.write_text("{not json")
    chain = bare(Archive(commits, vector=True))
    chain.subtensor = head(commits, {"hk1": (3, 5)})
    (shot, _) = chain.submissions(0, cache=cache)
    assert (shot.block, shot.submission) == (10, True)
