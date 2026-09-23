"""The Vector lane's intake, against a fake Hub and the real orchestrator contract."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from robotensor.chain import Commitment
from robotensor.config import VectorConfig
from robotensor.lanes.vector import Entry, VectorLane
from robotensor.protocol import commitment
from robotensor.state import State

pytest.importorskip("vector_orchestrator")

SPEC = Path("/root/robotensor/vector/vector-orchestrator/specs/vector_level1.json")
A, B, C = "a" * 40, "b" * 40, "c" * 40


class FakeHub:
    """`model_info` for a few repositories: {repo@sha: {file: lfs sha256 or None}}; a repository
    it does not know is private."""

    def __init__(self, repos):
        self.repos = repos

    def model_info(self, repo, revision, files_metadata):
        from huggingface_hub.errors import RepositoryNotFoundError

        files = self.repos.get(f"{repo}@{revision}")
        if files is None:
            raise RepositoryNotFoundError(
                "404", response=SimpleNamespace(headers={}, status_code=404, request=None)
            )
        siblings = [
            SimpleNamespace(
                rfilename=name,
                size=7,
                lfs=SimpleNamespace(sha256=sha) if sha else None,
            )
            for name, sha in files.items()
        ]
        return SimpleNamespace(sha=revision, siblings=siblings)


@pytest.fixture
def lane(tmp_path):
    if not SPEC.is_file():
        pytest.skip("the orchestrator checkout is not beside this one")
    cfg = VectorConfig(
        name="vector",
        share=0.3,
        spec=SPEC,
        store=tmp_path / "store",
        key=tmp_path / "keys" / "k.ed25519",
        run_dir=tmp_path / "runs",
        cache=tmp_path / "cache",
        policy_python="python",
        simulator_python="python",
        private_window_blocks=10,
    )
    return VectorLane(cfg, State(tmp_path / "state"))


def c(hotkey, repo, sha, block):
    return Commitment(hotkey=hotkey, block=block, data=commitment.encode(repo, sha))


def test_new_commitments_are_queued_oldest_first(lane):
    hub = FakeHub(
        {
            f"m/one@{A}": {"model.safetensors": "w1"},
            f"m/two@{B}": {"model.safetensors": "w2", "README.md": None},
        }
    )
    lane.intake([c("hk2", "m/two", B, 20), c("hk1", "m/one", A, 10)], 30, api=hub)
    assert [e.hotkey for e in lane.queue()] == ["hk1", "hk2"]


def test_a_commitment_this_subnet_does_not_read_is_skipped(lane):
    """A prefix from before the lane was named, or anything else that is not a commitment, is
    passed over without a word: the chain keeps what was written at a block forever, and one
    stale entry must not stop the rest of an intake."""
    hub = FakeHub({f"m/one@{A}": {"model.safetensors": "w1"}})
    stale = Commitment(hotkey="hk0", block=5, data=f"vector1:m/old@{C}")
    nonsense = Commitment(hotkey="hk3", block=6, data="hello")

    lane.intake([stale, nonsense, c("hk1", "m/one", A, 10)], 30, api=hub)

    assert [(e.hotkey, e.repo) for e in lane.queue()] == [("hk1", "m/one")]


def test_a_repository_holding_code_is_refused(lane):
    hub = FakeHub({f"m/one@{A}": {"model.safetensors": "w1", "policy.py": None}})
    (entry,) = lane.intake([c("hk1", "m/one", A, 10)], 30, api=hub)
    assert entry.status == "refused" and not lane.queue()


def test_a_private_repository_waits_then_is_refused(lane):
    hub = FakeHub({})
    (entry,) = lane.intake([c("hk1", "m/one", A, 10)], 15, api=hub)
    assert entry.status == "pending"
    (entry,) = lane.intake([c("hk1", "m/one", A, 10)], 25, api=hub)
    assert entry.status == "refused"


def test_a_private_repository_made_public_in_time_is_queued(lane):
    lane.intake([c("hk1", "m/one", A, 10)], 12, api=FakeHub({}))
    (entry,) = lane.intake(
        [c("hk1", "m/one", A, 10)], 14, api=FakeHub({f"m/one@{A}": {"model.safetensors": "w1"}})
    )
    assert entry.status == "queued"


def test_copied_weights_are_a_duplicate_and_the_earliest_commitment_keeps_them(lane):
    hub = FakeHub(
        {f"m/orig@{A}": {"model.safetensors": "same"}, f"x/copy@{B}": {"model.safetensors": "same"}}
    )
    lane.intake([c("copier", "x/copy", B, 12), c("author", "m/orig", A, 10)], 30, api=hub)
    queued = lane.queue()
    assert [e.hotkey for e in queued] == ["author"]
    entries = lane.state.lane("vector")["entries"]
    assert [r["status"] for r in entries.values() if r["hotkey"] == "copier"] == ["duplicate"]


def test_a_new_commitment_supersedes_the_hotkeys_waiting_entry(lane):
    lane.intake(
        [c("hk1", "m/one", A, 10)], 30, api=FakeHub({f"m/one@{A}": {"model.safetensors": "w1"}})
    )
    lane.intake(
        [c("hk1", "m/one", C, 40)], 50, api=FakeHub({f"m/one@{C}": {"model.safetensors": "w3"}})
    )
    assert [e.revision for e in lane.queue()] == [C]


def test_a_settled_entry_is_not_taken_in_again(lane):
    hub = FakeHub({f"m/one@{A}": {"model.safetensors": "w1"}})
    lane.intake([c("hk1", "m/one", A, 10)], 30, api=hub)
    assert lane.intake([c("hk1", "m/one", A, 10)], 31, api=hub) == []


def test_the_empty_store_has_no_king_and_no_champions(lane):
    assert not lane.ready() and lane.champions() == []


def test_the_lane_is_what_the_validator_asks_of_a_competition(lane):
    """The shape the loop drives, so a second competition can be driven by the same loop."""
    from robotensor.lanes.base import Lane

    assert isinstance(lane, Lane)
    assert lane.name == "vector"


def test_a_step_reports_the_engines_failure_rather_than_raising_it(lane, monkeypatch):
    """The loop cannot catch what it cannot import: a validator running one competition does not
    install the other's engine, so a lane's own failures come back as a `Progress`."""
    from vector_orchestrator.duel.orchestrate import DuelFailed

    from robotensor.lanes.base import FAILED

    monkeypatch.setattr(type(lane), "ready", lambda self: False)
    monkeypatch.setattr(
        type(lane), "genesis", lambda self, chain: (_ for _ in ()).throw(DuelFailed("no harness"))
    )

    progress = lane.step(chain=None)

    assert progress.outcome == FAILED and "no harness" in progress.detail
    assert progress.resting and progress.lane == "vector"


def test_a_seed_block_that_is_not_final_yet_is_waiting_not_a_failure(lane, monkeypatch):
    from robotensor.lanes.base import WAITING
    from robotensor.protocol import seed as seed_

    monkeypatch.setattr(type(lane), "ready", lambda self: True)
    entry = Entry("k", "hk", "m/one", A, 10, "queued")
    monkeypatch.setattr(type(lane), "queue", lambda self: [entry])
    monkeypatch.setattr(
        type(lane),
        "duel",
        lambda self, entry, chain, size=None: (_ for _ in ()).throw(seed_.NotYet("3 blocks to go")),
    )

    progress = lane.step(chain=None)

    assert progress.outcome == WAITING and "3 blocks to go" in progress.detail
