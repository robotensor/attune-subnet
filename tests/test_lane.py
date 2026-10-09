"""The Vector lane's intake, against a fake Hub and the real orchestrator contract."""

import dataclasses
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from robotensor.chain import Commitment
from robotensor.config import VectorConfig
from robotensor.lanes.vector import Entry, VectorLane
from robotensor.protocol import commitment
from robotensor.state import State

pytest.importorskip("vector_orchestrator")

A, B, C = "a" * 40, "b" * 40, "c" * 40
#: The weights each revision holds in these tests, by a label the fake Hub hashes to a digest.
WEIGHTS = {A: "w1", B: "w2", C: "w3"}


def digest(label):
    """The sha256 the fake Hub reports for weights labelled `label`."""
    return hashlib.sha256(label.encode()).hexdigest()


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
                lfs=SimpleNamespace(sha256=digest(sha)) if sha else None,
            )
            for name, sha in files.items()
        ]
        return SimpleNamespace(sha=revision, siblings=siblings)


@pytest.fixture
def lane(tmp_path):
    if not Path("/root/robotensor/vector/vector-orchestrator/spec.json").is_file():
        pytest.skip("the orchestrator checkout is not beside this one")
    cfg = VectorConfig(
        name="vector",
        share=0.3,
        store=tmp_path / "store",
        run_dir=tmp_path / "runs",
        cache=tmp_path / "cache",
        policy_python="python",
        simulator_python="python",
        simulator_root="/checkout/RoboTwin-Vector",
    )
    return VectorLane(cfg, State(tmp_path / "state"))


@pytest.fixture
def hub_of(monkeypatch):
    """Point the lane's Hub lookups at a `FakeHub` of `repos`; the fake, to change later."""
    from robotensor import hub

    real = hub.inspect

    def install(repos):
        fake = FakeHub(repos)
        monkeypatch.setattr(hub, "inspect", lambda *a, **kw: real(*a, **{**kw, "api": fake}))
        return fake

    return install


def c(hotkey, repo, sha, block, weights=None):
    """A commitment of `repo@sha` naming the weights labelled `weights` (by default the ones
    `WEIGHTS` says that revision holds)."""
    data = commitment.encode(repo, sha, digest(weights or WEIGHTS[sha]))
    return Commitment(hotkey=hotkey, block=block, data=data)


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
    stale = Commitment(hotkey="hk0", block=5, data=f"old1:m/old@{C}")
    nonsense = Commitment(hotkey="hk3", block=6, data="hello")

    lane.intake([stale, nonsense, c("hk1", "m/one", A, 10)], 30, api=hub)

    assert [(e.hotkey, e.repo) for e in lane.queue()] == [("hk1", "m/one")]


def test_a_commitment_made_before_the_start_block_is_not_read(lane):
    """What miners committed before the competition started does not enter it, and leaves no
    entry behind: from the start block on, a commitment is read as usual."""
    lane.cfg = dataclasses.replace(lane.cfg, start_block=20)
    lane.intake([c("hk1", "m/one", A, 19), c("hk2", "m/two", B, 20)], 30)

    assert [(e.hotkey, e.repo) for e in lane.queue()] == [("hk2", "m/two")]
    assert [r["hotkey"] for r in lane.state.lane("vector")["entries"].values()] == ["hk2"]


def test_a_hotkey_ignored_before_the_start_block_enters_by_committing_again(lane):
    """An ignored commitment spends nothing: the same hotkey's next one, made from the start block
    on, is queued rather than refused, even naming the same revision."""
    lane.cfg = dataclasses.replace(lane.cfg, start_block=20)
    assert lane.intake([c("hk1", "m/one", A, 10)], 15) == []

    (entry,) = lane.intake([c("hk1", "m/one", A, 25)], 30)

    assert (entry.status, entry.commit_block) == ("queued", 25)
    assert [e.hotkey for e in lane.queue()] == ["hk1"]


def test_a_hotkey_makes_one_submission_and_a_later_commitment_is_refused(lane):
    lane.intake(
        [c("hk1", "m/one", A, 10)], 30, api=FakeHub({f"m/one@{A}": {"model.safetensors": "w1"}})
    )
    (later,) = lane.intake(
        [c("hk1", "m/one", C, 40)], 50, api=FakeHub({f"m/one@{C}": {"model.safetensors": "w3"}})
    )
    assert later.status == "refused" and later.revision == C
    entries = lane.state.lane("vector")["entries"]
    assert "one submission" in entries[later.key]["reason"]
    assert [e.revision for e in lane.queue()] == [A], "the first submission keeps its place"


def test_the_one_submission_stays_spent_after_its_duel_won_or_lost(lane):
    lane.intake(
        [c("hk1", "m/one", A, 10)], 30, api=FakeHub({f"m/one@{A}": {"model.safetensors": "w1"}})
    )
    (first,) = lane.queue()
    for settled in ("duelled", "crowned", "void", "refused"):
        lane.state.lane("vector")["entries"][first.key]["status"] = settled
        (again,) = lane.intake(
            [c("hk1", "m/one", B, 40 + len(settled))],
            60,
            api=FakeHub({f"m/one@{B}": {"model.safetensors": "w2"}}),
        )
        assert again.status == "refused", settled
        del lane.state.lane("vector")["entries"][again.key]


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

    entry = Entry("k", "hk", "m/one", A, 10, "queued")
    monkeypatch.setattr(type(lane), "queue", lambda self: [entry])
    monkeypatch.setattr(
        type(lane),
        "duel",
        lambda self, entry, chain: (_ for _ in ()).throw(DuelFailed("no harness")),
    )

    progress = lane.step(chain=None)

    assert progress.outcome == FAILED and "no harness" in progress.detail
    assert progress.resting and progress.lane == "vector"


class FakeChain:
    """A head at 1000 whose finalized head is 997."""

    def block(self):
        return 1000

    def block_hash(self, block):
        return "0x" + "ab" * 32

    def finalized(self):
        return 997, "0x" + "CD" * 32


def test_an_empty_queue_leaves_the_throne_waiting_and_publishes_an_empty_queue(lane):
    from robotensor.lanes.base import IDLE

    assert lane.step(chain=FakeChain()).outcome == IDLE
    assert json.loads(lane.engine.store.queue_path().read_text()) == {"entries": []}


def test_the_oldest_entry_takes_the_empty_throne_and_the_queue_is_published(
    lane, monkeypatch, hub_of
):
    """The first entry duels no king, which the orchestrator runs as a genesis; it is crowned
    though a genesis record is never `dethroned`."""
    from robotensor.lanes.base import WORKED

    hub_of({f"m/one@{A}": {"model.safetensors": "w1"}, f"m/two@{B}": {"model.safetensors": "w2"}})
    lane.intake([c("hk2", "m/two", B, 20), c("hk1", "m/one", A, 10)], 30)
    asked = []

    def run(req):
        asked.append(req)
        record = {"kind": "genesis", "dethroned": False}
        return SimpleNamespace(
            published=True, status="published", reason="genesis", event_id="e1",
            as_dict=lambda: {"reason": "genesis", "record": record},
        )  # fmt: skip

    monkeypatch.setattr(lane.engine, "run", run)
    progress = lane.step(chain=FakeChain())

    assert progress.outcome == WORKED
    (req,) = asked
    assert req.king is None and req.challenger.repo == "m/one" and req.kind == "genesis"
    # Seeded from the finalized head and the hash read with it, not from a block behind the head.
    assert (req.seed_block, req.seed_block_hash) == (997, "0x" + "cd" * 32)
    entries = lane.state.lane("vector")["entries"]
    assert {r["hotkey"]: r["status"] for r in entries.values()} == {
        "hk1": "crowned",
        "hk2": "queued",
    }
    queue = json.loads(lane.engine.store.queue_path().read_text())
    assert [e["repo"] for e in queue["entries"]] == ["m/two"]


def test_champions_are_the_hotkeys_of_what_was_crowned_newest_first(lane, monkeypatch):
    from vector_orchestrator.ids import submission_key

    lane.intake(
        [c("hk1", "m/one", A, 10), c("hk2", "m/two", B, 20)],
        30,
        api=FakeHub(
            {f"m/one@{A}": {"model.safetensors": "w1"}, f"m/two@{B}": {"model.safetensors": "w2"}}
        ),
    )
    one = {"key": submission_key("m/one", A), "repo": "m/one", "revision": A}
    two = {"key": submission_key("m/two", B), "repo": "m/two", "revision": B}
    stranger = {"key": submission_key("m/other", C), "repo": "m/other", "revision": C}
    records = [
        {"kind": "genesis", "king": one, "dethroned": False},
        {"kind": "duel", "king": one, "challenger": stranger, "dethroned": False},
        {"kind": "duel", "king": one, "challenger": two, "dethroned": True},
        {"kind": "duel", "king": two, "challenger": stranger, "dethroned": True},
    ]
    monkeypatch.setattr(lane.engine.store, "iter_index", lambda: iter(records))
    assert lane.champions() == [None, "hk2", "hk1"]


def test_a_seed_block_that_is_not_final_yet_is_waiting_not_a_failure(lane, monkeypatch):
    from robotensor.lanes.base import WAITING
    from robotensor.protocol import seed as seed_

    monkeypatch.setattr(type(lane), "ready", lambda self: True)
    entry = Entry("k", "hk", "m/one", A, 10, "queued")
    monkeypatch.setattr(type(lane), "queue", lambda self: [entry])
    monkeypatch.setattr(
        type(lane),
        "duel",
        lambda self, entry, chain: (_ for _ in ()).throw(seed_.NotYet("3 blocks to go")),
    )

    progress = lane.step(chain=None)

    assert progress.outcome == WAITING and "3 blocks to go" in progress.detail


def test_the_queue_names_each_entrys_hotkey(lane):
    lane.intake(
        [c("hk1", "m/one", A, 10)], 30, api=FakeHub({f"m/one@{A}": {"model.safetensors": "w1"}})
    )
    lane.publish_queue(lane.queue())
    (entry,) = json.loads(lane.engine.store.queue_path().read_text())["entries"]
    assert (entry["repo"], entry["hotkey"], entry["block"]) == ("m/one", "hk1", 10)


def test_the_champions_paid_now_are_published_with_their_part_of_the_lane(lane, monkeypatch):
    """The newest four a miner committed, 40/30/20/10; what no entry committed holds no place, and
    with fewer than four the filled places share the whole lane."""
    from vector_orchestrator.ids import submission_key

    lane.intake(
        [c("hk1", "m/one", A, 10), c("hk2", "m/two", B, 20)],
        30,
        api=FakeHub(
            {f"m/one@{A}": {"model.safetensors": "w1"}, f"m/two@{B}": {"model.safetensors": "w2"}}
        ),
    )
    one = {"key": submission_key("m/one", A), "repo": "m/one", "revision": A}
    two = {"key": submission_key("m/two", B), "repo": "m/two", "revision": B}
    stranger = {"key": submission_key("m/other", C), "repo": "m/other", "revision": C}
    records = [
        {"kind": "genesis", "king": stranger, "event_id": "e0", "finished_at": "t0"},
        {"kind": "duel", "king": stranger, "challenger": one, "dethroned": True, "event_id": "e1",
         "finished_at": "t1"},
        {"kind": "duel", "king": one, "challenger": two, "dethroned": True, "event_id": "e2",
         "finished_at": "t2"},
    ]  # fmt: skip
    monkeypatch.setattr(lane.engine.store, "iter_index", lambda: iter(records))

    lane.publish_champions()

    doc = json.loads(lane.engine.store.champions_path().read_text())
    assert doc["schema"] == 1 and doc["lane_share"] == 0.3
    assert doc["split"] == [0.4, 0.3, 0.2, 0.1]
    assert [(p["place"], p["repo"], p["hotkey"], p["event_id"]) for p in doc["champions"]] == [
        (1, "m/two", "hk2", "e2"),
        (2, "m/one", "hk1", "e1"),
    ]
    assert [round(p["share"], 6) for p in doc["champions"]] == [
        round(0.4 / 0.7, 6),
        round(0.3 / 0.7, 6),
    ]
    assert doc["champions"][0]["key"] == two["key"] and doc["champions"][0]["crowned_at"] == "t2"


def test_an_empty_store_publishes_no_champions(lane):
    lane.publish_champions()
    assert json.loads(lane.engine.store.champions_path().read_text())["champions"] == []


def test_a_refresh_takes_new_commitments_in_and_republishes_the_queue(lane, monkeypatch):
    from robotensor import hub

    fake = FakeHub({f"m/one@{A}": {"model.safetensors": "w1"}})
    real = hub.inspect
    monkeypatch.setattr(hub, "inspect", lambda *a, **kw: real(*a, **{**kw, "api": fake}))

    class Chain:
        def commitments(self):
            return [c("hk1", "m/one", A, 10)]

        def block(self):
            return 30

    (changed,) = lane.refresh(Chain())

    assert changed.status == "queued"
    queue = json.loads(lane.engine.store.queue_path().read_text())
    assert [e["hotkey"] for e in queue["entries"]] == ["hk1"]


# -- intake queues at once; the Hub is asked at an entry's turn -----------------------------------


def genesis_run(asked):
    """An `Orchestrator.run` that publishes a genesis for whatever it is asked, and notes it."""

    def run(req):
        asked.append(req)
        record = {"kind": req.kind, "dethroned": False}
        return SimpleNamespace(
            published=True, status="published", reason=req.kind, event_id=f"e{len(asked)}",
            as_dict=lambda: {"reason": req.kind, "record": record},
        )  # fmt: skip

    return run


def statuses(lane):
    return {r["hotkey"]: r["status"] for r in lane.state.lane("vector")["entries"].values()}


def test_a_commitment_is_queued_without_asking_the_hub(lane, monkeypatch):
    from robotensor import hub

    def no(*a, **kw):
        raise AssertionError("intake asked the Hub")

    monkeypatch.setattr(hub, "inspect", no)
    (entry,) = lane.intake([c("hk1", "m/one", A, 10)], 30)
    assert entry.status == "queued" and [e.hotkey for e in lane.queue()] == ["hk1"]


def test_a_queued_commitment_is_the_hotkeys_one_shot_whatever_its_repository_holds(lane, hub_of):
    hub_of({f"m/one@{A}": {"model.safetensors": "w1", "run.py": None}})
    lane.intake([c("hk1", "m/one", A, 10)], 30)
    lane.step(chain=FakeChain())
    assert statuses(lane) == {"hk1": "refused"}
    (later,) = lane.intake([c("hk1", "m/one", B, 40)], 50)
    assert later.status == "refused"


def test_a_challenger_the_hub_does_not_show_is_skipped_for_the_next(lane, hub_of, monkeypatch):
    from robotensor.lanes.base import WORKED

    hub_of({f"m/two@{B}": {"model.safetensors": "w2"}})
    lane.intake([c("hk1", "m/gone", A, 10), c("hk2", "m/two", B, 20)], 30)
    asked = []
    monkeypatch.setattr(lane.engine, "run", genesis_run(asked))

    progress = lane.step(chain=FakeChain())

    assert progress.outcome == WORKED
    assert [r.challenger.repo for r in asked] == ["m/two"]
    assert statuses(lane) == {"hk1": "missing", "hk2": "crowned"}
    (gone,) = [r for r in lane.state.lane("vector")["entries"].values() if r["hotkey"] == "hk1"]
    assert "not on the Hub" in gone["reason"]
    assert not lane.queue()


def test_a_hub_that_does_not_answer_is_a_missing_challenger(lane, monkeypatch):
    from robotensor import hub
    from robotensor.lanes.base import IDLE

    def down(*a, **kw):
        raise hub.HubUnreachable("503")

    monkeypatch.setattr(hub, "inspect", down)
    lane.intake([c("hk1", "m/one", A, 10)], 30)
    progress = lane.step(chain=FakeChain())
    assert progress.outcome == IDLE and "503" in progress.detail
    assert statuses(lane) == {"hk1": "missing"} and not lane.queue()


def test_weights_that_are_not_the_committed_digest_are_refused_at_their_turn(lane, hub_of):
    hub_of({f"m/one@{A}": {"model.safetensors": "w1"}})
    lane.intake([c("hk1", "m/one", A, 10, "something-else")], 30)
    lane.step(chain=FakeChain())
    (record,) = lane.state.lane("vector")["entries"].values()
    assert record["status"] == "refused"
    assert digest("something-else") in record["reason"] and digest("w1") in record["reason"]


def test_copied_weights_are_a_duplicate_and_the_earliest_commitment_keeps_them(
    lane, hub_of, monkeypatch
):
    hub_of(
        {f"m/orig@{A}": {"model.safetensors": "same"}, f"x/copy@{B}": {"model.safetensors": "same"}}
    )
    lane.intake([c("copier", "x/copy", B, 12, "same"), c("author", "m/orig", A, 10, "same")], 30)
    asked = []
    monkeypatch.setattr(lane.engine, "run", genesis_run(asked))
    lane.step(chain=FakeChain())
    lane.step(chain=FakeChain())
    assert statuses(lane) == {"author": "crowned", "copier": "duplicate"}
    assert [r.challenger.repo for r in asked] == ["m/orig"]


def crown(lane, repo, sha):
    """Publish a genesis crowning `repo@sha` in the lane's real store."""
    from vector_orchestrator.ids import SubmissionRef
    from vector_orchestrator.store.records import index_record

    ref = SubmissionRef.make(repo, sha)
    record = index_record(
        event_id="g" * 64, kind="genesis", block=1, finished_at="t0", king=ref.as_dict(),
        challenger=None, king_score=0.5, challenger_score=None, dethroned=False, units=[],
    )  # fmt: skip
    lane.engine.store.write_event({"event_id": record["event_id"]})
    lane.engine.store.append(record)
    return ref


def test_a_king_whose_repository_is_gone_is_vacated_and_its_place_burns(lane, hub_of, monkeypatch):
    from robotensor.lanes.base import WORKED
    from robotensor.protocol import weights as weights_

    hub_of({f"m/two@{B}": {"model.safetensors": "w2"}})
    lane.intake([c("hk1", "m/one", A, 10)], 30)
    lane.state.lane("vector")["entries"][next(iter(lane.state.lane("vector")["entries"]))][
        "status"
    ] = "crowned"
    crown(lane, "m/one", A)
    lane.intake([c("hk2", "m/two", B, 20)], 30)
    asked = []
    monkeypatch.setattr(lane.engine, "run", genesis_run(asked))

    progress = lane.step(chain=FakeChain())

    # The king is gone from the Hub: vacated, and the challenger takes the empty throne by genesis.
    kinds = [r["kind"] for r in lane.engine.store.iter_index()]
    assert kinds == ["genesis", "vacate"]
    assert progress.outcome == WORKED
    (req,) = asked
    assert req.king is None and req.challenger.repo == "m/two"
    assert statuses(lane) == {"hk1": "vacated", "hk2": "crowned"}
    # Its place among the champions holds, and its part goes to the burn UID.
    assert lane.champions() == [weights_.BURNED]
    (paid,) = lane.paid()
    assert paid["hotkey"] == "hk1" and paid["burned"] is True
    vector = weights_.weight_vector(
        [weights_.Lane("vector", 0.3, lane.champions())], {"hk1": 1, "owner": 0}, 0
    )
    assert vector == {0: 1.0}


def test_a_king_still_on_the_hub_keeps_the_crown(lane, hub_of):
    hub_of({f"m/one@{A}": {"model.safetensors": "w1"}})
    crown(lane, "m/one", A)
    assert lane.check_king() is None
    assert lane.engine.store.head()["king"]["repo"] == "m/one"


def test_a_king_is_vacated_when_the_hub_does_not_answer(lane, monkeypatch):
    from robotensor import hub

    crown(lane, "m/one", A)

    def down(*a, **kw):
        raise hub.HubUnreachable("timeout")

    monkeypatch.setattr(hub, "inspect", down)
    assert "timeout" in lane.check_king()
    assert lane.engine.store.head()["king"] is None
