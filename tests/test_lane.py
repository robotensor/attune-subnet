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
    from vector_orchestrator.spec import load_spec

    try:
        load_spec()
    except OSError:
        pytest.skip("the installed orchestrator has no spec.json")
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


def history(lane, *names):
    """Crown `names` in that order, oldest first - a genesis, then a duel won by each next one -
    each committed by the hotkey `hk-<name>`; their refs, oldest first."""
    from vector_orchestrator.ids import SubmissionRef
    from vector_orchestrator.store.records import index_record

    shas = {name: f"{n:x}" * 40 for n, name in enumerate(names, start=1)}
    lane.intake(
        [
            c(f"hk-{name}", f"m/{name}", sha, 10 + n, name)
            for n, (name, sha) in enumerate(shas.items())
        ],
        100,
    )
    for record in lane.state.lane("vector")["entries"].values():
        record["status"] = "crowned"
    refs, king = [], None
    for n, (name, sha) in enumerate(shas.items(), start=1):
        ref = SubmissionRef.make(f"m/{name}", sha)
        record = index_record(
            event_id=f"{n:064x}", kind="duel" if king else "genesis", block=n,
            finished_at=f"t{n}", king=(king or ref).as_dict(),
            challenger=ref.as_dict() if king else None, king_score=0.5,
            challenger_score=0.6 if king else None, dethroned=king is not None, units=[],
        )  # fmt: skip
        lane.engine.store.write_event({"event_id": record["event_id"]})
        lane.engine.store.append(record)
        refs.append(ref)
        king = ref
    return refs


def on_the_hub(hub_of, refs):
    """A Hub that shows `refs`, and nothing else."""
    return hub_of({r.entry: {"model.safetensors": r.repo.split("/")[1]} for r in refs})


def watched(lane, times):
    """`times` looks at the champions, one after another; whether the last found one gone."""
    return [lane.watch(every=0) for _ in range(times)][-1]


def test_gone_champions_leave_the_pool_and_the_crown_returns_to_the_previous_king(lane, hub_of):
    """The king history, newest first, is A B C D E F; A's and D's repositories are deleted. B,
    C, E and F are paid, and B is king again."""
    from robotensor.lanes import vector
    from robotensor.protocol import weights as weights_

    f, e, d, c_, b, a = history(lane, "f", "e", "d", "c", "b", "a")
    on_the_hub(hub_of, [b, c_, e, f])
    hk = {r.repo.split("/")[1]: f"hk-{r.repo.split('/')[1]}" for r in (a, b, c_, d, e, f)}
    assert lane.champions() == [hk[x] for x in "abcdef"]

    # Missing once or twice is not gone yet: the pool stays as it was.
    assert watched(lane, vector.MISSES - 1) is False
    assert lane.champions() == [hk[x] for x in "abcdef"]
    # Missing MISSES times in a row is gone: out of the pool at once, before the store says so.
    assert watched(lane, 1) is True
    assert lane.champions() == [hk[x] for x in "bcef"]
    assert [p["hotkey"] for p in lane.paid()] == [hk[x] for x in "bcef"]
    assert [p["share"] for p in lane.paid()] == [0.4, 0.3, 0.2, 0.1]
    assert "burned" not in lane.paid()[0]
    assert lane.engine.store.head()["king"] == a.as_dict()

    # Recorded in the store between duels: A vacated with B its successor, D retired.
    assert sorted(lane.strike()) == sorted([a.key, d.key])
    kinds = [(r["kind"], r["king"]["repo"]) for r in lane.engine.store.iter_index()][-2:]
    assert kinds == [("vacate", "m/a"), ("retire", "m/d")]
    assert lane.engine.store.head()["king"] == b.as_dict()
    assert lane.champions() == [hk[x] for x in "bcef"]
    status = statuses(lane)
    assert (status["hk-a"], status["hk-d"], status["hk-b"]) == ("vacated", "retired", "crowned")
    assert lane.strike() == []  # nothing more to record

    uids = {hk[x]: n for n, x in enumerate("abcdef", start=1)}
    vector_ = weights_.weight_vector([weights_.Lane("vector", 0.3, lane.champions())], uids, 0)
    assert vector_ == pytest.approx({0: 0.7, 2: 0.12, 3: 0.09, 5: 0.06, 6: 0.03})


def test_the_watch_looks_no_further_than_a_full_pool(lane, hub_of, monkeypatch):
    from robotensor import hub

    refs = history(lane, "f", "e", "d", "c", "b", "a")
    on_the_hub(hub_of, refs)
    looked = []
    inspect = hub.inspect
    monkeypatch.setattr(
        hub, "inspect", lambda repo, *a, **kw: looked.append(repo) or inspect(repo, *a, **kw)
    )
    assert lane.watch(every=0) is False
    assert looked == ["m/a", "m/b", "m/c", "m/d"]
    # A second look within the minute is not made.
    assert lane.watch() is False and len(looked) == 4


def test_a_repository_seen_again_before_it_is_gone_starts_its_count_over(lane, hub_of):
    from robotensor.lanes import vector

    (one,) = history(lane, "one")
    on_the_hub(hub_of, [])
    watched(lane, vector.MISSES - 1)
    on_the_hub(hub_of, [one])
    watched(lane, 1)
    on_the_hub(hub_of, [])
    assert watched(lane, vector.MISSES - 1) is False
    assert lane.champions() == ["hk-one"]


def test_a_hub_that_does_not_answer_counts_neither_way(lane, hub_of, monkeypatch):
    from robotensor import hub
    from robotensor.lanes import vector

    (one,) = history(lane, "one")
    on_the_hub(hub_of, [])
    watched(lane, vector.MISSES - 1)
    missing = hub.inspect

    def down(*a, **kw):
        raise hub.HubUnreachable("503")

    monkeypatch.setattr(hub, "inspect", down)
    assert watched(lane, 5) is False
    assert lane.champions() == ["hk-one"] and lane.strike() == []
    monkeypatch.setattr(hub, "inspect", missing)
    assert watched(lane, 1) is True
    assert lane.champions() == []


def test_a_king_gone_with_no_earlier_king_left_leaves_the_throne_to_the_next_entry(
    lane, hub_of, monkeypatch
):
    from robotensor.lanes import vector
    from robotensor.lanes.base import WORKED

    (one,) = history(lane, "one")
    lane.intake([c("hk2", "m/two", B, 120)], 130)
    hub_of({f"m/two@{B}": {"model.safetensors": "w2"}})
    watched(lane, vector.MISSES - 1)
    monkeypatch.setattr(vector, "WATCH_S", 0)
    asked = []
    monkeypatch.setattr(lane.engine, "run", genesis_run(asked))

    # The step's own look is the last miss: the king is vacated, and the throne left empty.
    progress = lane.step(chain=FakeChain())

    assert [r["kind"] for r in lane.engine.store.iter_index()] == ["genesis", "vacate"]
    assert lane.engine.store.head()["king"] is None
    assert progress.outcome == WORKED
    (req,) = asked
    assert req.king is None and req.challenger.repo == "m/two"
    assert statuses(lane) == {"hk-one": "vacated", "hk2": "crowned"}


def test_a_king_still_on_the_hub_keeps_the_crown(lane, hub_of):
    from robotensor.lanes import vector

    (one,) = history(lane, "one")
    on_the_hub(hub_of, [one])
    assert watched(lane, vector.MISSES) is False
    assert lane.strike() == []
    assert lane.engine.store.head()["king"] == one.as_dict()


# -- another host takes the lane up from the mirror -----------------------------------------------


class FakeMirror:
    """A Hugging Face dataset as the orchestrator's `Mirror` is asked of it: files by path, shared
    by every host that names it; a push copies what it names from the host's store."""

    def __init__(self, remote, root):
        self.remote = remote
        self.root = Path(root)
        self.repo = "org/results"

    def read(self, path):
        from vector_orchestrator.store.mirror import MirrorUnreachable

        if self.remote.get("unreachable"):
            raise MirrorUnreachable("no route to host")
        return self.remote["files"].get(path)

    def head(self):
        data = self.read("head.json")
        return json.loads(data) if data else None

    def push(self, files, message="publish"):
        for name in files:
            if (self.root / name).is_file():
                self.remote["files"][name] = (self.root / name).read_bytes()


@pytest.fixture
def remote(monkeypatch):
    from vector_orchestrator.store import mirror

    shared = {"files": {}}
    monkeypatch.setattr(mirror, "Mirror", lambda root, repo, token=None: FakeMirror(shared, root))
    return shared


def host(path):
    """A validator's lane on a host of its own (`path`), mirroring to the shared dataset."""
    cfg = VectorConfig(
        name="vector",
        share=0.3,
        store=path / "store",
        run_dir=path / "runs",
        cache=path / "cache",
        policy_python="python",
        simulator_python="python",
        simulator_root="/checkout/RoboTwin-Vector",
        mirror="org/results",
    )
    return VectorLane(cfg, State(path / "state"))


def crowning_run(lane, asked):
    """An `Orchestrator.run` that publishes for real what `genesis_run` only says it did: the
    challenger crowned, its event and record in the store."""

    def run(req):
        asked.append(req)
        store = lane.engine.store
        event_id = f"{len(asked):064x}"
        record = {
            "kind": req.kind,
            "event_id": event_id,
            "block": req.block,
            "king": req.challenger.as_dict(),
            "dethroned": False,
        }
        store.write_event({"event_id": event_id})
        store.append(record)
        return SimpleNamespace(
            published=True, status="published", reason=req.kind, event_id=event_id,
            as_dict=lambda: {"reason": req.kind, "record": record},
        )  # fmt: skip

    return run


def test_a_validator_on_a_new_host_takes_the_lane_up_where_the_mirror_has_it(
    tmp_path, remote, hub_of, monkeypatch
):
    hub_of({f"m/one@{A}": {"model.safetensors": "w1"}, f"m/two@{B}": {"model.safetensors": "w2"}})
    first = host(tmp_path / "first")
    first.intake([c("hk1", "m/one", A, 10), c("hk2", "m/two", B, 20)], 30)
    monkeypatch.setattr(first.engine, "run", crowning_run(first, []))
    first.step(chain=FakeChain())
    assert first.champions() == ["hk1"]

    second = host(tmp_path / "second")
    asked = []
    monkeypatch.setattr(second.engine, "run", genesis_run(asked))

    # The crown, the index and who committed what, as the first host left them.
    assert second.engine.store.head()["king"]["repo"] == "m/one"
    assert second.engine.store.next_seq() == 2
    assert second.champions() == ["hk1"]
    assert statuses(second) == {"hk1": "crowned", "hk2": "queued"}
    # No events or clips came with it.
    assert not (tmp_path / "second" / "store" / "events").exists()
    # hk1 has made its one submission; hk2 duels the king rather than taking an empty throne.
    (again,) = second.intake([c("hk1", "m/three", C, 40)], 50)
    assert again.status == "refused"
    second.step(chain=FakeChain())
    (req,) = asked
    assert req.king.repo == "m/one" and req.challenger.repo == "m/two"


def test_the_lane_state_is_mirrored_with_the_queue(tmp_path, remote):
    lane = host(tmp_path)
    lane.intake([c("hk1", "m/one", A, 10)], 30)
    lane.publish_queue(lane.queue())
    kept = json.loads(remote["files"]["validator/vector.json"])
    assert {r["hotkey"] for r in kept["entries"].values()} == {"hk1"}


def test_an_empty_store_whose_mirror_cannot_be_read_is_not_started(tmp_path, remote):
    remote["unreachable"] = True
    lane = host(tmp_path)
    with pytest.raises(RuntimeError, match="not starting"):
        lane.engine  # noqa: B018
    assert not (tmp_path / "store" / "manifest.json").exists()


def test_a_store_with_a_history_goes_on_while_its_mirror_cannot_be_read(
    tmp_path, remote, hub_of, monkeypatch
):
    hub_of({f"m/one@{A}": {"model.safetensors": "w1"}})
    first = host(tmp_path / "first")
    first.intake([c("hk1", "m/one", A, 10)], 30)
    monkeypatch.setattr(first.engine, "run", crowning_run(first, []))
    first.step(chain=FakeChain())

    remote["unreachable"] = True
    again = host(tmp_path / "first")
    assert again.engine.store.head()["king"]["repo"] == "m/one"
    assert again.step(chain=FakeChain()).outcome == "idle"


def test_a_store_that_published_its_own_history_stops_rather_than_overwrite(
    tmp_path, remote, hub_of, monkeypatch
):
    from robotensor.lanes.base import FAILED

    hub_of({f"m/one@{A}": {"model.safetensors": "w1"}, f"m/two@{B}": {"model.safetensors": "w2"}})
    stray = host(tmp_path / "stray")
    stray.intake([c("hk2", "m/two", B, 20)], 30)
    monkeypatch.setattr(stray.engine, "run", crowning_run(stray, []))
    stray.step(chain=FakeChain())

    remote["files"] = {}
    first = host(tmp_path / "first")
    first.intake([c("hk1", "m/one", A, 10)], 30)
    monkeypatch.setattr(first.engine, "run", crowning_run(first, []))
    first.step(chain=FakeChain())

    progress = stray.step(chain=FakeChain())
    assert progress.outcome == FAILED and "disagree" in progress.detail


def published_by_a_first_host(tmp_path, hub_of, monkeypatch):
    """hk1's m/one crowned by genesis, hk2's m/two waiting, as a first host left the mirror."""
    hub_of({f"m/one@{A}": {"model.safetensors": "w1"}, f"m/two@{B}": {"model.safetensors": "w2"}})
    first = host(tmp_path / "first")
    first.intake([c("hk1", "m/one", A, 10), c("hk2", "m/two", B, 20)], 30)
    monkeypatch.setattr(first.engine, "run", crowning_run(first, []))
    first.step(chain=FakeChain())
    return first


def test_the_first_thing_a_new_host_reads_is_already_the_mirrors(
    tmp_path, remote, hub_of, monkeypatch
):
    """The weights thread's first look on a new host: the champions, before anything else built
    the engine. Read before the take-up, hk1's place would burn."""
    published_by_a_first_host(tmp_path, hub_of, monkeypatch)
    assert host(tmp_path / "weights").champions() == ["hk1"]
    assert [e.hotkey for e in host(tmp_path / "queue").queue()] == ["hk2"]


def test_a_mirror_kept_before_the_lane_state_was_does_not_duel_its_entries_again(
    tmp_path, remote, hub_of, monkeypatch
):
    """A dataset published before `validator/vector.json` existed: the index alone says m/one was
    crowned, so taking it in from the chain again crowns rather than queues it."""
    published_by_a_first_host(tmp_path, hub_of, monkeypatch)
    del remote["files"]["validator/vector.json"]

    second = host(tmp_path / "second")
    second.intake([c("hk1", "m/one", A, 10), c("hk2", "m/two", B, 20)], 50)
    assert statuses(second) == {"hk1": "crowned", "hk2": "queued"}
    assert second.champions() == ["hk1"]
    (again,) = second.intake([c("hk1", "m/three", C, 40)], 60)
    assert again.status == "refused"


def test_a_lane_level_with_its_mirror_writes_no_state(tmp_path, remote, hub_of, monkeypatch):
    first = published_by_a_first_host(tmp_path, hub_of, monkeypatch)

    def no(self, name):
        raise AssertionError("the lane's state was written")

    monkeypatch.setattr(State, "writing", no)
    first.sync()
    host(tmp_path / "first").engine  # noqa: B018 - a restart on the same host
