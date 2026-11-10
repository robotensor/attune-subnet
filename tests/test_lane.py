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
        #: Repositories gated without this account let in: metadata shows, the download does not.
        self.not_let_in = set()

    def auth_check(self, repo, repo_type=None, token=None):
        from huggingface_hub.errors import GatedRepoError

        if repo in self.not_let_in:
            raise GatedRepoError(
                "403", response=SimpleNamespace(headers={}, status_code=403, request=None)
            )

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


def listing(names, sha256=None):
    """A repository as `model_info` shows it: `names`, each file's LFS sha256 from `sha256`."""
    sha256 = sha256 or {}
    return SimpleNamespace(
        siblings=[
            SimpleNamespace(
                rfilename=name,
                size=1,
                lfs=SimpleNamespace(sha256=sha256[name]) if name in sha256 else None,
            )
            for name in names
        ]
    )


@pytest.fixture(autouse=True)
def sound_token(monkeypatch):
    """The validator's own token works, unless a test says otherwise."""
    from robotensor import hub

    monkeypatch.setattr(hub, "token_problem", lambda **kw: None)


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
        kings=tmp_path / "kings",
    )
    lane = VectorLane(cfg, State(tmp_path / "state"))
    # No Hub here: whatever the kings would ask of it fails at once, as if it did not answer.
    lane.kings._api = SimpleNamespace()
    return lane


@pytest.fixture
def hub_of(monkeypatch):
    """Point the lane's Hub lookups at a `FakeHub` of `repos`; the fake, to change later."""
    from robotensor import hub

    real, really = hub.inspect, hub.accessible

    def install(repos):
        fake = FakeHub(repos)
        monkeypatch.setattr(hub, "inspect", lambda *a, **kw: real(*a, **{**kw, "api": fake}))
        monkeypatch.setattr(hub, "accessible", lambda *a, **kw: really(*a, **{**kw, "api": fake}))
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


def test_an_entry_that_keeps_failing_goes_behind_the_rest_and_keeps_its_submission(
    lane, hub_of, monkeypatch
):
    """Whatever the cause: one entry must not hold the queue, and a harness down loses nobody's."""
    from vector_orchestrator.duel.orchestrate import DuelFailed

    from robotensor.lanes import vector
    from robotensor.lanes.base import FAILED, WORKED

    hub_of({f"m/bad@{A}": {"model.safetensors": "w1"}, f"m/two@{B}": {"model.safetensors": "w2"}})
    lane.intake([c("hk1", "m/bad", A, 10), c("hk2", "m/two", B, 20)], 30)
    asked = []
    good = genesis_run(asked)

    def run(req):
        if req.challenger.repo == "m/bad":
            raise DuelFailed("the check crashed")
        return good(req)

    monkeypatch.setattr(lane.engine, "run", run)
    for _ in range(vector.DUEL_TRIES):
        assert lane.step(chain=FakeChain()).outcome == FAILED
    assert [e.repo for e in lane.queue()] == ["m/two", "m/bad"]
    assert lane.step(chain=FakeChain()).outcome == WORKED
    assert [r.challenger.repo for r in asked] == ["m/two"]
    assert statuses(lane) == {"hk1": "queued", "hk2": "crowned"}


def test_an_entry_that_sets_off_anything_unexpected_is_held_the_same_way(lane, hub_of, monkeypatch):
    from robotensor.lanes import vector
    from robotensor.lanes.base import FAILED

    hub_of({f"m/bad@{A}": {"model.safetensors": "w1"}, f"m/two@{B}": {"model.safetensors": "w2"}})
    lane.intake([c("hk1", "m/bad", A, 10), c("hk2", "m/two", B, 20)], 30)

    def run(req):
        raise TypeError("unhashable type: list")

    monkeypatch.setattr(lane.engine, "run", run)
    progress = lane.step(chain=FakeChain())
    assert progress.outcome == FAILED and "unhashable" in progress.detail
    for _ in range(vector.DUEL_TRIES - 1):
        lane.step(chain=FakeChain())
    assert lane.queue()[0].repo == "m/two"


def past_the_tries(lane, monkeypatch):
    """Steps, `HUB_GAP_S` apart, until the entry at the head has had all its tries."""
    from robotensor.lanes import vector

    clock = [1000.0]
    monkeypatch.setattr(vector, "_now", lambda: clock[0])
    for _ in range(vector.HUB_TRIES):
        progress = lane.step(chain=FakeChain())
        clock[0] += vector.HUB_GAP_S
    return progress


def test_a_challenger_the_hub_does_not_show_is_skipped_for_the_next(lane, hub_of, monkeypatch):
    from robotensor.lanes.base import WORKED

    hub_of({f"m/two@{B}": {"model.safetensors": "w2"}})
    lane.intake([c("hk1", "m/gone", A, 10), c("hk2", "m/two", B, 20)], 30)
    asked = []
    monkeypatch.setattr(lane.engine, "run", genesis_run(asked))

    progress = past_the_tries(lane, monkeypatch)

    assert progress.outcome == WORKED
    assert [r.challenger.repo for r in asked] == ["m/two"]
    assert statuses(lane) == {"hk1": "missing", "hk2": "crowned"}
    (gone,) = [r for r in lane.state.lane("vector")["entries"].values() if r["hotkey"] == "hk1"]
    assert "not downloadable" in gone["reason"]
    assert not lane.queue()


def test_a_hub_that_does_not_answer_is_asked_again_three_times_in_all(lane, monkeypatch):
    from robotensor import hub
    from robotensor.lanes import vector
    from robotensor.lanes.base import IDLE, WAITING

    def down(*a, **kw):
        raise hub.HubUnreachable("503")

    monkeypatch.setattr(hub, "inspect", down)
    clock = [1000.0]
    monkeypatch.setattr(vector, "_now", lambda: clock[0])
    lane.intake([c("hk1", "m/one", A, 10)], 30)
    for n in range(1, vector.HUB_TRIES):
        progress = lane.step(chain=FakeChain())
        assert progress.outcome == WAITING and f"({n} of {vector.HUB_TRIES})" in progress.detail
        assert statuses(lane) == {"hk1": "queued"} and lane.queue()
        clock[0] += vector.HUB_GAP_S
    progress = lane.step(chain=FakeChain())
    assert progress.outcome == IDLE and "503" in progress.detail
    assert statuses(lane) == {"hk1": "missing"} and not lane.queue()


def test_a_step_sooner_than_the_gap_after_a_silent_hub_waits_without_asking(lane, monkeypatch):
    """Three tries in three consecutive steps would span 90 seconds; a short outage must not use
    a challenger's submission up. A step within `HUB_GAP_S` of the last try does not count."""
    from robotensor import hub
    from robotensor.lanes import vector
    from robotensor.lanes.base import WAITING

    asked = []

    def down(*a, **kw):
        asked.append(1)
        raise hub.HubUnreachable("503")

    monkeypatch.setattr(hub, "inspect", down)
    clock = [1000.0]
    monkeypatch.setattr(vector, "_now", lambda: clock[0])
    lane.intake([c("hk1", "m/one", A, 10)], 30)
    assert lane.step(chain=FakeChain()).outcome == WAITING
    for _ in range(5):
        clock[0] += 10
        progress = lane.step(chain=FakeChain())
        assert progress.outcome == WAITING and "again in" in progress.detail
    assert len(asked) == 1 and statuses(lane) == {"hk1": "queued"}
    assert vector.HUB_GAP_S == 90
    clock[0] += vector.HUB_GAP_S
    lane.step(chain=FakeChain())
    assert len(asked) == 2


def test_a_token_that_stopped_working_uses_nobodys_submission_up(lane, hub_of, monkeypatch):
    """Revoked or regenerated while the validator runs: every repository then looks shut. The queue
    waits, no try is counted, and with a working token all of it goes on."""
    from robotensor import hub
    from robotensor.lanes.base import WAITING, WORKED

    hub_of({f"m/one@{A}": {"model.safetensors": "w1"}, f"m/two@{B}": {"model.safetensors": "w2"}})
    fake_refusal = hub.inspect

    def refused(*a, **kw):
        raise hub.NotVisible("401")

    monkeypatch.setattr(hub, "inspect", refused)
    monkeypatch.setattr(hub, "token_problem", lambda **kw: "the Hub refuses this token")
    lane.intake([c("hk1", "m/one", A, 10), c("hk2", "m/two", B, 20)], 30)
    for _ in range(10):
        progress = lane.step(chain=FakeChain())
        assert progress.outcome == WAITING and "refuses this token" in progress.detail
    assert statuses(lane) == {"hk1": "queued", "hk2": "queued"}
    assert all("hub_tries" not in r for r in lane.state.lane("vector")["entries"].values())

    monkeypatch.setattr(hub, "inspect", fake_refusal)
    monkeypatch.setattr(hub, "token_problem", lambda **kw: None)
    monkeypatch.setattr(lane.engine, "run", genesis_run([]))
    assert lane.step(chain=FakeChain()).outcome == WORKED


def test_a_hub_down_for_everything_uses_nobodys_submission_up(lane, monkeypatch):
    from robotensor import hub
    from robotensor.lanes import vector

    def down(*a, **kw):
        raise hub.HubUnreachable("503")

    clock = [1000.0]
    monkeypatch.setattr(vector, "_now", lambda: clock[0])
    monkeypatch.setattr(hub, "inspect", down)
    monkeypatch.setattr(hub, "token_problem", lambda **kw: "the Hub did not answer")
    lane.intake([c("hk1", "m/one", A, 10)], 30)
    for _ in range(20):
        lane.step(chain=FakeChain())
        clock[0] += vector.HUB_GAP_S
    assert statuses(lane) == {"hk1": "queued"}


def test_a_repository_still_private_after_its_commit_gets_its_three_tries(
    lane, hub_of, monkeypatch
):
    """The chain has the commitment and `reveal` is a moment behind it, or was run by hand."""
    from robotensor.lanes import vector
    from robotensor.lanes.base import WAITING, WORKED

    clock = [1000.0]
    monkeypatch.setattr(vector, "_now", lambda: clock[0])
    fake = hub_of({})
    lane.intake([c("hk1", "m/one", A, 10)], 30)
    monkeypatch.setattr(lane.engine, "run", genesis_run([]))
    assert lane.step(chain=FakeChain()).outcome == WAITING
    assert statuses(lane) == {"hk1": "queued"}
    clock[0] += vector.HUB_GAP_S
    fake.repos[f"m/one@{A}"] = {"model.safetensors": "w1"}
    assert lane.step(chain=FakeChain()).outcome == WORKED
    assert statuses(lane) == {"hk1": "crowned"}


def test_a_gated_repository_the_validator_is_not_let_into_is_missing(lane, hub_of, monkeypatch):
    """Its metadata shows to anyone, so the digest check passes; the download would not."""
    fake = hub_of(
        {f"m/shut@{A}": {"model.safetensors": "w1"}, f"m/two@{B}": {"model.safetensors": "w2"}}
    )
    fake.not_let_in.add("m/shut")
    lane.intake([c("hk1", "m/shut", A, 10), c("hk2", "m/two", B, 20)], 30)
    asked = []
    monkeypatch.setattr(lane.engine, "run", genesis_run(asked))

    past_the_tries(lane, monkeypatch)

    assert [r.challenger.repo for r in asked] == ["m/two"]
    assert statuses(lane) == {"hk1": "missing", "hk2": "crowned"}
    (shut,) = [r for r in lane.state.lane("vector")["entries"].values() if r["hotkey"] == "hk1"]
    assert "may not download" in shut["reason"]


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
        del record["digest"]  # these tests' files stand for weights and hash to nothing named
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


def test_the_four_newest_crowned_models_are_kept_and_the_oldest_pushed_out(lane):
    """The kings' weights come from the orchestrator's cache, where each one's duel left them;
    `kings/` holds the four newest and the fifth is dropped."""
    from vector_orchestrator.submissions.resolve import Resolved

    from robotensor import kings as kings_

    refs = history(lane, "a", "b", "c", "d", "e")
    cache = lane.engine.runtime.cache
    for ref in refs:
        resolved = Resolved(repo=ref.repo, revision=ref.revision, sha=ref.revision)
        cache.store(resolved, lambda into, r=ref: (into / "model.safetensors").write_text(r.repo))

    lane.keep_kings()

    assert lane.kings.kept() == [r.key for r in reversed(refs)][:4]
    assert sorted(p.name for p in lane.kings.root.iterdir() if p.is_dir()) == sorted(
        r.key for r in refs[1:]
    )
    assert lane.kings.path(refs[-1].key).read_text() == "m/e"
    assert lane.kings.manifest()["kings"][0] == {
        "place": 1, "key": refs[-1].key, "repo": "m/e", "revision": refs[-1].revision,
        "sha256": None, "weights": True,
    }  # fmt: skip
    assert kings_.KEPT == 4


def test_weights_that_do_not_hash_to_what_the_duel_recorded_are_not_kept(tmp_path):
    """A wrong copy - on the Hub, in a cache - is refused, and removed when it had been fetched
    into `kings/`; the king is wanted again next time."""
    import hashlib

    from robotensor.kings import Kings

    kings = Kings(tmp_path / "kings", "", api=object())
    good = tmp_path / "good"
    good.write_bytes(b"weights")
    sha = hashlib.sha256(b"weights").hexdigest()
    model = {"key": "a", "repo": "m/a", "revision": "1" * 40, "sha256": sha}

    def from_hub(m):  # as `fetch_from_hub` does: straight into its place
        fetched = kings.path(m["key"])
        fetched.parent.mkdir(parents=True, exist_ok=True)
        fetched.write_bytes(b"tampered")
        return fetched

    assert kings.keep([model], from_hub) == ["a"]
    assert not kings.path("a").exists() and kings.manifest()["kings"][0]["weights"] is False

    assert kings.keep([model], lambda m: good) == []
    assert kings.path("a").read_bytes() == b"weights"
    assert kings.manifest()["kings"][0]["sha256"] == sha


def test_a_kept_king_carries_the_sha256_its_admission_recorded(lane):
    (one,) = history(lane, "one")
    lane.state.lane("vector")["entries"][one.key] = {"hotkey": "hk1", "weights_sha256": "f" * 64}

    (model,) = lane.kept_models()
    assert model["key"] == one.key and model["sha256"] == "f" * 64


def test_a_host_restored_from_the_hub_copy_takes_the_kings_without_uploading_them_again(tmp_path):
    """`kings/` empty, the Hub copy ahead of it: the weights come down, and bringing the Hub
    level is the manifest alone."""
    from robotensor.kings import Kings

    calls = []

    class Api:
        def whoami(self):
            return {"name": "v"}

        def hf_hub_download(self, repo, filename, repo_type=None, local_dir=None):
            calls.append(("download", filename))
            target = Path(local_dir) / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"weights")
            return str(target)

        def create_repo(self, repo, **kw):
            pass

        def model_info(self, repo, files_metadata=None):
            return listing(["kings.json", "a/model.safetensors"])

        def create_commit(self, repo, operations, commit_message, repo_type=None):
            calls.append(("commit", [o.path_in_repo for o in operations]))

    kings = Kings(tmp_path / "kings", "kings", api=Api())
    model = {"key": "a", "repo": "m/a", "revision": "1" * 40}
    assert kings.keep([model], lambda m: kings.fetch_from_hub(m["key"])) == []
    assert kings.path("a").read_bytes() == b"weights"
    assert kings.publish() is True
    assert calls == [("download", "a/model.safetensors"), ("commit", ["kings.json"])]


def test_a_king_without_weights_anywhere_is_tried_again_next_step(lane, monkeypatch):
    from robotensor.lanes import vector

    (one,) = history(lane, "one")
    monkeypatch.setattr(vector.VectorLane, "_king_weights", lambda self, model: None)
    lane.keep_kings()
    assert lane.kings.kept() == [one.key] and not lane.kings.path(one.key).exists()
    assert lane.kings.manifest()["kings"][0]["weights"] is False

    found = lane.engine.runtime.cache.root / "w"
    found.parent.mkdir(parents=True, exist_ok=True)
    found.write_text("weights")
    monkeypatch.setattr(vector.VectorLane, "_king_weights", lambda self, model: found)
    lane.keep_kings()
    assert lane.kings.path(one.key).read_text() == "weights"


def test_a_folder_that_is_not_a_kings_is_never_removed(tmp_path):
    """Whatever the directory is pointed at, only a king's folder is this code's to drop."""
    from robotensor.kings import Kings

    root = tmp_path / "kings"
    (root / "src").mkdir(parents=True)
    (root / "src" / "code.py").write_text("x")
    (root / "old").mkdir()
    (root / "old" / "model.safetensors").write_text("w")
    kings = Kings(root, api=SimpleNamespace())
    kings.keep([{"key": "new"}], lambda m: None)
    assert (root / "src" / "code.py").is_file() and not (root / "old").exists()
    with pytest.raises(ValueError):
        Kings("relative/kings")


def test_a_badly_named_entry_is_dropped_and_the_next_one_duels(lane, hub_of, monkeypatch):
    """`a/b..c` passes the commitment's format; the Hub's client raises before asking anything."""
    from huggingface_hub.utils import validate_repo_id

    fake = hub_of({f"m/two@{B}": {"model.safetensors": "w2"}})
    plain = fake.model_info

    def model_info(repo, revision, files_metadata):
        validate_repo_id(repo)
        return plain(repo, revision, files_metadata)

    fake.model_info = model_info
    lane.intake([c("hk1", "a/b..c", A, 10), c("hk2", "m/two", B, 20)], 30)
    asked = []
    monkeypatch.setattr(lane.engine, "run", genesis_run(asked))

    past_the_tries(lane, monkeypatch)

    entries = lane.state.lane("vector")["entries"].values()
    bad = next(e for e in entries if e["repo"] == "a/b..c")
    assert bad["status"] == "missing" and "not a repository name" in bad["reason"]
    assert [r.challenger.repo for r in asked] == ["m/two"]


def test_a_new_pod_takes_the_kings_from_the_hub_copy(lane, hub_of, monkeypatch):
    """Nothing on this host: `keep_kings` fills `kings/` from the kings repository, and the king's
    duel is served from there, without its miner's repository being asked for anything."""
    from vector_orchestrator.submissions.resolve import Resolved

    (king,) = history(lane, "king")

    class Api:
        def whoami(self):
            return {"name": "v"}

        def hf_hub_download(self, repo, filename, repo_type=None, local_dir=None, revision=None):
            assert repo == "v/vector-kings", "asked the miner, not the kings repository"
            target = Path(local_dir) / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("mirrored weights")
            return str(target)

        def create_repo(self, repo, **kw):
            pass

        def model_info(self, repo, files_metadata=None):
            return listing(["kings.json", f"{king.key}/model.safetensors"])

        def create_commit(self, repo, operations, commit_message, repo_type=None):
            pass

    lane.kings.name, lane.kings._api = "vector-kings", Api()
    hub_of({f"m/two@{B}": {"model.safetensors": "w2"}})
    lane.intake([c("hk2", "m/two", B, 20)], 30)
    asked = []
    monkeypatch.setattr(lane.engine, "run", genesis_run(asked))

    lane.step(chain=FakeChain())

    assert lane.kings.path(king.key).read_text() == "mirrored weights"
    kept = lane.engine.runtime.cache.lookup(
        Resolved(repo=king.repo, revision=king.revision, sha=king.revision)
    )
    assert kept is not None and (kept.root / "model.safetensors").read_text() == "mirrored weights"
    assert [r.king.key for r in asked] == [king.key]


def test_a_king_is_restored_into_the_cache_from_kings_before_its_duel(lane, hub_of, monkeypatch):
    """A host with no checkout of the king fetches it from `kings/`, never from the miner."""
    from vector_orchestrator.submissions.resolve import Resolved

    (king,) = history(lane, "king")
    lane.kings.path(king.key).parent.mkdir(parents=True)
    lane.kings.path(king.key).write_text("kept weights")
    hub_of({f"m/two@{B}": {"model.safetensors": "w2"}})
    lane.intake([c("hk2", "m/two", B, 20)], 30)
    asked = []
    monkeypatch.setattr(lane.engine, "run", genesis_run(asked))

    lane.step(chain=FakeChain())

    cache = lane.engine.runtime.cache
    kept = cache.lookup(Resolved(repo=king.repo, revision=king.revision, sha=king.revision))
    assert kept is not None and (kept.root / "model.safetensors").read_text() == "kept weights"
    assert [r.king.key for r in asked] == [king.key]


def test_the_kings_hub_copy_is_brought_level_in_one_commit(tmp_path):
    """Weights not there uploaded, folders no longer kept deleted, the manifest with them; and
    nothing asked of the Hub while it already matches."""
    from robotensor.kings import Kings

    calls = []

    class Api:
        def whoami(self):
            return {"name": "v"}

        def create_repo(self, repo, **kw):
            calls.append(("create", repo))

        def model_info(self, repo, files_metadata=None):
            return listing(["kings.json", "old/model.safetensors", "b/model.safetensors"])

        def create_commit(self, repo, operations, commit_message, repo_type=None):
            calls.append(("commit", [(type(o).__name__, o.path_in_repo) for o in operations]))

    kings = Kings(tmp_path / "kings", "kings", api=Api())
    (tmp_path / "w").write_text("weights")
    newest = [{"key": k, "repo": f"m/{k}", "revision": "1" * 40} for k in ("a", "b")]
    assert kings.keep(newest, lambda m: tmp_path / "w") == []

    assert kings.publish() is True
    assert calls == [
        ("create", "v/kings"),
        (
            "commit",
            [
                ("CommitOperationAdd", "kings.json"),
                ("CommitOperationAdd", "a/model.safetensors"),
                ("CommitOperationDelete", "old/model.safetensors"),
            ],
        ),
    ]
    calls.clear()
    assert kings.publish() is True and calls == []


def test_a_king_without_weights_holds_the_others_back_from_the_hub_no_longer(tmp_path):
    """Three of four here: those go up, and the fourth's folder on the Hub is not touched."""
    from robotensor.kings import Kings

    commits = []

    class Api:
        def whoami(self):
            return {"name": "v"}

        def create_repo(self, repo, **kw):
            pass

        def model_info(self, repo, files_metadata=None):
            return listing(["d/model.safetensors"])

        def create_commit(self, repo, operations, commit_message, repo_type=None):
            commits.append([(type(o).__name__, o.path_in_repo) for o in operations])

    kings = Kings(tmp_path / "kings", "kings", api=Api())
    (tmp_path / "w").write_text("weights")
    newest = [{"key": k, "repo": f"m/{k}", "revision": "1" * 40} for k in "abcd"]
    assert kings.keep(newest, lambda m: None if m["key"] == "d" else tmp_path / "w") == ["d"]

    assert kings.publish() is False  # one still to come: the next step tries again
    assert commits == [
        [
            ("CommitOperationAdd", "kings.json"),
            ("CommitOperationAdd", "a/model.safetensors"),
            ("CommitOperationAdd", "b/model.safetensors"),
            ("CommitOperationAdd", "c/model.safetensors"),
        ]
    ]
    assert kings.publish() is False and len(commits) == 1  # nothing new: nothing asked


def test_a_hub_copy_that_cannot_be_written_is_tried_again_next_time(tmp_path):
    from robotensor.kings import Kings

    class Api:
        def whoami(self):
            return {"name": "v"}

        def create_repo(self, repo, **kw):
            raise OSError("no network")

    kings = Kings(tmp_path / "kings", "kings", api=Api())
    (tmp_path / "w").write_text("weights")
    kings.keep([{"key": "a", "repo": "m/a", "revision": "1" * 40}], lambda m: tmp_path / "w")
    assert kings.publish() is False
    assert not (tmp_path / "kings" / "published.json").exists()


def test_the_kings_repository_is_made_under_the_token_s_account_and_again_on_a_new_one(tmp_path):
    from robotensor.kings import Kings

    created = []

    def api(name):
        class Api:
            def whoami(self):
                return {"name": name}

            def create_repo(self, repo, **kw):
                created.append(repo)

            def model_info(self, repo, files_metadata=None):
                return listing([])

            def create_commit(self, repo, operations, commit_message, repo_type=None):
                pass

        return Api()

    (tmp_path / "w").write_text("weights")
    first = Kings(tmp_path / "kings", "vector-kings", api=api("alice"))
    first.keep([{"key": "a", "repo": "m/a", "revision": "1" * 40}], lambda m: tmp_path / "w")
    assert first.publish() is True and first.publish() is True
    assert created == ["alice/vector-kings"]

    assert Kings(tmp_path / "kings", "vector-kings", api=api("bob")).publish() is True
    assert created == ["alice/vector-kings", "bob/vector-kings"]


def test_a_kings_owner_puts_the_repository_under_it_whatever_the_token_s_account(lane, tmp_path):
    from dataclasses import replace

    from robotensor.kings import Kings

    created = []

    class Api:
        def whoami(self):
            pytest.fail("the token's account was asked")

        def create_repo(self, repo, **kw):
            created.append(repo)

        def model_info(self, repo, files_metadata=None):
            return listing([])

        def create_commit(self, repo, operations, commit_message, repo_type=None):
            pass

    (tmp_path / "w").write_text("weights")
    kings = Kings(tmp_path / "own", "vector-kings", owner="robotensor", api=Api())
    kings.keep([{"key": "a", "repo": "m/a", "revision": "1" * 40}], lambda m: tmp_path / "w")
    assert kings.publish() is True and created == ["robotensor/vector-kings"]
    cfg = replace(lane.cfg, kings_repo="vector-kings", kings_owner="robotensor")
    assert VectorLane(cfg, lane.state).kings.repo == "robotensor/vector-kings"


def test_a_kings_repository_with_an_owner_is_refused(tmp_path):
    from robotensor.kings import Kings

    with pytest.raises(ValueError, match="a name"):
        Kings(tmp_path / "kings", "someone/vector-kings")


def test_a_hub_that_cannot_name_the_account_is_tried_again_next_time(tmp_path):
    from robotensor import hub
    from robotensor.kings import Kings

    class Api:
        def whoami(self):
            raise OSError("no network")

    kings = Kings(tmp_path / "kings", "vector-kings", api=Api())
    (tmp_path / "w").write_text("weights")
    kings.keep([{"key": "a", "repo": "m/a", "revision": "1" * 40}], lambda m: tmp_path / "w")
    assert kings.publish() is False
    with pytest.raises(hub.HubUnreachable):
        hub.account(api=Api())


def test_an_index_naming_no_king_empties_nothing(tmp_path):
    """A new pod whose results are off or empty: the kings held, here and on the Hub, stand."""
    from robotensor.kings import Kings

    class Api:
        def __getattr__(self, name):
            pytest.fail(f"the Hub was asked: {name}")

    (tmp_path / "w").write_text("weights")
    kings = Kings(tmp_path / "kings", "vector-kings", api=Api())
    kings.keep([{"key": "a", "repo": "m/a", "revision": "1" * 40}], lambda m: tmp_path / "w")
    kings.keep([], lambda m: None)
    assert kings.path("a").is_file()
    assert Kings(tmp_path / "other", "vector-kings", api=Api()).publish() is True


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def test_the_hub_copy_is_compared_once_per_process_whatever_published_json_says(tmp_path):
    """A wrong copy with weights here is replaced; one with none here is removed, not left public;
    and the comparison is made even though an earlier process wrote that it was level."""
    from robotensor.kings import Kings

    commits = []

    class Api:
        def whoami(self):
            return {"name": "v"}

        def create_repo(self, repo, **kw):
            pass

        def model_info(self, repo, files_metadata=None):
            return listing(
                ["a/model.safetensors", "b/model.safetensors", "c/model.safetensors"],
                {
                    "a/model.safetensors": _sha("tampered"),
                    "b/model.safetensors": _sha("tampered"),
                    "c/model.safetensors": _sha("c"),
                },
            )

        def create_commit(self, repo, operations, commit_message, repo_type=None):
            commits.append([(type(o).__name__, o.path_in_repo) for o in operations])

    root = tmp_path / "kings"
    newest = [
        {"key": "a", "repo": "m/a", "revision": "1" * 40, "sha256": _sha("a")},
        {"key": "b", "repo": "m/b", "revision": "1" * 40, "sha256": _sha("b")},
        {"key": "c", "repo": "m/c", "revision": "1" * 40, "sha256": _sha("c")},
    ]
    (tmp_path / "a").write_text("a")
    (tmp_path / "c").write_text("c")
    source = {"a": tmp_path / "a", "b": None, "c": tmp_path / "c"}
    kings = Kings(root, "vector-kings", api=Api())
    kings.keep(newest, lambda m: source[m["key"]])
    assert kings.publish() is False
    assert commits == [
        [
            ("CommitOperationAdd", "kings.json"),
            ("CommitOperationAdd", "a/model.safetensors"),
            ("CommitOperationDelete", "b/model.safetensors"),
        ]
    ]
    assert kings.publish() is False and len(commits) == 1

    again = Kings(root, "vector-kings", api=Api())
    again.keep(newest, lambda m: source[m["key"]])
    again.publish()
    assert len(commits) == 2


def test_a_kept_file_is_hashed_once_per_process_and_fetched_again_when_wrong(tmp_path):
    from robotensor.kings import Kings

    model = {"key": "a", "repo": "m/a", "revision": "1" * 40, "sha256": _sha("a")}
    (tmp_path / "a").write_text("a")
    root = tmp_path / "kings"
    (root / "a").mkdir(parents=True)
    (root / "a" / "model.safetensors").write_text("rotted")
    kings = Kings(root)
    assert kings.keep([model], lambda m: tmp_path / "a") == []
    assert kings.path("a").read_text() == "a"


def test_a_new_pod_works_out_the_kings_from_the_results_and_puts_the_kings_repo_right(
    tmp_path, remote, hub_of, monkeypatch
):
    """Five crowned on one host. On a new one, with only the token: the results say who the newest
    four are and the sha256 of each; the kings repository, one king missing, one wrong and one
    dethroned, is put right. The right one is neither fetched from a miner nor uploaded again."""
    from robotensor.kings import Kings

    revisions = [str(n) * 40 for n in range(1, 6)]
    weights = {r: f"weights {r[0]}" for r in revisions}
    hub_of({f"m/{r[0]}@{r}": {"model.safetensors": weights[r]} for r in revisions})
    first = host(tmp_path / "first")
    first.intake(
        [c(f"hk{r[0]}", f"m/{r[0]}", r, 10 + n, weights[r]) for n, r in enumerate(revisions)],
        30,
    )
    asked = []

    def run(req):
        asked.append(req)
        store = first.engine.store
        event_id = f"{len(asked):064x}"
        record = {"kind": req.kind, "event_id": event_id, "block": req.block}
        if req.king is None:
            record.update(king=req.challenger.as_dict(), dethroned=False)
        else:
            record.update(
                king=req.king.as_dict(), challenger=req.challenger.as_dict(), dethroned=True
            )
        store.write_event({"event_id": event_id})
        store.append(record)
        return SimpleNamespace(
            published=True, status="published", reason=req.kind, event_id=event_id,
            as_dict=lambda: {"reason": req.kind, "record": record},
        )  # fmt: skip

    monkeypatch.setattr(first.engine, "run", run)
    for _ in revisions:
        first.step(chain=FakeChain())
    assert len(asked) == 5

    entries = first.state.lane("vector")["entries"]
    key_of = {e["revision"]: k for k, e in entries.items()}
    five, four, three, two, one = (key_of[r] for r in reversed(revisions))
    by_key = dict(entries)

    downloads = []

    class Api:
        def whoami(self):
            return {"name": "org"}

        def create_repo(self, repo, **kw):
            pass

        def model_info(self, repo, files_metadata=None):
            return listing(
                [f"{k}/model.safetensors" for k in (five, four, one)],
                {
                    f"{five}/model.safetensors": by_key[five]["weights_sha256"],
                    f"{four}/model.safetensors": _sha("tampered"),
                    f"{one}/model.safetensors": by_key[one]["weights_sha256"],
                },
            )

        def hf_hub_download(self, repo, filename, repo_type=None, local_dir=None, revision=None):
            downloads.append((repo, filename))
            if repo == "org/vector-kings":
                key = filename.split("/")[0]
                if key == three:
                    raise OSError("not there")
                text = "tampered" if key == four else weights[by_key[key]["revision"]]
                target = Path(local_dir) / filename
            else:
                text = weights[revision]
                target = tmp_path / "miner" / revision / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)
            return str(target)

        def create_commit(self, repo, operations, commit_message, repo_type=None):
            commits.append([(type(o).__name__, o.path_in_repo) for o in operations])

    commits = []
    new = host(tmp_path / "new")
    new.kings = Kings(new.cfg.kings, "vector-kings", token="t", api=Api())
    new.keep_kings()

    assert new.kings.kept() == [five, four, three, two]
    for key in (five, four, three, two):
        assert new.kings.path(key).read_text() == weights[by_key[key]["revision"]]
    assert not new.kings.path(one).exists()
    assert ("org/vector-kings", f"{five}/model.safetensors") in downloads
    assert ("m/5", "model.safetensors") not in downloads
    (ops,) = commits
    assert sorted(ops) == sorted(
        [
            ("CommitOperationAdd", "kings.json"),
            ("CommitOperationAdd", f"{four}/model.safetensors"),
            ("CommitOperationAdd", f"{three}/model.safetensors"),
            ("CommitOperationAdd", f"{two}/model.safetensors"),
            ("CommitOperationDelete", f"{one}/model.safetensors"),
        ]
    )


def test_a_store_older_than_its_mirror_is_pushed_whole_and_a_new_pod_takes_it_up(
    tmp_path, remote, hub_of, monkeypatch
):
    """An upgrade: duels were published with no mirror. Turned on, the mirror gets all of it, so a
    new pod finds the king and the spent submissions rather than an empty history."""
    hub_of({f"m/one@{A}": {"model.safetensors": "w1"}})
    cfg = host(tmp_path / "old").cfg
    old = VectorLane(dataclasses.replace(cfg, mirror=""), State(tmp_path / "old" / "state"))
    old.kings._api = SimpleNamespace()
    old.intake([c("hk1", "m/one", A, 10)], 30)
    monkeypatch.setattr(old.engine, "run", crowning_run(old, []))
    old.step(chain=FakeChain())
    assert not remote["files"]

    upgraded = host(tmp_path / "old")
    upgraded.step(chain=FakeChain())
    assert {"manifest.json", "head.json"} <= set(remote["files"])
    pushes = dict(remote["files"])
    upgraded.step(chain=FakeChain())
    assert remote["files"] == pushes

    new = host(tmp_path / "new")
    new.intake([c("hk1", "m/other", B, 40)], 50)
    assert new.ready() and statuses(new)["hk1"] in ("crowned", "refused")
    assert [e["status"] for e in new.state.lane("vector")["entries"].values()].count("crowned") == 1


def test_an_account_not_permitted_takes_nothing_up(tmp_path, remote, monkeypatch):
    """`status` under the wrong token must not fill this store with that account's history."""
    from robotensor import hub

    monkeypatch.setattr(hub, "account", lambda **kw: "someone")
    with pytest.raises(RuntimeError, match="not a permitted account"):
        host(tmp_path / "a").engine  # noqa: B018
    assert not (tmp_path / "a" / ".hub_account").exists()


def test_the_account_is_remembered_for_a_hub_that_does_not_answer(tmp_path, remote, monkeypatch):
    from robotensor import hub

    host(tmp_path / "a").engine  # noqa: B018

    def down(**kw):
        raise hub.HubUnreachable("503")

    monkeypatch.setattr(hub, "account", down)
    assert host(tmp_path / "a")._account() == "org"
    with pytest.raises(hub.HubUnreachable):
        host(tmp_path / "b")._account()


def test_a_resumed_duel_waits_for_a_token_that_works_too(lane, hub_of, monkeypatch):
    """Admitted, then the token went: the download would read as the entry being shut."""
    from robotensor import hub
    from robotensor.lanes.base import WAITING, WORKED

    hub_of({f"m/one@{A}": {"model.safetensors": "w1"}})
    lane.intake([c("hk1", "m/one", A, 10)], 30)
    from vector_orchestrator.duel.orchestrate import DuelFailed

    def fails(req):
        raise DuelFailed("harness")

    monkeypatch.setattr(lane.engine, "run", fails)
    lane.step(chain=FakeChain())
    asked = []
    monkeypatch.setattr(lane.engine, "run", genesis_run(asked))
    monkeypatch.setattr(hub, "token_problem", lambda **kw: "the Hub refuses this token")
    assert lane.step(chain=FakeChain()).outcome == WAITING and not asked
    assert statuses(lane) == {"hk1": "queued"}
    monkeypatch.setattr(hub, "token_problem", lambda **kw: None)
    assert lane.step(chain=FakeChain()).outcome == WORKED and asked


def test_the_tries_start_again_after_an_admission(lane, hub_of, monkeypatch):
    from robotensor.lanes import vector

    clock = [1000.0]
    monkeypatch.setattr(vector, "_now", lambda: clock[0])
    fake = hub_of({})
    lane.intake([c("hk1", "m/one", A, 10)], 30)
    lane.step(chain=FakeChain())
    fake.repos[f"m/one@{A}"] = {"model.safetensors": "w1"}
    clock[0] += vector.HUB_GAP_S
    monkeypatch.setattr(lane.engine, "run", genesis_run([]))
    lane.step(chain=FakeChain())
    (record,) = lane.state.lane("vector")["entries"].values()
    assert "hub_tries" not in record and "hub_tried_at" not in record


def test_a_king_with_no_recorded_sha256_is_held_to_its_commitments_digest(lane):
    from robotensor.lanes import vector  # noqa: F401

    (king,) = history(lane, "king")
    entries = lane.state.lane("vector")["entries"]
    entries[king.key] = {"hotkey": "h", "digest": "d" * 64}
    assert lane.kept_models()[0]["sha256"] == "d" * 64


# -- the kings repository: always the newest four, always what the results say ---------------------


class KingsHub:
    """The Hub as `Kings` and a miner's repository are asked of it, holding files for real; `down`
    is a Hub that fails every call. Weights are text whose sha256 is what `digest` gives."""

    def __init__(self, labels=None):
        self.files = {}
        self.labels = labels or {}
        self.down = False
        self.write_refused = False

    def _ask(self):
        if self.down:
            raise OSError("the Hub is down")

    def whoami(self):
        self._ask()
        return {"name": "org"}

    def create_repo(self, repo, **kw):
        self._ask()

    def model_info(self, repo, files_metadata=None):
        self._ask()
        return listing(sorted(self.files), {n: _sha(v) for n, v in self.files.items()})

    def create_commit(self, repo, operations, commit_message, repo_type=None):
        self._ask()
        if self.write_refused:
            raise OSError("403")
        for op in operations:
            if type(op).__name__ == "CommitOperationDelete":
                self.files.pop(op.path_in_repo, None)
            elif isinstance(op.path_or_fileobj, bytes):
                self.files[op.path_in_repo] = op.path_or_fileobj.decode()
            else:
                self.files[op.path_in_repo] = Path(op.path_or_fileobj).read_text()

    def hf_hub_download(
        self, repo, filename, repo_type=None, local_dir=None, revision=None, cache_dir=None
    ):
        from huggingface_hub.errors import RemoteEntryNotFoundError

        self._ask()
        if revision is not None:  # the miner's own repository
            text = self.labels[revision]
        elif filename in self.files:
            text = self.files[filename]
        else:
            raise RemoteEntryNotFoundError(
                "404", response=SimpleNamespace(headers={}, status_code=404, request=None)
            )
        target = Path(local_dir or cache_dir or self.labels["tmp"]) / (revision or "") / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        return str(target)

    def kings(self):
        return sorted(n.split("/")[0] for n in self.files if n.endswith("/model.safetensors"))


def pod(path, api):
    """A validator on a pod of its own, with a real `Kings` on the Hub `api`."""
    from robotensor.kings import Kings

    cfg = VectorConfig(
        name="vector",
        share=0.3,
        store=path / "store",
        run_dir=path / "runs",
        cache=path / "cache",
        policy_python="python",
        simulator_python="python",
        simulator_root="/checkout/RoboTwin-Vector",
        mirror="results",
        kings_repo="vector-kings",
        kings=path / "kings",
    )
    lane = VectorLane(cfg, State(path / "state"), hub_token="t")
    lane.kings = Kings(cfg.kings, "vector-kings", token="t", api=api)
    return lane


def really_crowns(lane, labels, asked, bare=None):
    """What a duel leaves: the challenger's weights in the cache, and its crowning in the results,
    pushed. A king duelled without its weights on the host goes on `bare`, for the test to read:
    the lane catches whatever a duel raises."""
    from vector_orchestrator.submissions.resolve import Resolved

    def run(req):
        cache = lane.engine.runtime.cache
        if req.king is not None:
            held = Resolved(repo=req.king.repo, revision=req.king.revision, sha=req.king.revision)
            if cache.lookup(held) is None:
                assert bare is not None, "a king was duelled without its weights"
                bare.append(req.king.key)
        c = req.challenger
        cache.store(
            Resolved(repo=c.repo, revision=c.revision, sha=c.revision),
            lambda into: (into / "model.safetensors").write_text(labels[c.revision]),
        )
        store, event_id = lane.engine.store, _sha(f"{c.key}{req.block}")
        record = {"kind": req.kind, "event_id": event_id, "block": req.block}
        if req.king is None:
            record.update(king=c.as_dict(), dethroned=False)
        else:
            record.update(king=req.king.as_dict(), challenger=c.as_dict(), dethroned=True)
        store.write_event({"event_id": event_id})
        store.append(record)
        lane.engine.push_touched()
        asked.append(c.key)
        return SimpleNamespace(
            published=True, status="published", reason=req.kind, event_id=event_id,
            as_dict=lambda: {"reason": req.kind, "record": record},
        )  # fmt: skip

    lane.engine.run = run


def contest(hub_of, n, tmp_path):
    revisions = [f"{i:x}" * 40 for i in range(1, n + 1)]
    labels = {r: f"weights {r[0]}" for r in revisions}
    labels["tmp"] = str(tmp_path / "downloads")
    hub_of({f"m/{r[0]}@{r}": {"model.safetensors": labels[r]} for r in revisions})
    commitments = [
        c(f"hk{r[0]}", f"m/{r[0]}", r, 10 + i, labels[r]) for i, r in enumerate(revisions)
    ]
    return revisions, labels, commitments


def newest_four(lane):
    return [m["key"] for m in lane.kept_models()]


def test_every_crowning_leaves_the_kings_repo_holding_the_newest_four(tmp_path, remote, hub_of):
    revisions, labels, commitments = contest(hub_of, 7, tmp_path)
    api = KingsHub(labels)
    lane = pod(tmp_path / "a", api)
    lane.intake(commitments, 100)
    asked = []
    really_crowns(lane, labels, asked)
    for n in range(1, len(revisions) + 1):
        lane.step(chain=FakeChain())
        assert len(asked) == n
        want = newest_four(lane)
        assert len(want) == min(n, 4) and sorted(want) == api.kings()
        assert json.loads(api.files["kings.json"])["kings"][0]["key"] == asked[-1]
        for model in lane.kept_models():
            assert _sha(api.files[f"{model['key']}/model.safetensors"]) == model["sha256"]


def test_a_king_whose_weights_cannot_be_found_never_duels_and_duels_once_they_are_back(
    tmp_path, remote, hub_of
):
    """A new pod, one failed download: the queue waits; nothing forfeits, nobody loses a turn."""
    from robotensor.lanes.base import WAITING, WORKED

    revisions, labels, commitments = contest(hub_of, 2, tmp_path)
    api = KingsHub(labels)
    first = pod(tmp_path / "a", api)
    first.intake(commitments[:1], 100)
    really_crowns(first, labels, [])
    first.step(chain=FakeChain())

    new = pod(tmp_path / "new", api)
    new.engine  # noqa: B018 - results read
    api.down = True
    new.intake(commitments, 100)
    asked = []
    really_crowns(new, labels, asked)
    for _ in range(3):
        progress = new.step(chain=FakeChain())
        assert progress.outcome == WAITING and "not resolved" in progress.detail and not asked
    assert [e.repo for e in new.queue()] == ["m/2"]
    api.down = False
    assert new.step(chain=FakeChain()).outcome == WORKED and len(asked) == 1


def test_a_kings_repo_that_cannot_be_written_holds_the_duels_until_it_can(tmp_path, remote, hub_of):
    from robotensor.lanes.base import WAITING, WORKED

    revisions, labels, commitments = contest(hub_of, 2, tmp_path)
    api = KingsHub(labels)
    lane = pod(tmp_path / "a", api)
    lane.intake(commitments[:1], 100)
    asked = []
    really_crowns(lane, labels, asked)
    api.write_refused = True
    lane.step(chain=FakeChain())
    assert not api.kings()
    lane.intake(commitments, 100)
    progress = lane.step(chain=FakeChain())
    assert progress.outcome == WAITING and len(asked) == 1
    api.write_refused = False
    assert lane.step(chain=FakeChain()).outcome == WORKED and len(asked) == 2
    assert sorted(newest_four(lane)) == api.kings()


def test_results_that_cannot_be_written_hold_the_kings_repo_back_too(tmp_path, remote, hub_of):
    """It may never name a king the results do not, or a new pod would strike that king off."""
    from robotensor.lanes.base import WAITING

    revisions, labels, commitments = contest(hub_of, 2, tmp_path)
    api = KingsHub(labels)
    lane = pod(tmp_path / "a", api)
    lane.engine  # noqa: B018
    lane.intake(commitments, 100)
    really_crowns(lane, labels, [])
    remote["unreachable"] = True
    lane.step(chain=FakeChain())
    assert not api.kings()
    assert lane.step(chain=FakeChain()).outcome == WAITING
    remote.pop("unreachable")
    lane.step(chain=FakeChain())
    assert sorted(newest_four(lane)) == api.kings()
    assert remote["files"]["head.json"]


def test_results_empty_and_kings_named_is_not_a_new_competition(tmp_path, remote, hub_of):
    """A first crowning here would strike all four off the kings repository."""
    from robotensor.lanes.base import WAITING

    revisions, labels, commitments = contest(hub_of, 1, tmp_path)
    api = KingsHub(labels)
    api.files["kings.json"] = json.dumps({"kings": [{"key": "old"}]})
    api.files["old/model.safetensors"] = "w"
    lane = pod(tmp_path / "a", api)
    lane.intake(commitments, 100)
    asked = []
    really_crowns(lane, labels, asked)
    progress = lane.step(chain=FakeChain())
    assert progress.outcome == WAITING and "names ['old']" in progress.detail and not asked
    assert api.kings() == ["old"]


def test_a_kill_between_a_records_append_and_the_heads_write_does_not_stop_the_queue(
    tmp_path, remote, hub_of
):
    from robotensor.lanes.base import WORKED

    revisions, labels, commitments = contest(hub_of, 2, tmp_path)
    api = KingsHub(labels)
    lane = pod(tmp_path / "a", api)
    lane.intake(commitments[:1], 100)
    asked = []
    really_crowns(lane, labels, asked)
    lane.step(chain=FakeChain())
    store = lane.engine.store
    head = store.head()
    record = {"kind": "duel", "event_id": "f" * 64, "block": head["block"] + 1,
              "king": head["king"], "challenger": head["king"], "dethroned": False}  # fmt: skip
    store.write_event({"event_id": record["event_id"]})
    store.append(record)
    store.head_path().write_text(json.dumps(head))  # as the kill left it

    again = pod(tmp_path / "a", api)
    again.intake(commitments, 100)
    really_crowns(again, labels, asked)
    assert again.step(chain=FakeChain()).outcome == WORKED and len(asked) == 2


def test_the_wait_for_a_kings_weights_says_which_and_what_to_do(tmp_path, remote, hub_of):
    revisions, labels, commitments = contest(hub_of, 2, tmp_path)
    api = KingsHub(labels)
    first = pod(tmp_path / "a", api)
    first.intake(commitments[:1], 100)
    really_crowns(first, labels, [])
    first.step(chain=FakeChain())

    # The pod died before its king was uploaded, and the miner's repository is gone.
    king = first.engine.store.head()["king"]
    api.files.clear()
    del labels[king["revision"]]
    new = pod(tmp_path / "new", api)
    new.intake(commitments, 100)
    really_crowns(new, labels, [])
    detail = new.step(chain=FakeChain()).detail
    assert king["repo"] in detail and f"{king['key']}/model.safetensors" in detail

    # Someone uploads them, and it goes on by itself.
    api.files[f"{king['key']}/model.safetensors"] = f"weights {king['revision'][0]}"
    assert new.step(chain=FakeChain()).outcome == "worked"


@pytest.mark.parametrize("seed", range(25))
def test_the_kings_stay_right_through_failures_and_pods_coming_and_going(
    tmp_path, remote, hub_of, seed
):
    """The Hub and the results failing at random, and the pod replaced by an empty one at random.
    No king is ever duelled without its weights (`really_crowns`), and once things are quiet the
    kings repository is exactly the newest four the results name, right to the byte."""
    import random

    rng = random.Random(seed)
    revisions, labels, commitments = contest(hub_of, 8, tmp_path)
    api = KingsHub(labels)
    pods, asked, bare = 0, [], []
    lane = pod(tmp_path / "pod0", api)

    def step():
        try:
            lane.intake(commitments, 100)
            really_crowns(lane, labels, asked, bare)
            lane.step(chain=FakeChain())
        except Exception:  # noqa: BLE001 - the worker logs it and steps again
            pass

    for _ in range(60):
        api.down = rng.random() < 0.3
        api.write_refused = rng.random() < 0.2
        remote["unreachable"] = rng.random() < 0.2
        if rng.random() < 0.15:
            pods += 1
            lane = pod(tmp_path / f"pod{pods}", api)
        step()

    api.down, api.write_refused = False, False
    remote.pop("unreachable")
    for _ in range(len(revisions) + 3):
        step()

    final = pod(tmp_path / "final", api)
    final.engine  # noqa: B018
    really_crowns(final, labels, asked)
    final.step(chain=FakeChain())
    assert not bare, "a king was duelled without its weights"
    assert not final.queue() and len(final.kept_models()) == 4
    assert sorted(newest_four(final)) == api.kings()
    assert [k["key"] for k in json.loads(api.files["kings.json"])["kings"]] == newest_four(final)
    for model in final.kept_models():
        text = api.files[f"{model['key']}/model.safetensors"]
        assert _sha(text) == model["sha256"]
        assert final.kings.path(model["key"]).read_text() == text


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
        if self.remote.get("unreachable"):
            raise OSError("no route to host")
        for name in files:
            if (self.root / name).is_file():
                self.remote["files"][name] = (self.root / name).read_bytes()

    def push_all(self, message="full mirror"):
        from vector_orchestrator.store.mirror import check_is_store

        self.push(check_is_store(self.root), message)


@pytest.fixture
def remote(monkeypatch):
    from vector_orchestrator.store import mirror

    from robotensor import hub

    shared = {"files": {}}
    monkeypatch.setattr(hub, "account", lambda **kw: "org")
    monkeypatch.setattr(hub, "PERMITTED_VALIDATORS", ("org",))

    def named(root, repo, token=None):
        assert repo == "org/results", "the results go under the token's account"
        return FakeMirror(shared, root)

    monkeypatch.setattr(mirror, "Mirror", named)
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
        mirror="results",
        kings=path / "kings",
    )
    lane = VectorLane(cfg, State(path / "state"))
    lane.kings._api = SimpleNamespace()
    # These tests are about the results: a king's weights are taken as resolved.
    lane._restore_king = lambda king: True
    lane.kings.resolved = lambda key: True
    return lane


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
