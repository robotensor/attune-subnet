"""Horizon's chain side: which round a commitment entered, and what the lane says it does."""

import hashlib
from types import SimpleNamespace

import pytest

from robotensor.chain import Commitment
from robotensor.config import ConfigError, HorizonConfig
from robotensor.lanes.base import IDLE, WAITING, Lane
from robotensor.lanes.horizon import HorizonLane
from robotensor.protocol import commitment
from robotensor.state import State

A, B = "a" * 40, "b" * 40
SHARDS = "transformer/diffusion_pytorch_model-00001-of-00002.safetensors"


class FakeHub:
    """`model_info` for a few repositories: {repo@sha: {file: bytes or None}}; None is a plain git
    blob, which the Hub has no sha256 for."""

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
                size=len(data or b""),
                lfs=SimpleNamespace(sha256=hashlib.sha256(data).hexdigest()) if data else None,
            )
            for name, data in files.items()
        ]
        return SimpleNamespace(sha=revision, siblings=siblings)

    def fetch(self, repo, revision):
        """What intake reads of the kilobyte files the Hub has no sha256 for."""
        files = self.repos[f"{repo}@{revision}"]
        return lambda name: files[name] or b""


def submission(**extra):
    return {
        SHARDS: b"shard one",
        "transformer/config.json": None,
        "norm_stats/robotwin.json": None,
        "zerowam.yaml": None,
        **extra,
    }


@pytest.fixture
def lane(tmp_path):
    cfg = HorizonConfig(
        name="horizon",
        share=0.0,
        genesis_block=1000,
        window_blocks=100,
        store=tmp_path / "store",
        rounds=tmp_path / "rounds",
        private_window_blocks=10,
    )
    return HorizonLane(cfg, State(tmp_path / "state"))


def c(hotkey, repo, sha, block):
    return Commitment(hotkey=hotkey, block=block, data=commitment.encode(repo, sha, lane="horizon"))


def test_the_lane_is_what_the_validator_asks_of_a_competition(lane):
    assert isinstance(lane, Lane) and lane.name == "horizon"


def test_a_commitment_enters_the_round_that_was_open_when_it_was_made(lane, monkeypatch):
    """Not the one open when a validator got round to reading it."""
    monkeypatch.setattr(
        "robotensor.hub.inspect",
        lambda *a, **k: SimpleNamespace(content_key="k1", weights_bytes=9, files=(), manifest=()),
    )

    [entry] = lane.intake([c("hk1", "m/one", A, 1150)], block=9000)

    assert entry.round == 1 and entry.status == "queued"
    assert [e.key for e in lane.entrants(1)] == ["m/one@" + A]
    assert lane.entrants(0) == []


def test_a_commitment_from_before_the_first_round_entered_nothing(lane):
    assert lane.intake([c("hk1", "m/one", A, 999)], block=9000) == []


def test_a_submission_is_judged_from_the_hubs_metadata_alone(lane):
    """50 GB of shards: what it holds, and whether it is a copy, are read from the file list."""
    hub = FakeHub({f"m/one@{A}": submission(), f"m/two@{B}": submission()})

    changed = lane.intake(
        [c("hk1", "m/one", A, 1150), c("hk2", "m/two", B, 1160)],
        1200,
        api=hub,
        fetch=hub.fetch("m/one", A),
    )

    first, second = changed
    assert first.status == "queued"
    assert second.status == "duplicate", "the same bytes under another name is one model"


def test_a_repository_holding_what_a_submission_may_not_is_refused(lane):
    hub = FakeHub({f"m/one@{A}": submission(**{"train.py": None})})

    [entry] = lane.intake([c("hk1", "m/one", A, 1150)], 1200, api=hub, fetch=lambda n: b"")

    assert entry.status == "refused"


def test_a_shard_that_is_not_in_lfs_is_refused_rather_than_downloaded(lane):
    hub = FakeHub({f"m/one@{A}": {**submission(), SHARDS: None}})

    [entry] = lane.intake([c("hk1", "m/one", A, 1150)], 1200, api=hub, fetch=lambda n: b"")

    assert entry.status == "refused"


def test_a_step_says_where_the_open_round_stands(lane):
    progress = lane.step(SimpleNamespace(block=lambda: 1150))

    assert progress.outcome == IDLE
    assert "round 1 is open until block 1200" in progress.detail


def test_with_no_engine_configured_nothing_scores_the_closed_round(lane):
    """A validator that runs only the other competition says so, rather than failing."""
    progress = lane.step(SimpleNamespace(block=lambda: 1150))

    assert "no engine configured to score them" in progress.detail


def test_a_round_waits_until_its_seed_block_is_final(lane):
    """The units are drawn from a block after the window closed; reading it early would read a
    block that can still be reorganised."""
    progress = lane.step(SimpleNamespace(block=lambda: 1101))

    assert progress.outcome == WAITING and "seed is not final" in progress.detail


def test_the_first_round_has_nothing_before_it(lane):
    progress = lane.step(SimpleNamespace(block=lambda: 1050))

    assert progress.outcome == IDLE and "none has closed yet" in progress.detail


def test_with_no_closed_round_there_is_nobody_to_pay(lane):
    award = lane.award()

    assert list(award.entries) == [] and award.keep == 1 and award.decay == 0.0


def test_a_share_above_zero_without_serve_as_is_refused(tmp_path, monkeypatch):
    """A model served as the user running the validator can read `private/expert.npz` - the
    answer key for the episode it is being scored on."""
    from robotensor import config
    from robotensor.config import load

    monkeypatch.setitem(config.SHARES, "horizon", 0.70)
    path = tmp_path / "c.toml"
    path.write_text('network = "test"\nnetuid = 2\n[horizon]\n')
    with pytest.raises(ConfigError, match="serve_as"):
        load(path)
    path.write_text('network = "test"\nnetuid = 2\n[horizon]\nserve_as = "horizon"\n')
    assert load(path).horizon.serve_as == "horizon"


def test_the_shape_comes_from_the_runtime_that_will_serve_it(lane):
    """Intake and the runtime must agree on what two submissions of the same bytes are."""
    pytest.importorskip("horizon_runtime_zerowam")
    from horizon_runtime_zerowam import family as family_module

    family = family_module.load()

    assert lane.shape.key == "listing"
    assert f"{family.weights['directory']}/*.safetensors" in lane.shape.hashed
    assert family_module.KNOBS_FILE in lane.shape.hashed


def test_a_step_runs_one_verb_and_notes_it(lane, tmp_path, monkeypatch):
    """One of them can take hours, so the loop does one and comes back."""
    from robotensor.lanes.base import WORKED

    engine_config = tmp_path / "competition.yml"
    engine_config.write_text("axes: {}\n")
    object.__setattr__(lane.cfg, "competition", engine_config)
    object.__setattr__(lane.cfg, "rounds", tmp_path / "rounds")
    ran = []

    def fake_run(argv, **kwargs):
        ran.append(argv)
        (tmp_path / "rounds" / "e00001").mkdir(parents=True, exist_ok=True)
        (tmp_path / "rounds" / "e00001" / "round.json").write_text("{}")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("robotensor.lanes.horizon.subprocess.run", fake_run)
    monkeypatch.setattr(type(lane), "_register", lambda self, engine, window: None)

    progress = lane.step(SimpleNamespace(block=lambda: 1250))

    assert progress.outcome == WORKED and progress.detail == "round e00001: open"
    # The store and the key that signs it are made once, before the first round is opened.
    assert [argv[3:5] for argv in ran] == [["store", "init"], ["round", "open"]]
    assert lane.state.lane("horizon")["rounds"]["e00001"]["stage"] == "open"

    # The next step picks the round up where the engine left it.
    ran.clear()
    monkeypatch.setattr("robotensor.lanes.horizon.subprocess.run", fake_run)
    assert lane.step(SimpleNamespace(block=lambda: 1250)).detail == "round e00001: pool"


def opening(lane, tmp_path, monkeypatch):
    """Step a lane whose round e00001 is next to open; the `round open` it ran."""
    engine_config = tmp_path / "competition.yml"
    engine_config.write_text("axes: {}\n")
    object.__setattr__(lane.cfg, "competition", engine_config)
    object.__setattr__(lane.cfg, "rounds", tmp_path / "rounds")
    (tmp_path / "store").mkdir(exist_ok=True)
    (tmp_path / "store" / "index.json").write_text("{}")
    ran = []

    def fake_run(argv, **kwargs):
        ran.append(argv)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("robotensor.lanes.horizon.subprocess.run", fake_run)
    monkeypatch.setattr(type(lane), "_register", lambda self, engine, window: None)
    lane.step(SimpleNamespace(block=lambda: 1250))
    [opened] = [argv for argv in ran if argv[3:5] == ["round", "open"]]
    return opened


@pytest.mark.parametrize("marker", ["round.json", "epoch.json"])
def test_a_round_opens_where_the_one_before_it_closed(lane, tmp_path, monkeypatch, marker):
    """Back to back, like the chain's windows: after the previous round's directory, whichever
    name its marker has - one opened before the rename holds `epoch.json`."""
    previous = tmp_path / "rounds" / "e00000"
    previous.mkdir(parents=True)
    (previous / marker).write_text("{}")

    argv = opening(lane, tmp_path, monkeypatch)

    assert argv[argv.index("--after") + 1] == str(previous)


def test_a_round_with_no_round_opened_before_it_follows_nothing(lane, tmp_path, monkeypatch):
    """A week this validator never opened, or an open that never finished, has no close to
    start from: the round opens when it is opened."""
    assert "--after" not in opening(lane, tmp_path, monkeypatch)

    (tmp_path / "rounds" / "e00000").mkdir(parents=True)

    assert "--after" not in opening(lane, tmp_path, monkeypatch)


def test_a_verb_that_fails_is_reported_and_not_noted(lane, tmp_path, monkeypatch):
    from robotensor.lanes.base import FAILED

    engine_config = tmp_path / "competition.yml"
    engine_config.write_text("axes: {}\n")
    object.__setattr__(lane.cfg, "competition", engine_config)
    object.__setattr__(lane.cfg, "rounds", tmp_path / "rounds")
    monkeypatch.setattr(
        "robotensor.lanes.horizon.subprocess.run", lambda argv, **k: SimpleNamespace(returncode=2)
    )
    monkeypatch.setattr(type(lane), "_register", lambda self, engine, window: None)

    progress = lane.step(SimpleNamespace(block=lambda: 1250))

    assert progress.outcome == FAILED and "exited 2" in progress.detail
    assert not (lane.state.lane("horizon").get("rounds") or {}).get("e00001")


def closed_round(lane, tmp_path, monkeypatch, *, rank=(), records=None, dry_run=False):
    """Run the last verb of round e00001, whose store holds `records` (by default its close
    record, ranking `rank`), and return the lane's winners."""
    if records is None:
        records = {"round-e00001": {"round_id": "e00001", "scores": {"rank": list(rank)}}}
    engine_config = tmp_path / "competition.yml"
    engine_config.write_text("axes: {}\n")
    object.__setattr__(lane.cfg, "competition", engine_config)
    object.__setattr__(lane.cfg, "rounds", tmp_path / "rounds")
    object.__setattr__(lane.cfg, "store", tmp_path / "store")
    object.__setattr__(lane.cfg, "dry_run", dry_run)
    directory = tmp_path / "rounds" / "e00001"
    directory.mkdir(parents=True)
    for name in ("round.json", "pool_manifest.json", "shortlist.json", "scores.json"):
        (directory / name).write_text("{}")
    # The drains leave no file of their own, so the lane noted them as they finished.
    lane.state.lane("horizon").setdefault("rounds", {})["e00001"] = {"stage": "full"}
    lane.state.save()  # as intake leaves it: on disk, because `writing` re-reads under the lock
    monkeypatch.setattr(
        "robotensor.lanes.horizon.subprocess.run", lambda argv, **k: SimpleNamespace(returncode=0)
    )

    def read(store, name):
        if name not in records:
            raise LookupError(f"{name}: no such record")
        return records[name]

    monkeypatch.setattr("horizon_competition.store.read", read)
    lane.step(SimpleNamespace(block=lambda: 1250))
    return lane.state.lane("horizon").get("winners") or []


def entered(lane):
    """A queued entry of round 1 from hk1; its key, as the engine's register names it."""
    from horizon_competition import submissions

    lane.state.lane("horizon")["entries"]["m/one@" + A] = {
        "hotkey": "hk1",
        "repo": "m/one",
        "revision": A,
        "commit_block": 1150,
        "round": 1,
        "status": "queued",
    }
    return submissions.key_for("m/one", A)


def test_the_winner_is_read_back_from_the_published_record(lane, tmp_path, monkeypatch):
    """Never from what this validator thought was happening: the record is the thing every other
    reader can check."""
    key = entered(lane)

    winners = closed_round(lane, tmp_path, monkeypatch, rank=[key, "other"])

    assert winners == [{"round": 1, "key": key, "hotkey": "hk1"}]
    award = lane.award()
    assert list(award.entries) == ["hk1"] and award.keep == 1 and award.decay == 0.0


def test_a_round_closed_before_the_rename_is_read_from_its_epoch_record(
    lane, tmp_path, monkeypatch
):
    """Nothing published is renamed: its record is still `epoch-<id>`, naming `epoch_id`."""
    key = entered(lane)
    old = {"epoch-e00001": {"epoch_id": "e00001", "scores": {"rank": [key]}}}

    winners = closed_round(lane, tmp_path, monkeypatch, records=old)

    assert winners == [{"round": 1, "key": key, "hotkey": "hk1"}]


def test_the_round_record_is_read_before_the_epoch_one(lane, tmp_path, monkeypatch):
    key = entered(lane)
    both = {
        "round-e00001": {"round_id": "e00001", "scores": {"rank": [key]}},
        "epoch-e00001": {"epoch_id": "e00001", "scores": {"rank": ["somebody__else@1-2"]}},
    }

    winners = closed_round(lane, tmp_path, monkeypatch, records=both)

    assert [w["key"] for w in winners] == [key]


def test_a_record_that_is_another_rounds_pays_nobody(lane, tmp_path, monkeypatch):
    key = entered(lane)
    other = {"round-e00001": {"round_id": "e00002", "scores": {"rank": [key]}}}

    assert closed_round(lane, tmp_path, monkeypatch, records=other) == []


def test_a_round_with_no_close_record_pays_nobody(lane, tmp_path, monkeypatch):
    entered(lane)

    assert closed_round(lane, tmp_path, monkeypatch, records={}) == []


def test_a_rehearsal_pays_nobody(lane, tmp_path, monkeypatch):
    """A dry run publishes no record; recording a winner from one would pay for a rehearsal."""
    key = entered(lane)

    winners = closed_round(lane, tmp_path, monkeypatch, rank=[key], dry_run=True)

    assert winners == [] and list(lane.award().entries) == []


def test_a_winner_no_commitment_names_is_not_paid(lane, tmp_path, monkeypatch):
    """The engine's register can hold an entry this chain never accepted; it is not a hotkey."""
    winners = closed_round(lane, tmp_path, monkeypatch, rank=["somebody__else@1234abcd-0f0f0f"])

    assert winners == []


def test_what_the_lane_noted_before_the_rename_is_still_read(lane, tmp_path, monkeypatch):
    """Its document said `epoch` where it now says `round`: an entry's round, the notes of a
    round's verbs and a winner's round are read under either name, and nothing is rewritten."""
    doc = lane.state.lane("horizon")
    doc["entries"]["m/one@" + A] = {
        "hotkey": "hk1",
        "repo": "m/one",
        "revision": A,
        "commit_block": 1150,
        "epoch": 1,
        "status": "queued",
    }
    doc["epochs"] = {"e00001": {"stage": "screen"}}
    doc["winners"] = [{"epoch": 0, "key": "k0", "hotkey": "hk0"}]
    lane.state.save()

    assert [(e.key, e.round) for e in lane.entrants(1)] == [("m/one@" + A, 1)]

    engine_config = tmp_path / "competition.yml"
    engine_config.write_text("axes: {}\n")
    object.__setattr__(lane.cfg, "competition", engine_config)
    object.__setattr__(lane.cfg, "rounds", tmp_path / "rounds")
    directory = tmp_path / "rounds" / "e00001"
    directory.mkdir(parents=True)
    for name in ("epoch.json", "pool_manifest.json"):
        (directory / name).write_text("{}")
    monkeypatch.setattr(
        "robotensor.lanes.horizon.subprocess.run", lambda argv, **k: SimpleNamespace(returncode=0)
    )

    assert lane.step(SimpleNamespace(block=lambda: 1250)).detail == "round e00001: shortlist"
    assert lane.state.lane("horizon")["rounds"]["e00001"] == {"stage": "shortlist"}
    assert list(lane.award().entries) == ["hk0"]


def test_a_round_whose_winner_was_noted_before_the_rename_is_not_paid_twice(
    lane, tmp_path, monkeypatch
):
    key = entered(lane)
    lane.state.lane("horizon")["winners"] = [{"epoch": 1, "key": key, "hotkey": "hk1"}]

    winners = closed_round(lane, tmp_path, monkeypatch, rank=[key])

    assert winners == [{"epoch": 1, "key": key, "hotkey": "hk1"}]
