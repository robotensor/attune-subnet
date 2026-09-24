"""Horizon's chain side: which epoch a commitment entered, and what the lane says it does."""

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
        epochs=tmp_path / "epochs",
        key=tmp_path / "keys" / "k.ed25519",
        private_window_blocks=10,
    )
    return HorizonLane(cfg, State(tmp_path / "state"))


def c(hotkey, repo, sha, block):
    return Commitment(hotkey=hotkey, block=block, data=commitment.encode(repo, sha, lane="horizon"))


def test_the_lane_is_what_the_validator_asks_of_a_competition(lane):
    assert isinstance(lane, Lane) and lane.name == "horizon"


def test_a_commitment_enters_the_epoch_that_was_open_when_it_was_made(lane, monkeypatch):
    """Not the one open when a validator got round to reading it."""
    monkeypatch.setattr(
        "robotensor.hub.inspect",
        lambda *a, **k: SimpleNamespace(content_key="k1", weights_bytes=9, files=(), manifest=()),
    )

    [entry] = lane.intake([c("hk1", "m/one", A, 1150)], block=9000)

    assert entry.epoch == 1 and entry.status == "queued"
    assert [e.key for e in lane.entrants(1)] == ["m/one@" + A]
    assert lane.entrants(0) == []


def test_a_commitment_from_before_the_first_epoch_entered_nothing(lane):
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


def test_a_step_says_where_the_open_epoch_stands(lane):
    progress = lane.step(SimpleNamespace(block=lambda: 1150))

    assert progress.outcome == IDLE
    assert "epoch 1 is open until block 1200" in progress.detail


def test_with_no_engine_configured_nothing_scores_the_closed_epoch(lane):
    """A validator that runs only the other competition says so, rather than failing."""
    progress = lane.step(SimpleNamespace(block=lambda: 1150))

    assert "no engine configured to score them" in progress.detail


def test_an_epoch_waits_until_its_seed_block_is_final(lane):
    """The units are drawn from a block after the window closed; reading it early would read a
    block that can still be reorganised."""
    progress = lane.step(SimpleNamespace(block=lambda: 1101))

    assert progress.outcome == WAITING and "seed is not final" in progress.detail


def test_the_first_epoch_has_nothing_before_it(lane):
    progress = lane.step(SimpleNamespace(block=lambda: 1050))

    assert progress.outcome == IDLE and "none has closed yet" in progress.detail


def test_with_no_closed_epoch_there_is_nobody_to_pay(lane):
    award = lane.award()

    assert list(award.entries) == [] and award.keep == 1 and award.decay == 0.0


def test_a_share_above_zero_without_serve_as_is_refused(tmp_path):
    """A model served as the user running the validator can read `private/expert.npz` - the
    answer key for the episode it is being scored on."""
    from robotensor.config import load

    path = tmp_path / "c.toml"
    path.write_text(
        '[chain]\nnetwork = "test"\nnetuid = 2\n'
        "[validator]\nburn_remainder = true\n"
        "[lanes.horizon]\nshare = 0.70\n"
    )

    with pytest.raises(ConfigError, match="serve_as"):
        load(path)

    path.write_text(
        '[chain]\nnetwork = "test"\nnetuid = 2\n'
        "[validator]\nburn_remainder = true\n"
        '[lanes.horizon]\nshare = 0.70\nserve_as = "horizon"\n'
    )
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
    object.__setattr__(lane.cfg, "epochs", tmp_path / "epochs")
    ran = []

    def fake_run(argv, **kwargs):
        ran.append(argv)
        (tmp_path / "epochs" / "e00001").mkdir(parents=True, exist_ok=True)
        (tmp_path / "epochs" / "e00001" / "epoch.json").write_text("{}")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("robotensor.lanes.horizon.subprocess.run", fake_run)
    monkeypatch.setattr(type(lane), "_register", lambda self, engine, window: None)

    progress = lane.step(SimpleNamespace(block=lambda: 1250))

    assert progress.outcome == WORKED and progress.detail == "epoch e00001: open"
    # The store and the key that signs it are made once, before the first epoch is opened.
    assert [argv[3:5] for argv in ran] == [["store", "init"], ["epoch", "open"]]
    assert lane.state.lane("horizon")["epochs"]["e00001"]["stage"] == "open"

    # The next step picks the epoch up where the engine left it.
    ran.clear()
    monkeypatch.setattr("robotensor.lanes.horizon.subprocess.run", fake_run)
    assert lane.step(SimpleNamespace(block=lambda: 1250)).detail == "epoch e00001: pool"


def test_a_verb_that_fails_is_reported_and_not_noted(lane, tmp_path, monkeypatch):
    from robotensor.lanes.base import FAILED

    engine_config = tmp_path / "competition.yml"
    engine_config.write_text("axes: {}\n")
    object.__setattr__(lane.cfg, "competition", engine_config)
    object.__setattr__(lane.cfg, "epochs", tmp_path / "epochs")
    monkeypatch.setattr(
        "robotensor.lanes.horizon.subprocess.run", lambda argv, **k: SimpleNamespace(returncode=2)
    )
    monkeypatch.setattr(type(lane), "_register", lambda self, engine, window: None)

    progress = lane.step(SimpleNamespace(block=lambda: 1250))

    assert progress.outcome == FAILED and "exited 2" in progress.detail
    assert not (lane.state.lane("horizon").get("epochs") or {}).get("e00001")
