"""Reading a config: what it refuses, and where its relative paths hang from."""

import pytest

from robotensor.config import ConfigError, load

GOOD = """
[chain]
network = "ws://127.0.0.1:9944"
netuid = 2

[validator]
burn_remainder = true

[lanes.vector]
share = 0.30
spec = "specs/vector_level1.json"
policy_python = "python"
simulator_python = "python"
"""


def write(tmp_path, text, name="config.toml"):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_relative_paths_hang_from_the_config_file_unless_it_says_otherwise(tmp_path):
    cfg = load(write(tmp_path, GOOD))

    assert cfg.root == tmp_path
    assert cfg.vector.spec == tmp_path / "specs" / "vector_level1.json"
    assert cfg.state == tmp_path / "var" / "validator-state"


def test_paths_root_moves_where_everything_hangs_from(tmp_path):
    (tmp_path / "config").mkdir()
    path = write(tmp_path / "config", '[paths]\nroot = ".."\n' + GOOD)

    cfg = load(path)

    assert cfg.root == tmp_path and cfg.vector.spec == tmp_path / "specs" / "vector_level1.json"


def test_a_key_this_build_does_not_know_is_refused_by_name(tmp_path):
    """A typo is otherwise read as "not set", and the validator runs something else."""
    with pytest.raises(ConfigError, match="policy_pythonn"):
        load(write(tmp_path, GOOD.replace("policy_python =", "policy_pythonn =")))

    with pytest.raises(ConfigError, match="burn_hotkeys"):
        load(write(tmp_path, GOOD.replace("burn_remainder = true", "burn_hotkeys = 'x'")))


def test_a_competition_this_build_does_not_run_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="horizon"):
        load(write(tmp_path, GOOD + "\n[lanes.horizon]\nshare = 0.70\n"))


def test_shares_that_do_not_claim_the_whole_emission_need_saying_so(tmp_path):
    """Burning most of a subnet's emission is a decision, not a default: without
    `burn_remainder` the shares must add up."""
    with pytest.raises(ConfigError, match="0.3 of the emission"):
        load(write(tmp_path, GOOD.replace("burn_remainder = true", "")))

    whole = GOOD.replace("burn_remainder = true", "").replace("share = 0.30", "share = 1.0")
    assert load(write(tmp_path, whole)).shares == {"vector": 1.0}


def test_shares_over_the_whole_emission_are_refused(tmp_path):
    with pytest.raises(ConfigError, match="sum to at most 1"):
        load(write(tmp_path, GOOD.replace("share = 0.30", "share = 1.30")))


def test_a_lane_that_names_no_contract_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="needs spec"):
        load(write(tmp_path, GOOD.replace('spec = "specs/vector_level1.json"', "")))


HORIZON = GOOD + "\n[lanes.horizon]\nshare = 0.0\n"


def test_horizon_keeps_its_rounds_under_the_root_unless_the_config_says_where(tmp_path):
    assert load(write(tmp_path, HORIZON)).horizon.rounds == tmp_path / "var" / "horizon" / "rounds"

    cfg = load(write(tmp_path, HORIZON + 'rounds = "runs/rounds"\n'))

    assert cfg.horizon.rounds == tmp_path / "runs" / "rounds"


def test_a_config_written_when_rounds_were_epochs_still_runs(tmp_path):
    """`epochs` is what `[lanes.horizon]` called the rounds' directory: read as `rounds`."""
    cfg = load(write(tmp_path, HORIZON + 'epochs = "var/horizon/epochs"\n'))

    assert cfg.horizon.rounds == tmp_path / "var" / "horizon" / "epochs"


def test_a_key_under_both_its_names_is_refused(tmp_path):
    """Which of the two was meant cannot be told."""
    both = HORIZON + 'epochs = "var/horizon/epochs"\nrounds = "var/horizon/rounds"\n'

    with pytest.raises(ConfigError, match="both rounds and epochs"):
        load(write(tmp_path, both))
