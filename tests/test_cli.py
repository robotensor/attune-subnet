"""One command for both halves of the subnet: what it routes, and which competition it means."""

import pytest

from robotensor.cli import (
    COMPETITION_ENV,
    LEGACY_COMPETITION_ENV,
    CompetitionError,
    build_parser,
    competition,
)


def test_the_flag_wins_over_the_environment_and_the_config():
    chosen = competition("vector", {"vector": 1, "horizon": 2}, {COMPETITION_ENV: "horizon"})

    assert chosen == "vector"


def test_the_environment_saves_saying_it_on_every_command():
    assert competition(None, {"vector": 1, "horizon": 2}, {COMPETITION_ENV: "horizon"}) == "horizon"


def test_the_environment_variable_s_old_name_still_counts():
    old = {LEGACY_COMPETITION_ENV: "horizon"}
    assert competition(None, {"vector": 1, "horizon": 2}, old) == "horizon"
    both = {COMPETITION_ENV: "vector", LEGACY_COMPETITION_ENV: "horizon"}
    assert competition(None, {"vector": 1, "horizon": 2}, both) == "vector"


def test_one_competition_enabled_needs_no_saying():
    assert competition(None, {"vector": 1}, {}) == "vector"


def test_two_enabled_and_nothing_said_is_refused_by_name():
    """A miner who meant the other one should not find out from a refused submission."""
    with pytest.raises(CompetitionError, match="vector, horizon"):
        competition(None, {"vector": 1, "horizon": 2}, {})


def test_a_competition_this_validator_does_not_run_is_refused():
    with pytest.raises(CompetitionError, match="horizon is not"):
        competition("horizon", {"vector": 1}, {})


def test_the_command_routes_both_halves():
    parser = build_parser()

    miner = parser.parse_args(["miner", "check", "--dir", "submission/"])
    validator = parser.parse_args(["validator", "--config", "c.toml", "status"])

    assert miner.part == "miner" and miner.command == "check"
    assert validator.part == "validator" and validator.command == "status"
    assert validator.config == "c.toml"


def test_the_version_is_the_packages():
    from robotensor import __version__

    with pytest.raises(SystemExit) as exit_:
        build_parser().parse_args(["--version"])

    assert exit_.value.code == 0 and __version__


def test_init_writes_a_config_that_loads(tmp_path):
    """`init` writes a config holding only what differs between hosts, and the directories its data
    is kept in."""
    from robotensor.config import load
    from robotensor.init import validator

    path = validator(tmp_path / "node")

    cfg = load(path)
    assert cfg.root == tmp_path / "node"
    assert cfg.vector.store == tmp_path / "node" / "var" / "vector" / "store"
    assert cfg.shares == {"vector": 0.30}
    assert (tmp_path / "node" / "var" / "state").is_dir()
    assert "wallet" in path.read_text() and "policy_python" in path.read_text()


def test_init_will_not_write_over_a_config(tmp_path):
    from robotensor.init import validator

    validator(tmp_path / "node")

    with pytest.raises(FileExistsError):
        validator(tmp_path / "node")


def test_submit_is_one_command_in_the_order_that_keeps_weights_yours():
    parser = build_parser()

    args = parser.parse_args(
        [
            "miner",
            "submit",
            "--dir",
            "s/",
            "--repo",
            "me/mine",
            "--netuid",
            "2",
            "--wallet.name",
            "w",
            "--wallet.hotkey",
            "h",
        ]
    )

    assert args.command == "submit" and not args.keep_private
