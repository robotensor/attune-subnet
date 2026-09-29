"""The validator's config: only what differs between hosts, everything else fixed in code."""

from pathlib import Path

import pytest

from robotensor.config import SHARES, ConfigError, load

GOOD = """\
network = "test"
netuid = 7
wallet = { name = "owner", hotkey = "hot" }

[vector]
policy_python = "/env/policy/bin/python"
simulator_python = "/env/sim/bin/python"
simulator_root = "/checkout/RoboTwin-Vector"
"""


def write(directory: Path, text: str, name: str = "c.toml") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(text)
    return path


def test_a_host_config_loads_and_the_rest_is_fixed(tmp_path):
    cfg = load(write(tmp_path, GOOD))
    assert (cfg.network, cfg.netuid, cfg.wallet_name, cfg.wallet_hotkey, cfg.wallet_path) == (
        "test",
        7,
        "owner",
        "hot",
        None,
    )
    assert cfg.shares == {"vector": SHARES["vector"]} == {"vector": 0.30}
    assert cfg.weights_interval_blocks == 360
    vector = cfg.vector
    assert vector.simulator_root == "/checkout/RoboTwin-Vector" and vector.mirror == ""
    assert (vector.workers, vector.policy_kwargs, vector.private_window_blocks) == (
        1,
        {"device": "cuda:0"},
        300,
    )


def test_data_is_kept_beside_the_config_or_in_var_name_for_one_in_config(tmp_path):
    here = load(write(tmp_path / "work", GOOD))
    assert (
        here.data == tmp_path / "work" / "var" and here.state == tmp_path / "work" / "var" / "state"
    )
    assert here.vector.store == tmp_path / "work" / "var" / "vector" / "store"
    repo = load(write(tmp_path / "repo" / "config", GOOD, "testnet.toml"))
    assert repo.data == tmp_path / "repo" / "var" / "testnet" and repo.root == tmp_path / "repo"
    assert repo.vector.run_dir == tmp_path / "repo" / "var" / "testnet" / "vector" / "runs"


def test_relative_paths_hang_from_the_root(tmp_path):
    text = GOOD.replace('"/checkout/RoboTwin-Vector"', '"RoboTwin-Vector"').replace(
        'hotkey = "hot" }', 'hotkey = "hot", path = "var/wallets" }'
    )
    cfg = load(write(tmp_path / "repo" / "config", text, "localnet.toml"))
    assert cfg.vector.simulator_root == str(tmp_path / "repo" / "RoboTwin-Vector")
    assert cfg.wallet_path == str(tmp_path / "repo" / "var" / "wallets")


def test_a_host_sets_how_many_units_each_of_its_gpus_runs(tmp_path):
    assert load(write(tmp_path, GOOD + "workers = 3\n")).vector.workers == 3


def test_a_local_chain_counts_more_blocks(tmp_path):
    cfg = load(write(tmp_path, GOOD.replace('"test"', '"ws://127.0.0.1:9944"')))
    assert cfg.weights_interval_blocks == 100 and cfg.vector.private_window_blocks == 1200


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (GOOD.replace("policy_python =", "policy_pythonn ="), "does not take policy_pythonn"),
        (GOOD + 'store = "x"\n', "does not take store"),
        (GOOD + "workers = 0\n", "workers must be a whole number"),
        (GOOD + "workers = 1.5\n", "workers must be a whole number"),
        (GOOD.replace("netuid = 7\n", ""), "needs netuid"),
        (GOOD.replace('policy_python = "/env/policy/bin/python"\n', ""), "needs policy_python"),
        ("network = 'test'\nnetuid = 1\n", "a validator runs at least one"),
        (GOOD + "[lanes.vector]\n", "the file does not take lanes"),
        (
            GOOD.replace('hotkey = "hot" }', 'hotkey = "hot", pat = "x" }'),
            "wallet does not take pat",
        ),
    ],
)
def test_a_config_this_build_cannot_run_is_refused_by_name(tmp_path, text, message):
    with pytest.raises(ConfigError, match=message):
        load(write(tmp_path, text))


def test_horizon_keeps_its_rounds_in_the_data_directory(tmp_path):
    cfg = load(write(tmp_path, GOOD + "\n[horizon]\ngenesis_block = 5\nwindow_blocks = 100\n"))
    assert cfg.horizon.rounds == tmp_path / "var" / "horizon" / "rounds"
    assert (cfg.horizon.genesis_block, cfg.horizon.window_blocks, cfg.horizon.share) == (
        5,
        100,
        0.0,
    )
    assert cfg.shares == {"vector": 0.30, "horizon": 0.0}


def test_the_repositorys_configs_load():
    root = Path(__file__).resolve().parents[1] / "config"
    for name in ("testnet.toml", "localnet.toml"):
        cfg = load(root / name)
        assert cfg.vector.simulator_root and cfg.data == root.parent / "var" / name[: -len(".toml")]
