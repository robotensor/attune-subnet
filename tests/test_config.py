"""The validator's config: only what differs between hosts, everything else fixed in code."""

import os
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
start_block = 1000
"""


def write(directory: Path, text: str, name: str = "c.toml") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(text)
    return path


def test_the_env_beside_a_config_is_read_and_an_exported_variable_wins(tmp_path, monkeypatch):
    from robotensor.config import read_env

    path = write(tmp_path, GOOD)
    (path.parent / ".env").write_text(
        '# the token\nHF_TOKEN="hf_abc"\nexport OTHER=x=y\nSET=file\n\nnot a line\n'
    )
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("OTHER", raising=False)
    monkeypatch.setenv("SET", "shell")

    load(path)

    assert read_env(path.parent / ".env") == {"HF_TOKEN": "hf_abc", "OTHER": "x=y", "SET": "file"}
    assert os.environ["HF_TOKEN"] == "hf_abc" and os.environ["OTHER"] == "x=y"
    assert os.environ["SET"] == "shell"
    assert read_env(tmp_path / "none") == {}


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
    assert vector.simulator_root == "/checkout/RoboTwin-Vector"
    assert (vector.mirror, vector.kings_repo) == ("vector-results-test", "vector-kings-test")
    assert (vector.workers, vector.policy_kwargs) == (1, {"device": "cuda:0"})
    assert vector.start_block == 1000


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


def test_a_local_chain_may_leave_the_start_block_out_and_reads_every_commitment(tmp_path):
    text = GOOD.replace('"test"', '"ws://127.0.0.1:9944"').replace("start_block = 1000\n", "")
    assert load(write(tmp_path, text)).vector.start_block == 0


def test_a_local_chain_counts_more_blocks(tmp_path):
    cfg = load(write(tmp_path, GOOD.replace('"test"', '"ws://127.0.0.1:9944"')))
    assert cfg.weights_interval_blocks == 100


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (GOOD.replace("policy_python =", "policy_pythonn ="), "does not take policy_pythonn"),
        (GOOD + 'store = "x"\n', "does not take store"),
        (GOOD + "workers = 0\n", "workers must be a whole number"),
        (GOOD + "workers = 1.5\n", "workers must be a whole number"),
        (GOOD.replace("start_block = 1000\n", ""), "needs start_block"),
        (GOOD.replace("start_block = 1000", "start_block = -1"), "start_block must be a whole"),
        (GOOD.replace("start_block = 1000", "start_block = 10.5"), "start_block must be a whole"),
        (GOOD.replace("start_block = 1000", "start_block = true"), "start_block must be a whole"),
        (
            GOOD.replace("[vector]\n", '[vector]\nkings_repo = "a/b"\n'),
            "kings_repo is a repository",
        ),
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


def test_kings_repo_and_mirror_are_names_and_may_be_turned_off(tmp_path):
    text = GOOD.replace("[vector]\n", '[vector]\nkings_repo = ""\nmirror = "mine"\n')
    cfg = load(write(tmp_path, text))
    assert (cfg.vector.kings_repo, cfg.vector.mirror) == ("", "mine")


@pytest.mark.parametrize("key", ["kings_repo", "mirror"])
def test_a_repository_with_an_owner_is_refused(tmp_path, key):
    text = GOOD.replace("[vector]\n", f'[vector]\n{key} = "someone/name"\n')
    with pytest.raises(ConfigError, match=f"{key} is a repository name"):
        load(write(tmp_path, text))


@pytest.mark.parametrize(
    ("network", "suffix"),
    [("finney", ""), ("test", "-test"), ("ws://127.0.0.1:9944", "-local"), ("local", "-local")],
)
def test_a_test_or_local_chain_never_defaults_to_the_real_repositories(tmp_path, network, suffix):
    text = GOOD.replace('network = "test"', f'network = "{network}"').replace(
        'network = "finney"', f'network = "{network}"'
    )
    vector = load(write(tmp_path, text)).vector
    assert (vector.mirror, vector.kings_repo) == (
        f"vector-results{suffix}",
        f"vector-kings{suffix}",
    )


def test_validator_user_is_gone(tmp_path):
    text = GOOD.replace("[vector]\n", '[vector]\nvalidator_user = "me"\n')
    with pytest.raises(ConfigError, match="validator_user"):
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
        if not (root / name).is_file():
            pytest.skip(f"no config/{name} in this checkout")
        cfg = load(root / name)
        assert cfg.vector.simulator_root and cfg.data == root.parent / "var" / name[: -len(".toml")]
