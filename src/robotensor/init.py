"""`attune init`: a working directory, and a config that only needs your wallet filled in.

What is fiddly about running a validator is not the code, it is the paths around it. This writes a
config holding only what differs between hosts, and the directories its data is kept in.

What it will not do is guess a wallet or a netuid. Those are the two lines it leaves for you, and
`attune doctor` tells you the moment either is wrong.
"""

from __future__ import annotations

from pathlib import Path

CONFIG = "attune.toml"

VALIDATOR_TOML = """\
# Written by `attune init`. State, stores and runs are kept in var/ beside this file.
# `finney` is mainnet, `test` the test network, or a ws:// address of your own.
network = "{network}"
netuid = {netuid}
# The wallet this validator sets weights with. `btcli wallet list` shows what you have.
wallet = {{ name = "{wallet_name}", hotkey = "{wallet_hotkey}" }}

[vector]
# The two environments a duel needs, and the RoboTwin-Vector checkout the simulator runs from; see
# the README. `attune doctor` checks all three.
policy_python = "{policy_python}"
simulator_python = "{simulator_python}"
simulator_root = "{simulator_root}"
# A Hugging Face dataset the store is published to; empty to keep it local.
mirror = ""
"""

MINER_NEXT = """\
A submission is weights and nothing else. Three commands, in this order:

    attune miner check  --dir {directory}/submission
    attune miner upload --dir {directory}/submission --repo <you>/<name> --private
    attune miner commit --repo <you>/<name> --revision <the sha upload printed> \\
        --netuid <N> --network test --wallet.name <w> --wallet.hotkey <h>

Then make the repository public. Committing while it is private is what keeps your weights from
being copied and committed by somebody else first: the earliest commitment of a set of weights
keeps them.

This subnet charges nothing to submit.
"""


def validator(directory: Path, **values: str) -> Path:
    """Write a validator's config and the directories it names; the config's path."""
    fields = {
        "network": "test",
        "netuid": "0",
        "wallet_name": "validator",
        "wallet_hotkey": "default",
        "policy_python": "/path/to/policy-env/bin/python",
        "simulator_python": "/path/to/simulator-env/bin/python",
        "simulator_root": "/path/to/RoboTwin-Vector",
        **values,
    }
    directory.mkdir(parents=True, exist_ok=True)
    for made in ("var/state", "var/vector/store", "var/vector/runs", "var/vector/cache"):
        (directory / made).mkdir(parents=True, exist_ok=True)
    path = directory / CONFIG
    if path.exists():
        raise FileExistsError(f"{path} is already there; init will not write over a config")
    path.write_text(VALIDATOR_TOML.format(**fields), encoding="utf-8")
    return path


def miner(directory: Path) -> str:
    """Make a miner's working directory; what to run next."""
    (directory / "submission").mkdir(parents=True, exist_ok=True)
    return MINER_NEXT.format(directory=directory)
