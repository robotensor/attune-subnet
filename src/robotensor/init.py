"""`robotensor init`: a working directory, and a config that only needs your wallet filled in.

What is fiddly about running a validator is not the code, it is the twenty paths around it. This
writes them: a config where everything hangs from the directory it is in, the directories
themselves, and a contract named as the installed engine ships it rather than as a path into
somebody else's checkout.

What it will not do is guess a wallet or a netuid. Those are the two lines it leaves for you, and
`robotensor doctor` tells you the moment either is wrong.
"""

from __future__ import annotations

from pathlib import Path

CONFIG = "robotensor.toml"

VALIDATOR_TOML = """\
# Written by `robotensor init`. Everything relative hangs from this file's own directory.
[paths]
root = "."

[chain]
# `finney` is mainnet, `test` the test network, or a ws:// address of your own.
network = "{network}"
netuid = {netuid}

[validator]
# The wallet this validator sets weights with. `btcli wallet list` shows what you have.
wallet_name = "{wallet_name}"
wallet_hotkey = "{wallet_hotkey}"
state = "var/state"
weights_interval_blocks = 360
# Empty: whatever no competition claims goes to the subnet owner's hotkey.
burn_hotkey = ""
# Competition 2 (Horizon) is not open yet, so its share of the emission burns. Said out loud,
# because a share that does not add up should never be a typo nobody noticed.
burn_remainder = true

[lanes.vector]
# Robotensor Vector: one demonstration shown as context, the task done in another scene.
share = 0.30
# The contract as the installed engine ships it. A path works too, if you run from a checkout.
spec = "@vector_orchestrator/specs/vector_level1.json"
store = "var/vector/store"
key = "var/vector/keys/orchestrator.ed25519"
run_dir = "var/vector/runs"
cache = "var/vector/cache"
# The two environments a duel needs, and the RoboTwin-Vector checkout the simulator runs from; see
# docs/VALIDATOR.md. `robotensor doctor` checks all three.
policy_python = "{policy_python}"
simulator_python = "{simulator_python}"
simulator_root = "{simulator_root}"
duel_size = "launch"
workers = 4
# A Hugging Face dataset the signed store is published to; empty to keep it local.
mirror = ""
"""

MINER_NEXT = """\
A submission is weights and nothing else. Three commands, in this order:

    robotensor miner check  --dir {directory}/submission
    robotensor miner upload --dir {directory}/submission --repo <you>/<name> --private
    robotensor miner commit --repo <you>/<name> --revision <the sha upload printed> \\
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
