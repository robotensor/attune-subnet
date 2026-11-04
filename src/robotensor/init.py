"""`attune init`: a working directory, and a config that only needs your wallet filled in.

What is fiddly about running a validator is not the code, it is the paths around it. This writes a
config holding only what differs between hosts, and the directories its data is kept in.

What it will not do is guess a wallet, a netuid or the start block. Those are the lines it leaves
for you.
"""

from __future__ import annotations

from pathlib import Path

CONFIG = "attune.toml"
#: The file beside a config whose `KEY=VALUE` lines go into the environment (secrets: HF_TOKEN).
#: Here and not in `config`, which the miner's package does not hold.
ENV_FILE = ".env"

VALIDATOR_TOML = """\
# Written by `attune init`. State, stores and runs are kept in var/ beside this file.
# `finney` is mainnet, `test` the test network, or a ws:// address of your own.
network = "{network}"
netuid = {netuid}
# The wallet this validator sets weights with. `btcli wallet list` shows what you have.
wallet = {{ name = "{wallet_name}", hotkey = "{wallet_hotkey}" }}

# Horizon's vault takes 70% of the weights. A network whose vault is fixed in code
# (`VAULT_HOTKEYS`) needs nothing here; elsewhere name the vault's hotkey (left out, it burns).
# [horizon]
# vault_hotkey = "5..."

[vector]
# The two environments a duel needs, and the RoboTwin-Vector checkout the simulator runs from; see
# the README.
policy_python = "{policy_python}"
simulator_python = "{simulator_python}"
simulator_root = "{simulator_root}"
# The results dataset (`mirror`, default vector-results) and the public kings repository
# (`kings_repo`, default vector-kings) are made under the account HF_TOKEN (in .env beside this
# file) belongs to, and a new pod restores from them. Off finney the defaults end in -test or -local.
# The block the competition starts at: commitments made before it are ignored. Set it to the block
# announced for this network; 0 reads every commitment on chain.
start_block = {start_block}
"""

ENV = """\
# Read into the environment by the validator and its workers; a variable already set wins.
# The Hugging Face token of a permitted validator account (`hub.PERMITTED_VALIDATORS`), or the
# validator does not start. It downloads miners' gated repositories and writes the results mirror
# and kings_repo with write access.
HF_TOKEN=
"""

MINER_NEXT = """\
A submission is weights and nothing else. Three commands, in this order:

    attune miner check  --dir {directory}/submission
    attune miner upload --dir {directory}/submission --repo <you>/<name>
    attune miner commit --repo <you>/<name> --revision <the sha upload printed> \\
        --digest <the digest upload printed> \\
        --netuid <N> --network test --wallet.name <w> --wallet.hotkey <h>

`upload` makes the repository private and gated, and lets the permitted validator accounts in
before uploading. `commit` takes it out of private once the chain has your commitment, gated still:
nobody else can take your weights. Keep it that way until the validator has downloaded it, which
happens when its turn comes.

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
        "start_block": "0",
        **values,
    }
    directory.mkdir(parents=True, exist_ok=True)
    for made in ("var/state", "var/vector/store", "var/vector/runs", "var/vector/cache"):
        (directory / made).mkdir(parents=True, exist_ok=True)
    path = directory / CONFIG
    if path.exists():
        raise FileExistsError(f"{path} is already there; init will not write over a config")
    path.write_text(VALIDATOR_TOML.format(**fields), encoding="utf-8")
    env = directory / ENV_FILE
    if not env.exists():
        env.write_text(ENV, encoding="utf-8")
    return path


def miner(directory: Path) -> str:
    """Make a miner's working directory; what to run next."""
    (directory / "submission").mkdir(parents=True, exist_ok=True)
    return MINER_NEXT.format(directory=directory)
