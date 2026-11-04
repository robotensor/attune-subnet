"""The validator's configuration: one small TOML file per host and network (`config/*.toml`).

    network = "test"                     # finney, test, or a ws:// address
    netuid = 0
    wallet = { name = "validator", hotkey = "default" }    # and `path`, where btcli keeps it

    [vector]                             # runs Vector
    policy_python = "/abs/policy-env/bin/python"
    simulator_python = "/abs/simulator-env/bin/python"
    simulator_root = "/abs/RoboTwin-Vector"
    mirror = "vector-results"            # optional: the dataset the store is published to, and
                                         # restored from on a new pod; "" for none
    kings_repo = "vector-kings"          # optional: the public model repo the four kings are
                                         # mirrored to; "" for none
                                         # both are names, under the account HF_TOKEN belongs to
    workers = 1                          # optional: units at once on each GPU, for its memory
    start_block = 4000000                # commitments made before this block are ignored; a
                                         # local chain may leave it out (0)

    [horizon]                            # Horizon's vault, where the network does not pin one in
    vault_hotkey = "5F..."               # code (`VAULT_HOTKEYS`)

Only what differs between hosts is here. Everything else is fixed in code: each share of the
emission, the data directory (`var/<config name>/` beside `config/`, or `var/` beside a config
elsewhere), the weights interval (per network) and the device.
A key this build does not know is refused, with its name.

Secrets are not in the TOML: a `.env` beside it (`KEY=VALUE` lines, `#` comments) is read by
`load` into the environment for whatever is not already set there, so `HF_TOKEN=hf_...` in
`config/.env` serves the validator and its workers alike; a variable already exported wins.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .init import ENV_FILE

#: Each competition's share of the miner emission. What the competitions run here do not claim
#: burns (to the subnet owner's hotkey).
SHARES = {"vector": 0.30}

#: Horizon's share of the miner emission: one vault UID, owned by Robotensor, which pays the people
#: who collect robot data on Attune Horizon, Robotensor's data platform. The payouts are made from
#: the vault off chain; validators only weight it. With Vector's 30% it is the whole of it.
HORIZON_SHARE = 0.70

#: The vault hotkey each public network weights, fixed here so that every validator agrees on it.
#: A network without one takes `[horizon] vault_hotkey` from the config; with neither, or while
#: the vault is not registered, Horizon's share burns.
VAULT_HOTKEYS: dict[str, str] = {}


class ConfigError(ValueError):
    """A configuration this build cannot run, and why."""


def local(network: str) -> bool:
    """A chain of one's own (a ws:// address): fast blocks, so block counts are larger."""
    return network.startswith(("ws://", "wss://"))


@dataclass(frozen=True)
class LaneConfig:
    name: str
    share: float


@dataclass(frozen=True)
class VectorConfig(LaneConfig):
    policy_python: str = ""
    simulator_python: str = ""
    #: The RoboTwin-Vector checkout the simulator runs from.
    simulator_root: str = ""
    #: The name of the Hugging Face dataset the store is mirrored to, under the account HF_TOKEN
    #: belongs to; empty for none.
    mirror: str = ""
    #: The name of the public Hugging Face model repository the four kings' weights are mirrored
    #: to, under the account HF_TOKEN belongs to; empty keeps them on this host alone.
    kings_repo: str = ""
    store: Path = Path()
    #: Where the four kings' weights are kept (`kings.Kings`): a directory of their own.
    kings: Path = Path()
    run_dir: Path = Path()
    cache: Path = Path()
    #: Units run at once on each GPU of the lane's lease, each a simulator and a policy server:
    #: as many as one card's memory holds (one unit alone took up to 34 GB on 2026-09-29).
    workers: int = 1
    #: Commitments made before this block are not read: what miners committed before the
    #: competition started does not enter it. 0 reads them all.
    start_block: int = 0
    policy_kwargs: dict[str, str] = field(default_factory=lambda: {"device": "cuda:0"})


VECTOR_KEYS = (
    "policy_python", "simulator_python", "simulator_root", "mirror", "kings_repo", "workers",
    "start_block",
)  # fmt: skip


@dataclass(frozen=True)
class HorizonConfig:
    """Horizon's vault, as the validator weights it: one hotkey takes the whole share."""

    share: float = HORIZON_SHARE
    #: The vault's hotkey; empty while the network has none, and its share then burns.
    vault_hotkey: str = ""


HORIZON_KEYS = ("vault_hotkey",)
#: What `[horizon]` held when Horizon was a model competition. A config still naming them is told
#: so, rather than refused key by key.
RETIRED_HORIZON_KEYS = (
    "genesis_block", "window_blocks", "competition", "engine_python", "runtime_python", "serve_as",
    "mirror", "devices", "profile", "dry_run",
)  # fmt: skip


@dataclass(frozen=True)
class Config:
    network: str
    netuid: int
    wallet_name: str
    wallet_hotkey: str
    wallet_path: str | None
    #: Where this config's state, stores and runs live.
    data: Path
    lanes: dict[str, LaneConfig]
    path: Path
    #: What relative paths in the file hang from.
    root: Path
    horizon: HorizonConfig = HorizonConfig()

    @property
    def state(self) -> Path:
        return self.data / "state"

    @property
    def weights_interval_blocks(self) -> int:
        """A tempo of 12-second blocks (72 minutes); 25 s of a local chain's 250 ms blocks."""
        return 100 if local(self.network) else 360

    @property
    def shares(self) -> dict[str, float]:
        return {name: lane.share for name, lane in self.lanes.items()}

    @property
    def vector(self) -> VectorConfig:
        lane = self.lanes["vector"]
        assert isinstance(lane, VectorConfig)
        return lane


def _path(base: Path, value: Any) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return path if path.is_absolute() else (base / path).resolve()


def _refuse_unknown(where: str, table: Any, known: tuple[str, ...], path: Path) -> None:
    if not isinstance(table, dict):
        raise ConfigError(f"{path}: {where} is a table")
    unknown = sorted(set(table) - set(known))
    if unknown:
        raise ConfigError(
            f"{path}: {where} does not take {', '.join(unknown)}; it takes {', '.join(known)}"
        )


def _vector(
    table: dict[str, Any], root: Path, data: Path, network: str, path: Path
) -> VectorConfig:
    _refuse_unknown("[vector]", table, VECTOR_KEYS, path)
    for required in ("policy_python", "simulator_python", "simulator_root"):
        if not table.get(required):
            raise ConfigError(f"{path}: [vector] needs {required}")
    workers = table.get("workers", 1)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ConfigError(f"{path}: [vector] workers must be a whole number of at least 1")
    # A public chain names it: left out, every commitment made before launch would enter.
    if "start_block" not in table and not local(network):
        raise ConfigError(
            f"{path}: [vector] needs start_block, the block from which commitments are read"
        )
    start_block = table.get("start_block", 0)
    if isinstance(start_block, bool) or not isinstance(start_block, int) or start_block < 0:
        raise ConfigError(f"{path}: [vector] start_block must be a whole number of at least 0")
    # Under the validator's own account, whatever the host; a test or local chain's are apart from
    # the real competition's, which the same token could otherwise write over.
    suffix = "-local" if local(network) or network == "local" else ""
    suffix = "-test" if network == "test" else suffix
    repos = {"kings_repo": f"vector-kings{suffix}", "mirror": f"vector-results{suffix}"}
    for key, default in repos.items():
        repos[key] = str(table.get(key, default))
        if "/" in repos[key]:
            raise ConfigError(
                f"{path}: [vector] {key} is a repository name, without an owner: the account is "
                f"the one HF_TOKEN belongs to (got {repos[key]!r})"
            )
    return VectorConfig(
        name="vector",
        share=SHARES["vector"],
        policy_python=os.path.expandvars(str(table["policy_python"])),
        simulator_python=os.path.expandvars(str(table["simulator_python"])),
        simulator_root=str(_path(root, table["simulator_root"])),
        mirror=repos["mirror"],
        kings_repo=repos["kings_repo"],
        workers=workers,
        start_block=start_block,
        store=data / "vector" / "store",
        kings=data / "vector" / "kings",
        run_dir=data / "vector" / "runs",
        cache=data / "vector" / "cache",
    )


def _ss58(value: Any) -> bool:
    """Whether `value` looks like an SS58 address (the chain checks the rest)."""
    text = str(value)
    alphabet = set("123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz")
    return 46 <= len(text) <= 48 and set(text) <= alphabet


def _horizon(table: dict[str, Any], network: str, path: Path) -> HorizonConfig:
    retired = sorted(set(table) & set(RETIRED_HORIZON_KEYS)) if isinstance(table, dict) else []
    if retired:
        raise ConfigError(
            f"{path}: [horizon] names {', '.join(retired)}, which were the Horizon model "
            "competition's. It is gone: Horizon is Robotensor's data platform, and [horizon] takes "
            "only vault_hotkey."
        )
    _refuse_unknown("[horizon]", table, HORIZON_KEYS, path)
    pinned = VAULT_HOTKEYS.get(network, "")
    named = str(table.get("vault_hotkey", ""))
    if named and not _ss58(named):
        raise ConfigError(f"{path}: [horizon] vault_hotkey is not an SS58 address: {named}")
    if pinned and named and named != pinned:
        raise ConfigError(
            f"{path}: [horizon] vault_hotkey is {named}, but {network}'s vault is {pinned}; "
            "every validator weights the same one. Leave it out."
        )
    return HorizonConfig(share=HORIZON_SHARE, vault_hotkey=pinned or named)


#: The competitions this build runs, and how each reads its table.
LANES = {"vector": _vector}


def read_env(path: Path) -> dict[str, str]:
    """`KEY=VALUE` lines of `path` (blank lines and `#` comments skipped, an optional `export `,
    one layer of matching quotes stripped); nothing when there is no such file."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    out: dict[str, str] = {}
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            out[key] = value
    return out


def load(path: str | os.PathLike[str]) -> Config:
    """Read a config file, or `ConfigError` saying what is wrong with it. The `.env` beside it,
    if any, goes into the environment first (`read_env`), without writing over what is set."""
    path = Path(path).resolve()
    for key, value in read_env(path.parent / ENV_FILE).items():
        os.environ.setdefault(key, value)
    doc = tomllib.loads(path.read_text(encoding="utf-8"))
    _refuse_unknown("the file", doc, ("network", "netuid", "wallet", "horizon", *LANES), path)
    for key in ("network", "netuid"):
        if key not in doc:
            raise ConfigError(f"{path}: needs {key}")
    wallet = doc.get("wallet", {})
    _refuse_unknown("wallet", wallet, ("name", "hotkey", "path"), path)
    # A config in the repository's config/ keeps its data in var/<its name>/ at the repository root.
    root = path.parent.parent if path.parent.name == "config" else path.parent
    data = root / "var" / path.stem if path.parent.name == "config" else root / "var"
    network = str(doc["network"])
    lanes = {
        name: read(doc[name], root, data, network, path)
        for name, read in LANES.items()
        if name in doc
    }
    if not lanes:
        raise ConfigError(f"{path}: no [vector] table; a validator runs at least one competition")
    horizon = _horizon(doc.get("horizon", {}), network, path)
    return Config(
        network=network,
        netuid=int(doc["netuid"]),
        wallet_name=str(wallet.get("name", "validator")),
        wallet_hotkey=str(wallet.get("hotkey", "default")),
        wallet_path=str(_path(root, wallet["path"])) if wallet.get("path") else None,
        data=data,
        lanes=lanes,
        path=path,
        root=root,
        horizon=horizon,
    )
