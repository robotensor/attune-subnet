"""The validator's configuration: one small TOML file per host and network (`config/*.toml`).

    network = "test"                     # finney, test, or a ws:// address
    netuid = 0
    wallet = { name = "validator", hotkey = "default" }    # and `path`, where btcli keeps it

    [vector]                             # runs Robotensor Vector
    policy_python = "/abs/policy-env/bin/python"
    simulator_python = "/abs/simulator-env/bin/python"
    simulator_root = "/abs/RoboTwin-Vector"
    mirror = "owner/vector-results"      # optional: the dataset the store is published to

    [horizon]                            # runs Robotensor Horizon; see HorizonConfig

Only what differs between hosts is here. Everything else is fixed in code: each competition's share
of the emission, the data directory (`var/<config name>/` beside `config/`, or `var/` beside a
config elsewhere), the weights interval and private window (per network), the workers and device.
A key this build does not know is refused, with its name.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Each competition's share of the miner emission. What the competitions run here do not claim
#: burns (to the subnet owner's hotkey). Horizon pays nothing until it opens.
SHARES = {"vector": 0.30, "horizon": 0.0}


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
    #: A Hugging Face dataset the store is mirrored to; empty for none.
    mirror: str = ""
    store: Path = Path()
    run_dir: Path = Path()
    cache: Path = Path()
    #: How long a commitment whose repository the Hub will not show yet (still private) waits.
    private_window_blocks: int = 300
    #: Units played at once, each a simulator and a policy server (about 9 GB of GPU memory).
    workers: int = 4
    policy_kwargs: dict[str, str] = field(default_factory=lambda: {"device": "cuda:0"})


VECTOR_KEYS = ("policy_python", "simulator_python", "simulator_root", "mirror")


@dataclass(frozen=True)
class HorizonConfig(LaneConfig):
    """Robotensor Horizon. Its schedule is the chain's: `genesis_block` and `window_blocks` are all
    two validators need to agree on which round is open (`protocol.schedule`)."""

    genesis_block: int = 0
    window_blocks: int = 50400  # a week of 12-second blocks
    #: The engine's own configuration, which says what a round evaluates.
    competition: Path | None = None
    #: An interpreter for the engine, when it cannot share this one.
    engine_python: str = ""
    #: The interpreter of the model runtime environment, which serves a submission.
    runtime_python: str = ""
    #: The unprivileged user a served model runs as; a lane that pays must name one.
    serve_as: str = ""
    mirror: str = ""
    #: The cards this competition may use, when the box is shared with the other one.
    devices: tuple[int, ...] = ()
    #: Which of the engine's profiles a round runs; empty for its own default.
    profile: str = ""
    #: Rehearse: score beside the real scores and publish nothing at close.
    dry_run: bool = False
    store: Path = Path()
    rounds: Path = Path()
    models: Path = Path()
    private_window_blocks: int = 300
    #: A submission's cap: shards, statistics and the knobs file together.
    max_repo_bytes: int = 50 << 30


HORIZON_KEYS = (
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
    def horizon(self) -> HorizonConfig:
        lane = self.lanes["horizon"]
        assert isinstance(lane, HorizonConfig)
        return lane

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
    return VectorConfig(
        name="vector",
        share=SHARES["vector"],
        policy_python=os.path.expandvars(str(table["policy_python"])),
        simulator_python=os.path.expandvars(str(table["simulator_python"])),
        simulator_root=str(_path(root, table["simulator_root"])),
        mirror=str(table.get("mirror", "")),
        store=data / "vector" / "store",
        run_dir=data / "vector" / "runs",
        cache=data / "vector" / "cache",
        private_window_blocks=1200 if local(network) else 300,
    )


def _horizon(
    table: dict[str, Any], root: Path, data: Path, network: str, path: Path
) -> HorizonConfig:
    _refuse_unknown("[horizon]", table, HORIZON_KEYS, path)
    share = SHARES["horizon"]
    serve_as = str(table.get("serve_as", ""))
    if share > 0 and not serve_as:
        raise ConfigError(
            f"{path}: Horizon pays {share:.0%} of the emission and [horizon] names no serve_as. A "
            "model served as the user running the validator can read the answer key of the episode "
            "it is being scored on; name an unprivileged user."
        )
    return HorizonConfig(
        name="horizon",
        share=share,
        genesis_block=int(table.get("genesis_block", 0)),
        window_blocks=int(table.get("window_blocks", 50400)),
        competition=_path(root, table["competition"]) if table.get("competition") else None,
        engine_python=os.path.expandvars(str(table.get("engine_python", ""))),
        runtime_python=os.path.expandvars(str(table.get("runtime_python", ""))),
        serve_as=serve_as,
        mirror=str(table.get("mirror", "")),
        devices=tuple(int(d) for d in (table.get("devices") or ())),
        profile=str(table.get("profile", "")),
        dry_run=bool(table.get("dry_run", False)),
        store=data / "horizon" / "store",
        rounds=data / "horizon" / "rounds",
        models=data / "horizon" / "models",
        private_window_blocks=1200 if local(network) else 300,
    )


#: The competitions this build runs, and how each reads its table.
LANES = {"vector": _vector, "horizon": _horizon}


def load(path: str | os.PathLike[str]) -> Config:
    """Read a config file, or `ConfigError` saying what is wrong with it."""
    path = Path(path).resolve()
    doc = tomllib.loads(path.read_text(encoding="utf-8"))
    _refuse_unknown("the file", doc, ("network", "netuid", "wallet", *LANES), path)
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
        raise ConfigError(f"{path}: no [vector] or [horizon] table; a validator runs at least one")
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
    )
