"""The validator's configuration: one TOML file per network (`config/*.toml`).

A file names the chain, the validator's wallet and one `[lanes.<name>]` table per competition it
runs. Each lane's table is read by that lane's own shape, so a competition can ask for what it
needs without every other one knowing about it.

Two rules that are easy to get wrong and expensive to debug:

- **Relative paths hang from `[paths].root`**, which defaults to the config file's own directory.
  Nothing resolves against the package, because an installed validator has no checkout around it.
- **A key this build does not know is refused**, with the name and where it was. A typo in a path
  or a share would otherwise be read as "not set" and quietly run something else.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: What the file may hold at the top, beside `[lanes]`.
TABLES = ("paths", "chain", "validator", "lanes")
PATHS_KEYS = ("root",)
CHAIN_KEYS = ("network", "netuid")
VALIDATOR_KEYS = (
    "wallet_name",
    "wallet_hotkey",
    "wallet_path",
    "state",
    "weights_interval_blocks",
    "burn_hotkey",
    "burn_remainder",
)


class ConfigError(ValueError):
    """A configuration this build cannot run, and why."""


@dataclass(frozen=True)
class LaneConfig:
    """What every competition's table holds, whatever else it adds."""

    #: The lane this configures: its `[lanes.<name>]` table, and its commitments' prefix.
    name: str
    #: Its share of the miner emissions, in [0, 1].
    share: float


@dataclass(frozen=True)
class VectorConfig(LaneConfig):
    """Robotensor Vector: the orchestrator contract, and where its signed store and runs live."""

    spec: Path = Path()
    store: Path = Path()
    key: Path = Path()
    run_dir: Path = Path()
    cache: Path = Path()
    policy_python: str = ""
    simulator_python: str = ""
    duel_size: str | None = None
    #: The genesis baseline's commit, when the spec's is null.
    genesis_revision: str | None = None
    #: How long a commitment whose repository the Hub will not show yet (still private) waits.
    private_window_blocks: int = 300
    policy_kwargs: dict[str, str] = field(default_factory=dict)
    #: A Hugging Face dataset the signed store is mirrored to; empty for none.
    mirror: str = ""
    #: Units materialized and played at once, each a simulator and a policy server of its own
    #: (about 9 GB of GPU memory each).
    workers: int = 1


VECTOR_KEYS = tuple(f.name for f in VectorConfig.__dataclass_fields__.values() if f.name != "name")


@dataclass(frozen=True)
class Config:
    network: str
    netuid: int
    wallet_name: str
    wallet_hotkey: str
    wallet_path: str | None
    state: Path
    weights_interval_blocks: int
    #: The hotkey the unclaimed emission goes to; empty for the subnet owner's.
    burn_hotkey: str
    #: Whether what the lanes do not claim may burn. Off, their shares must sum to exactly 1, so
    #: that emission is never burnt by a typo in a share.
    burn_remainder: bool
    lanes: dict[str, LaneConfig]
    path: Path
    root: Path

    @property
    def shares(self) -> dict[str, float]:
        return {name: lane.share for name, lane in self.lanes.items()}

    @property
    def vector(self) -> VectorConfig:
        """The Vector lane's config; `KeyError` when this validator does not run it."""
        lane = self.lanes["vector"]
        assert isinstance(lane, VectorConfig)
        return lane


#: What a path starting with this names: a file the installed engine ships, rather than one in a
#: checkout. `@vector_orchestrator/specs/vector_level1.json` is the contract in its wheel.
PACKAGED = "@"


def _packaged(value: str) -> Path:
    """A `@<package>/<path>` spelling, resolved inside the installed package's `data/`."""
    from importlib.resources import files

    package, _, rest = value[len(PACKAGED) :].partition("/")
    if not package or not rest:
        raise ConfigError(f"{value!r} is not @<package>/<path inside its data>")
    try:
        root = files(package)
    except ModuleNotFoundError:
        raise ConfigError(f"{value!r}: {package} is not installed") from None
    here = Path(str(root))
    # A wheel carries the contracts under the package's own `data/`; an editable install is the
    # checkout itself, where they sit beside `src/`. Both are "what this engine ships".
    for found in (here / "data" / rest, here.parent.parent / rest):
        if found.is_file():
            return found
    raise ConfigError(f"{value!r}: {package} ships no {rest}")


def _path(base: Path, value: Any) -> Path:
    text = str(value)
    if text.startswith(PACKAGED):
        return _packaged(text)
    path = Path(os.path.expandvars(os.path.expanduser(text)))
    return path if path.is_absolute() else (base / path).resolve()


def _refuse_unknown(where: str, table: dict[str, Any], known: tuple[str, ...], path: Path) -> None:
    unknown = sorted(set(table) - set(known))
    if unknown:
        raise ConfigError(
            f"{path}: {where} does not take {', '.join(unknown)}; it takes {', '.join(known)}"
        )


def _vector(name: str, table: dict[str, Any], base: Path, path: Path) -> VectorConfig:
    _refuse_unknown(f"[lanes.{name}]", table, VECTOR_KEYS, path)
    for required in ("spec", "policy_python", "simulator_python"):
        if not table.get(required):
            raise ConfigError(f"{path}: [lanes.{name}] needs {required}")
    return VectorConfig(
        name=name,
        share=float(table.get("share", 0.30)),
        spec=_path(base, table["spec"]),
        store=_path(base, table.get("store", "var/vector/store")),
        key=_path(base, table.get("key", "var/vector/keys/orchestrator.ed25519")),
        run_dir=_path(base, table.get("run_dir", "var/vector/runs")),
        cache=_path(base, table.get("cache", "var/vector/cache")),
        policy_python=os.path.expandvars(str(table["policy_python"])),
        simulator_python=os.path.expandvars(str(table["simulator_python"])),
        duel_size=table.get("duel_size") or None,
        genesis_revision=table.get("genesis_revision") or None,
        private_window_blocks=int(table.get("private_window_blocks", 300)),
        policy_kwargs={str(k): str(v) for k, v in (table.get("policy_kwargs") or {}).items()},
        mirror=str(table.get("mirror", "")),
        workers=int(table.get("workers", 1)),
    )


#: The competitions this build can be configured for, and how each reads its table. A lane is
#: added here, to `protocol.commitment.LANES` and as its own `lanes/` module.
LANES = {"vector": _vector}


def load(path: str | os.PathLike[str]) -> Config:
    """Read a config file, or `ConfigError` saying what is wrong with it."""
    path = Path(path).resolve()
    doc = tomllib.loads(path.read_text(encoding="utf-8"))
    _refuse_unknown("the file", doc, TABLES, path)
    paths, chain = doc.get("paths", {}), doc.get("chain", {})
    validator, lanes = doc.get("validator", {}), doc.get("lanes", {})
    _refuse_unknown("[paths]", paths, PATHS_KEYS, path)
    _refuse_unknown("[chain]", chain, CHAIN_KEYS, path)
    _refuse_unknown("[validator]", validator, VALIDATOR_KEYS, path)
    if not lanes:
        raise ConfigError(f"{path}: no [lanes.<name>] table; a validator runs at least one")
    # The file's own directory, unless it says otherwise: an installed validator has no checkout.
    base = _path(path.parent, paths.get("root", "."))

    built: dict[str, LaneConfig] = {}
    for name, table in lanes.items():
        read = LANES.get(name)
        if read is None:
            raise ConfigError(
                f"{path}: [lanes.{name}] is not a competition this build runs "
                f"({', '.join(sorted(LANES))})"
            )
        built[name] = read(name, table, base, path)

    burn_remainder = bool(validator.get("burn_remainder", False))
    total = sum(lane.share for lane in built.values())
    if any(lane.share < 0 for lane in built.values()) or total > 1 + 1e-9:
        raise ConfigError(
            f"{path}: lane shares must be non-negative and sum to at most 1, not {total}"
        )
    if not burn_remainder and abs(total - 1.0) > 1e-9:
        raise ConfigError(
            f"{path}: the lanes claim {total} of the emission, not all of it. Set "
            "[validator].burn_remainder = true to burn the rest on purpose, or fix the shares."
        )
    return Config(
        network=str(chain["network"]),
        netuid=int(chain["netuid"]),
        wallet_name=str(validator.get("wallet_name", "validator")),
        wallet_hotkey=str(validator.get("wallet_hotkey", "default")),
        wallet_path=validator.get("wallet_path") or None,
        # A directory of one document per lane. A config naming the single file it used to be is
        # read as the directory beside it, and that file is split into it once.
        state=_path(base, validator.get("state", "var/validator-state")),
        weights_interval_blocks=int(validator.get("weights_interval_blocks", 360)),
        burn_hotkey=str(validator.get("burn_hotkey", "")),
        burn_remainder=burn_remainder,
        lanes=built,
        path=path,
        root=base,
    )
