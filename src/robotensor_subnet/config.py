"""The validator's configuration: one TOML file per network (`config/*.toml`)."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VectorConfig:
    """The Vector lane: the orchestrator contract and where its signed store and runs live."""

    share: float
    spec: Path
    store: Path
    key: Path
    run_dir: Path
    cache: Path
    policy_python: str
    simulator_python: str
    duel_size: str | None = None
    #: The genesis baseline's commit, when the spec's is null.
    genesis_revision: str | None = None
    #: How long a commitment whose repository the Hub will not show yet (still private) waits.
    private_window_blocks: int = 300
    policy_kwargs: dict[str, str] = field(default_factory=dict)
    #: A Hugging Face dataset the signed store is mirrored to; empty for none.
    mirror: str = ""


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
    vector: VectorConfig
    path: Path

    @property
    def lanes(self) -> dict[str, float]:
        return {"vector": self.vector.share}


def _path(base: Path, value: Any) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return path if path.is_absolute() else (base / path).resolve()


def load(path: str | os.PathLike[str]) -> Config:
    """A config file; relative paths in it are relative to the file's directory's parent (the
    repository root, for the files under `config/`)."""
    path = Path(path).resolve()
    doc = tomllib.loads(path.read_text(encoding="utf-8"))
    base = path.parent.parent
    chain, validator = doc["chain"], doc.get("validator", {})
    lane = doc["lanes"]["vector"]
    vector = VectorConfig(
        share=float(lane.get("share", 0.30)),
        spec=_path(base, lane["spec"]),
        store=_path(base, lane.get("store", "var/vector/store")),
        key=_path(base, lane.get("key", "var/vector/keys/orchestrator.ed25519")),
        run_dir=_path(base, lane.get("run_dir", "var/vector/runs")),
        cache=_path(base, lane.get("cache", "var/vector/cache")),
        policy_python=os.path.expandvars(str(lane["policy_python"])),
        simulator_python=os.path.expandvars(str(lane["simulator_python"])),
        duel_size=lane.get("duel_size") or None,
        genesis_revision=lane.get("genesis_revision") or None,
        private_window_blocks=int(lane.get("private_window_blocks", 300)),
        policy_kwargs={str(k): str(v) for k, v in (lane.get("policy_kwargs") or {}).items()},
        mirror=str(lane.get("mirror", "")),
    )
    total = sum([vector.share])
    if not 0 <= total <= 1:
        raise ValueError(f"{path}: lane shares sum to {total}, not within [0, 1]")
    return Config(
        network=str(chain["network"]),
        netuid=int(chain["netuid"]),
        wallet_name=str(validator.get("wallet_name", "validator")),
        wallet_hotkey=str(validator.get("wallet_hotkey", "default")),
        wallet_path=validator.get("wallet_path") or None,
        state=_path(base, validator.get("state", "var/validator-state.json")),
        weights_interval_blocks=int(validator.get("weights_interval_blocks", 360)),
        burn_hotkey=str(validator.get("burn_hotkey", "")),
        vector=vector,
        path=path,
    )
