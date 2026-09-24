"""`robotensor doctor`: can this host do what it is configured to do?

Everything a validator needs is checked here, once, in seconds, and said in one place: the config
as it was actually read, the chain, the wallet, the Hub token, the disk, the clock, and for each
competition its interpreters, its engine, its contract and its cards. A miner's report is the
subset that needs no config and no chain.

Every check catches its own failure and becomes a line of the report: a doctor that stops at the
first problem makes an operator run it five times to find five things.

It also says the thing a miner most wants to know and no page states plainly: **this subnet
charges no fee to submit.**
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

#: What a validator should have free where it keeps runs and weights.
DISK_WARN_GB = 200
#: A clock further off than this makes a seed block's finality judgement wrong.
SKEW_WARN_S = 120


@dataclass(frozen=True)
class Check:
    """One thing that is either so or not, and what to do when it is not."""

    name: str
    ok: bool
    detail: str = ""
    #: A check that is worth knowing about but does not stop a validator running.
    advisory: bool = False


@dataclass(frozen=True)
class Report:
    role: str
    competition: str | None
    checks: list[Check] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.ok or c.advisory for c in self.checks)

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "competition": self.competition,
            "ok": self.ok,
            "checks": [asdict(c) for c in self.checks],
        }


def _check(name: str, run: Any, *, advisory: bool = False) -> Check:
    """Run one check, and turn whatever it raises into its answer."""
    try:
        detail = run()
    except Exception as exc:  # noqa: BLE001 - a failed check is a line of the report
        return Check(name, False, f"{type(exc).__name__}: {exc}", advisory)
    if isinstance(detail, tuple):
        ok, said = detail
        return Check(name, bool(ok), str(said), advisory)
    return Check(name, True, str(detail or ""), advisory)


# -- the checks -------------------------------------------------------------------------------


def _python() -> str:
    version = ".".join(str(p) for p in sys.version_info[:3])
    return f"{version} at {sys.executable}"


def _no_fee() -> str:
    return "this subnet charges nothing to submit: a commitment costs the chain's fee and no more"


def _hub_token(write_to: str = "") -> tuple[bool, str]:
    token = os.environ.get("HF_TOKEN") or ""
    if not token:
        return False, "HF_TOKEN is not set: intake reads public repositories only"
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    who = api.whoami()
    name = who.get("name", "?")
    if not write_to:
        return True, f"read as {name}"
    permission = who.get("auth", {}).get("accessToken", {}).get("role", "?")
    return True, f"as {name} ({permission}); publishing to {write_to}"


def _disk(path: Path) -> tuple[bool, str]:
    path.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(path).free / 2**30
    return free_gb >= DISK_WARN_GB, f"{free_gb:.0f} GB free at {path}"


def _clock() -> tuple[bool, str]:
    """How far this host's clock is from the Hub's, which is what its signatures are read against."""
    import urllib.request
    from email.utils import parsedate_to_datetime

    request = urllib.request.Request("https://huggingface.co", method="HEAD")
    with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310 - a fixed https url
        served = response.headers.get("Date")
    if not served:
        return True, "no Date header to compare against"
    skew = abs(time.time() - parsedate_to_datetime(served).timestamp())
    return skew <= SKEW_WARN_S, f"{skew:.0f} s from the Hub's clock"


def _gpus() -> tuple[bool, str]:
    from .compute import visible_devices

    devices = visible_devices()
    if not devices:
        return False, "no GPU: a duel and an epoch both need one"
    return True, f"{len(devices)} visible: {', '.join(str(d) for d in devices)}"


def _chain(cfg: Any) -> tuple[bool, str]:
    from . import chain as chain_

    chain = chain_.Chain(cfg.network, cfg.netuid)
    try:
        return True, f"{cfg.network} netuid {cfg.netuid} at block {chain.block()}"
    finally:
        chain.close()


def _wallet(cfg: Any) -> tuple[bool, str]:
    from . import chain as chain_

    wallet = chain_.wallet(cfg.wallet_name, cfg.wallet_hotkey, cfg.wallet_path)
    hotkey = wallet.hotkey.ss58_address
    chain = chain_.Chain(cfg.network, cfg.netuid)
    try:
        uids = chain.uids()
    finally:
        chain.close()
    if hotkey not in uids:
        return False, f"{hotkey} is not registered on netuid {cfg.netuid}"
    return True, f"{cfg.wallet_name}/{cfg.wallet_hotkey} is uid {uids[hotkey]} ({hotkey[:8]}...)"


def _shares(cfg: Any) -> tuple[bool, str]:
    total = sum(cfg.shares.values())
    said = ", ".join(f"{name} {share:.0%}" for name, share in sorted(cfg.shares.items()))
    if abs(total - 1.0) <= 1e-9:
        return True, said
    if cfg.burn_remainder:
        return True, f"{said}; the rest burns ({1 - total:.0%}), as the config says it may"
    return False, f"{said}: they do not claim the whole emission"


def _interpreter(path: str, imports: tuple[str, ...]) -> tuple[bool, str]:
    import subprocess

    if not Path(path).is_file():
        return False, f"{path} is not there"
    code = "import " + ", ".join(imports) + "\nprint('ok')"
    done = subprocess.run([path, "-c", code], capture_output=True, text=True, timeout=120)
    if done.returncode != 0:
        return False, f"{path} cannot import {', '.join(imports)}: {done.stderr.strip()[-200:]}"
    return True, f"{path} imports {', '.join(imports)}"


def _serve_as(lane_cfg: Any) -> tuple[bool, str]:
    """A served model must not run as the user that can read the episode's answer key."""
    if lane_cfg.share <= 0:
        return True, "not needed: this competition pays nothing yet"
    if not lane_cfg.serve_as:
        return False, "nobody named: a served model could read the episode's answer key"
    import pwd

    pwd.getpwnam(lane_cfg.serve_as)  # raises if the user is not on this host
    return True, f"models are served as {lane_cfg.serve_as}"


def _horizon_engine(lane_cfg: Any) -> tuple[bool, str]:
    """What the engine says about this host, in the engine's own words.

    `config check` already answers "can this machine run an epoch?" better than anything here
    could: it resolves every path the config names and asks each benchmark fork about its task
    config and its tasks. Repeating that badly would be worse than quoting it.
    """
    import subprocess

    if not lane_cfg.competition:
        return True, "no engine config named ([lanes.horizon].competition); nothing to run yet"
    python = lane_cfg.engine_python or sys.executable
    done = subprocess.run(
        [
            python,
            "-m",
            "horizon_competition.cli",
            "config",
            "check",
            "--config",
            str(lane_cfg.competition),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if done.returncode == 0:
        return True, f"{lane_cfg.competition.name}: every axis this host is asked for"
    problems = []
    for line in (done.stderr + done.stdout).splitlines():  # it writes its refusals to stderr
        if line.startswith("not checked"):  # what the problems above stopped it looking at
            break
        if line.startswith("  - "):
            problems.append(line[4:].split(" (WORKSPACE")[0])
    if not problems:
        return False, done.stderr.strip()[-200:] or "the engine refused its config"
    said = "; ".join(problems[:2])
    more = f" (+{len(problems) - 2} more)" if len(problems) > 2 else ""
    return False, f"{said}{more}"


def _vector_contract(lane_cfg: Any) -> tuple[bool, str]:
    from vector_orchestrator.spec import load_spec_file

    spec = load_spec_file(lane_cfg.spec)
    tracks = ", ".join(spec.tracks)
    return True, f"{lane_cfg.spec.name}: {tracks} at {spec.fingerprint[:12]}"


def _vector_benchmark(lane_cfg: Any) -> tuple[bool, str]:
    from vector_orchestrator.benchmarks.plugins import discover
    from vector_orchestrator.spec import load_spec_file

    spec = load_spec_file(lane_cfg.spec)
    found = discover(spec)
    problems = [f"{name}: {'; '.join(p.problems)}" for name, p in found.items() if not p.ok]
    if problems:
        return False, " | ".join(problems)
    return True, ", ".join(f"{name} {p.version}" for name, p in found.items())


def miner_checks(competition: str) -> list[Check]:
    """What a miner needs, and nothing that takes a chain or a config to answer."""
    checks = [
        _check("python", _python),
        _check("submission fee", _no_fee),
        _check("hugging face token", _hub_token, advisory=True),
    ]
    if competition == "vector":
        checks.append(
            _check(
                "runtime",
                lambda: _interpreter(sys.executable, ("vector_runtime.check",)),
            )
        )
    return checks


def validator_checks(cfg: Any, competition: str) -> list[Check]:
    lane_cfg = cfg.lanes[competition]
    checks = [
        _check("config", lambda: f"{cfg.path} (paths from {cfg.root})"),
        _check("shares", lambda: _shares(cfg)),
        _check("state", lambda: (os.access(cfg.state.parent, os.W_OK), f"{cfg.state}")),
        _check("chain", lambda: _chain(cfg)),
        _check("wallet", lambda: _wallet(cfg)),
        _check("hugging face token", lambda: _hub_token(getattr(lane_cfg, "mirror", ""))),
        _check("gpu", _gpus),
        _check("disk", lambda: _disk(Path(getattr(lane_cfg, "run_dir", cfg.state)))),
        _check("clock", _clock, advisory=True),
        _check("submission fee", _no_fee),
    ]
    if competition == "horizon":
        checks += [
            _check(
                "schedule",
                lambda: (
                    True,
                    f"epochs of {lane_cfg.window_blocks} blocks from block "
                    f"{lane_cfg.genesis_block}",
                ),
            ),
            _check(
                "serve_as",
                lambda: _serve_as(lane_cfg),
            ),
            _check("runtime", lambda: _interpreter(sys.executable, ("horizon_runtime_zerowam",))),
            _check("engine", lambda: _horizon_engine(lane_cfg)),
        ]
    if competition == "vector":
        checks += [
            _check("contract", lambda: _vector_contract(lane_cfg)),
            _check("benchmark", lambda: _vector_benchmark(lane_cfg)),
            _check(
                "simulator interpreter",
                lambda: _interpreter(
                    lane_cfg.simulator_python, ("robotwin_bench", "robotensor_benchmark_robotwin")
                ),
            ),
            _check(
                "policy interpreter",
                lambda: _interpreter(
                    lane_cfg.policy_python, ("vector_runtime.policy", "vector_policy")
                ),
            ),
        ]
    return checks


def report(config: str | None, competition: str | None, *, role: str = "validator") -> Report:
    """Everything this host can be asked about the work it is configured for."""
    from .cli import competition as resolve

    if role == "miner":
        name = resolve(competition, None)
        return Report(role, name, miner_checks(name))
    if not config:
        raise ValueError("a validator's doctor needs --config")
    from .config import load

    cfg = load(config)
    name = resolve(competition, cfg.lanes)
    return Report(role, name, validator_checks(cfg, name))


def render(found: Report) -> str:
    """The report as an operator reads it: one line per check, the failures last in the eye."""
    lines = [f"robotensor doctor: {found.role}, {found.competition}"]
    for check in found.checks:
        mark = "ok  " if check.ok else ("warn" if check.advisory else "FAIL")
        lines.append(f"  [{mark}] {check.name}: {check.detail}")
    lines.append("" if found.ok else "\nSomething above must be fixed before this host can run.")
    return "\n".join(lines)
