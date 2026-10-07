"""The miner's package, robotensor-attune: its command, and what `miner/build.sh` puts in it."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from robotensor import miner_cli

ROOT = Path(__file__).resolve().parents[1]
ORCHESTRATOR = next(
    (
        p
        for p in (
            ROOT.parent / "vector" / "vector-orchestrator",
            ROOT.parent / "vector-orchestrator",
        )
        if (p / ".git").is_dir()
    ),
    None,
)


def test_the_miner_command_has_the_miner_s_parts_and_no_validator():
    parser = miner_cli.build_parser()

    check = parser.parse_args(["miner", "check", "--dir", "submission/"])
    assert check.part == "miner" and check.command == "check"
    assert parser.parse_args(["init", "--role", "miner"]).func is miner_cli.cmd_init
    assert parser.parse_args(["doctor", "--json"]).func is miner_cli.cmd_doctor
    with pytest.raises(SystemExit):
        parser.parse_args(["validator", "--config", "c.toml", "status"])


def test_the_competition_comes_from_the_flag_then_the_environment(monkeypatch):
    monkeypatch.delenv("ATTUNE_COMPETITION", raising=False)
    monkeypatch.setenv("ROBOTENSOR_COMPETITION", "horizon")
    assert miner_cli.competition(None) == "horizon"
    monkeypatch.setenv("ATTUNE_COMPETITION", "vector")
    assert miner_cli.competition(None) == "vector"
    assert miner_cli.competition("horizon") == "horizon"
    with pytest.raises(miner_cli.CompetitionError):
        miner_cli.competition("nope")


@pytest.mark.skipif(ORCHESTRATOR is None, reason="no vector-orchestrator checkout beside this one")
def test_the_package_holds_the_miner_and_runs_on_its_own(tmp_path):
    stage = tmp_path / "stage"
    env = {**os.environ, "ORCHESTRATOR": str(ORCHESTRATOR)}
    subprocess.run(
        ["bash", str(ROOT / "miner" / "build.sh"), "--stage", str(stage)],
        env=env,
        check=True,
        capture_output=True,
    )

    files = {str(p.relative_to(stage / "src")) for p in (stage / "src").rglob("*") if p.is_file()}
    assert files == {
        "robotensor/__init__.py",
        "robotensor/miner_cli.py",
        "robotensor/miner.py",
        "robotensor/hub.py",
        "robotensor/chain.py",
        "robotensor/init.py",
        "robotensor/doctor.py",
        "robotensor/protocol/__init__.py",
        "robotensor/protocol/commitment.py",
        "vector_runtime/__init__.py",
        "vector_runtime/check.py",
        "vector_runtime/header.py",
        "vector_runtime/vector_v1.1.json",
    }
    for name in ("__init__.py", "check.py", "header.py", "vector_v1.1.json"):
        copied = (stage / "src" / "vector_runtime" / name).read_bytes()
        source = ORCHESTRATOR / "packages" / "vector-runtime" / "src" / "vector_runtime" / name
        assert copied == source.read_bytes(), name

    # Only what was staged, and no site-packages: nothing of the full install can stand in for it.
    run = {
        "env": {**os.environ, "PYTHONPATH": str(stage / "src")},
        "capture_output": True,
        "text": True,
    }
    helped = subprocess.run([sys.executable, "-S", "-m", "robotensor.miner_cli", "--help"], **run)
    assert helped.returncode == 0 and "{init,doctor,miner}" in helped.stdout

    junk = tmp_path / "model.safetensors"
    junk.write_bytes(b"\x08\x00\x00\x00\x00\x00\x00\x00{}      ")
    checked = subprocess.run(
        [sys.executable, "-S", "-m", "robotensor.miner_cli", "miner", "check", "--dir", str(junk)],
        **run,
    )
    assert checked.returncode == 1, checked.stderr
    assert json.loads(checked.stdout)["ok"] is False
