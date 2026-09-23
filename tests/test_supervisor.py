"""One process per competition: what the supervisor starts, restarts and stops."""

import sys
import time

from robotensor_subnet.supervisor import RESTART_S, Worker, worker_argv


class Cfg:
    """Enough of a config for the argv and the lane list."""

    def __init__(self, path, lanes):
        self.path = path
        self.lanes = dict.fromkeys(lanes)


def test_a_worker_is_started_for_the_lane_it_runs(tmp_path):
    argv = worker_argv(Cfg(tmp_path / "c.toml", ["vector"]), "vector", once=True)

    assert argv[:3] == [sys.executable, "-m", "robotensor_subnet.worker"]
    assert argv[-3:] == ["--lane", "vector", "--once"]
    assert str(tmp_path / "c.toml") in argv


def test_a_worker_that_finishes_reports_its_status():
    worker = Worker("vector", [sys.executable, "-c", "raise SystemExit(3)"])

    worker.start()
    while worker.running:
        time.sleep(0.02)

    assert worker.reap() == 3 and worker.starts == 1


def test_stopping_a_worker_that_will_not_listen_kills_it():
    """A validator must come down even when an engine is wedged inside a simulator."""
    worker = Worker(
        "vector",
        [
            sys.executable,
            "-c",
            "import signal, time\nsignal.signal(signal.SIGTERM, lambda *a: None)\ntime.sleep(60)",
        ],
    )
    worker.start()
    time.sleep(0.4)

    worker.stop(timeout=0.5)

    assert not worker.running


def test_the_pause_before_starting_a_dead_worker_again_grows():
    """A competition whose engine is missing dies at once, every time; without a growing pause
    the supervisor would spin on it."""
    worker = Worker("vector", [sys.executable, "-c", ""])
    first = worker.backoff
    worker.backoff = min(worker.backoff * 2, 300.0)

    assert first == RESTART_S and worker.backoff == 2 * RESTART_S
