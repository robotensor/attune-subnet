"""One process per competition: what the supervisor starts, restarts and stops."""

import sys
import time

from robotensor.supervisor import RESTART_S, Worker, worker_argv


class Cfg:
    """Enough of a config for the argv and the lane list."""

    def __init__(self, path, lanes):
        self.path = path
        self.lanes = dict.fromkeys(lanes)


def test_a_worker_is_started_for_the_lane_it_runs(tmp_path):
    argv = worker_argv(Cfg(tmp_path / "c.toml", ["vector"]), "vector", once=True)

    assert argv[:3] == [sys.executable, "-m", "robotensor.worker"]
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


def test_the_refresher_refreshes_its_lane_on_a_connection_of_its_own(monkeypatch):
    """What runs beside a step that takes hours: a failure is logged, and the next tick tries
    again on a fresh connection."""
    import threading
    from types import SimpleNamespace

    from robotensor import worker

    opened, closed = [], []

    class Chain:
        def __init__(self, network, netuid):
            opened.append((network, netuid))

        def close(self):
            closed.append(True)

    calls = []
    done = threading.Event()

    class Lane:
        name = "vector"

        def refresh(self, chain):
            calls.append(chain)
            if len(calls) == 1:
                raise RuntimeError("the chain went away")
            if len(calls) == 3:
                done.set()
            return []

    monkeypatch.setattr(worker.chain_, "Chain", Chain)
    refresher = worker.Refresher(SimpleNamespace(network="test", netuid=7), Lane(), interval_s=0.01)
    refresher.start()
    assert done.wait(5)
    refresher.stop.set()
    refresher.join(5)

    assert opened[0] == ("test", 7) and len(opened) == 2, "a failed tick reconnects"
    assert calls[1] is calls[2] and calls[0] is not calls[1]
    assert len(closed) == 2
