"""`validator run`: a worker process per competition, and the weights thread beside them.

What the supervisor is for: the competitions are independent, and the one thing they share - the
chain, and the weight vector built from what each publishes - is read, not written, by them. So
each runs in a process of its own and the supervisor does three things:

- **starts one worker per configured lane**, and starts it again if it dies, with a pause that
  grows so a competition whose engine is missing does not spin;
- **sets weights**, in a thread with a chain connection of its own, because one duel can take
  hours and a validator must stay inside the chain's activity cutoff meanwhile. The vector is read
  from what the lanes published, so a wedged worker never stops the subnet being paid;
- **stops them all** on a signal, and waits for them: a killed worker leaves its engine's own
  lock behind, and the next one picks the work up from disk.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field

from .config import Config

log = logging.getLogger("robotensor.supervisor")

#: How long to wait before starting a worker that has just died, and the most it grows to.
RESTART_S = 5.0
RESTART_MAX_S = 300.0


@dataclass
class Worker:
    """One competition's process, and what it has done lately."""

    lane: str
    argv: list[str]
    process: subprocess.Popen[bytes] | None = None
    starts: int = 0
    #: When it may be started again, as a monotonic time.
    not_before: float = 0.0
    backoff: float = RESTART_S
    env: dict[str, str] = field(default_factory=dict)

    def start(self) -> None:
        log.info("starting the %s worker: %s", self.lane, " ".join(self.argv))
        self.process = subprocess.Popen(self.argv, env={**os.environ, **self.env})
        self.starts += 1

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def reap(self) -> int | None:
        """The exit status if it has stopped, and None while it runs."""
        if self.process is None:
            return None
        return self.process.poll()

    def stop(self, timeout: float = 30.0) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout)
        except subprocess.TimeoutExpired:
            log.warning("the %s worker did not stop; killing it", self.lane)
            self.process.kill()
            self.process.wait()  # collected here, so a stopped supervisor leaves no child behind


def worker_argv(cfg: Config, lane: str, *, once: bool) -> list[str]:
    """How to start a lane's worker. Its own interpreter, when its engine needs one."""
    argv = [
        sys.executable,
        "-m",
        "robotensor.worker",
        "--config",
        str(cfg.path),
        "--lane",
        lane,
    ]
    return [*argv, "--once"] if once else argv


def run(cfg: Config, *, once: bool = False, weights: object | None = None) -> int:
    """Run every configured competition until they are stopped; the worst exit status seen."""
    workers = [Worker(lane, worker_argv(cfg, lane, once=once)) for lane in sorted(cfg.lanes)]
    stopping = False

    def handle(signum: int, _frame: object) -> None:
        nonlocal stopping
        log.info("stopping on %s", signal.Signals(signum).name)
        stopping = True

    previous = {sig: signal.signal(sig, handle) for sig in (signal.SIGINT, signal.SIGTERM)}
    worst = 0
    try:
        for worker in workers:
            worker.start()
        while not stopping:
            running = 0
            for worker in workers:
                if worker.running:
                    running += 1
                    continue
                status = worker.reap()
                if status is not None:  # it has just stopped: take its status and let it go
                    worst = max(worst, abs(status))
                    if status != 0:
                        log.warning("the %s worker exited with %s", worker.lane, status)
                    worker.process = None
                    if once:
                        continue
                    worker.not_before = time.monotonic() + worker.backoff
                    worker.backoff = min(worker.backoff * 2, RESTART_MAX_S)
                if not once and time.monotonic() >= worker.not_before:
                    worker.start()
                    running += 1
            if once and running == 0:
                break
            time.sleep(1.0)
    finally:
        for worker in workers:
            worker.stop()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if weights is not None and hasattr(weights, "stop"):
            weights.stop.set()  # type: ignore[attr-defined]
    return worst
