"""One competition, in a process of its own: intake, then step after step until it is stopped.

A worker is what the supervisor starts per lane. It exists because both engines read
`CUDA_VISIBLE_DEVICES` from the process they run in, so two competitions in one process could not
hold different cards - and because one wedged competition should be restartable without touching
the other.

It holds a chain connection of its own (a websocket is not thread-safe, let alone process-shared),
writes only its own lane's state document, and takes a GPU lease around each step.

A lane whose step can take hours (a Vector duel) also offers `refresh(chain)`: the worker runs it
every `REFRESH_S` on a thread and a chain connection of its own, so what the chain committed meanwhile
reaches the lane - and the queue the dashboard shows - while the step is still running.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
from typing import Any

from . import chain as chain_
from . import compute
from .config import Config, load
from .lanes.base import Lane, Progress
from .state import State

log = logging.getLogger("robotensor.worker")

#: How long a worker waits before stepping again when there was nothing to do.
IDLE_S = 30.0
#: How often a lane that offers `refresh` takes in what the chain committed, beside its step.
REFRESH_S = 60.0


def hub_token() -> str | None:
    return os.environ.get("HF_TOKEN") or None


def build(cfg: Config, name: str, state: State) -> Lane:
    """The lane `name`, built from its own table. Imported here, so a validator running one
    competition never imports the other's engine."""
    if name == "vector":
        from .lanes.vector import VectorLane

        return VectorLane(cfg.vector, state, hub_token=hub_token())
    if name == "horizon":
        from .lanes.horizon import HorizonLane

        return HorizonLane(cfg.horizon, state, hub_token=hub_token())
    raise ValueError(f"{name} is not a competition this build runs ({', '.join(cfg.lanes)})")


def one_step(lane: Lane, chain: chain_.Chain, broker: compute.Broker) -> Progress:
    """Intake, then one piece of the lane's own work, with the cards it is allowed for it."""
    changed = lane.intake(chain.commitments(), chain.block())
    for entry in changed:
        log.info("intake: %s from %s is %s", entry.entry, entry.hotkey, entry.status)
    with broker.lease(lane.name) as lease, compute.applied(lease):
        return lane.step(chain)


class Refresher(threading.Thread):
    """`lane.refresh(chain)` every `interval_s`, on a chain connection of its own (a websocket is
    not thread-safe). A failure is logged and the next tick tries again."""

    def __init__(self, cfg: Config, lane: Any, interval_s: float = REFRESH_S) -> None:
        super().__init__(name=f"robotensor-refresh-{lane.name}", daemon=True)
        self.cfg = cfg
        self.lane = lane
        self.interval_s = interval_s
        self.stop = threading.Event()

    def run(self) -> None:
        chain = None
        while not self.stop.wait(self.interval_s):
            try:
                if chain is None:
                    chain = chain_.Chain(self.cfg.network, self.cfg.netuid)
                for entry in self.lane.refresh(chain):
                    log.info("intake: %s from %s is %s", entry.entry, entry.hotkey, entry.status)
            except Exception:  # noqa: BLE001 - logged; the next tick tries again
                log.exception("%s: the refresh failed", self.lane.name)
                if chain is not None:
                    chain.close()
                chain = None
        if chain is not None:
            chain.close()


def run(cfg: Config, name: str, *, once: bool = False) -> int:
    state = State(cfg.state)
    lane = build(cfg, name, state)
    chain = chain_.Chain(cfg.network, cfg.netuid)
    broker = compute.Broker(cfg.state)
    refresher = Refresher(cfg, lane) if hasattr(lane, "refresh") and not once else None
    if refresher is not None:
        refresher.start()
    try:
        while True:
            try:
                what = one_step(lane, chain, broker)
            except Exception as exc:  # noqa: BLE001 - logged; the worker goes on after a pause
                log.exception("%s: the step failed", name)
                what = Progress(name, "failed", f"the step raised: {exc}")
            log.info("step: %s", what)
            if once:
                return 0
            if what.resting:
                time.sleep(IDLE_S)
    except KeyboardInterrupt:
        return 0
    finally:
        if refresher is not None:
            refresher.stop.set()
        chain.close()


def main(argv: list[str] | None = None) -> int:
    from .validator import _logging

    _logging()
    parser = argparse.ArgumentParser(prog="robotensor-worker", description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True, help="config/<network>.toml")
    parser.add_argument("--lane", required=True, help="the competition this worker runs")
    parser.add_argument("--once", action="store_true", help="one step, then exit")
    args = parser.parse_args(argv)
    cfg = load(args.config)
    return run(cfg, args.lane, once=args.once)


if __name__ == "__main__":
    sys.exit(main())
