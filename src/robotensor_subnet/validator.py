"""`robotensor-validator`: read commitments, run the lanes' duels, set weights.

    run      the loop: intake -> genesis (once) -> the oldest queued duel -> ... ; a thread of its
             own keeps setting weights every `weights_interval_blocks`, because one duel can take
             hours and a validator must stay inside the chain's activity cutoff meanwhile
    intake   read the chain's commitments into the lanes once, and print what changed
    duel     run one duel now: the queue's next, or --challenger repo@sha (a manual duel)
    weights  compute the weight vector and set it (--dry-run prints it)
    status   the lanes' queues, champions and kings

The chain is `chain.Chain`, the only bittensor code; the Vector lane is `lanes.vector`.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from typing import Any

from . import chain as chain_
from .config import Config, load
from .lanes.vector import VectorLane, Entry
from .protocol import seed as seed_
from .protocol.weights import Lane, weight_vector
from .state import State

log = logging.getLogger("robotensor.validator")

#: How long the loop sleeps when there is nothing to duel.
IDLE_S = 30.0


def hub_token() -> str | None:
    return os.environ.get("HF_TOKEN") or None


def lane(cfg: Config, state: State) -> VectorLane:
    return VectorLane(cfg.vector, state, hub_token=hub_token())


def compute_weights(cfg: Config, vector: VectorLane, chain: chain_.Chain) -> dict[int, float]:
    uids = chain.uids()
    burn = cfg.burn_hotkey or chain.owner_hotkey()
    if burn not in uids:
        raise RuntimeError(f"the burn hotkey {burn} is not registered on netuid {cfg.netuid}")
    lanes = [Lane("vector", cfg.vector.share, vector.champions())]
    return weight_vector(lanes, uids, uids[burn])


def set_weights(cfg: Config, vector: VectorLane, chain: chain_.Chain, wallet: Any) -> dict[int, float]:
    weights = compute_weights(cfg, vector, chain)
    chain.set_weights(wallet, weights)
    return weights


class WeightsThread(threading.Thread):
    """Sets weights every `weights_interval_blocks`, on a chain connection of its own."""

    def __init__(self, cfg: Config, state_path: str, wallet: Any) -> None:
        super().__init__(name="robotensor-weights", daemon=True)
        self.cfg = cfg
        self.state_path = state_path
        self.wallet = wallet
        self.stop = threading.Event()

    def run(self) -> None:
        chain = chain_.Chain(self.cfg.network, self.cfg.netuid)
        last = 0
        while not self.stop.is_set():
            try:
                block = chain.block()
                if last == 0 or block - last >= self.cfg.weights_interval_blocks:
                    # A read-only view of the lane: its own state object, never saved from here.
                    vector = lane(self.cfg, State(self.state_path))
                    weights = set_weights(self.cfg, vector, chain, self.wallet)
                    last = block
                    log.info("set weights at block %s: %s", block, weights)
            except chain_.RateLimited as exc:
                log.info("weights wait: %s", exc)
            except Exception:  # noqa: BLE001 - logged; the next tick tries again
                log.exception("setting weights failed")
            self.stop.wait(12.0)
        chain.close()


def step(cfg: Config, state: State, vector: VectorLane, chain: chain_.Chain) -> str:
    """One pass of the loop: intake, then genesis or the oldest queued duel. What it did."""
    from icil_orchestrator.duel.orchestrate import CrownMoved, DuelFailed

    changed = vector.intake(chain.commitments(), chain.block())
    for entry in changed:
        log.info("intake: %s from %s is %s", entry.entry, entry.hotkey, entry.status)
    if not vector.ready():
        log.info("the throne is empty: crowning the baseline by genesis")
        result = vector.genesis(chain)
        return f"genesis: {result['status']} ({result['reason']})"
    queue = vector.queue()
    if not queue:
        return "idle"
    entry = queue[0]
    log.info("duel: %s from %s (committed at %s)", entry.entry, entry.hotkey, entry.commit_block)
    try:
        result = vector.duel(entry, chain)
    except seed_.NotYet as exc:
        return f"waiting: {exc}"
    except CrownMoved as exc:
        return f"crown moved, duel again: {exc}"
    except DuelFailed as exc:
        log.error("duel of %s failed (it stays queued): %s", entry.entry, exc)
        return f"failed: {exc}"
    record = result.get("record") or {}
    return (
        f"{result['status']}: {result['reason']}; dethroned={record.get('dethroned')} "
        f"king={record.get('king_scores', {}).get('average') if record else None} "
        f"challenger={record.get('challenger_scores', {}).get('average') if record else None}"
    )


def cmd_run(args: argparse.Namespace, cfg: Config) -> int:
    wallet = chain_.wallet(cfg.wallet_name, cfg.wallet_hotkey, cfg.wallet_path)
    state = State(cfg.state)
    vector = lane(cfg, state)
    chain = chain_.Chain(cfg.network, cfg.netuid)
    weights = WeightsThread(cfg, str(cfg.state), wallet) if not args.no_weights else None
    if weights is not None:
        weights.start()
    try:
        while True:
            try:
                what = step(cfg, state, vector, chain)
            except Exception:  # noqa: BLE001 - logged; the loop goes on after a pause
                log.exception("the loop's step failed")
                what = "error"
            log.info("step: %s", what)
            if args.once:
                return 0
            if what in ("idle", "error") or what.startswith(("waiting", "failed")):
                time.sleep(IDLE_S)
    except KeyboardInterrupt:
        return 0
    finally:
        if weights is not None:
            weights.stop.set()
        chain.close()


def cmd_intake(args: argparse.Namespace, cfg: Config) -> int:
    state = State(cfg.state)
    vector = lane(cfg, state)
    chain = chain_.Chain(cfg.network, cfg.netuid)
    changed = vector.intake(chain.commitments(), chain.block())
    print(json.dumps([e.__dict__ for e in changed], indent=1))
    print(json.dumps([e.__dict__ for e in vector.queue()], indent=1))
    return 0


def cmd_duel(args: argparse.Namespace, cfg: Config) -> int:
    state = State(cfg.state)
    vector = lane(cfg, state)
    chain = chain_.Chain(cfg.network, cfg.netuid)
    if not vector.ready():
        print(json.dumps(vector.genesis(chain), indent=1, default=str))
    if args.challenger:
        repo, _, revision = args.challenger.partition("@")
        from icil_orchestrator.ids import submission_key

        key = submission_key(repo, revision)
        entries = state.lane("vector")["entries"]
        entries.setdefault(
            key,
            {
                "hotkey": args.hotkey or "manual",
                "repo": repo,
                "revision": revision,
                "commit_block": chain.block() - seed_.FINALITY - 1,
                "status": "queued",
                "reason": "manual",
            },
        )
        state.save()
        entry = Entry(
            key, entries[key]["hotkey"], repo, revision, entries[key]["commit_block"], "queued"
        )
    else:
        queue = vector.queue()
        if not queue:
            print("nothing queued")
            return 0
        entry = queue[0]
    result = vector.duel(entry, chain, size=args.size)
    print(json.dumps({k: v for k, v in result.items() if k != "units"}, indent=1, default=str))
    return 0


def cmd_weights(args: argparse.Namespace, cfg: Config) -> int:
    state = State(cfg.state)
    vector = lane(cfg, state)
    chain = chain_.Chain(cfg.network, cfg.netuid)
    if args.dry_run:
        weights = compute_weights(cfg, vector, chain)
    else:
        wallet = chain_.wallet(cfg.wallet_name, cfg.wallet_hotkey, cfg.wallet_path)
        weights = set_weights(cfg, vector, chain, wallet)
    print(json.dumps({"block": chain.block(), "weights": weights}, indent=1))
    return 0


def cmd_status(args: argparse.Namespace, cfg: Config) -> int:
    state = State(cfg.state)
    vector = lane(cfg, state)
    head = vector.engine.store.head("vector_l1") or {}
    print(
        json.dumps(
            {
                "king": head.get("king"),
                "champions": vector.champions(),
                "queue": [e.__dict__ for e in vector.queue()],
                "entries": state.lane("vector")["entries"],
            },
            indent=1,
            default=str,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="robotensor-validator", description=__doc__.split("\n")[0]
    )
    parser.add_argument("--config", required=True, help="config/<network>.toml")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="the validator loop")
    run.add_argument("--once", action="store_true", help="one step, then exit")
    run.add_argument("--no-weights", action="store_true", help="never set weights")
    run.set_defaults(func=cmd_run)
    sub.add_parser("intake", help="read commitments into the lanes once").set_defaults(
        func=cmd_intake
    )
    duel = sub.add_parser("duel", help="run one duel now (after genesis if needed)")
    duel.add_argument(
        "--challenger", default=None, help="owner/name@sha, instead of the queue's next"
    )
    duel.add_argument(
        "--hotkey", default=None, help="the hotkey a manual challenger is credited to"
    )
    duel.add_argument("--size", default=None, help="the duel size (default: the config's)")
    duel.set_defaults(func=cmd_duel)
    weights = sub.add_parser("weights", help="compute and set weights")
    weights.add_argument("--dry-run", action="store_true", help="print, do not set")
    weights.set_defaults(func=cmd_weights)
    sub.add_parser("status", help="queue, king and champions").set_defaults(func=cmd_status)
    return parser


def _logging() -> None:
    """Our loggers with a handler of their own. Importing bittensor sets every logger that exists
    then to CRITICAL, which would hide every line of the loop's progress, so it is imported first
    and our loggers are put back after it."""
    import bittensor  # noqa: F401 - for its logging setup, before ours

    for name, logger in list(logging.root.manager.loggerDict.items()):
        if name.startswith(("robotensor", "icil_orchestrator")) and isinstance(
            logger, logging.Logger
        ):
            logger.setLevel(logging.NOTSET)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    for name in ("robotensor", "icil_orchestrator"):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.handlers[:] = [handler]
        logger.propagate = False


def main(argv: list[str] | None = None) -> int:
    _logging()
    args = build_parser().parse_args(argv)
    cfg = load(args.config)
    return int(args.func(args, cfg) or 0)


if __name__ == "__main__":
    sys.exit(main())
