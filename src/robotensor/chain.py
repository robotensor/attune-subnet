"""The chain, and the only module that imports bittensor.

Pinned to bittensor 10.5 (`pyproject.toml`): version 11 replaced the SDK's API, and the subnets this
one follows (SN3, SN80, SN98) all still pin 10.x. Moving to 11 changes this file and nothing else.

Everything here is a thin call to `Subtensor`:

- `commitments()`: every hotkey's current commitment on the subnet with the block it was made at,
  from one query of the `Commitments.CommitmentOf` storage map. The chain keeps one commitment per
  hotkey; a new one replaces it and resets its block.
- `submissions(start_block, cache)`: each registration's one submission and the Vector commitments
  its hotkeys hold now (`protocol.slots`). A registration's first commitment is found on an archive
  node from `Commitments.LastCommitment`, which only grows while a UID is held: a few steps back
  from the newest (`LastCommitment` at `b - 1` is the commitment before the one made at `b`), then
  a binary search over the blocks, so a hotkey that commits a thousand times costs as much as one
  that commits twenty. What is found never changes and is kept in `cache`, a file a validator may
  lose: the chain gives the same answer again.
- `block()`, `block_hash(n)`, `finalized()`, `uids()`: the head, a block's hash, the finalized
  head and its hash, and the metagraph's hotkeys.
- `commit(wallet, data)` and `set_weights(wallet, weights)`: the two writes.

`Subtensor` holds a websocket that is not safe to share between threads, so a `Chain` opens its own
and a thread that needs the chain makes its own `Chain`.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Commitment:
    hotkey: str
    block: int
    data: str
    #: The registration it was made from, `uid@block registered at` (`protocol.slots`); None when
    #: it was not read with one.
    slot: str | None = None
    #: Whether it is its registration's one submission, as the chain's history says.
    submission: bool = True


#: Where a network's past state is read (`Chain.first`), when not from its own endpoint: a lite
#: node keeps only the last few hundred blocks' state. `SUBTENSOR_ARCHIVE` in the environment (or
#: the config's `.env`) names another, such as one's own archive node.
ARCHIVES = {"finney": "archive"}
ARCHIVE_ENV = "SUBTENSOR_ARCHIVE"
#: How many commitments back from the newest are stepped through before searching by bisection:
#: a miner commits once or twice, and each step is one read.
STEPS_BACK = 4
#: How many times one past read is tried, the first wait between tries doubling after each: public
#: archives answer a burst of them with "Historical work rate limit exceeded".
ARCHIVE_TRIES = 6
ARCHIVE_GAP_S = 2.0
#: How long one past read is waited for before it is tried again.
ARCHIVE_TIMEOUT_S = 20.0


class ChainError(RuntimeError):
    """The chain refused a write, or could not be read."""


class RateLimited(ChainError):
    """Weights were set too recently: `blocks` more must pass (the subnet's weights rate limit)."""

    def __init__(self, blocks: int) -> None:
        self.blocks = blocks
        super().__init__(f"weights were set too recently; {blocks} more block(s) must pass")


class Chain:
    def __init__(self, network: str, netuid: int) -> None:
        import bittensor as bt

        self.network = network
        self.netuid = int(netuid)
        self.subtensor = bt.Subtensor(network=network)
        self._archive: Any = None
        #: The past does not change, so nothing of it is read twice: `LastCommitment` at a block,
        #: per (hotkey, block); each first commitment found, per (hotkey, floor); and what a
        #: hotkey committed at a block. The last two are also kept in `submissions`' cache file.
        self._last_at: dict[tuple[str, int], int | None] = {}
        self._firsts: dict[tuple[str, int], int] = {}
        self._made: dict[tuple[str, int], str] = {}
        self._cache: Path | None = None

    # -- reads ------------------------------------------------------------------------------

    def block(self) -> int:
        return int(self.subtensor.get_current_block())

    def block_hash(self, block: int) -> str:
        value = self.subtensor.get_block_hash(int(block))
        if not value:
            raise ChainError(f"the chain has no hash for block {block}")
        return str(value)

    def finalized(self) -> tuple[int, str]:
        """The finalized head: its number and its hash, read together so they name one block."""
        substrate = self.subtensor.substrate
        value = substrate.get_chain_finalised_head()
        if not value:
            raise ChainError("the chain reported no finalized head")
        return int(substrate.get_block_number(value)), str(value)

    def commitments(self) -> list[Commitment]:
        """Every hotkey's commitment on the subnet, with the block it was made at, oldest first."""
        from bittensor.core.chain_data.utils import decode_metadata

        out = []
        for hotkey, value in self.subtensor.query_map(
            module="Commitments", name="CommitmentOf", params=[self.netuid]
        ):
            record = getattr(value, "value", value)
            try:
                data = decode_metadata(record)
                block = int(record["block"])
            except Exception:  # noqa: BLE001 - a commitment we cannot read is no submission
                continue
            key = getattr(hotkey, "value", hotkey)
            out.append(Commitment(hotkey=str(key), block=block, data=data))
        return sorted(out, key=lambda c: (c.block, c.hotkey))

    def submissions(self, start_block: int, cache: Path | None = None) -> list[Commitment]:
        """Each registration's submission from `start_block` on, and the Vector commitments its
        hotkeys hold now, oldest first, each with its `slot` and whether it is the `submission`.
        A registration whose past the archive does not give now is left out until it does."""
        from .protocol import slots

        self._take_up(cache)
        uids = {str(_v(k)): int(_v(x)) for k, x in self._map("SubtensorModule", "Uids")}
        since = {
            int(_v(k)): int(_v(x)) for k, x in self._map("SubtensorModule", "BlockAtRegistration")
        }
        registered = {hk: (uid, since[uid]) for hk, uid in uids.items() if uid in since}
        last = {str(_v(k)): int(_v(x)) for k, x in self._map("Commitments", "LastCommitment")}
        current = {c.hotkey: c for c in self.commitments()}
        found = len(self._firsts) + len(self._made)
        out = slots.submissions(
            registered, self.roots(), last, current, self._first, self._read, start_block
        )
        if cache is not None and len(self._firsts) + len(self._made) != found:
            _write_json(cache, self._keep())
        return out

    def roots(self) -> dict[str, str]:
        """Each hotkey swapped into on the subnet, with the first hotkey of its swaps
        (`HotkeyRoot`); none on a chain that does not keep them."""
        try:
            pairs = self._map("SubtensorModule", "HotkeyRoot")
            return {str(_v(k)): str(_v(x)) for k, x in pairs}
        except Exception as exc:  # noqa: BLE001 - an older runtime: swaps are not followed
            log.warning("the chain keeps no HotkeyRoot (%s): hotkey swaps are not followed", exc)
            return {}

    def first(self, hotkey: str, newest: int, floor: int) -> int:
        """The block of `hotkey`'s first commitment from `floor` on, its newest made at `newest`
        (at or after `floor`). `ChainError` when the past cannot be read."""
        if (hotkey, floor) in self._firsts:
            return self._firsts[(hotkey, floor)]
        block = newest
        for _ in range(STEPS_BACK):
            before = self._last(hotkey, block - 1)
            if before is None or before < floor:
                self._firsts[(hotkey, floor)] = block
                return block
            block = before
        # `LastCommitment` only grows while the UID is held: the first block at which it reached
        # `floor` is the block of the first commitment from `floor` on.
        low, high = floor, block
        while low < high:
            middle = (low + high) // 2
            at = self._last(hotkey, middle)
            if at is not None and at >= floor:
                high = middle
            else:
                low = middle + 1
        self._firsts[(hotkey, floor)] = low
        return low

    def made(self, hotkey: str, block: int) -> str:
        """What `hotkey` committed at `block`; empty when it cannot be read as a commitment (a
        timelocked one, say). `ChainError` when the past cannot be read."""
        from bittensor.core.chain_data.utils import decode_metadata

        if (hotkey, block) not in self._made:
            record = self._at("Commitments", "CommitmentOf", hotkey, block)
            data = ""
            if isinstance(record, Mapping) and int(record.get("block", -1)) == block:
                try:
                    data = decode_metadata(record)
                except Exception:  # noqa: BLE001 - what we cannot read is no submission
                    data = ""
            self._made[(hotkey, block)] = data
        return self._made[(hotkey, block)]

    def _first(self, hotkey: str, newest: int, floor: int) -> int | None:
        try:
            return self.first(hotkey, newest, floor)
        except ChainError as exc:
            log.warning("left out until the archive answers: %s", exc)
            return None

    def _read(self, hotkey: str, block: int) -> str | None:
        try:
            return self.made(hotkey, block)
        except ChainError as exc:
            log.warning("left out until the archive answers: %s", exc)
            return None

    def _last(self, hotkey: str, block: int) -> int | None:
        if (hotkey, block) not in self._last_at:
            value = self._at("Commitments", "LastCommitment", hotkey, block)
            self._last_at[(hotkey, block)] = None if value is None else int(value)
        return self._last_at[(hotkey, block)]

    def _take_up(self, cache: Path | None) -> None:
        """What `cache` keeps of this netuid's past, once per connection; nothing when it is
        missing or unreadable, which only costs reads."""
        if cache is None or cache == self._cache:
            return
        self._cache = cache
        try:
            doc = json.loads(cache.read_text(encoding="utf-8"))
            if doc.get("netuid") != self.netuid:
                return
            for key, block in doc["firsts"].items():
                hotkey, _, floor = key.rpartition("@")
                self._firsts.setdefault((hotkey, int(floor)), int(block))
            for key, data in doc["made"].items():
                hotkey, _, block = key.rpartition("@")
                self._made.setdefault((hotkey, int(block)), str(data))
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return

    def _keep(self) -> dict[str, Any]:
        return {
            "netuid": self.netuid,
            "firsts": {f"{h}@{f}": b for (h, f), b in sorted(self._firsts.items())},
            "made": {f"{h}@{b}": d for (h, b), d in sorted(self._made.items())},
        }

    @property
    def archive(self) -> str:
        return os.environ.get(ARCHIVE_ENV) or ARCHIVES.get(self.network, self.network)

    def _map(self, module: str, name: str) -> list[Any]:
        return list(self.subtensor.query_map(module=module, name=name, params=[self.netuid]))

    def _at(self, module: str, name: str, hotkey: str, block: int) -> Any:
        """`module.name(netuid, hotkey)` as it stood at the end of `block`, from the archive."""
        if self._archive is None:
            import bittensor as bt

            # A connection of its own, quick to give up: a public archive may drop a past read it
            # rate-limits without a word, which the SDK would wait out for minutes.
            self._archive = bt.Subtensor(network=self.archive)
            self._archive.substrate.retry_timeout = ARCHIVE_TIMEOUT_S
            self._archive.substrate.max_retries = 1
        for attempt in range(ARCHIVE_TRIES):
            try:
                value = self._archive.query_module(
                    module, name, params=[self.netuid, hotkey], block=int(block)
                )
                return _v(value)
            except Exception as exc:  # noqa: BLE001 - public archives rate-limit past reads
                if attempt + 1 == ARCHIVE_TRIES:
                    raise ChainError(
                        f"could not read {module}.{name} for {hotkey} at block {block} (does "
                        f"{self.archive} keep past state? {ARCHIVE_ENV} names another): {exc}"
                    ) from exc
                time.sleep(ARCHIVE_GAP_S * 2**attempt)
        raise AssertionError("unreachable")

    def uids(self) -> dict[str, int]:
        """The metagraph's hotkeys, each with its UID now. UIDs are recycled: map at the end."""
        metagraph = self.subtensor.metagraph(self.netuid, lite=True)
        return {
            str(hk): int(uid) for uid, hk in zip(metagraph.uids, metagraph.hotkeys, strict=True)
        }

    def owner_hotkey(self) -> str | None:
        return self.subtensor.get_subnet_owner_hotkey(self.netuid)

    def weights_rate_limit(self) -> int | None:
        return self.subtensor.weights_rate_limit(self.netuid)

    # -- writes -----------------------------------------------------------------------------

    def commit(self, wallet: Any, data: str) -> None:
        response = self.subtensor.set_commitment(
            wallet=wallet,
            netuid=self.netuid,
            data=data,
            wait_for_inclusion=True,
            wait_for_finalization=True,
        )
        if not getattr(response, "success", False):
            raise ChainError(f"the chain refused the commitment: {_message(response)}")

    def set_weights(self, wallet: Any, weights: Mapping[int, float]) -> None:
        """Set `weights` for `wallet`'s hotkey. `RateLimited` while the subnet's weights rate limit
        has not passed since its last update: the SDK then returns a bare failure without trying,
        so the limit is read here first and said plainly."""
        uid = self.subtensor.get_uid_for_hotkey_on_subnet(wallet.hotkey.ss58_address, self.netuid)
        if uid is None:
            raise ChainError(f"{wallet.hotkey.ss58_address} is not registered on {self.netuid}")
        since = self.subtensor.blocks_since_last_update(self.netuid, uid)
        limit = self.subtensor.weights_rate_limit(self.netuid)
        if since is not None and limit is not None and since <= limit:
            raise RateLimited(int(limit) - int(since) + 1)
        uids = sorted(weights)
        response = self.subtensor.set_weights(
            wallet=wallet,
            netuid=self.netuid,
            uids=uids,
            weights=[float(weights[u]) for u in uids],
            wait_for_inclusion=True,
            wait_for_finalization=False,
        )
        if not getattr(response, "success", False):
            raise ChainError(f"the chain refused the weights: {_message(response)}")

    def close(self) -> None:
        for subtensor in (self.subtensor, self._archive):
            if subtensor is None:
                continue
            try:
                subtensor.close()
            except Exception:  # noqa: BLE001 - closing a socket that already went is fine
                pass


def wallet(name: str, hotkey: str, path: str | None = None) -> Any:
    """A bittensor wallet by name, as `btcli` keeps them (`~/.bittensor/wallets`)."""
    import bittensor as bt

    return (
        bt.Wallet(name=name, hotkey=hotkey, path=path)
        if path
        else bt.Wallet(name=name, hotkey=hotkey)
    )


def _write_json(path: Path, doc: dict[str, Any]) -> None:
    """`doc` at `path`, whole or not at all; a cache that cannot be written is only slower."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}")
        tmp.write_text(json.dumps(doc), encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:
        log.warning("could not keep %s: %s", path, exc)


def _v(value: Any) -> Any:
    """A chain value as plain data: the SDK wraps some in an object holding `.value`."""
    return getattr(value, "value", value)


def _message(response: Any) -> str:
    return str(getattr(response, "message", None) or response)
