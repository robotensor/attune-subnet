"""The chain, and the only module that imports bittensor.

Pinned to bittensor 10.5 (`pyproject.toml`): version 11 replaced the SDK's API, and the subnets this
one follows (SN3, SN80, SN98) all still pin 10.x. Moving to 11 changes this file and nothing else.

Everything here is a thin call to `Subtensor`:

- `commitments()`: every hotkey's current commitment on the subnet with the block it was made at,
  from one query of the `Commitments.CommitmentOf` storage map. The chain keeps one commitment per
  hotkey; a new one replaces it and resets its block.
- `block()`, `block_hash(n)`, `finalized()`, `uids()`: the head, a block's hash, the finalized
  head and its hash, and the metagraph's hotkeys.
- `commit(wallet, data)` and `set_weights(wallet, weights)`: the two writes.

`Subtensor` holds a websocket that is not safe to share between threads, so a `Chain` opens its own
and a thread that needs the chain makes its own `Chain`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Commitment:
    hotkey: str
    block: int
    data: str


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
        try:
            self.subtensor.close()
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


def _message(response: Any) -> str:
    return str(getattr(response, "message", None) or response)
