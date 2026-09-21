"""Create the rehearsal subnet on a localnet: wallets, the subnet, its activation, registrations.

    python scripts/localnet_setup.py [--network ws://127.0.0.1:9944] [--wallets var/wallets]

Throwaway keys only, under `--wallets` (git-ignored): the owner's coldkey is the localnet's funded
dev account `//Alice`, which also validates (a subnet owner needs no validator permit); `miner1` and
`miner2` get fresh coldkeys funded from it. Prints the netuid and writes it into
config/localnet.toml. Idempotent: an existing subnet owned by the owner is reused.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import bittensor as bt
from bittensor_wallet import Wallet

ROOT = Path(__file__).resolve().parents[1]


def wallet(path: Path, name: str, coldkey_uri: str | None = None) -> Wallet:
    w = Wallet(name=name, hotkey="default", path=str(path))
    if coldkey_uri is not None:
        w.create_coldkey_from_uri(coldkey_uri, use_password=False, overwrite=True, suppress=True)
    elif not Path(w.coldkeypub_file.path).exists():
        w.create_new_coldkey(n_words=12, use_password=False, overwrite=True, suppress=True)
    if not Path(w.hotkey_file.path).exists():
        w.create_new_hotkey(n_words=12, use_password=False, overwrite=True, suppress=True)
    return w


def ok(response, what: str) -> None:
    if not getattr(response, "success", False):
        raise SystemExit(f"{what} failed: {getattr(response, 'message', response)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", default="ws://127.0.0.1:9944")
    parser.add_argument("--wallets", default=str(ROOT / "var" / "wallets"))
    parser.add_argument("--config", default=str(ROOT / "config" / "localnet.toml"))
    args = parser.parse_args()
    path = Path(args.wallets)
    path.mkdir(parents=True, exist_ok=True)

    sub = bt.Subtensor(network=args.network)
    owner = wallet(path, "owner", "//Alice")
    miners = [wallet(path, name) for name in ("miner1", "miner2")]

    for m in miners:
        if sub.get_balance(m.coldkeypub.ss58_address).tao < 100:
            ok(
                sub.transfer(owner, m.coldkeypub.ss58_address, bt.Balance.from_tao(1000)),
                "transfer",
            )

    netuid = None
    for n in sub.get_all_subnets_netuid():
        if int(n) != 0 and sub.get_subnet_owner_hotkey(int(n)) == owner.hotkey.ss58_address:
            netuid = int(n)
    if netuid is None:
        before = set(int(n) for n in sub.get_all_subnets_netuid())
        ok(sub.register_subnet(owner), "register_subnet")
        created = set(int(n) for n in sub.get_all_subnets_netuid()) - before
        netuid = min(created)
        print(f"registered subnet {netuid}")
    if not sub.is_subnet_active(netuid):
        response = sub.start_call(owner, netuid, wait_for_finalization=True)
        print(
            f"start_call: {getattr(response, 'success', response)} {getattr(response, 'message', '')}"
        )

    for m in miners:
        if not sub.is_hotkey_registered(m.hotkey.ss58_address, netuid):
            ok(sub.burned_register(m, netuid), f"burned_register {m.name}")
    uids = {
        w.name: sub.get_uid_for_hotkey_on_subnet(w.hotkey.ss58_address, netuid)
        for w in [owner, *miners]
    }
    print(f"netuid {netuid}; uids {uids}")
    config = Path(args.config)
    text = config.read_text(encoding="utf-8")
    config.write_text(re.sub(r"(?m)^netuid = \d+", f"netuid = {netuid}", text), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
