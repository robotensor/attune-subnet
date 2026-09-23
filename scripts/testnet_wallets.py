"""Create the testnet wallets: one coldkey, the owner/validator hotkey, and a test miner hotkey.

    python scripts/testnet_wallets.py [--name robotensor-owner] [--path ~/.bittensor/wallets]

Unencrypted keys, for a test network only; never run this for a coldkey that will hold real TAO.
The coldkey pays for everything (the subnet lock and both registrations), so only its address needs
test TAO. Existing keys are kept, so this is safe to run twice; it prints addresses, never secrets.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bittensor_wallet import Wallet

#: The owner's own hotkey validates (a subnet owner needs no validator permit), and `miner1` stands
#: in for a miner during the rehearsal.
HOTKEYS = ("default", "miner1")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="robotensor-owner", help="the coldkey's wallet name")
    parser.add_argument("--path", default=None, help="where wallets live (default: btcli's)")
    parser.add_argument("--hotkeys", nargs="*", default=list(HOTKEYS))
    args = parser.parse_args()

    kwargs = {"path": args.path} if args.path else {}
    coldkey = Wallet(name=args.name, hotkey=args.hotkeys[0], **kwargs)
    if not Path(coldkey.coldkeypub_file.path).exists():
        coldkey.create_new_coldkey(n_words=24, use_password=False, overwrite=False, suppress=True)
    out = {"wallet": args.name, "coldkey": coldkey.coldkeypub.ss58_address, "hotkeys": {}}
    for name in args.hotkeys:
        w = Wallet(name=args.name, hotkey=name, **kwargs)
        if not Path(w.hotkey_file.path).exists():
            w.create_new_hotkey(n_words=24, use_password=False, overwrite=False, suppress=True)
        out["hotkeys"][name] = w.hotkey.ss58_address
    out["files"] = str(Path(coldkey.coldkeypub_file.path).parent)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
