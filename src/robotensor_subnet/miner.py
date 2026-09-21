
    check    the file against the architecture the validator serves, as the validator checks it
    upload   model.safetensors (and an optional README) to your Hugging Face model repository;
             prints the commit sha to commit
    commit   write `vector1:<repo>@<sha>` as your hotkey's commitment on the subnet
    status   your hotkey's commitment on chain, and what the Hub shows for it

Order matters for copy protection: upload to a PRIVATE repository, commit the sha on chain, then
make the repository public. The validator waits for a private repository for a while after the
commitment; weights byte-identical to an earlier commitment's are refused as a duplicate, and the
earlier commitment keeps them, so whoever commits first owns the weights.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from . import hub
from .protocol import commitment as commitment_

DEFAULT_NETWORK = "test"


def cmd_check(args: argparse.Namespace) -> int:
    from vector_runtime import check

    report = check.check_weights(
        Path(args.dir) / hub.WEIGHTS_FILE if Path(args.dir).is_dir() else Path(args.dir)
    )
    print(
        json.dumps(
            {
                "ok": report.ok,
                "errors": list(report.errors),
                "weights_sha256": report.weights_sha256,
            },
            indent=1,
        )
    )
    return 0 if report.ok else 1


def cmd_upload(args: argparse.Namespace) -> int:
    weights = Path(args.dir) / hub.WEIGHTS_FILE if Path(args.dir).is_dir() else Path(args.dir)
    if not weights.is_file():
        print(f"error: no {hub.WEIGHTS_FILE} at {weights}", file=sys.stderr)
        return 2
    readme = Path(args.readme).read_text(encoding="utf-8") if args.readme else None
    sha = hub.upload_weights(
        args.repo,
        str(weights),
        token=os.environ.get("HF_TOKEN"),
        private=args.private,
        readme=readme,
    )
    print(
        json.dumps(
            {"repo": args.repo, "revision": sha, "commitment": commitment_.encode(args.repo, sha)}
        )
    )
    return 0


def cmd_commit(args: argparse.Namespace) -> int:
    from . import chain as chain_

    data = commitment_.encode(args.repo, args.revision)
    wallet = chain_.wallet(args.wallet_name, args.wallet_hotkey, args.wallet_path)
    chain = chain_.Chain(args.network, args.netuid)
    try:
        chain.commit(wallet, data)
        block = chain.block()
    finally:
        chain.close()
    print(json.dumps({"committed": data, "hotkey": wallet.hotkey.ss58_address, "block": block}))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    from . import chain as chain_

    chain = chain_.Chain(args.network, args.netuid)
    try:
        mine = [c for c in chain.commitments() if c.hotkey == args.hotkey]
    finally:
        chain.close()
    out = []
    for c in mine:
        entry = {"hotkey": c.hotkey, "block": c.block, "data": c.data}
        try:
            sub = commitment_.parse(c.data)
            found = hub.inspect(
                sub.repo,
                sub.revision,
                frozenset({hub.WEIGHTS_FILE, "README.md", ".gitattributes"}),
                token=os.environ.get("HF_TOKEN"),
            )
            entry["weights_sha256"] = found.weights_sha256
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            entry["problem"] = str(exc)
        out.append(entry)
    print(json.dumps(out, indent=1))
    return 0


def _chain_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--network", default=DEFAULT_NETWORK, help="finney, test, or ws://host:port"
    )
    parser.add_argument("--netuid", type=int, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="robotensor-miner", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="check model.safetensors as the validator does")
    check.add_argument(
        "--dir", required=True, help="the directory holding model.safetensors, or the file"
    )
    check.set_defaults(func=cmd_check)

    upload = sub.add_parser("upload", help="upload model.safetensors; prints the sha to commit")
    upload.add_argument(
        "--dir", required=True, help="the directory holding model.safetensors, or the file"
    )
    upload.add_argument(
        "--repo", required=True, help="your Hugging Face model repository, owner/name"
    )
    upload.add_argument("--readme", default=None, help="a README.md to upload beside it")
    upload.add_argument("--private", action="store_true", help="create the repository private")
    upload.set_defaults(func=cmd_upload)

    commit = sub.add_parser("commit", help="commit repo@sha on chain for your hotkey")
    commit.add_argument("--repo", required=True)
    commit.add_argument("--revision", required=True, help="the 40-hex commit sha upload printed")
    _chain_args(commit)
    commit.add_argument("--wallet.name", dest="wallet_name", required=True)
    commit.add_argument("--wallet.hotkey", dest="wallet_hotkey", required=True)
    commit.add_argument("--wallet.path", dest="wallet_path", default=None)
    commit.set_defaults(func=cmd_commit)

    status = sub.add_parser("status", help="your hotkey's commitment on chain")
    status.add_argument("--hotkey", required=True, help="your hotkey's ss58 address")
    _chain_args(status)
    status.set_defaults(func=cmd_status)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
