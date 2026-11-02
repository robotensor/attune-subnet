"""`attune miner`: check a `vector_v1.1` weights file, submit it and commit it on chain.

    check    model.safetensors, the only file a submission holds, against the architecture the
             validator serves, as the validator checks it
    upload   model.safetensors (and an optional README) to your Hugging Face model repository;
             prints the commit sha to commit
    commit   write your repository, its revision and your weights' sha256 as your hotkey's
             commitment on the subnet (`vector:<repo>@<commit>.<digest>`)
    status   your hotkey's commitment on chain, and what the Hub shows for it

Your repository is gated: `upload` and `submit` create it with manual access approval and let
the validator's account (`hub.VALIDATOR_USER`, or each `--validator-user` given) in before a byte
is uploaded, and read the Hub's accepted list back to be sure, so nobody else can take your
weights. The validator looks your repository up when its turn comes and downloads the weights
then - that is the one moment it must answer: a repository it cannot download (deleted, private,
gated without the validator let in) is dropped as missing and the next entry duels instead; a
Hub that does not answer is asked again a few times first. From then on the duel runs
on the validator's copy, and a crowned model's weights are published by the validator itself,
with the three kings before it. Weights byte-identical to an earlier commitment's are refused as a
duplicate, and the earlier commitment keeps them.

Each hotkey makes one submission: once a commitment of yours is queued, anything else the hotkey
commits is refused, before its duel and after it - even if its repository is then skipped. Check the
weights (`check`) before you commit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from . import hub
from .protocol import commitment as commitment_

DEFAULT_NETWORK = "test"


def weights_digest(path: Path) -> str:
    """The sha256 of the weights file, as the commitment names it."""
    return hub.sha256_of(path)


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


def gate_to(args: argparse.Namespace) -> list[str]:
    """The accounts a submission is gated to: `--validator-user` as given, else the validator's;
    none with `--no-gate`, which is said out loud, as it leaves the weights open to everyone."""
    if args.no_gate:
        print(
            "warning: --no-gate: the repository is left open, anyone can take the weights",
            file=sys.stderr,
        )
        return []
    return list(args.validator_user or [hub.VALIDATOR_USER])


def _upload(args: argparse.Namespace, weights: Path) -> tuple[str, list[str]]:
    """The weights (and README) uploaded, gated as `args` say: the commit sha, and who is let
    in. `hub.NotLetIn` when the Hub would not let an account in: nothing uploaded."""
    users = gate_to(args)
    readme = Path(args.readme).read_text(encoding="utf-8") if args.readme else None
    sha = hub.upload_weights(
        args.repo,
        str(weights),
        token=os.environ.get("HF_TOKEN"),
        gate_to=users or None,
        readme=readme,
    )
    return sha, users


def cmd_upload(args: argparse.Namespace) -> int:
    weights = Path(args.dir) / hub.WEIGHTS_FILE if Path(args.dir).is_dir() else Path(args.dir)
    if not weights.is_file():
        print(f"error: no {hub.WEIGHTS_FILE} at {weights}", file=sys.stderr)
        return 2
    try:
        sha, users = _upload(args, weights)
    except hub.NotLetIn as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    digest = weights_digest(weights)
    print(
        json.dumps(
            {
                "repo": args.repo,
                "revision": sha,
                "digest": digest,
                "gated": bool(users),
                "granted": users,
                "commitment": commitment_.encode(args.repo, sha, digest),
            },
            indent=1,
        )
    )
    return 0


def cmd_commit(args: argparse.Namespace) -> int:
    from . import chain as chain_

    data = commitment_.encode(args.repo, args.revision, args.digest)
    wallet = chain_.wallet(args.wallet_name, args.wallet_hotkey, args.wallet_path)
    chain = chain_.Chain(args.network, args.netuid)
    try:
        chain.commit(wallet, data)
        block = chain.block()
    finally:
        chain.close()
    print(json.dumps({"committed": data, "hotkey": wallet.hotkey.ss58_address, "block": block}))
    return 0


def cmd_submit(args: argparse.Namespace) -> int:
    """check -> upload to a repository gated to the validator -> commit on chain. The weights
    are never open: the validator alone downloads them, and publishes them if they are crowned."""
    from . import chain as chain_

    if cmd_check(args) != 0:
        print("error: the weights did not pass the check the validator runs", file=sys.stderr)
        return 1
    weights = Path(args.dir) / hub.WEIGHTS_FILE if Path(args.dir).is_dir() else Path(args.dir)
    try:
        sha, users = _upload(args, weights)
    except hub.NotLetIn as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    digest = weights_digest(weights)
    data = commitment_.encode(args.repo, sha, digest)
    wallet = chain_.wallet(args.wallet_name, args.wallet_hotkey, args.wallet_path)
    chain = chain_.Chain(args.network, args.netuid)
    try:
        chain.commit(wallet, data)
        block = chain.block()
    finally:
        chain.close()
    print(
        json.dumps(
            {
                "repo": args.repo,
                "revision": sha,
                "digest": digest,
                "gated": bool(users),
                "granted": users,
                "hotkey": wallet.hotkey.ss58_address,
                "committed": data,
                "block": block,
                "fee": "none: this subnet charges nothing to submit",
            },
            indent=1,
        )
    )
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
                hub.vector_shape({hub.WEIGHTS_FILE, "README.md", ".gitattributes"}),
                token=os.environ.get("HF_TOKEN"),
            )
            entry["weights_sha256"] = found.weights_sha256
            entry["digest"] = sub.digest
            entry["digest_matches"] = sub.digest == found.weights_sha256
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            entry["problem"] = str(exc)
        out.append(entry)
    print(json.dumps(out, indent=1))
    return 0


def _gate_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--validator-user",
        action="append",
        default=None,
        help=(
            f"a Hub account let into your gated repository (default {hub.VALIDATOR_USER}); "
            "may be given more than once"
        ),
    )
    parser.add_argument(
        "--no-gate",
        action="store_true",
        help="leave the repository open to everyone (for a public test submission)",
    )


def _chain_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--network", default=DEFAULT_NETWORK, help="finney, test, or ws://host:port"
    )
    parser.add_argument("--netuid", type=int, required=True)


def add_subcommands(sub: Any) -> None:
    """Hang a miner's verbs off `sub`, so `attune miner ...` and the standalone command are
    the same parser rather than two that drift."""
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
    _gate_args(upload)
    upload.set_defaults(func=cmd_upload)

    commit = sub.add_parser("commit", help="commit repo@sha on chain for your hotkey")
    commit.add_argument("--repo", required=True)
    commit.add_argument("--revision", required=True, help="the 40-hex commit sha upload printed")
    commit.add_argument(
        "--digest",
        required=True,
        help="the sha256 of model.safetensors (check and upload print it)",
    )
    _chain_args(commit)
    commit.add_argument("--wallet.name", dest="wallet_name", required=True)
    commit.add_argument("--wallet.hotkey", dest="wallet_hotkey", required=True)
    commit.add_argument("--wallet.path", dest="wallet_path", default=None)
    commit.set_defaults(func=cmd_commit)

    submit = sub.add_parser(
        "submit", help="check, upload gated to the validator, then commit on chain"
    )
    submit.add_argument(
        "--dir", required=True, help="the directory holding model.safetensors, or the file"
    )
    submit.add_argument("--repo", required=True, help="your Hugging Face model repository")
    submit.add_argument("--readme", default=None, help="a README.md to upload beside it")
    _gate_args(submit)
    _chain_args(submit)
    submit.add_argument("--wallet.name", dest="wallet_name", required=True)
    submit.add_argument("--wallet.hotkey", dest="wallet_hotkey", required=True)
    submit.add_argument("--wallet.path", dest="wallet_path", default=None)
    submit.set_defaults(func=cmd_submit)

    status = sub.add_parser("status", help="your hotkey's commitment on chain")
    status.add_argument("--hotkey", required=True, help="your hotkey's ss58 address")
    _chain_args(status)
    status.set_defaults(func=cmd_status)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="robotensor-miner", description=__doc__.split("\n")[0])
    add_subcommands(parser.add_subparsers(dest="command", required=True))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
