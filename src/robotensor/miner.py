"""`attune miner`: check a `vector_v1.1` weights file, submit it and commit it on chain.

    check    model.safetensors, the only file a submission holds, against the architecture the
             validator serves, as the validator checks it
    upload   model.safetensors (and an optional README) to your Hugging Face model repository,
             which stays private; prints the commit sha to commit
    commit   write your repository, its revision and your weights' sha256 as your hotkey's
             commitment on the subnet (`vector:<repo>@<commit>.<digest>`), then take the
             repository out of private
    reveal   take the repository out of private, if `commit` could not
    status   your hotkey's commitment on chain, and what the Hub shows for it

Your repository stays private until your commitment is on chain: a gated repository still shows
anyone its commit and the weights' sha256, which is all it takes to commit them first. `upload` and
`submit` make it private and gated with manual access approval, let the validator's account
(`hub.PERMITTED_VALIDATORS`) in and read the Hub's accepted list back
before a byte is uploaded; `commit` and `submit` take it out of private once the chain has the
commitment. If that last step fails, `reveal` does it. Nobody else can take your weights either
way. The validator looks your repository up when its turn comes and downloads the weights
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
import time
from pathlib import Path
from typing import Any

from . import hub
from .protocol import commitment as commitment_

DEFAULT_NETWORK = "test"
#: How many times, and how far apart, a repository is taken out of private after its commitment.
REVEAL_TRIES = 3
REVEAL_GAP_S = 5.0


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


def _upload(args: argparse.Namespace, weights: Path) -> str:
    """The weights (and README) uploaded, gated to the permitted validator accounts: the commit
    sha. `hub.NotLetIn` when the Hub would not let one in: nothing uploaded."""
    readme = Path(args.readme).read_text(encoding="utf-8") if args.readme else None
    return hub.upload_weights(
        args.repo, str(weights), token=os.environ.get("HF_TOKEN"), readme=readme
    )


def _commit_on_chain(args: argparse.Namespace, data: str) -> tuple[Any, int]:
    """`data` as the hotkey's commitment: its wallet, and the head after. A chain call that fails
    may still have landed, with the repository private; the miner must hear what to do then."""
    from . import chain as chain_

    wallet = chain_.wallet(args.wallet_name, args.wallet_hotkey, args.wallet_path)
    chain = chain_.Chain(args.network, args.netuid)
    try:
        chain.commit(wallet, data)
        return wallet, chain.block()
    except Exception:
        print(
            f"error: the commit failed, but it may still have landed. Check `attune miner status "
            f"--hotkey {wallet.hotkey.ss58_address}`; if it is there, run `attune miner reveal "
            f"--repo {args.repo}` at once",
            file=sys.stderr,
        )
        raise
    finally:
        chain.close()


def _reveal(repo: str) -> bool:
    """`repo` out of private, a few tries in; what went wrong said, with what to run. A committed
    repository still private at its turn is dropped as missing, and the submission with it."""
    for tries_left in reversed(range(REVEAL_TRIES)):
        try:
            hub.reveal(repo, token=os.environ.get("HF_TOKEN"))
            return True
        except Exception as exc:  # noqa: BLE001 - whatever it was, the miner must hear of it
            if not tries_left:
                print(
                    f"error: committed, but {repo} is still private ({type(exc).__name__}: {exc}). "
                    f"The validator cannot download it: run `attune miner reveal --repo {repo}` now",
                    file=sys.stderr,
                )
            else:
                time.sleep(REVEAL_GAP_S)
    return False


def cmd_reveal(args: argparse.Namespace) -> int:
    if not _reveal(args.repo):
        return 1
    print(json.dumps({"repo": args.repo, "revealed": True}))
    return 0


def cmd_upload(args: argparse.Namespace) -> int:
    weights = Path(args.dir) / hub.WEIGHTS_FILE if Path(args.dir).is_dir() else Path(args.dir)
    if not weights.is_file():
        print(f"error: no {hub.WEIGHTS_FILE} at {weights}", file=sys.stderr)
        return 2
    try:
        sha = _upload(args, weights)
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
                "granted": list(hub.PERMITTED_VALIDATORS),
                "private": "until `commit` has put it on chain",
                "commitment": commitment_.encode(args.repo, sha, digest),
            },
            indent=1,
        )
    )
    return 0


def cmd_commit(args: argparse.Namespace) -> int:
    data = commitment_.encode(args.repo, args.revision, args.digest)
    wallet, block = _commit_on_chain(args, data)
    revealed = _reveal(args.repo)
    print(
        json.dumps(
            {
                "committed": data,
                "hotkey": wallet.hotkey.ss58_address,
                "block": block,
                "revealed": revealed,
            }
        )
    )
    return 0 if revealed else 1


def cmd_submit(args: argparse.Namespace) -> int:
    """check -> upload to a private repository gated to the validator -> commit on chain -> out
    of private. Nobody sees the commit or the digest until the chain says they are yours; nobody
    but the validator downloads the weights, and it publishes them if they are crowned."""
    if cmd_check(args) != 0:
        print("error: the weights did not pass the check the validator runs", file=sys.stderr)
        return 1
    weights = Path(args.dir) / hub.WEIGHTS_FILE if Path(args.dir).is_dir() else Path(args.dir)
    try:
        sha = _upload(args, weights)
    except hub.NotLetIn as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    digest = weights_digest(weights)
    data = commitment_.encode(args.repo, sha, digest)
    wallet, block = _commit_on_chain(args, data)
    revealed = _reveal(args.repo)
    print(
        json.dumps(
            {
                "repo": args.repo,
                "revision": sha,
                "digest": digest,
                "granted": list(hub.PERMITTED_VALIDATORS),
                "hotkey": wallet.hotkey.ss58_address,
                "committed": data,
                "block": block,
                "revealed": revealed,
                "fee": "none: this subnet charges nothing to submit",
            },
            indent=1,
        )
    )
    return 0 if revealed else 1


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

    reveal = sub.add_parser("reveal", help="take your repository out of private, after commit")
    reveal.add_argument("--repo", required=True)
    reveal.set_defaults(func=cmd_reveal)

    submit = sub.add_parser(
        "submit", help="check, upload private and gated to the validator, commit, then reveal"
    )
    submit.add_argument(
        "--dir", required=True, help="the directory holding model.safetensors, or the file"
    )
    submit.add_argument("--repo", required=True, help="your Hugging Face model repository")
    submit.add_argument("--readme", default=None, help="a README.md to upload beside it")
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
