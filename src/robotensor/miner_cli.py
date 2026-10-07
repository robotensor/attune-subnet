"""`attune`, as a miner installs it (`pip install robotensor-attune`): a submission, start to end.

    attune --version
    attune init [DIR]
    attune miner check | upload | commit | submit | status

The same `init` and `miner` commands as the validator's install, which has the validator's
besides. `--role miner` is accepted, so a command written for either one runs here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__


def cmd_init(args: argparse.Namespace) -> int:
    from . import init as init_

    print(init_.miner(Path(args.directory).expanduser().resolve()))
    return 0


def build_parser() -> argparse.ArgumentParser:
    from . import miner as miner_

    parser = argparse.ArgumentParser(prog="attune", description=__doc__.split("\n")[0])
    parser.add_argument("--version", action="version", version=f"attune {__version__}")
    sub = parser.add_subparsers(dest="part", required=True)

    start = sub.add_parser("init", help="a working directory for a submission")
    start.add_argument("directory", nargs="?", default=".", help="where (default: .)")
    start.add_argument("--role", default="miner", choices=("miner",), help=argparse.SUPPRESS)
    start.add_argument("--competition", default=None, help=argparse.SUPPRESS)
    start.set_defaults(func=cmd_init)

    miner = sub.add_parser("miner", help="check, upload and commit a submission")
    miner_.add_subcommands(miner.add_subparsers(dest="command", required=True))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
