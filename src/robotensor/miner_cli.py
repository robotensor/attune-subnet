"""`attune`, as a miner installs it (`pip install robotensor-attune`): a submission, start to end.

    attune --version
    attune init [DIR]
    attune doctor [--competition vector] [--json]
    attune miner check | upload | commit | submit | status

The same `init`, `doctor` and `miner` commands as the validator's install, which has the
validator's besides. `--role miner` is accepted, so a command written for either one runs here.
Which competition `doctor` checks for: `--competition`, then `$ATTUNE_COMPETITION` (or the older
`$ROBOTENSOR_COMPETITION`), then the only one open.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__

COMPETITION_ENVS = ("ATTUNE_COMPETITION", "ROBOTENSOR_COMPETITION")


class CompetitionError(ValueError):
    """The command does not know which competition it is about."""


def competition(chosen: str | None) -> str:
    """The competition named by the flag or the environment, or the only one there is."""
    from .protocol.commitment import LANES

    named = chosen or next((os.environ[e] for e in COMPETITION_ENVS if os.environ.get(e)), "")
    if named:
        if named not in LANES:
            raise CompetitionError(f"{named} is not a competition ({', '.join(LANES)})")
        return named
    if len(LANES) == 1:
        return LANES[0]
    raise CompetitionError(f"say which competition: --competition {'|'.join(LANES)}")


def cmd_init(args: argparse.Namespace) -> int:
    from . import init as init_

    print(init_.miner(Path(args.directory).expanduser().resolve()))
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    from .doctor import Report, miner_checks, render

    name = competition(args.competition)
    found = Report("miner", name, miner_checks(name))
    print(json.dumps(found.as_dict(), indent=1, default=str) if args.json else render(found))
    return 0 if found.ok else 1


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

    doctor = sub.add_parser("doctor", help="can this host check, upload and commit?")
    doctor.add_argument("--competition", default=None, help="vector (the only one open)")
    doctor.add_argument("--role", default="miner", choices=("miner",), help=argparse.SUPPRESS)
    doctor.add_argument("--json", action="store_true", help="the report as JSON")
    doctor.set_defaults(func=cmd_doctor)

    miner = sub.add_parser("miner", help="check, upload and commit a submission")
    miner_.add_subcommands(miner.add_subparsers(dest="command", required=True))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except CompetitionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
