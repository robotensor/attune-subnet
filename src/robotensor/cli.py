"""`robotensor`: one command for everything a miner or a validator of this subnet does.

    robotensor init [DIR] [--role miner|validator]
    robotensor --version
    robotensor doctor [--competition vector] [--role miner|validator] [--json]
    robotensor status [--competition vector]
    robotensor miner     check | upload | commit | status
    robotensor validator run | intake | duel | weights | status

Which competition a command means is resolved once, in this order: `--competition`, then
`$ROBOTENSOR_COMPETITION`, then the only one the config enables. Two enabled and nothing said is
an error naming them, because a miner who meant the other one should not find out from a refused
submission.

`robotensor-miner` and `robotensor-validator` are the same parsers under their old names, and stay
through 0.2 so a running validator's unit file keeps working.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from . import __version__

#: The variable a host can set instead of passing `--competition` to everything.
COMPETITION_ENV = "ROBOTENSOR_COMPETITION"


class CompetitionError(ValueError):
    """The command does not know which competition it is about."""


def competition(
    chosen: str | None,
    enabled: dict[str, Any] | None = None,
    environ: dict[str, str] | None = None,
) -> str:
    """Which competition a command means: the flag, the environment, or the only one enabled."""
    environ = os.environ if environ is None else environ
    named = chosen or environ.get(COMPETITION_ENV) or ""
    if named:
        if enabled is not None and named not in enabled:
            raise CompetitionError(
                f"{named} is not a competition this validator runs ({', '.join(enabled)})"
            )
        return named
    if enabled is None:
        from .protocol.commitment import LANES

        if len(LANES) == 1:
            return LANES[0]
        raise CompetitionError(
            f"say which competition: --competition {'|'.join(LANES)}, or ${COMPETITION_ENV}"
        )
    if len(enabled) == 1:
        return next(iter(enabled))
    raise CompetitionError(
        f"this validator runs {', '.join(enabled)}: say which with --competition, "
        f"or ${COMPETITION_ENV}"
    )


def cmd_init(args: argparse.Namespace) -> int:
    from . import init as init_

    directory = Path(args.directory).expanduser().resolve()
    if args.role == "miner":
        print(init_.miner(directory))
        return 0
    path = init_.validator(directory)
    print(f"wrote {path}")
    print("Fill in the wallet and the netuid, then: robotensor doctor --config", path)
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    from .doctor import render, report

    found = report(args.config, args.competition, role=args.role)
    if args.json:
        print(json.dumps(found.as_dict(), indent=1, default=str))
    else:
        print(render(found))
    return 0 if found.ok else 1


def cmd_status(args: argparse.Namespace) -> int:
    from .config import load
    from .state import State
    from .worker import build

    cfg = load(args.config)
    name = competition(args.competition, cfg.lanes)
    lane = build(cfg, name, State(cfg.state))
    print(json.dumps({name: lane.snapshot()}, indent=1, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    from . import miner as miner_
    from . import validator as validator_

    parser = argparse.ArgumentParser(prog="robotensor", description=__doc__.split("\n")[0])
    parser.add_argument("--version", action="version", version=f"robotensor {__version__}")
    sub = parser.add_subparsers(dest="part", required=True)

    start = sub.add_parser("init", help="a working directory, and a config to fill in")
    start.add_argument("directory", nargs="?", default=".", help="where to write it (default: .)")
    start.add_argument(
        "--role", default="validator", choices=("miner", "validator"), help="what it is for"
    )
    start.add_argument("--competition", default=None, help="vector (the only one open)")
    start.set_defaults(func=cmd_init)

    doctor = sub.add_parser("doctor", help="can this host do what it is configured to do?")
    doctor.add_argument("--config", default=None, help="config/<network>.toml")
    doctor.add_argument("--competition", default=None, help="vector (the only one open)")
    doctor.add_argument(
        "--role", default="validator", choices=("miner", "validator"), help="what to check for"
    )
    doctor.add_argument("--json", action="store_true", help="the report as JSON")
    doctor.set_defaults(func=cmd_doctor)

    status = sub.add_parser("status", help="where a competition stands")
    status.add_argument("--config", required=True, help="config/<network>.toml")
    status.add_argument("--competition", default=None)
    status.set_defaults(func=cmd_status)

    miner = sub.add_parser("miner", help="build, check, upload and commit a submission")
    miner_.add_subcommands(miner.add_subparsers(dest="command", required=True))

    validator = sub.add_parser("validator", help="run the competitions and set weights")
    validator.add_argument("--config", required=True, help="config/<network>.toml")
    validator_.add_subcommands(validator.add_subparsers(dest="command", required=True))
    return parser


def main(argv: list[str] | None = None) -> int:
    from .config import load
    from .validator import _logging

    args = build_parser().parse_args(argv)
    if args.part == "validator":
        _logging()
        return int(args.func(args, load(args.config)) or 0)
    try:
        return int(args.func(args) or 0)
    except CompetitionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
