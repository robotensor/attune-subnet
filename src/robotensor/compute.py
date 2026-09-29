"""Which GPUs a competition may use while it takes a step.

Both engines read `CUDA_VISIBLE_DEVICES` themselves to decide where to put work, and that variable
is process-wide, so two competitions cannot hold different cards inside one process. They run in
processes of their own; this decides what each one's says.

Two boxes, one rule:

- **Cards named per lane** (`[<name>].devices`): each competition has its own, nothing is
  shared, and a lease is free - no waiting, no lock.
- **Nothing named**: the box has one pool and a lane takes all of it for the length of one step,
  under a lock beside the state. Vector at `workers = 4` is about 36 GB and one Horizon episode
  wants 80; overlapping them on one card means both fail slowly rather than one finishing.

A lease is held for one step, never longer, so a competition that runs for hours between steps
cannot starve the other for days.
"""

from __future__ import annotations

import fcntl
import os
import subprocess
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

#: What the lock is called, beside the lanes' state documents.
LOCK = "gpu.lock"


@dataclass(frozen=True)
class Lease:
    """The cards a lane may use for one step."""

    devices: tuple[int, ...]
    #: Whether anything else could want them; false when the config named this lane's own.
    shared: bool = True

    @property
    def environ(self) -> dict[str, str]:
        """What to put in the environment before the engine places any work."""
        if not self.devices:
            return {}
        return {"CUDA_VISIBLE_DEVICES": ",".join(str(d) for d in self.devices)}


def visible_devices() -> tuple[int, ...]:
    """The cards this process may use: `CUDA_VISIBLE_DEVICES` if it is set, else what the driver
    reports, else none - a box with no GPU runs the pure paths and says so when it needs one."""
    named = os.environ.get("CUDA_VISIBLE_DEVICES")
    if named is not None:
        return tuple(int(d) for d in named.split(",") if d.strip().isdigit())
    try:
        listing = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    return tuple(range(sum(1 for line in listing.stdout.splitlines() if line.startswith("GPU "))))


class Broker:
    """Hands out the box's cards, one lane at a time, unless each lane has its own."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        devices: Mapping[str, Sequence[int]] | None = None,
        pool: Sequence[int] | None = None,
    ) -> None:
        self.root = Path(root)
        self.devices = {name: tuple(d) for name, d in (devices or {}).items()}
        self.pool = tuple(pool) if pool is not None else visible_devices()

    @contextmanager
    def lease(self, lane: str) -> Iterator[Lease]:
        """The cards `lane` may use for the step about to run, waiting for them if it must."""
        own = self.devices.get(lane)
        if own:
            yield Lease(own, shared=False)
            return
        self.root.mkdir(parents=True, exist_ok=True)
        with open(self.root / LOCK, "w", encoding="utf-8") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                yield Lease(self.pool)
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)


@contextmanager
def applied(lease: Lease) -> Iterator[None]:
    """`lease.environ` in this process for the length of the block, and what was there after."""
    before = {key: os.environ.get(key) for key in lease.environ}
    os.environ.update(lease.environ)
    try:
        yield
    finally:
        for key, value in before.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
