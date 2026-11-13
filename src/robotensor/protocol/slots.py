"""One submission per registration: which commitment is each UID's one shot.

A registration is a UID from the block its holder registered it at (`BlockAtRegistration`). Its
one submission is the first commitment made from it at or after `max(registered at, start_block)`,
of any kind: one that is not a Vector commitment spends the shot all the same, so junk cannot be
used to push the real one back. Every other commitment made from it is refused, whichever hotkey
made it:

- a hotkey swap moves the UID, and its registration block, to the new hotkey, and the old hotkey's
  commitments stay on chain under it (`HotkeyRoot` links the two), so they still count;
- registering again starts a new registration, paid for with the registration fee.

The hotkeys that held one registration did so one after another, so their commitments do not
interleave: the first of them all is the first of the hotkey whose newest commitment is oldest.
Only that one hotkey's past is searched (`first`), whatever the others did.

All of it is read from the chain (`chain.Chain.submissions`), so a validator starting on an empty
disk decides as one that ran from the start: nothing here depends on what a validator remembers.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping

from ..chain import Commitment
from . import commitment as commitment_


def slot(uid: int, registered_at: int) -> str:
    return f"{uid}@{registered_at}"


def submissions(
    registered: Mapping[str, tuple[int, int]],
    roots: Mapping[str, str],
    last: Mapping[str, int],
    current: Mapping[str, Commitment],
    first: Callable[[str, int, int], int | None],
    read: Callable[[str, int], str | None],
    start_block: int,
) -> list[Commitment]:
    """Each registration's submission, when its first commitment is a Vector one, and the Vector
    commitments its hotkeys hold now, oldest first, each with `slot` and whether it is the
    `submission`.

    `registered` maps each registered hotkey to its (uid, block at registration); `roots` a hotkey
    swapped into to the first hotkey of its swaps (absent: its own root); `last` a hotkey to the
    block of its newest commitment, registered or not; `current` a hotkey to the commitment it
    holds now. `first(hotkey, newest, floor)` is the block of its first commitment from `floor`
    on, and `read(hotkey, block)` what it committed there; either is None when the chain could not
    say, and that registration is left out until it can."""
    out = []
    for hotkey, (uid, registered_at) in registered.items():
        root = roots.get(hotkey, hotkey)
        floor = max(registered_at, start_block)
        name = slot(uid, registered_at)
        # The hotkeys it was swapped from, unless one holds a registration of its own again.
        lineage = [hotkey] + [
            h for h in last if h != hotkey and h not in registered and roots.get(h, h) == root
        ]
        committed = sorted((last[h], h) for h in lineage if last.get(h, -1) >= floor)
        if not committed:
            continue
        earliest = committed[0][1]
        block = first(earliest, last[earliest], floor)
        if block is None:
            continue
        now = current.get(earliest)
        data = now.data if now is not None and now.block == block else read(earliest, block)
        if data is None:
            continue
        if _vector(data):
            out.append(Commitment(earliest, block, data, slot=name))
        for h in lineage:
            now = current.get(h)
            if now is None or now.block < floor or not _vector(now.data):
                continue
            if (h, now.block) != (earliest, block):
                out.append(dataclasses.replace(now, slot=name, submission=False))
    return sorted(out, key=lambda c: (c.block, c.hotkey))


def _vector(data: str) -> bool:
    try:
        return commitment_.parse(data).lane == commitment_.VECTOR
    except commitment_.CommitmentError:
        return False
