"""What the validator remembers between runs: one JSON file, written atomically.

Everything is keyed by hotkey or by submission key, never by UID (UIDs are recycled). The signed
store the lane engine publishes is the record of what was decided; this file only remembers what
the chain does not keep (a commitment is overwritten by the next one) and what the store does not
name (which hotkey committed a submission).
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

#: An entry's life: waiting for the Hub, queued for a duel, or settled.
PENDING = "pending"
QUEUED = "queued"
SETTLED = ("duelled", "refused", "duplicate", "superseded", "void", "crowned")


class State:
    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        try:
            self.doc: dict[str, Any] = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            self.doc = {}
        self.doc.setdefault("version", 1)
        self.doc.setdefault("lanes", {})
        self.doc.setdefault("weights", {"last_block": 0})

    def lane(self, name: str) -> dict[str, Any]:
        lane = self.doc["lanes"].setdefault(name, {})
        lane.setdefault("entries", {})
        lane.setdefault("by_weights", {})
        lane.setdefault("block_counter", 0)
        return lane

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".state-", dir=self.path.parent)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(self.doc, fh, indent=1, sort_keys=True)
            fh.write("\n")
        os.replace(tmp, self.path)
