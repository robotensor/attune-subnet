"""What the validator remembers between runs: one JSON document per competition, in a directory.

Everything is keyed by hotkey or by submission key, never by UID (UIDs are recycled). The signed
store a lane engine publishes is the record of what was decided; these documents only remember
what the chain does not keep (a commitment is overwritten by the next one) and what the store does
not name (which hotkey committed a submission).

**One document per lane, one writer each.** Two competitions run in processes of their own, and a
single file would have them overwriting each other's work between a read and a write. A lane's
document is written only by the process running that lane; writes take its lock and replace the
file atomically, so a reader - `status`, the weights thread - always sees a whole document and
never blocks.

A directory left over from the single-file days is migrated on first use: `<root>.json` is split
into `<root>/<lane>.json` and kept, renamed, in case a human wants to look at it.
"""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

#: An entry's life: waiting for the Hub, queued for a duel, or settled.
PENDING = "pending"
QUEUED = "queued"
SETTLED = ("duelled", "refused", "duplicate", "superseded", "void", "crowned")

#: The document every lane starts from.
EMPTY: dict[str, Any] = {"version": 1, "entries": {}, "by_weights": {}, "block_counter": 0}


class State:
    """The lanes' documents under one directory, loaded as they are asked for."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)
        if self.root.suffix == ".json":  # a path that named the single file it used to be
            self.root = self.root.with_suffix("")
        self._docs: dict[str, dict[str, Any]] = {}
        self._migrate()

    def path(self, name: str) -> Path:
        return self.root / f"{name}.json"

    def lane(self, name: str) -> dict[str, Any]:
        """The lane's document, read from disk the first time it is asked for."""
        doc = self._docs.get(name)
        if doc is None:
            doc = self._read(self.path(name))
            self._docs[name] = doc
        return doc

    def save(self) -> None:
        """Write every document this object has touched, each atomically under its own lock."""
        for name, doc in self._docs.items():
            path = self.path(name)
            with _locked(path):
                _write(path, doc)

    # -- the read-modify-write a worker owns ---------------------------------------------------

    @contextmanager
    def writing(self, name: str) -> Iterator[dict[str, Any]]:
        """The lane's document, held under its lock for as long as the block runs and written at
        the end. Use it where a read and the write that follows must not be split by anyone
        else - taking a block number, settling an entry.

        It re-reads the document from disk, so anything changed in memory and not saved is gone:
        that is the point, because what another process wrote must not be written over by a copy
        this one has been holding.
        """
        path = self.path(name)
        with _locked(path):
            doc = self._read(path)
            self._docs[name] = doc
            yield doc
            _write(path, doc)

    # -- internals -----------------------------------------------------------------------------

    @staticmethod
    def _read(path: Path) -> dict[str, Any]:
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            doc = {}
        for key, value in EMPTY.items():
            doc.setdefault(key, value if not isinstance(value, dict) else dict(value))
        return doc

    def _migrate(self) -> None:
        """Split the single document this used to be, once, and keep it under another name."""
        legacy = self.root.with_suffix(".json")
        if self.root.exists() or not legacy.is_file():
            return
        try:
            doc = json.loads(legacy.read_text(encoding="utf-8"))
        except ValueError:
            return
        lanes = doc.get("lanes")
        if not isinstance(lanes, dict):
            return
        for name, lane in lanes.items():
            if isinstance(lane, dict):
                path = self.path(name)
                merged = dict(EMPTY) | lane
                _write(path, merged)
        legacy.replace(legacy.with_name(f"{legacy.name}.migrated"))


def _write(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    """An advisory exclusive lock on the lane's document, beside it. Readers take none: a document
    is replaced atomically, so what they open is always a whole one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_name(f"{path.name}.lock")
    with open(lock, "w", encoding="utf-8") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)
