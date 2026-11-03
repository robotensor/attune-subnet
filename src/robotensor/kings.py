"""The four kings kept: the newest crowned models' weights, as the lane's own state.

    <root>/<key>/model.safetensors    one of the four newest crowned models
    <root>/kings.json                 which four, newest first
    <root>/published.json             what the Hub copy last matched

A miner's repository is gated to the validator alone, and nothing asks the miner to keep it up
once its duel ran. So a king's weights are the validator's to serve: the four newest crowned
models - the four the subnet pays - are kept here, and mirrored to one public Hugging Face model
repository (`kings_repo`, `<key>/model.safetensors` beside `kings.json`), so that anyone can take
the weights a crown was won with and a validator on another host can take them up. A crowning
brings the new king in and drops the oldest of five; the Hub copy follows at the next step, and
again at every step until it matches. Nothing here decides who is king: the store's index does.

The duel history (run directories, the fetch cache) is something else and is not touched: that is
history, this is state. The directory is the kings' alone: `keep` removes only a folder holding
nothing but a weights file, and refuses a relative root, so that a misconfiguration can never
point it at anything else. Weights are taken in only when they hash to the sha256 the duel
recorded for them (`sha256` on the model), so a wrong or tampered copy - on the Hub, in a cache -
is never kept, nor served as a king.
"""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .hub import WEIGHTS_FILE, sha256_of

log = logging.getLogger(__name__)

#: How many kings are kept: the subnet's champion pool (`protocol.weights.CHAMPION_SPLIT`).
KEPT = 4
MANIFEST = "kings.json"
PUBLISHED = "published.json"
SCHEMA = 1
#: What a king's folder may hold; a folder holding anything else is nobody's to remove.
FOLDER_FILES = frozenset({WEIGHTS_FILE, f"{WEIGHTS_FILE}.partial"})


class Kings:
    def __init__(
        self, root: str | Path, repo: str = "", *, token: str | None = None, api: Any = None
    ) -> None:
        self.root = Path(root)
        if not self.root.is_absolute() or self.root == Path("/"):
            raise ValueError(f"the kings directory must be an absolute path of its own: {root!r}")
        self.repo = repo
        self.token = token
        self._api = api

    @property
    def api(self) -> Any:
        if self._api is None:
            from huggingface_hub import HfApi

            self._api = HfApi(token=self.token or None)
        return self._api

    def path(self, key: str) -> Path:
        return self.root / key / WEIGHTS_FILE

    def manifest(self) -> dict[str, Any]:
        try:
            doc = json.loads((self.root / MANIFEST).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"schema": SCHEMA, "kings": []}
        if not isinstance(doc, dict) or not isinstance(doc.get("kings"), list):
            return {"schema": SCHEMA, "kings": []}
        return doc

    def kept(self) -> list[str]:
        """The keys kept, newest first, as `kings.json` names them."""
        return [str(k["key"]) for k in self.manifest()["kings"] if isinstance(k, dict)]

    def keep(
        self,
        newest: list[dict[str, Any]],
        source: Callable[[dict[str, Any]], Path | None],
    ) -> list[str]:
        """Bring this directory to the newest `KEPT` of `newest` (models newest first, each
        `{key, repo, revision, sha256?}`): the weights of one not here yet come from
        `source(model)`, a file to copy, or None when it cannot be found now (it is wanted again
        next time); a file that does not hash to the model's `sha256` is refused (and removed
        when it was fetched into this directory); the folder of any other king is removed. The
        keys still without weights."""
        wanted = [m for m in newest[:KEPT] if isinstance(m, dict) and m.get("key")]
        keys = [str(m["key"]) for m in wanted]
        self.root.mkdir(parents=True, exist_ok=True)
        for child in self.root.iterdir():
            if not child.is_dir() or child.name in keys:
                continue
            held = {p.name for p in child.iterdir()}
            if held - FOLDER_FILES:
                log.warning("kings: %s is not a king's folder; left alone", child)
                continue
            shutil.rmtree(child, ignore_errors=True)
            log.info("kings: %s dropped, no longer among the newest %d", child.name, KEPT)
        missing = []
        for model in wanted:
            target = self.path(str(model["key"]))
            if target.is_file():
                continue
            found = source(model)
            if found is None or not Path(found).is_file():
                missing.append(str(model["key"]))
                continue
            expected = model.get("sha256")
            if expected and (got := sha256_of(found)) != expected:
                log.warning(
                    "kings: %s from %s hashes to %s, not the %s its duel recorded; refused",
                    model["key"], found, got[:12], expected[:12],
                )  # fmt: skip
                if self.root.resolve() in Path(found).resolve().parents:
                    Path(found).unlink(missing_ok=True)
                missing.append(str(model["key"]))
                continue
            if Path(found).resolve() == target.resolve():
                continue  # fetched from the Hub copy straight to its place
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.parent / f"{WEIGHTS_FILE}.partial"
            shutil.copyfile(found, partial)
            partial.replace(target)
            log.info("kings: %s kept from %s", model["key"], found)
        doc = {
            "schema": SCHEMA,
            "kings": [
                {
                    "place": place,
                    "key": str(m["key"]),
                    "repo": m.get("repo"),
                    "revision": m.get("revision"),
                    "sha256": m.get("sha256"),
                    "weights": self.path(str(m["key"])).is_file(),
                }
                for place, m in enumerate(wanted, start=1)
            ],
        }
        text = json.dumps(doc, indent=1, sort_keys=True)
        manifest = self.root / MANIFEST
        if not (manifest.is_file() and manifest.read_text(encoding="utf-8") == text):
            manifest.write_text(text, encoding="utf-8")
        return missing

    # -- the Hub ---------------------------------------------------------------------------

    def fetch_from_hub(self, key: str) -> Path | None:
        """`<key>/model.safetensors` from the Hub copy, into this directory; None when the Hub
        does not have it or cannot be asked."""
        if not self.repo:
            return None
        target = self.path(key)
        try:
            got = self.api.hf_hub_download(
                self.repo, f"{key}/{WEIGHTS_FILE}", repo_type="model", local_dir=self.root
            )
        except Exception as exc:  # noqa: BLE001 - the Hub's many ways of saying no, or nothing
            log.info("kings: %s is not on %s: %s", key, self.repo, type(exc).__name__)
            return None
        shutil.rmtree(self.root / ".cache", ignore_errors=True)
        return Path(got) if Path(got).is_file() else target if target.is_file() else None

    def fetch_from_miner(self, repo: str, revision: str) -> Path | None:
        """The weights from the miner's own repository at its commit, while it still answers
        this token; None otherwise. For a champion crowned before the kings were kept."""
        try:
            got = self.api.hf_hub_download(repo, WEIGHTS_FILE, revision=revision, repo_type="model")
        except Exception as exc:  # noqa: BLE001 - the miner's repository is not owed to us
            log.info("kings: %s@%s not fetched from its repository: %s", repo, revision[:12], exc)
            return None
        return Path(got) if Path(got).is_file() else None

    def publish(self) -> bool:
        """The Hub repository brought level with this directory: the kept kings' weights that are
        not there uploaded, the folders of kings no longer kept deleted, `kings.json` with them,
        in one commit. True when level (and nothing asked when it already was); False when the
        Hub could not be asked or refused, which the next step tries again. Never raises."""
        if not self.repo:
            return True
        doc = self.manifest()
        text = json.dumps(doc, indent=1, sort_keys=True)
        kept = [str(k["key"]) for k in doc["kings"] if self.path(str(k["key"])).is_file()]
        if len(kept) != len(doc["kings"]):
            return False  # weights still to find: published once they are all here
        published = self.root / PUBLISHED
        try:
            if published.is_file() and published.read_text(encoding="utf-8") == text:
                return True
        except OSError:
            pass
        try:
            from huggingface_hub import CommitOperationAdd, CommitOperationDelete

            self.api.create_repo(self.repo, repo_type="model", private=False, exist_ok=True)
            there = set(self.api.list_repo_files(self.repo, repo_type="model"))
            operations: list[Any] = [
                CommitOperationAdd(path_in_repo=MANIFEST, path_or_fileobj=text.encode())
            ]
            for key in kept:
                if f"{key}/{WEIGHTS_FILE}" not in there:
                    operations.append(
                        CommitOperationAdd(
                            path_in_repo=f"{key}/{WEIGHTS_FILE}",
                            path_or_fileobj=str(self.path(key)),
                        )
                    )
            for name in sorted(there):
                folder, _, file = name.partition("/")
                if file == WEIGHTS_FILE and folder not in kept:
                    operations.append(CommitOperationDelete(path_in_repo=name))
            self.api.create_commit(
                self.repo,
                operations=operations,
                commit_message=f"Kings: {', '.join(k[:12] for k in kept)}",
                repo_type="model",
            )
        except Exception as exc:  # noqa: BLE001 - the next step tries again
            log.warning("kings: %s could not be brought level: %s", self.repo, exc)
            return False
        published.write_text(text, encoding="utf-8")
        log.info("kings: %s published to %s", kept, self.repo)
        return True
