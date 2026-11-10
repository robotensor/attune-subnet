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
again at every step until it matches. Nothing here decides who is king: the store's index does,
which a new pod takes up from the results mirror before anything here runs.

The lane holds every duel until the reigning king is resolved (`resolved`: its weights here and on
the Hub), and writes here only once the results mirror holds the record that crowned it, so this
copy can never name a king the results do not.

Once per process, what is held is verified rather than trusted: every kept file is hashed, and
what the Hub holds is compared with what the index says (the kings, and the sha256 of their
weights), so a wrong or missing copy on either side is put right. An index naming no king at all
says nothing, so it never empties this directory or the Hub copy.

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
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .hub import WEIGHTS_FILE, NotVisible, account, manifest_of, sha256_of

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
        self,
        root: str | Path,
        repo: str = "",
        *,
        owner: str = "",
        token: str | None = None,
        api: Any = None,
    ) -> None:
        self.root = Path(root)
        if not self.root.is_absolute() or self.root == Path("/"):
            raise ValueError(f"the kings directory must be an absolute path of its own: {root!r}")
        if "/" in repo:
            raise ValueError(f"the kings repository is a name, not owner/name: {repo!r}")
        self.name = repo
        self.owner = owner
        self.token = token
        self._api = api
        self._repo: str | None = None
        #: Whether this process has hashed the kept files, and compared the Hub copy.
        self._hashed = False
        self._compared = False
        #: The kings the Hub copy was last seen, or just made, to hold with the right weights.
        self._on_hub: set[str] = set()

    @property
    def api(self) -> Any:
        if self._api is None:
            from huggingface_hub import HfApi

            self._api = HfApi(token=self.token or None)
        return self._api

    @property
    def repo(self) -> str:
        """The repository's id: its name under `owner`, else the account the token belongs to,
        asked once."""
        if self._repo is None:
            self._repo = f"{self.owner or account(api=self.api)}/{self.name}"
        return self._repo

    def path(self, key: str) -> Path:
        return self.root / key / WEIGHTS_FILE

    def resolved(self, key: str) -> bool:
        """Whether king `key` is safe: its weights here and, unless there is no Hub copy, there."""
        return self.path(key).is_file() and (not self.name or key in self._on_hub)

    def named_on_hub(self) -> list[str]:
        """The kings the Hub copy's `kings.json` names; none when it has no such file. Raises when
        the Hub cannot be asked: not knowing is not the same as none."""
        if not self.name:
            return []
        from huggingface_hub.errors import (
            RemoteEntryNotFoundError,
            RepositoryNotFoundError,
            RevisionNotFoundError,
        )

        with tempfile.TemporaryDirectory() as cache:
            try:
                got = self.api.hf_hub_download(
                    self.repo, MANIFEST, repo_type="model", cache_dir=cache
                )
            except (RemoteEntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError):
                return []
            doc = json.loads(Path(got).read_text(encoding="utf-8"))
        return [str(k["key"]) for k in doc.get("kings", []) if isinstance(k, dict)]

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
        if not wanted:
            return []  # no king named says nothing about the ones held
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
            expected = model.get("sha256")
            if target.is_file():
                if self._hashed or not expected or sha256_of(target) == expected:
                    continue
                log.warning("kings: %s does not hash to %s; fetched again", target, expected[:12])
                target.unlink()
            found = source(model)
            if found is None or not Path(found).is_file():
                missing.append(str(model["key"]))
                continue
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
        self._hashed = True
        return missing

    # -- the Hub ---------------------------------------------------------------------------

    def fetch_from_hub(self, key: str) -> Path | None:
        """`<key>/model.safetensors` from the Hub copy, into this directory; None when the Hub
        does not have it or cannot be asked."""
        if not self.name:
            return None
        target = self.path(key)
        try:
            got = self.api.hf_hub_download(
                self.repo, f"{key}/{WEIGHTS_FILE}", repo_type="model", local_dir=self.root
            )
        except Exception as exc:  # noqa: BLE001 - the Hub's many ways of saying no, or nothing
            log.info("kings: %s is not on %s: %s", key, self.name, type(exc).__name__)
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
        in one commit. A king whose weights are not here yet does not hold the others back, and
        its folder on the Hub, if any, stays. True when level with all of them (and nothing asked
        when it already was); False when one is still to come, or the Hub could not be asked or
        refused, which the next step tries again. Never raises."""
        if not self.name:
            return True
        doc = self.manifest()
        text = json.dumps(doc, indent=1, sort_keys=True)
        named = {str(k["key"]) for k in doc["kings"]}
        kept = [str(k["key"]) for k in doc["kings"] if self.path(str(k["key"])).is_file()]
        complete = len(kept) == len(doc["kings"])
        if not doc["kings"]:
            return True  # never empties the Hub copy
        try:
            repo = self.repo
        except NotVisible as exc:  # the next step tries again
            log.warning("kings: %s has no account to go under: %s", self.name, exc)
            return False
        # Whose repository too: a token of another account has it all to publish again.
        level = json.dumps({"repo": repo, **doc}, indent=1, sort_keys=True)
        published = self.root / PUBLISHED
        try:
            if (
                self._compared
                and published.is_file()
                and published.read_text(encoding="utf-8") == level
            ):
                return complete
        except OSError:
            pass
        try:
            from huggingface_hub import CommitOperationAdd, CommitOperationDelete

            self.api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
            there = {f.name: f for f in manifest_of(self.api.model_info(repo, files_metadata=True))}
            operations: list[Any] = [
                CommitOperationAdd(path_in_repo=MANIFEST, path_or_fileobj=text.encode())
            ]
            on_hub = set()
            for king in doc["kings"]:
                key, expected = str(king["key"]), king.get("sha256")
                name = f"{key}/{WEIGHTS_FILE}"
                file = there.get(name)
                wrong = file is not None and expected and file.lfs_sha256 != expected
                if wrong:
                    log.warning("kings: %s on %s is not the crowned weights", name, repo)
                if self.path(key).is_file() and (file is None or wrong):
                    operations.append(
                        CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(self.path(key)))
                    )
                    on_hub.add(key)
                elif wrong:
                    operations.append(CommitOperationDelete(path_in_repo=name))
                elif file is not None:
                    on_hub.add(key)
            for name in sorted(there):
                folder, _, file = name.partition("/")
                if file == WEIGHTS_FILE and folder not in named:
                    operations.append(CommitOperationDelete(path_in_repo=name))
            self.api.create_commit(
                repo,
                operations=operations,
                commit_message=f"Kings: {', '.join(k[:12] for k in kept)}",
                repo_type="model",
            )
        except Exception as exc:  # noqa: BLE001 - the next step tries again
            log.warning("kings: %s could not be brought level: %s", repo, exc)
            return False
        self._on_hub = on_hub
        published.write_text(level, encoding="utf-8")
        self._compared = True
        log.info("kings: %s published to %s", kept, repo)
        return complete
