"""The Hugging Face Hub, as the subnet sees a submission there.

A submission's weights hash is the sha256 of `model.safetensors`, which the Hub already knows (Git
LFS stores files by their sha256), so the validator can tell a copy from an original without
downloading either. The lane engine downloads and checks the file itself before a duel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

WEIGHTS_FILE = "model.safetensors"


class NotVisible(RuntimeError):
    """The Hub will not show the repository at that commit (missing, private, or a bad sha)."""


class NotASubmission(ValueError):
    """The repository holds what a weights submission may not, or lacks its weights file."""


@dataclass(frozen=True)
class HubSubmission:
    repo: str
    revision: str
    weights_sha256: str
    weights_bytes: int
    files: tuple[str, ...]


def inspect(
    repo: str, revision: str, allowed: frozenset[str], *, api: Any = None, token: str | None = None
) -> HubSubmission:
    """What `repo@revision` holds, from the Hub's metadata alone."""
    from huggingface_hub import HfApi
    from huggingface_hub.errors import (
        GatedRepoError,
        HfHubHTTPError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    api = api or HfApi(token=token)
    try:
        info = api.model_info(repo, revision=revision, files_metadata=True)
    except (RepositoryNotFoundError, RevisionNotFoundError, GatedRepoError) as exc:
        raise NotVisible(f"{repo}@{revision}: {type(exc).__name__}") from None
    except HfHubHTTPError as exc:
        raise NotVisible(f"{repo}@{revision}: {exc}") from None
    if getattr(info, "sha", None) != revision:
        raise NotVisible(f"{repo}@{revision}: the Hub resolved it to {getattr(info, 'sha', None)}")
    files = tuple(sorted(s.rfilename for s in info.siblings or []))
    extra = [f for f in files if f not in allowed]
    if extra:
        raise NotASubmission(f"holds files a weights submission may not: {', '.join(extra[:5])}")
    weights = next((s for s in info.siblings or [] if s.rfilename == WEIGHTS_FILE), None)
    lfs = getattr(weights, "lfs", None) if weights is not None else None
    sha = getattr(lfs, "sha256", None) if lfs is not None else None
    if weights is None or not sha:
        raise NotASubmission(f"holds no {WEIGHTS_FILE} stored in LFS")
    return HubSubmission(
        repo=repo,
        revision=revision,
        weights_sha256=str(sha),
        weights_bytes=int(getattr(weights, "size", 0) or 0),
        files=files,
    )


def upload_weights(
    repo: str,
    weights: str,
    *,
    token: str | None = None,
    private: bool = False,
    readme: str | None = None,
    message: str = "Upload model.safetensors",
) -> str:
    """Upload `weights` as the repository's `model.safetensors` (and a README if given), creating
    the repository if needed; the commit sha of the upload."""
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi(token=token)
    api.create_repo(repo, repo_type="model", private=private, exist_ok=True)
    operations = [CommitOperationAdd(path_in_repo=WEIGHTS_FILE, path_or_fileobj=weights)]
    if readme is not None:
        operations.append(
            CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=readme.encode())
        )
    info = api.create_commit(repo, operations=operations, commit_message=message, repo_type="model")
    return str(info.oid)
