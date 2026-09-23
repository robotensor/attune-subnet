"""The Hugging Face Hub, as the subnet sees a submission there.

Intake decides from the Hub's metadata alone and never downloads a model. Two competitions ask
different things of a repository, so what each may hold and what makes two submissions the same
model is a `Shape` the lane brings.

**Why metadata is enough.** Git LFS stores a file under its own sha256, and the Hub reports it, so
the validator can tell a copy from an original without moving a byte. Vector's submission is one
`model.safetensors` and its content key is that file's sha256 - the value this subnet has always
used. Horizon's is tens of gigabytes of shards beside a few kilobyte files; its key is the digest
of the listing `sha256sum` would print for the files it covers, which is what its runtime computes
on disk. The kilobyte files are not in LFS, so those - and only those - are fetched.

A shard that was pushed as a plain git blob rather than through LFS has no sha256 on the Hub and
is refused, with the remedy: track `*.safetensors` in `.gitattributes`.
"""

from __future__ import annotations

import fnmatch
import hashlib
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

WEIGHTS_FILE = "model.safetensors"
#: A hashed file the Hub cannot give a sha256 for is fetched when it is no larger than this.
FETCH_MAX_BYTES = 1 << 20


class NotVisible(RuntimeError):
    """The Hub will not show the repository at that commit (missing, private, or a bad sha)."""


class NotASubmission(ValueError):
    """The repository holds what this competition's submission may not, or lacks what it must."""


@dataclass(frozen=True)
class HubFile:
    """One file of a repository, as the Hub describes it without downloading anything."""

    name: str
    size: int
    #: Its own sha256, when Git LFS stores it; `None` for a plain git blob, whose id is a sha1.
    lfs_sha256: str | None = None


@dataclass(frozen=True)
class Shape:
    """What a competition's submission may hold, and what makes two of them the same model.

    `allowed`, `required` and `hashed` are exact names or `fnmatch` globs. `key` is how the
    content key is built from the hashed files:

    - `"file"`: the one hashed file's own sha256. Vector's, and the value it has always had.
    - `"listing"`: `sha256` of `"<sha256>  <name>\\n"` for each hashed file, ordered by the bytes
      of its path - the digest `sha256sum <files> | sha256sum` gives, which is what Horizon's
      runtime computes over the same files on disk.
    """

    #: Files a repository may hold; one matching none of these is refused.
    allowed: tuple[str, ...]
    #: Patterns at least one file must match, or the repository is not a submission at all.
    required: tuple[str, ...]
    #: The files the content key covers.
    hashed: tuple[str, ...]
    key: str = "file"
    #: The repository's size cap, over the files that count; 0 for none.
    max_bytes: int = 0
    #: Patterns whose files must be in LFS, so their sha256 is known without downloading them.
    lfs_only: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.key not in ("file", "listing"):
            raise ValueError(f"{self.key!r} is not a content key rule (file, listing)")

    def matches(self, name: str, patterns: Iterable[str]) -> bool:
        return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


#: Vector's submission: the weights of one architecture, and nothing else that runs.
def vector_shape(allowed: Iterable[str], max_bytes: int = 0) -> Shape:
    return Shape(
        allowed=tuple(sorted(allowed)),
        required=(WEIGHTS_FILE,),
        hashed=(WEIGHTS_FILE,),
        key="file",
        max_bytes=max_bytes,
        lfs_only=(WEIGHTS_FILE,),
    )


@dataclass(frozen=True)
class HubSubmission:
    repo: str
    revision: str
    #: What two submissions of the same bytes share, whatever the competition.
    content_key: str
    #: The bytes the hashed files come to.
    weights_bytes: int
    files: tuple[str, ...] = ()
    #: Every file with what the Hub says about it, for a lane that wants more than the names.
    manifest: tuple[HubFile, ...] = field(default_factory=tuple)

    @property
    def weights_sha256(self) -> str:
        """What the content key was called when one file was all a submission could be."""
        return self.content_key


def content_key(
    files: Sequence[HubFile],
    shape: Shape,
    *,
    fetch: Callable[[str], bytes] | None = None,
) -> tuple[str, int]:
    """The submission's content key and the bytes it covers, from the Hub's metadata.

    `fetch` reads one small file whose sha256 the Hub does not know (a config, a statistics file);
    it is never called for anything matching `lfs_only`, and never for a large file.
    """
    covered = sorted(
        (f for f in files if shape.matches(f.name, shape.hashed)), key=lambda f: f.name.encode()
    )
    if not covered:
        raise NotASubmission(f"holds none of {', '.join(shape.hashed)}")
    digests = []
    for one in covered:
        if one.lfs_sha256:
            digests.append(one.lfs_sha256)
            continue
        if shape.matches(one.name, shape.lfs_only):
            raise NotASubmission(
                f"{one.name} is not stored in Git LFS, so the Hub cannot say what is in it: "
                "track it in .gitattributes and push it again"
            )
        if fetch is None or one.size > FETCH_MAX_BYTES:
            raise NotASubmission(
                f"{one.name} is {one.size} bytes and not in Git LFS; intake reads no more than "
                f"{FETCH_MAX_BYTES} bytes of a submission"
            )
        digests.append(hashlib.sha256(fetch(one.name)).hexdigest())
    size = sum(f.size for f in covered)
    if shape.key == "file":
        if len(covered) != 1:
            raise NotASubmission(
                f"holds {len(covered)} of {', '.join(shape.hashed)}; this competition takes one"
            )
        return digests[0], size
    listing = "".join(f"{d}  {f.name}\n" for d, f in zip(digests, covered, strict=True))
    return hashlib.sha256(listing.encode()).hexdigest(), size


def manifest_of(info: Any) -> tuple[HubFile, ...]:
    """What the Hub said the repository holds, as plain files."""
    out = []
    for sibling in info.siblings or []:
        lfs = getattr(sibling, "lfs", None)
        out.append(
            HubFile(
                name=str(sibling.rfilename),
                size=int(getattr(sibling, "size", 0) or 0),
                lfs_sha256=str(getattr(lfs, "sha256", "")) or None if lfs is not None else None,
            )
        )
    return tuple(sorted(out, key=lambda f: f.name))


def inspect(
    repo: str,
    revision: str,
    shape: Shape,
    *,
    api: Any = None,
    token: str | None = None,
    fetch: Callable[[str], bytes] | None = None,
) -> HubSubmission:
    """What `repo@revision` holds, from the Hub's metadata alone.

    `fetch` reads one small file the Hub knows no sha256 for; the default downloads it. Nothing
    large is ever fetched, and nothing at all for a competition whose key covers one LFS file.
    """
    from huggingface_hub import HfApi
    from huggingface_hub.errors import (
        GatedRepoError,
        HfHubHTTPError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    api = api or HfApi(token=token or None)  # an empty token is no token
    try:
        info = api.model_info(repo, revision=revision, files_metadata=True)
    except (RepositoryNotFoundError, RevisionNotFoundError, GatedRepoError) as exc:
        raise NotVisible(f"{repo}@{revision}: {type(exc).__name__}") from None
    except HfHubHTTPError as exc:
        raise NotVisible(f"{repo}@{revision}: {exc}") from None
    if getattr(info, "sha", None) != revision:
        raise NotVisible(f"{repo}@{revision}: the Hub resolved it to {getattr(info, 'sha', None)}")

    manifest = manifest_of(info)
    names = tuple(f.name for f in manifest)
    extra = [name for name in names if not shape.matches(name, shape.allowed)]
    if extra:
        raise NotASubmission(
            f"holds files this competition's submission may not: {', '.join(extra[:5])}"
        )
    for pattern in shape.required:
        if not any(shape.matches(name, [pattern]) for name in names):
            raise NotASubmission(f"holds no {pattern}")

    def download(name: str) -> bytes:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo, name, revision=revision, token=token or None)
        with open(path, "rb") as handle:
            return handle.read()

    key, size = content_key(manifest, shape, fetch=fetch or download)
    if shape.max_bytes and size > shape.max_bytes:
        raise NotASubmission(
            f"its weights are {size} bytes, over this competition's {shape.max_bytes}"
        )
    return HubSubmission(
        repo=repo,
        revision=revision,
        content_key=key,
        weights_bytes=size,
        files=names,
        manifest=manifest,
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

    api = HfApi(token=token or None)
    api.create_repo(repo, repo_type="model", private=private, exist_ok=True)
    operations = [CommitOperationAdd(path_in_repo=WEIGHTS_FILE, path_or_fileobj=weights)]
    if readme is not None:
        operations.append(
            CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=readme.encode())
        )
    info = api.create_commit(repo, operations=operations, commit_message=message, repo_type="model")
    return str(info.oid)
