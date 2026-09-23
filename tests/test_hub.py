"""What intake can tell about a submission from the Hub's metadata, without downloading it."""

import hashlib

import pytest

from robotensor_subnet.hub import (
    FETCH_MAX_BYTES,
    HubFile,
    NotASubmission,
    Shape,
    content_key,
    vector_shape,
)

#: Horizon's submission, as its runtime lays one out: shards and an index in `transformer/`, the
#: statistics and the knobs file beside them.
HORIZON = Shape(
    allowed=("transformer/*", "norm_stats/*.json", "zerowam.yaml", "README.md", ".gitattributes"),
    required=("transformer/*.safetensors",),
    hashed=(
        "transformer/*.safetensors",
        "transformer/config.json",
        "norm_stats/*.json",
        "zerowam.yaml",
    ),
    key="listing",
    lfs_only=("transformer/*.safetensors",),
)


def lfs(name, data: bytes):
    return HubFile(name, len(data), hashlib.sha256(data).hexdigest())


def blob(name, data: bytes):
    return HubFile(name, len(data))


def test_one_weights_file_keys_on_its_own_sha256(tmp_path):
    """Vector's key, and the one this subnet has always used: change it and every submission key,
    every duel id and every entry in the store changes with it."""
    weights = lfs("model.safetensors", b"tensors")
    shape = vector_shape(["model.safetensors", "README.md"])

    key, size = content_key([weights, blob("README.md", b"hello")], shape)

    assert key == hashlib.sha256(b"tensors").hexdigest() and size == len(b"tensors")


def test_the_listing_key_is_what_the_runtime_computes_on_disk(tmp_path):
    """Horizon's 50 GB submission is deduplicated before a byte moves, so the key intake works
    out from the Hub has to be the one its runtime computes from the files themselves."""
    weights_module = pytest.importorskip("horizon_runtime_zerowam.weights")
    files = {
        "transformer/diffusion_pytorch_model-00001-of-00002.safetensors": b"shard one",
        "transformer/diffusion_pytorch_model-00002-of-00002.safetensors": b"shard two",
        "transformer/diffusion_pytorch_model.safetensors.index.json": b'{"weight_map": {}}',
        "transformer/config.json": b'{"_class_name": "x"}',
        "norm_stats/robotwin.json": b'{"mean": 0}',
        "zerowam.yaml": b"knobs: {}\n",
    }
    for name, data in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    covered = [n for n in files if not n.endswith("index.json")]

    on_disk = weights_module.content_sha256(tmp_path, covered)
    from_hub, size = content_key(
        [
            lfs(name, data) if name.endswith(".safetensors") else blob(name, data)
            for name, data in files.items()
            if name in covered
        ],
        HORIZON,
        # What intake does with the kilobyte files the Hub has no sha256 for.
        fetch=lambda name: (tmp_path / name).read_bytes(),
    )

    assert from_hub == on_disk
    assert size == sum(len(files[name]) for name in covered)


def test_a_shard_pushed_as_a_plain_blob_is_refused_with_the_remedy():
    """Without LFS the Hub knows no sha256 for it, and intake will not download gigabytes to
    find one."""
    files = [blob("transformer/diffusion_pytorch_model.safetensors", b"shard")]

    with pytest.raises(NotASubmission, match="track it in .gitattributes"):
        content_key(files, HORIZON)


def test_a_hashed_file_too_large_to_read_is_refused_rather_than_fetched():
    big = HubFile("zerowam.yaml", FETCH_MAX_BYTES + 1)
    files = [lfs("transformer/diffusion_pytorch_model.safetensors", b"shard"), big]

    with pytest.raises(NotASubmission, match="no more than"):
        content_key(files, HORIZON, fetch=lambda name: b"")


def test_a_small_file_the_hub_cannot_hash_is_read_once():
    read = []

    def fetch(name):
        read.append(name)
        return b"knobs: {}\n"

    files = [
        lfs("transformer/diffusion_pytorch_model.safetensors", b"shard"),
        blob("zerowam.yaml", b"knobs: {}\n"),
    ]
    key, _ = content_key(files, HORIZON, fetch=fetch)

    assert read == ["zerowam.yaml"] and len(key) == 64


def test_a_competition_that_takes_one_file_refuses_two():
    shape = vector_shape(["model.safetensors", "other.safetensors"])
    shape = Shape(
        allowed=shape.allowed,
        required=shape.required,
        hashed=("*.safetensors",),
        key="file",
        lfs_only=("*.safetensors",),
    )

    with pytest.raises(NotASubmission, match="takes one"):
        content_key([lfs("model.safetensors", b"a"), lfs("other.safetensors", b"b")], shape)
