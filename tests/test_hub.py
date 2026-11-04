"""What intake can tell about a submission from the Hub's metadata, without downloading it."""

import hashlib

import pytest

from robotensor.hub import (
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


def test_upload_gates_the_repository_and_lets_the_validator_in_before_a_byte_goes_up(
    monkeypatch, tmp_path
):
    from types import SimpleNamespace

    import huggingface_hub

    from robotensor import hub

    calls = []

    class Api:
        def __init__(self, token=None):
            calls.append(("token", token))

        def create_repo(self, repo, **kw):
            calls.append(("create", repo, kw.get("private")))

        def update_repo_settings(self, repo, **kw):
            calls.append(("settings", repo, kw.get("gated"), kw.get("private")))

        def grant_access(self, repo, user, **kw):
            calls.append(("grant", repo, user))

        def list_accepted_access_requests(self, repo, **kw):
            calls.append(("accepted", repo))
            return [SimpleNamespace(username=hub.PERMITTED_VALIDATORS[0])]

        def create_commit(self, repo, operations, **kw):
            calls.append(("commit", repo, [o.path_in_repo for o in operations]))
            return SimpleNamespace(oid="c" * 40)

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    (tmp_path / "model.safetensors").write_bytes(b"w")
    sha = hub.upload_weights("me/mine", str(tmp_path / "model.safetensors"), token="t", readme="hi")

    assert sha == "c" * 40
    assert calls == [
        ("token", "t"),
        ("create", "me/mine", True),
        ("settings", "me/mine", "manual", True),
        ("grant", "me/mine", hub.PERMITTED_VALIDATORS[0]),
        ("accepted", "me/mine"),
        ("commit", "me/mine", ["model.safetensors", "README.md"]),
    ]


def _gating_api(monkeypatch, *, grant, accepted):
    """An `HfApi` whose `grant_access` does `grant(user)` and whose accepted list is `accepted`;
    the commits made."""
    from types import SimpleNamespace

    import huggingface_hub

    commits = []

    class Api:
        def __init__(self, token=None):
            pass

        def create_repo(self, repo, **kw):
            pass

        def update_repo_settings(self, repo, **kw):
            pass

        def grant_access(self, repo, user, **kw):
            grant(user)

        def list_accepted_access_requests(self, repo, **kw):
            return [SimpleNamespace(username=u) for u in accepted]

        def create_commit(self, repo, operations, **kw):
            commits.append([o.path_in_repo for o in operations])
            return SimpleNamespace(oid="c" * 40)

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    return commits


def _http(status):
    from types import SimpleNamespace

    from huggingface_hub.errors import HfHubHTTPError

    return HfHubHTTPError(
        f"{status}", response=SimpleNamespace(status_code=status, headers={}, request=None)
    )


def test_a_repository_already_gated_to_the_account_is_not_an_error(monkeypatch, tmp_path):
    """The Hub answers 400 to a grant already made; the accepted list shows the account."""
    from robotensor import hub

    def grant(user):
        raise _http(400)

    commits = _gating_api(monkeypatch, grant=grant, accepted=[hub.PERMITTED_VALIDATORS[0]])
    (tmp_path / "model.safetensors").write_bytes(b"w")
    assert hub.upload_weights("me/mine", str(tmp_path / "model.safetensors")) == "c" * 40
    assert commits == [["model.safetensors"]]


def test_an_account_the_hub_does_not_let_in_stops_the_upload(monkeypatch, tmp_path):
    """A misspelt account: the grant goes through or not, but the accepted list lacks it, and
    nothing is uploaded."""
    import pytest

    from robotensor import hub

    commits = _gating_api(monkeypatch, grant=lambda user: None, accepted=[])
    (tmp_path / "model.safetensors").write_bytes(b"w")
    with pytest.raises(hub.NotLetIn, match="angelecho is not let in"):
        hub.upload_weights("me/mine", str(tmp_path / "model.safetensors"))
    assert commits == []


def test_a_token_that_may_not_grant_stops_the_upload(monkeypatch, tmp_path):
    """A read-only token: the Hub refuses the grant with 403, and nothing is uploaded."""
    import pytest
    from huggingface_hub.errors import HfHubHTTPError

    from robotensor import hub

    def grant(user):
        raise _http(403)

    commits = _gating_api(monkeypatch, grant=grant, accepted=[])
    (tmp_path / "model.safetensors").write_bytes(b"w")
    with pytest.raises(HfHubHTTPError):
        hub.upload_weights("me/mine", str(tmp_path / "model.safetensors"))
    assert commits == []


def test_every_account_given_is_let_in(monkeypatch, tmp_path):
    from robotensor import hub

    granted = []
    commits = _gating_api(monkeypatch, grant=granted.append, accepted=["a", "b"])
    (tmp_path / "model.safetensors").write_bytes(b"w")
    hub.upload_weights("me/mine", str(tmp_path / "model.safetensors"), gate_to=["a", "b"])
    assert granted == ["a", "b"] and commits == [["model.safetensors"]]


def test_reveal_takes_the_repository_out_of_private_and_nothing_else(monkeypatch):
    import huggingface_hub

    from robotensor import hub

    calls = []

    class Api:
        def __init__(self, token=None):
            pass

        def update_repo_settings(self, repo, **kw):
            calls.append((kw.get("private"), kw.get("gated")))

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    hub.reveal("me/mine")
    assert calls == [(False, None)]


@pytest.mark.parametrize(
    ("whoami", "problem"),
    [
        ({"name": "v"}, None),
        ({"name": "v", "auth": {"accessToken": {"fineGrained": {}}}}, "may not read gated"),
        (OSError("no route"), "no route"),
    ],
)
def test_token_problem_says_what_is_wrong_with_the_validator_itself(monkeypatch, whoami, problem):
    import huggingface_hub

    from robotensor import hub

    class Api:
        def __init__(self, token=None):
            pass

        def whoami(self):
            if isinstance(whoami, Exception):
                raise whoami
            return whoami

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    found = hub.token_problem(token="t")
    assert found is None if problem is None else problem in found


