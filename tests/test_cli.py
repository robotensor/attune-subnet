"""One command for both halves of the subnet: what it routes, and which competition it means."""

import pytest

from robotensor.cli import (
    COMPETITION_ENV,
    LEGACY_COMPETITION_ENV,
    CompetitionError,
    build_parser,
    competition,
)


def test_the_flag_wins_over_the_environment_and_the_config():
    chosen = competition("vector", {"vector": 1, "horizon": 2}, {COMPETITION_ENV: "horizon"})

    assert chosen == "vector"


def test_the_environment_saves_saying_it_on_every_command():
    assert competition(None, {"vector": 1, "horizon": 2}, {COMPETITION_ENV: "horizon"}) == "horizon"


def test_the_environment_variable_s_old_name_still_counts():
    old = {LEGACY_COMPETITION_ENV: "horizon"}
    assert competition(None, {"vector": 1, "horizon": 2}, old) == "horizon"
    both = {COMPETITION_ENV: "vector", LEGACY_COMPETITION_ENV: "horizon"}
    assert competition(None, {"vector": 1, "horizon": 2}, both) == "vector"


def test_one_competition_enabled_needs_no_saying():
    assert competition(None, {"vector": 1}, {}) == "vector"


def test_two_enabled_and_nothing_said_is_refused_by_name():
    """A miner who meant the other one should not find out from a refused submission."""
    with pytest.raises(CompetitionError, match="vector, horizon"):
        competition(None, {"vector": 1, "horizon": 2}, {})


def test_a_competition_this_validator_does_not_run_is_refused():
    with pytest.raises(CompetitionError, match="horizon is not"):
        competition("horizon", {"vector": 1}, {})


def test_the_command_routes_both_halves():
    parser = build_parser()

    miner = parser.parse_args(["miner", "check", "--dir", "submission/"])
    validator = parser.parse_args(["validator", "--config", "c.toml", "status"])

    assert miner.part == "miner" and miner.command == "check"
    assert validator.part == "validator" and validator.command == "status"
    assert validator.config == "c.toml"


def test_the_version_is_the_packages():
    from robotensor import __version__

    with pytest.raises(SystemExit) as exit_:
        build_parser().parse_args(["--version"])

    assert exit_.value.code == 0 and __version__


def test_init_writes_a_config_that_loads(tmp_path):
    """`init` writes a config holding only what differs between hosts, and the directories its data
    is kept in."""
    from robotensor.config import load
    from robotensor.init import validator

    path = validator(tmp_path / "node")

    cfg = load(path)
    assert cfg.root == tmp_path / "node"
    assert cfg.vector.store == tmp_path / "node" / "var" / "vector" / "store"
    assert cfg.shares == {"vector": 0.30}
    assert (tmp_path / "node" / "var" / "state").is_dir()
    assert "wallet" in path.read_text() and "policy_python" in path.read_text()
    assert cfg.vector.start_block == 0, "a line to fill in, as the wallet and the netuid are"
    assert "HF_TOKEN=" in (tmp_path / "node" / ".env").read_text(), "the token's place, beside"


def test_init_will_not_write_over_a_config(tmp_path):
    from robotensor.init import validator

    validator(tmp_path / "node")

    with pytest.raises(FileExistsError):
        validator(tmp_path / "node")


def test_submit_is_one_command_gated_to_the_validator():
    parser = build_parser()

    args = parser.parse_args(
        [
            "miner",
            "submit",
            "--dir",
            "s/",
            "--repo",
            "me/mine",
            "--netuid",
            "2",
            "--wallet.name",
            "w",
            "--wallet.hotkey",
            "h",
        ]
    )

    assert args.command == "submit"


def test_a_miner_cannot_choose_who_is_let_in():
    """The permitted list is all there is: no flag adds an account or leaves the gate off."""
    import argparse

    from robotensor import miner

    parser = argparse.ArgumentParser()
    miner.add_subcommands(parser.add_subparsers(dest="command"))
    for flags in (["--validator-user", "a"], ["--no-gate"]):
        with pytest.raises(SystemExit):
            parser.parse_args(["upload", "--dir", "s/", "--repo", "me/mine", *flags])


def test_a_commit_that_fails_says_it_may_have_landed_and_what_to_run(monkeypatch, capsys):
    from argparse import Namespace
    from types import SimpleNamespace

    from robotensor import chain, miner

    class Chain:
        def __init__(self, network, netuid):
            pass

        def commit(self, wallet, data):
            raise TimeoutError("the socket went")

        def close(self):
            pass

    monkeypatch.setattr(chain, "Chain", Chain)
    wallet = SimpleNamespace(hotkey=SimpleNamespace(ss58_address="hk"))
    monkeypatch.setattr(chain, "wallet", lambda *a: wallet)
    args = Namespace(
        repo="me/mine", wallet_name="w", wallet_hotkey="h", wallet_path=None, network="t", netuid=1
    )
    with pytest.raises(TimeoutError):
        miner._commit_on_chain(args, "data")
    assert "attune miner reveal --repo me/mine" in capsys.readouterr().err


def test_status_reads_the_commitment_back_from_the_hub(monkeypatch, capsys):
    """`miner status` looks the commitment up with the real Hub check, so a submission that is
    fine reports its digest matching, not an error."""
    import hashlib
    import json
    from types import SimpleNamespace

    from robotensor import chain, hub, miner
    from robotensor.protocol import commitment

    sha = "a" * 40
    weights = hashlib.sha256(b"tensors").hexdigest()

    class FakeChain:
        def __init__(self, network, netuid):
            pass

        def commitments(self):
            data = commitment.encode("me/mine", sha, weights)
            return [chain.Commitment("hk", 10, data), chain.Commitment("other", 11, data)]

        def close(self):
            pass

    hub_api = SimpleNamespace(
        model_info=lambda repo, revision, files_metadata: SimpleNamespace(
            sha=revision,
            siblings=[
                SimpleNamespace(
                    rfilename="model.safetensors", size=7, lfs=SimpleNamespace(sha256=weights)
                )
            ],
        )
    )
    real = hub.inspect
    monkeypatch.setattr(chain, "Chain", FakeChain)
    monkeypatch.setattr(hub, "inspect", lambda *a, **kw: real(*a, **{**kw, "api": hub_api}))

    miner.cmd_status(SimpleNamespace(network="test", netuid=2, hotkey="hk"))

    (entry,) = json.loads(capsys.readouterr().out)
    assert "problem" not in entry, entry["problem"]
    assert entry["digest_matches"] and entry["weights_sha256"] == weights


# -- private until the chain has the commitment ----------------------------------------------------


def _committing(monkeypatch, order):
    from types import SimpleNamespace

    from robotensor import chain, hub

    class Chain:
        def __init__(self, *a):
            pass

        def commit(self, wallet, data):
            order.append("commit")

        def block(self):
            return 5

        def close(self):
            pass

    monkeypatch.setattr(chain, "Chain", Chain)
    monkeypatch.setattr(
        chain, "wallet", lambda *a: SimpleNamespace(hotkey=SimpleNamespace(ss58_address="hk"))
    )
    monkeypatch.setattr(hub, "reveal", lambda repo, token=None: order.append(("reveal", repo)))


def _commit_args(parser_args=()):
    import argparse

    from robotensor import miner

    parser = argparse.ArgumentParser()
    miner.add_subcommands(parser.add_subparsers(dest="command"))
    return parser.parse_args(
        ["commit", "--repo", "me/mine", "--revision", "a" * 40, "--digest", "b" * 64, "--netuid", "1",
         "--wallet.name", "w", "--wallet.hotkey", "h", *parser_args]
    )  # fmt: skip


def test_commit_reveals_the_repository_only_after_the_chain_has_it(monkeypatch, capsys):
    import json

    from robotensor import miner

    order = []
    _committing(monkeypatch, order)
    assert miner.cmd_commit(_commit_args()) == 0
    assert order == ["commit", ("reveal", "me/mine")]
    assert json.loads(capsys.readouterr().out)["revealed"] is True


def test_a_commit_that_failed_reveals_nothing(monkeypatch):
    from robotensor import chain, miner

    order = []
    _committing(monkeypatch, order)

    class Refused(chain.Chain):
        def commit(self, wallet, data):
            raise RuntimeError("no")

    monkeypatch.setattr(chain, "Chain", Refused)
    with pytest.raises(RuntimeError):
        miner.cmd_commit(_commit_args())
    assert order == []


def test_a_reveal_that_keeps_failing_is_said_with_what_to_run(monkeypatch, capsys):
    import json

    from robotensor import hub, miner

    tries = []

    def refused(repo, token=None):
        tries.append(repo)
        raise OSError("no network")

    _committing(monkeypatch, [])
    monkeypatch.setattr(hub, "reveal", refused)
    monkeypatch.setattr(miner, "REVEAL_GAP_S", 0.0)
    assert miner.cmd_commit(_commit_args()) == 1
    captured = capsys.readouterr()
    assert len(tries) == miner.REVEAL_TRIES
    assert "attune miner reveal --repo me/mine" in captured.err
    assert json.loads(captured.out)["revealed"] is False


def test_a_reveal_that_fails_once_is_tried_again(monkeypatch):
    from robotensor import hub, miner

    tries = []

    def flaky(repo, token=None):
        tries.append(repo)
        if len(tries) == 1:
            raise OSError("no network")

    _committing(monkeypatch, [])
    monkeypatch.setattr(hub, "reveal", flaky)
    monkeypatch.setattr(miner, "REVEAL_GAP_S", 0.0)
    assert miner.cmd_commit(_commit_args()) == 0 and len(tries) == 2


def test_init_s_next_steps_give_commit_everything_it_requires():
    from robotensor.init import MINER_NEXT

    assert "--digest" in MINER_NEXT


# -- the Hub account a validator starts on ---------------------------------------------------------


def _validator_config(tmp_path, **vector):
    from robotensor.config import load
    from robotensor.init import validator

    path = validator(tmp_path / "node")
    text = path.read_text()
    for key, value in vector.items():
        text = text.replace(f'{key} = ""', f'{key} = "{value}"')
    path.write_text(text)
    return load(path)


@pytest.mark.parametrize(
    ("token", "account", "wanted"),
    [
        (None, "angelecho", "HF_TOKEN is not set"),
        ("t", "someone", "belongs to someone, which is not a permitted account (angelecho)"),
        ("t", "angelecho", None),
    ],
)
def test_only_a_permitted_account_starts(monkeypatch, token, account, wanted):
    from robotensor import hub, validator

    monkeypatch.setattr(validator, "hub_token", lambda: token)
    monkeypatch.setattr(hub, "account", lambda **kw: account)
    problem = validator.hub_account_problem()
    assert problem is None if wanted is None else wanted in problem


def test_the_start_check_holds_on_a_local_chain_too(tmp_path, monkeypatch, capsys):
    from argparse import Namespace

    from robotensor import hub, validator
    from robotensor.config import load

    (tmp_path / "x.toml").write_text(
        'network = "ws://127.0.0.1:9944"\nnetuid = 1\n[vector]\npolicy_python = "p"\n'
        'simulator_python = "s"\nsimulator_root = "/r"\n'
    )
    monkeypatch.setattr(validator, "hub_token", lambda: "t")
    monkeypatch.setattr(hub, "account", lambda **kw: "someone")
    args = Namespace(no_weights=True, once=True, dry_run=False, challenger=None)
    cfg = load(tmp_path / "x.toml")
    for command in (
        validator.cmd_run,
        validator.cmd_duel,
        validator.cmd_intake,
        validator.cmd_weights,
    ):
        assert command(args, cfg) == 1
        assert "not a permitted account" in capsys.readouterr().err


def test_a_worker_started_by_hand_is_held_to_the_same_check(tmp_path, monkeypatch, capsys):
    from robotensor import hub, validator, worker

    (tmp_path / "x.toml").write_text(
        'network = "ws://127.0.0.1:9944"\nnetuid = 1\n[vector]\npolicy_python = "p"\n'
        'simulator_python = "s"\nsimulator_root = "/r"\n'
    )
    monkeypatch.setattr(validator, "hub_token", lambda: "t")
    monkeypatch.setattr(hub, "account", lambda **kw: "someone")
    monkeypatch.setattr(worker, "run", lambda *a, **kw: pytest.fail("it ran"))
    assert worker.main(["--config", str(tmp_path / "x.toml"), "--lane", "vector"]) == 1


@pytest.mark.parametrize(
    ("access", "reads"),
    [
        ({"role": "write"}, True),
        ({"role": "fineGrained", "fineGrained": {"canReadGatedRepos": True}}, True),
        ({"role": "fineGrained", "fineGrained": {"canReadGatedRepos": False}}, False),
        ({"role": "fineGrained", "fineGrained": {}}, False),
    ],
)
def test_reads_gated_is_what_the_hub_says_of_the_token(access, reads):
    from types import SimpleNamespace

    from robotensor import hub

    api = SimpleNamespace(whoami=lambda: {"name": "v", "auth": {"accessToken": access}})
    assert hub.reads_gated(api=api) is reads


def test_a_hub_that_cannot_say_whose_token_it_is_stops_the_start(tmp_path, monkeypatch):
    from robotensor import hub, validator

    def unreachable(**kw):
        raise hub.HubUnreachable("no answer")

    monkeypatch.setattr(validator, "hub_token", lambda: "t")
    monkeypatch.setattr(hub, "account", unreachable)
    assert "no answer" in validator.hub_account_problem()


def test_a_refused_start_starts_nothing(tmp_path, monkeypatch, capsys):
    from argparse import Namespace

    from robotensor import chain, validator

    def forbidden(*args, **kwargs):
        raise AssertionError("started")

    cfg = _validator_config(tmp_path)
    monkeypatch.setattr(validator, "hub_token", lambda: None)
    monkeypatch.setattr(chain, "wallet", forbidden)
    monkeypatch.setattr(validator.supervisor, "run", forbidden)
    assert validator.cmd_run(Namespace(no_weights=False, once=True), cfg) == 1
    assert "not starting: HF_TOKEN is not set" in capsys.readouterr().err
