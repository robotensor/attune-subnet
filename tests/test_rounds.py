"""Where a round has got to, and what the validator runs next."""

import pytest

from robotensor.lanes.rounds import STEPS, Engine, Plan, opened


def leave(directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text("{}")


def test_a_fresh_directory_starts_at_the_beginning(tmp_path):
    assert Plan(tmp_path / "e").next() == "open"
    assert Plan(tmp_path / "e").at() == "next: open"


def test_what_the_engine_left_behind_says_what_has_run(tmp_path):
    """Nothing is kept in memory: the loop can die between any two verbs."""
    directory = tmp_path / "e"
    leave(directory, "round.json")

    assert Plan(directory).next() == "pool"

    leave(directory, "pool_manifest.json")

    assert Plan(directory).next() == "screen"


def test_a_round_opened_before_the_rename_is_carried_on_not_opened_again(tmp_path):
    """Its directory holds `epoch.json`, what `open` wrote then: what is there decides."""
    directory = tmp_path / "e"
    leave(directory, "epoch.json")

    assert Plan(directory).finished("open")
    assert Plan(directory).next() == "pool"


def test_the_two_drains_are_read_from_the_note_they_leave_none(tmp_path):
    directory = tmp_path / "e"
    leave(directory, "round.json")
    leave(directory, "pool_manifest.json")

    assert Plan(directory, stage="screen").next() == "shortlist"

    leave(directory, "shortlist.json")

    assert Plan(directory, stage="screen").next() == "full"
    assert Plan(directory, stage="full").next() == "score"


def test_a_note_one_step_stale_costs_a_re_run_and_never_a_wrong_result(tmp_path):
    """The engine's drains skip what is already there, so re-running one is a re-scan."""
    directory = tmp_path / "e"
    for name in ("round.json", "pool_manifest.json", "shortlist.json"):
        leave(directory, name)

    # The note says the screen finished; the full stage's own note was lost.
    assert Plan(directory, stage="screen").next() == "full"


def test_a_dry_run_scores_beside_the_real_thing(tmp_path):
    directory = tmp_path / "e"
    for name in ("round.json", "pool_manifest.json", "shortlist.json"):
        leave(directory, name)
    leave(directory, "scores-dry-run.json")

    assert Plan(directory, stage="full", dry_run=True).next() == "close"
    assert Plan(directory, stage="full").next() == "score", "a real round wants the real scores"


def test_a_round_that_has_run_every_verb_is_finished(tmp_path):
    directory = tmp_path / "e"
    for name in ("round.json", "pool_manifest.json", "shortlist.json", "scores.json"):
        leave(directory, name)

    assert Plan(directory, stage="close").next() is None
    assert Plan(directory, stage="close").at() == "finished"


@pytest.fixture
def engine(tmp_path):
    return Engine(
        python="/usr/bin/python3",
        config=tmp_path / "competition.yml",
        store=tmp_path / "store",
        rounds=tmp_path / "rounds",
        models=tmp_path / "models",
        runtime_python="/opt/runtime/bin/python",
        serve_as="horizon",
        devices=(0, 1),
    )


def test_open_names_the_store_and_the_register(engine):
    argv = engine.argv("open", "2026-W39", profile="smoke")

    assert argv[:5] == ["/usr/bin/python3", "-m", "horizon_competition.cli", "round", "open"]
    assert "--profile" in argv and argv[argv.index("--profile") + 1] == "smoke"
    assert argv[argv.index("--store") + 1] == str(engine.store)
    assert argv[argv.index("--register") + 1] == str(engine.store)
    assert "--keys" not in argv


def test_open_after_the_previous_round_starts_where_that_one_closed(engine):
    """Back to back, as the chain's windows are, however late the round is opened."""
    previous = engine.directory("e00006")

    argv = engine.argv("open", "e00007", after=previous)

    assert argv[argv.index("--after") + 1] == str(previous)
    assert "--after" not in engine.argv("open", "e00007"), "the first round follows nothing"


@pytest.mark.parametrize("marker", ["round.json", "epoch.json"])
def test_a_directory_is_an_opened_round_by_either_marker(tmp_path, marker):
    directory = tmp_path / "e00006"
    directory.mkdir()

    assert not opened(directory), "a directory an open never finished in is not a round"
    assert not opened(tmp_path / "e00005")

    leave(directory, marker)

    assert opened(directory)


def test_the_screening_stage_is_the_drain_told_to_screen_only(engine):
    screen = engine.argv("screen", "2026-W39")
    full = engine.argv("full", "2026-W39")

    assert screen[:6] == [*full[:5], "--round"] and screen[-1] == "--screen-only"
    assert "--screen-only" not in full


@pytest.mark.parametrize("verb", STEPS)
def test_every_verb_is_the_engines_round_command_on_the_rounds_directory(engine, verb):
    """The engine has no `epoch` group any more, and no `--epoch` flag to take a directory."""
    argv = engine.argv(verb, "e00007")

    assert argv[3] == "round"
    assert argv[argv.index("--round") + 1] == str(engine.rounds / "e00007")
    assert "epoch" not in argv and "--epoch" not in argv


@pytest.mark.parametrize("stage", ["screen", "full"])
def test_each_drain_publishes_a_models_result_as_soon_as_it_finishes(engine, stage):
    """Live: each result goes into the round's store as soon as it is ready."""
    argv = engine.argv(stage, "e00007")

    assert argv[3:5] == ["round", "drain"]
    assert argv[argv.index("--store") + 1] == str(engine.store)
    assert "--keys" not in argv


@pytest.mark.parametrize("stage", ["screen", "full"])
def test_a_rehearsals_drains_publish_nothing(engine, stage):
    """A result is published once: a rehearsal's would take the real round's name."""
    argv = engine.argv(stage, "e00007", dry_run=True)

    assert "--store" not in argv and "--keys" not in argv


def test_a_served_model_runs_as_the_user_the_config_names(engine):
    """Not as the validator: a model served as that user could read the episode's answer key."""
    argv = engine.argv("full", "2026-W39")

    assert argv[argv.index("--serve-as") + 1] == "horizon"
    assert argv[argv.index("--gpus") + 1] == "0,1"


def test_a_dry_run_close_publishes_nothing(engine):
    assert "--dry-run" in engine.argv("close", "2026-W39", dry_run=True)
    assert "--dry-run" not in engine.argv("close", "2026-W39")


def test_a_verb_the_engine_does_not_have_is_refused(engine):
    with pytest.raises(ValueError, match=", ".join(STEPS)):
        engine.argv("publish", "2026-W39")
