"""The validator's memory: one document per competition, and what keeps two writers apart."""

import json
import multiprocessing as mp
import time

from robotensor_subnet.state import EMPTY, State


def test_each_lane_has_a_document_of_its_own(tmp_path):
    state = State(tmp_path / "state")
    state.lane("vector")["entries"]["k"] = {"hotkey": "hk"}
    state.lane("horizon")["entries"]["j"] = {"hotkey": "other"}

    state.save()

    assert json.loads((tmp_path / "state" / "vector.json").read_text())["entries"] == {
        "k": {"hotkey": "hk"}
    }
    assert json.loads((tmp_path / "state" / "horizon.json").read_text())["entries"] == {
        "j": {"hotkey": "other"}
    }


def test_a_lane_nothing_has_written_yet_reads_as_an_empty_one(tmp_path):
    assert State(tmp_path / "state").lane("horizon") == EMPTY


def test_writing_a_lane_does_not_touch_another_lanes_document(tmp_path):
    """The reason for a document each: a competition that knows nothing of the other cannot lose
    its work between a read and a write."""
    first = State(tmp_path / "state")
    first.lane("vector")["block_counter"] = 7
    first.save()

    second = State(tmp_path / "state")
    second.lane("horizon")["block_counter"] = 99
    second.save()

    assert State(tmp_path / "state").lane("vector")["block_counter"] == 7
    assert State(tmp_path / "state").lane("horizon")["block_counter"] == 99


def test_the_single_document_it_used_to_be_is_split_and_kept(tmp_path):
    legacy = tmp_path / "state.json"
    legacy.write_text(
        json.dumps(
            {
                "version": 1,
                "lanes": {"vector": {"entries": {"k": {"hotkey": "hk"}}, "block_counter": 3}},
                "weights": {"last_block": 5},
            }
        )
    )

    state = State(legacy)  # the path a config from before the split still names

    assert state.root == tmp_path / "state"
    assert state.lane("vector")["entries"] == {"k": {"hotkey": "hk"}}
    assert state.lane("vector")["block_counter"] == 3
    assert not legacy.exists() and (tmp_path / "state.json.migrated").is_file()


def test_a_directory_that_exists_is_never_overwritten_by_an_old_document(tmp_path):
    """Migration is for the one crossing, not for every start: what the lanes have written since
    wins over a file somebody put back."""
    State(tmp_path / "state").lane("vector")  # the directory exists from here on
    fresh = State(tmp_path / "state")
    fresh.lane("vector")["block_counter"] = 11
    fresh.save()
    (tmp_path / "state.json").write_text(json.dumps({"lanes": {"vector": {"block_counter": 3}}}))

    assert State(tmp_path / "state").lane("vector")["block_counter"] == 11
    assert (tmp_path / "state.json").is_file(), "the old document is left where it was"


def _hold(root, name, seconds, order):
    state = State(root)
    with state.writing(name) as doc:
        order.append(("held", time.monotonic()))
        time.sleep(seconds)
        doc["block_counter"] = 1


def test_a_second_writer_waits_for_the_one_holding_the_lane(tmp_path):
    """`writing` is the read-modify-write a worker owns: a manual command that wants the same
    lane waits for it rather than writing over what it did."""
    with mp.Manager() as manager:
        order = manager.list()
        root = str(tmp_path / "state")
        first = mp.Process(target=_hold, args=(root, "vector", 0.6, order))
        first.start()
        time.sleep(0.2)
        started = time.monotonic()
        with State(root).writing("vector") as doc:
            waited = time.monotonic() - started
            doc["entries"]["k"] = {"hotkey": "hk"}
        first.join(5)

    assert waited > 0.2, "the second writer took the lock while the first still held it"
    document = State(root).lane("vector")
    assert document["block_counter"] == 1 and document["entries"] == {"k": {"hotkey": "hk"}}
