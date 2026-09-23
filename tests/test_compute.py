"""Which cards a competition gets for a step, and what keeps two of them off the same one."""

import multiprocessing as mp
import os
import sys
import time

from robotensor.compute import Broker, Lease, applied


def test_a_lane_with_its_own_cards_waits_for_nobody(tmp_path):
    broker = Broker(tmp_path, devices={"vector": (0, 1), "horizon": (2, 3)}, pool=(0, 1, 2, 3))

    with broker.lease("vector") as vector, broker.lease("horizon") as horizon:
        assert vector.devices == (0, 1) and horizon.devices == (2, 3)
        assert not vector.shared and not horizon.shared


def test_with_nothing_named_a_lane_takes_the_whole_box(tmp_path):
    broker = Broker(tmp_path, pool=(0,))

    with broker.lease("vector") as lease:
        assert lease.devices == (0,) and lease.shared
        assert lease.environ == {"CUDA_VISIBLE_DEVICES": "0"}


def test_a_box_with_no_gpu_leases_nothing_rather_than_pretending(tmp_path):
    with Broker(tmp_path, pool=()).lease("vector") as lease:
        assert lease.devices == () and lease.environ == {}


def _hold(root, seconds, started):
    broker = Broker(root, pool=(0,))
    with broker.lease("vector"):
        started.value = 1
        time.sleep(seconds)


def test_two_lanes_sharing_a_box_do_not_overlap(tmp_path):
    """Vector at four workers is about 36 GB and one Horizon episode wants 80: on one card they
    must take turns, or both fail slowly instead of one finishing."""
    with mp.Manager() as manager:
        started = manager.Value("i", 0)
        first = mp.Process(target=_hold, args=(str(tmp_path), 0.6, started))
        first.start()
        while not started.value:
            time.sleep(0.02)
        began = time.monotonic()
        with Broker(tmp_path, pool=(0,)).lease("horizon"):
            waited = time.monotonic() - began
        first.join(5)

    assert waited > 0.2, "the second lane took the card while the first still held it"


def test_the_environment_goes_back_to_what_it_was(tmp_path):
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    try:
        with applied(Lease((0, 1))):
            assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1"
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "7"
    finally:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)


def test_a_lease_of_no_cards_leaves_the_environment_alone():
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    with applied(Lease(())):
        assert "CUDA_VISIBLE_DEVICES" not in os.environ


def test_the_pool_is_read_from_the_environment_when_it_says(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5")
    assert Broker(tmp_path).pool == (3, 5)
    assert sys.executable  # the worker a supervisor starts runs with this pool in its environment
