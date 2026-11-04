"""Horizon's vault: 70% of the weights to one hotkey, beside Vector's champions' 30%."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from robotensor import config, validator
from robotensor.config import ConfigError, load
from robotensor.protocol.weights import CHAMPION_SPLIT, weight_vector

VAULT = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"  # Bob
OTHER = "5FLSigC9HGRKVhB9FiEo4Y3koPsNmBmLJbpXg2mp1hXcS59Y"  # Charlie
OWNER = "5CiPPseXPECbkjWCa6MnjNokrgYjMqmKndv2rSnekmSK2DjL"  # Ferdie

VECTOR = '[vector]\npolicy_python = "p"\nsimulator_python = "s"\nsimulator_root = "/r"\nstart_block = 1\n'


def _config(tmp_path, extra="", network="test"):
    path = tmp_path / "config" / f"{network}.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'network = "{network}"\nnetuid = 7\n{extra}{VECTOR}', encoding="utf-8")
    return path


def test_the_vault_takes_70_percent_and_vectors_champions_30(tmp_path):
    cfg = load(_config(tmp_path, f'[horizon]\nvault_hotkey = "{VAULT}"\n'))
    award = SimpleNamespace(entries=["hk1", "hk2"], split=CHAMPION_SPLIT)
    uids = {OWNER: 0, VAULT: 3, "hk1": 5, "hk2": 6}

    w = weight_vector(validator.lanes(cfg, award), uids, 0)

    assert w[3] == pytest.approx(0.70)
    assert w[5] == pytest.approx(0.30 * 0.4 / 0.7) and w[6] == pytest.approx(0.30 * 0.3 / 0.7)
    assert sum(w.values()) == pytest.approx(1.0) and 0 not in w


def test_an_unset_or_unregistered_vault_burns_its_share(tmp_path):
    award = SimpleNamespace(entries=["hk1"], split=CHAMPION_SPLIT)

    unset = load(_config(tmp_path))
    assert weight_vector(validator.lanes(unset, award), {OWNER: 0, "hk1": 5}, 0) == {
        0: pytest.approx(0.70),
        5: pytest.approx(0.30),
    }

    named = load(_config(tmp_path, f'[horizon]\nvault_hotkey = "{VAULT}"\n'))
    unregistered = weight_vector(validator.lanes(named, award), {OWNER: 0, "hk1": 5}, 0)
    assert unregistered[0] == pytest.approx(0.70)


def test_a_network_that_pins_its_vault_weights_it_without_a_config(tmp_path, monkeypatch):
    monkeypatch.setitem(config.VAULT_HOTKEYS, "test", VAULT)

    assert load(_config(tmp_path)).horizon.vault_hotkey == VAULT


def test_a_network_that_pins_its_vault_refuses_another(tmp_path, monkeypatch):
    monkeypatch.setitem(config.VAULT_HOTKEYS, "test", VAULT)

    with pytest.raises(ConfigError, match="every validator weights the same one"):
        load(_config(tmp_path, f'[horizon]\nvault_hotkey = "{OTHER}"\n'))


@pytest.mark.parametrize(
    "table, message",
    [
        ('vault_hotkey = "nope"\n', "not an SS58 address"),
        (f'vault_hotkey = "{VAULT}"\nshare = 0.5\n', "does not take share"),
    ],
)
def test_a_horizon_table_this_build_cannot_read_is_refused(tmp_path, table, message):
    with pytest.raises(ConfigError, match=message):
        load(_config(tmp_path, "[horizon]\n" + table))


def test_horizon_and_vector_take_the_whole_emission():
    assert config.SHARES["vector"] + config.HORIZON_SHARE == pytest.approx(1.0)
