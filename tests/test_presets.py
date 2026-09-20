"""Jedes Preset zeigt, was sein Name und seine Hilfe behaupten (Bänder mit dem ausgelieferten Code kalibriert, bewusst weit: Training mit Zufallsstart ist chaotisch, die CI-Plattform rechnet anders als lokal)."""

import pytest

import ae_constants as C
from ae_evaluation import Settings, analyse, make_dataset, verdict


def _settings(p):
    return Settings(depth=p["depth"], width=p["width"], activation=p["activation"], lr=float(p["lr"]), n_epochs=p["n_epochs"], init_start=p["init_start"])


def _measure(p):
    dataset = make_dataset(p["n_tours"], p["q"], p["curvature"], p["noise"], p["outlier_pct"], p["seed"])
    s = _settings(p)
    a = analyse(dataset, s)
    code, data = verdict(a, dataset, s)[1:]
    return {"verdict": code, **data}


def test_every_preset_has_help_and_bands():
    assert set(C.PRESETS) == set(C.PRESET_HELP) == set(C.PRESET_EXPECTED_BANDS)
    assert len(C.PRESETS) == 6


def test_preset_settings_are_within_slider_bounds():
    for p in C.PRESETS.values():
        assert C.N_TOURS_MIN <= p["n_tours"] <= C.N_TOURS_MAX and C.Q_MIN <= p["q"] <= C.Q_MAX
        assert C.CURVATURE_MIN <= p["curvature"] <= C.CURVATURE_MAX and C.NOISE_MIN <= p["noise"] <= C.NOISE_MAX and C.OUTLIER_PCT_MIN <= p["outlier_pct"] <= C.OUTLIER_PCT_MAX
        assert C.DEPTH_MIN <= p["depth"] <= C.DEPTH_MAX and p["width"] in C.WIDTH_CHOICES and p["activation"] in C.ACTIVATIONS and p["lr"] in C.LR_CHOICES
        assert C.N_EPOCHS_MIN <= p["n_epochs"] <= C.N_EPOCHS_MAX and p["n_epochs"] % 50 == 0 and p["init_start"] in C.INIT_STARTS


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_preset_stays_inside_its_bands(name):
    measured = _measure(C.PRESETS[name])
    for key, expected in C.PRESET_EXPECTED_BANDS[name].items():
        value = measured[key]
        if isinstance(expected, str):
            assert value == expected, f"{key}: {value}"
        else:
            lo, hi = expected
            assert lo <= value <= hi, f"{key}: {value} nicht in [{lo}, {hi}]"
