"""Rauchtests der Streamlit-Oberfläche per AppTest: Standard, jedes Preset, Randgrößen, Schritt-Zustand, Breite bei Tiefe 0, Zusatz-Experimente, Achsensperre."""

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import ae_constants as C
from ae_presets import PRESET_KEYS

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app.py"
WARNING_PRESETS = ("Zu wenige Epochen", "Lernrate zu hoch")


def _run(setup=None, timeout=300):
    at = AppTest.from_file(str(APP), default_timeout=timeout)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    if setup is not None:
        setup(at)
        at.run()
        assert not at.exception, [e.value for e in at.exception]
    return at


def _apply(at, p):
    for key, state_key in PRESET_KEYS.items():
        at.session_state[state_key] = p[key]


def test_default_renders_without_exception():
    at = _run()
    assert any("Architektur" in h.value for h in at.subheader)
    assert not at.error and not at.warning


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_renders(name):
    at = _run(lambda a: _apply(a, C.PRESETS[name]))
    assert bool(at.warning) == (name in WARNING_PRESETS)


def test_extreme_settings_render():
    def small(at):
        at.session_state["n_tours_slider"] = C.N_TOURS_MIN
        at.session_state["q_slider"] = C.Q_MIN
        at.session_state["depth_slider"] = C.DEPTH_MAX
        at.session_state["width_select"] = min(C.WIDTH_CHOICES)
        at.session_state["n_epochs_slider"] = C.N_EPOCHS_MIN
        at.session_state["activation_select"] = "relu"
    at = _run(small)
    # Ein zusammengebrochener Autoencoder darf nirgends "nan" zeigen (früher: "Autoencoder nan" in der Bildunterschrift)
    texts = [m.value for m in at.markdown] + [c.value for c in at.caption]
    assert not [t for t in texts if re.search(r"\bnan\b", t, re.I)]

    def large(at):
        at.session_state["n_tours_slider"] = 400
        at.session_state["q_slider"] = C.Q_MAX
        at.session_state["noise_slider"] = C.NOISE_MAX
        at.session_state["outlier_slider"] = C.OUTLIER_PCT_MAX
        at.session_state["width_select"] = max(C.WIDTH_CHOICES)
        at.session_state["activation_select"] = "sigmoid"
        at.session_state["init_start_select"] = 5
        at.session_state["lr_select"] = max(C.LR_CHOICES)
    _run(large)


def test_width_control_is_hidden_at_depth_zero_and_its_value_is_kept():
    at = _run(lambda a: a.session_state.__setitem__("width_select", 32))
    assert any(s.key == "width_select" for s in at.select_slider)
    at.session_state["depth_slider"] = 0
    at.run()
    assert not at.exception and not any(s.key == "width_select" for s in at.select_slider)
    at.session_state["depth_slider"] = 2
    at.run()
    assert not at.exception and [s for s in at.select_slider if s.key == "width_select"][0].value == 32       # Auswahl kehrt zurück


def test_step_state_resets_when_the_data_or_settings_change_and_survives_reruns():
    at = _run()
    at.session_state["ae_step"] = 3
    at.run()
    assert not at.exception and at.session_state["ae_step"] == 3
    at.session_state["lr_select"] = 0.01
    at.run()
    assert not at.exception and at.session_state["ae_step"] == 1


@pytest.mark.parametrize("step", [1, 2, 3, 4])
def test_every_step_renders(step):
    def setup(at):
        at.session_state["ae_step"] = step
    _run(setup)


def test_snapshot_slider_survives_a_change_of_snapshot():
    at = _run(lambda a: a.session_state.__setitem__("ae_step", 3))
    at.session_state["ae_snap"] = 0
    at.run()
    assert not at.exception and at.session_state["ae_snap"] == 0


def test_extra_experiments_run_on_demand():
    at = _run()
    for key in ("oos_start", "stability_start"):
        [b for b in at.button if b.key == key][0].click()
        at.run()
        assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["oos_on"] and at.session_state["stability_on"]


def test_every_figure_of_the_visualisation_module_is_axis_locked():
    source = (ROOT / "ae_visualization.py").read_text(encoding="utf-8")
    assert len(re.findall(r"return lock_axes\(fig\)", source)) == len(re.findall(r"^def build_", source, flags=re.M)) == 12
    assert len(re.findall(r"^\s+return fig$", source, flags=re.M)) == 1


def test_play_runs_through_all_frames_without_duplicate_chart_keys():
    """Beim Abspielen entstehen in einem Lauf mehrere Diagramme mit demselben Namen - die Schlüssel tragen deshalb den Schritt (Regression: StreamlitDuplicateElementKey bei mehr als einem Bild)."""
    at = _run()
    [b for b in at.button if b.label == "▶️ Abspielen"][0].click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
