"""SETTING_SPECS-Permalink-Muster, Presets und Zufalls-Seed-Button (Standardmuster aus dem OR-Demo-Portfolio, siehe km_presets.py in kmeans-demo)."""

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import ae_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


def _choice(options):
    def cast(value):
        value = str(value)
        if value not in options:
            raise ValueError(value)
        return value
    return cast


SETTING_SPECS = {
    "n_tours_slider": SettingSpec("n", int, C.DEFAULT_N_TOURS, C.N_TOURS_MIN, C.N_TOURS_MAX),
    "q_slider": SettingSpec("q", int, C.DEFAULT_Q, C.Q_MIN, C.Q_MAX),
    "curvature_slider": SettingSpec("curv", float, C.DEFAULT_CURVATURE, C.CURVATURE_MIN, C.CURVATURE_MAX),
    "noise_slider": SettingSpec("noise", float, C.DEFAULT_NOISE, C.NOISE_MIN, C.NOISE_MAX),
    "outlier_slider": SettingSpec("out", int, C.DEFAULT_OUTLIER_PCT, C.OUTLIER_PCT_MIN, C.OUTLIER_PCT_MAX),
    "depth_slider": SettingSpec("depth", int, C.DEFAULT_DEPTH, C.DEPTH_MIN, C.DEPTH_MAX),
    "width_select": SettingSpec("width", int, C.DEFAULT_WIDTH, min(C.WIDTH_CHOICES), max(C.WIDTH_CHOICES)),
    "activation_select": SettingSpec("act", _choice(C.ACTIVATIONS), C.DEFAULT_ACTIVATION),
    "lr_select": SettingSpec("lr", float, C.DEFAULT_LR, min(C.LR_CHOICES), max(C.LR_CHOICES)),
    "n_epochs_slider": SettingSpec("ep", int, C.DEFAULT_N_EPOCHS, C.N_EPOCHS_MIN, C.N_EPOCHS_MAX),
    "init_start_select": SettingSpec("start", int, C.DEFAULT_INIT_START, min(C.INIT_STARTS), max(C.INIT_STARTS)),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, 2_000_000_000),
}
PRESET_KEYS = {"n_tours": "n_tours_slider", "q": "q_slider", "curvature": "curvature_slider", "noise": "noise_slider", "outlier_pct": "outlier_slider", "depth": "depth_slider", "width": "width_select",
               "activation": "activation_select", "lr": "lr_select", "n_epochs": "n_epochs_slider", "init_start": "init_start_select", "seed": "seed_input"}


def snap(choices, value):
    return min(choices, key=lambda choice: abs(choice - value))


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def init_session_state_defaults():
    """Fehlende Zustände auffüllen; die Breite (bei Tiefe 0 ausgeblendet, dann vom Widget-Zustand gelöscht) kehrt zum zuletzt gewählten Wert zurück."""
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = st.session_state.get("_width_kept", spec.default) if state_key == "width_select" else spec.default


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
                if isinstance(value, float) and not math.isfinite(value):
                    continue
                if spec.lo is not None:
                    value = max(spec.lo, value)
                if spec.hi is not None:
                    value = min(spec.hi, value)
                st.session_state[state_key] = value
            except (ValueError, TypeError):
                pass
    st.session_state["width_select"] = snap(C.WIDTH_CHOICES, st.session_state.get("width_select", C.DEFAULT_WIDTH))
    st.session_state["lr_select"] = snap(C.LR_CHOICES, st.session_state.get("lr_select", C.DEFAULT_LR))
    st.session_state["init_start_select"] = int(snap(C.INIT_STARTS, st.session_state.get("init_start_select", C.DEFAULT_INIT_START)))
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = C.PRESETS[name][key]
    st.session_state["_width_kept"] = C.PRESETS[name]["width"]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, 2_000_000_000)
