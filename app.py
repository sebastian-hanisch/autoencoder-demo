"""Autoencoder an Lieferrouten-Kennzahlen - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo EIN Verfahren - den Autoencoder - und lässt
stattdessen das Beispiel wachsen. Siebtes und letztes Stück der Dimensionsreduktion-Linie der "Konzepte"-Reihe: ein unabhängiger Ast direkt nach der PCA (nicht nach UMAP/PaCMAP)
und das erste neuronale Netz im Portfolio. Linear ist es die PCA; nichtlinear ein gelernter, parametrischer Encoder mit Decoder. Was das gegenüber den anderen Verfahren
bringt und kostet, wird hier gemessen. Siehe README für die Einordnung.

Lauffähig mit: streamlit run app.py
"""

import time

import numpy as np
import streamlit as st

import ae_constants as C
from ae_algorithm import layer_sizes, pca_subspace
from ae_evaluation import (
    Settings, analyse, anomaly_auc, convergence_rows, decoder_grid, depth_sweep, linear_equals_pca, lr_sweep, make_dataset, out_of_sample, reconstruct, stability, timing_sweep, verdict, width_sweep,
)
from ae_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from ae_visualization import (
    build_angle_curve,
    build_architecture,
    build_decoder_grid,
    build_distance_fidelity,
    build_embedding,
    build_error_map,
    build_out_of_sample,
    build_stability_compare,
    build_sweep,
    build_timing,
    build_tour_reconstruction,
    build_training,
)

st.set_page_config(page_title="Autoencoder – Sebastian Hanisch", layout="wide")

STEP_LABELS = {
    1: "1 · Architektur",
    2: "2 · Eine Tour hindurch",
    3: "3 · Training",
    4: "4 · Ergebnis",
}
FEATURE_NAMES = C.FEATURE_NAMES


def _lr_label(value):
    return f"{value:g}"


@st.cache_data(show_spinner=False)
def _dataset(n_tours, q, curvature, noise, outlier_pct, seed):
    return make_dataset(n_tours, q, curvature, noise, outlier_pct, seed)


@st.cache_data(show_spinner=False)
def _analysis(data_params, settings):
    return analyse(make_dataset(*data_params), settings)


@st.cache_data(show_spinner=False)
def _sweeps(q, curvature, noise, outlier_pct):
    return depth_sweep(q, curvature, noise, outlier_pct), width_sweep(q, curvature, noise, outlier_pct), lr_sweep(q, curvature, noise, outlier_pct)


@st.cache_data(show_spinner=False)
def _linear(data_params):
    return linear_equals_pca(make_dataset(*data_params))


@st.cache_data(show_spinner=False)
def _stability(q, curvature, noise, outlier_pct, settings):
    return stability(q, curvature, noise, outlier_pct, settings)


@st.cache_data(show_spinner=False)
def _oos(data_params, settings):
    return out_of_sample(make_dataset(*data_params), settings)


st.title("🧠 Autoencoder an Lieferrouten-Kennzahlen")
st.markdown(
    """
Dieselben **12 Kennzahlen je Lieferroute** wie in den Demos davor - erzeugt aus wenigen versteckten Faktoren, aber mit **gekrümmter** Struktur, an der die PCA scheiterte.
Der **Autoencoder** ist das siebte und letzte Stück der Dimensionsreduktion-Linie und geht einen ganz anderen Weg als Isomap, LLE, t-SNE, UMAP und PaCMAP: kein Nachbarschaftsgraph, keine Paare, sondern ein **neuronales Netz**,
das die 12 Kennzahlen durch einen Engpass von nur **2 Werten** zwängt und daraus wieder **rekonstruieren** soll. Die 2 Werte sind die Einbettung. Ohne versteckte Schichten *ist* das die PCA; mit ihnen ein gelernter,
**parametrischer** Encoder - neue Touren werden einfach durchgeschickt - **plus Decoder**, der aus jedem Punkt des 2-D-Raums wieder eine Tour erzeugt. Was das bringt und was es kostet (Stabilität, Datenhunger,
Faktoren-Rückgewinnung), misst die Demo direkt gegen PCA, Isomap, t-SNE, UMAP und PaCMAP - und sagt es, wenn eine Erwartung nicht aufgeht.
Wie das Verfahren funktioniert, erklärt der aufgeklappte Abschnitt direkt darunter.
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren vergleichen, zeigt diese Demo - siebtes und letztes Stück der "
    "Dimensionsreduktion-Linie der \"Konzepte\"-Reihe, ein unabhängiger Ast direkt nach der PCA - **ein** Verfahren an einem wachsenden Beispiel: das erste neuronale Netz im Portfolio."
)

with st.expander("So funktioniert der Autoencoder", expanded=True):
    st.markdown(
        """
Ein Autoencoder besteht aus zwei Teilen, die zusammen trainiert werden:

1. **Encoder**: schickt die 12 Kennzahlen (z-Werte) einer Tour durch null bis drei **versteckte Schichten** (je *Breite* Neuronen mit einer nichtlinearen Aktivierung) und schließlich durch den **Engpass** aus nur 2 Neuronen.
   Diese 2 Werte - der **Code** - sind die Einbettung.
2. **Decoder**: schickt den Code durch dieselben Schichten in umgekehrter Richtung zurück auf 12 Werte: die **Rekonstruktion**.
3. **Training**: das Netz minimiert den mittleren quadratischen Abstand zwischen Original und Rekonstruktion - per **Backpropagation** (die Fehler-Gradienten laufen rückwärts durch die Schichten) und dem Adam-Verfahren.
   Es gibt keinen Nachbarschaftsbegriff: der Code muss nur so viel Information tragen, dass der Decoder die Tour wiederherstellen kann.

Ohne versteckte Schichten (**Tiefe 0**) sind Encoder und Decoder lineare Abbildungen - dann lernt das Netz denselben 2-D-Unterraum wie die **PCA** (unten live nachgemessen). Mit versteckten Schichten kann es die gebogene Fläche entfalten und
deutlich besser rekonstruieren als jede lineare Methode. Anders als bei den Nachbarschaftsverfahren gibt es aber **keine Garantie**, dass der Code die wahren Faktoren nachzeichnet, und das Ergebnis hängt vom **zufälligen Start**
der Gewichte ab. Was der Autoencoder im Vergleich wirklich ändert, steht unten in "🆚" - gemessen, nicht behauptet.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_cols = st.columns(len(C.PRESETS))
for i, name in enumerate(C.PRESETS.keys()):
    with preset_cols[i]:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name])

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n_tours = st.slider("Anzahl Touren", *bounds("n_tours_slider"), key="n_tours_slider", step=50)
    q = st.slider(
        "Wahre Anzahl versteckter Faktoren (q)", *bounds("q_slider"), key="q_slider",
        help="So viele echte Einflussgrößen erzeugen die 12 Kennzahlen. Mit mehr Faktoren als Engpass-Werten (2) wird die Einbettung schwächer (im Test R² 0.32 bei q = 3 gegen 0.72 bei q = 2).",
    )
    curvature = st.slider(
        "Krümmung", *bounds("curvature_slider"), key="curvature_slider", step=0.05,
        help="0 = die Kennzahlen hängen linear von den Faktoren ab (dann hat das nichtlineare Netz keinen Vorteil vor der PCA). Größer = die Touren liegen auf einer zunehmend gebogenen Fläche.",
    )
    noise = st.slider(
        "Rauschen", *bounds("noise_slider"), key="noise_slider", step=0.05,
        help="Messrauschen je Kennzahl. Das Netz lernt es mit - im Test fiel das R² bei Rauschen 0.8 auf 0.54 (UMAP 0.90).",
    )
    outlier_pct = st.slider(
        "Sonderfahrten (%)", *bounds("outlier_slider"), key="outlier_slider",
        help="Anteil der Touren mit extremem Zeitdruck-Faktor (zehnfach vergrößert). t-SNE, UMAP und PaCMAP brechen ein; das Autoencoder im Test bei 5-10 % nicht (bei 2 % aber auch nicht besser).",
    )
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)

    st.markdown("**Autoencoder**")
    depth = st.slider(
        "Tiefe (versteckte Schichten je Seite)", *bounds("depth_slider"), key="depth_slider",
        help="0 = lineares Autoencoder (nur Engpass) - gleich der PCA. 1-3 = nichtlinear. Im Test war Tiefe 2 am besten; Tiefe 3 fiel beim R² wieder ab.",
    )
    if depth > 0:
        width = st.select_slider(
            "Breite der versteckten Schichten", options=C.WIDTH_CHOICES, key="width_select",
            help="Neuronen je versteckter Schicht. Breite 2 ist zu schmal (nicht besser als die PCA); ab 8 rekonstruiert das Netz gut; sehr breite Netze (64) rekonstruieren am besten, finden die Faktoren aber schlechter.",
        )
        st.session_state["_width_kept"] = int(width)
    else:
        width = int(st.session_state.get("_width_kept", C.DEFAULT_WIDTH))
    activation = st.selectbox(
        "Aktivierung", C.ACTIVATIONS, key="activation_select",
        help="Nichtlinearität der versteckten Schichten. Im Test waren tanh und sigmoid gleich gut, ReLU schlechter (Rekonstruktionsfehler 0.049 statt 0.022). Bei Tiefe 0 ohne Wirkung - dort gibt es keine versteckte Schicht.",
    )
    lr = st.select_slider(
        "Lernrate", options=C.LR_CHOICES, key="lr_select", format_func=_lr_label,
        help="Schrittweite von Adam. Zu klein: das Training braucht sehr lange. Zu groß (0.3): es springt über das Minimum und bleibt so schlecht wie die PCA.",
    )
    n_epochs = st.slider(
        "Epochen", *bounds("n_epochs_slider"), key="n_epochs_slider", step=50,
        help="Trainingsdurchläufe über alle Touren. Nach 100 Epochen ist der Rekonstruktionsfehler noch mehrfach so hoch wie nach 2000 (im Test 4-9-fach); ab etwa 500 ändert sich das R² kaum noch.",
    )
    init_start = st.selectbox(
        "Start der Gewichte", C.INIT_STARTS, key="init_start_select", format_func=lambda i: f"Start {i}",
        help="Zufällige Anfangsgewichte. Verschiedene Starts liefern verschiedene Bilder - im Test wichen sie um einen mittleren Procrustes-Abstand von 0.4-0.6 voneinander ab (siehe Stabilität unten).",
    )

    st.button("🎲 Neue Touren generieren", width="stretch", on_click=randomize_seed, help="Würfelt einen neuen Zufalls-Seed für die Touren.")

sync_query_params({
    "n_tours_slider": int(n_tours), "q_slider": int(q), "curvature_slider": curvature, "noise_slider": noise, "outlier_slider": int(outlier_pct), "depth_slider": int(depth), "width_select": int(width),
    "activation_select": activation, "lr_select": lr, "n_epochs_slider": int(n_epochs), "init_start_select": int(init_start), "seed_input": int(seed),
})

data_params = (int(n_tours), int(q), float(curvature), float(noise), int(outlier_pct), int(seed))
settings = Settings(depth=int(depth), width=int(width), activation=activation, lr=float(lr), n_epochs=int(n_epochs), init_start=int(init_start))
with st.spinner("Trainiere das Netz und die Vergleichsverfahren..."):
    dataset = _dataset(*data_params)
    analysis = _analysis(data_params, settings)
model = analysis.ae
metrics = analysis.metrics
z_color = dataset.z[:, 0]
iso_idx = analysis.iso_indices
data_key = data_params + (settings,)
snap_epochs = sorted(model.snapshots)
with st.spinner("Prüfe Tiefe, Breite und Lernrate über feste Sweep-Datensätze..."):
    depth_rows, width_rows, lr_rows = _sweeps(int(q), float(curvature), float(noise), int(outlier_pct))
level, code, vd = verdict(analysis, dataset, settings)
r2_points = convergence_rows(analysis)

# --- Autoencoder in Aktion --------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Autoencoder in Aktion")
if "ae_step" not in st.session_state or st.session_state.get("ae_step_owner") != data_key:
    st.session_state["ae_step"] = 1
    st.session_state["ae_snap"] = snap_epochs[-1]
    st.session_state["ae_step_owner"] = data_key
step_col, play_col = st.columns([5, 1])
with step_col:
    step = st.select_slider("Schritt", options=list(STEP_LABELS), key="ae_step", format_func=lambda s: STEP_LABELS[s])
with play_col:
    auto_play = st.button("▶️ Abspielen", width="stretch")

sizes = layer_sizes(len(FEATURE_NAMES), model.depth, model.width)
if st.session_state.get("ae_snap") not in model.snapshots:
    st.session_state["ae_snap"] = snap_epochs[-1]
if step == 3 and not auto_play:
    snap_ep = st.select_slider("Epoche", options=snap_epochs, key="ae_snap", format_func=lambda e: "Start (Zufallsgewichte)" if e == 0 else f"Epoche {e}")
else:
    snap_ep = st.session_state.get("ae_snap", snap_epochs[-1])
    if snap_ep not in model.snapshots:
        snap_ep = snap_epochs[-1]

axes, mse_floor = pca_subspace(model.Z, C.N_CODE)
focus = int(np.argmin(((analysis.pca_2d - analysis.pca_2d.mean(0)) ** 2).sum(1)))
recon_all, err_all = reconstruct(model, dataset.X)
pca_recon_focus = (model.Z[focus] @ axes.T) @ axes
view_slot = st.empty()


def _render(current_step, epoch=None):
    if current_step == 1:
        with view_slot.container():
            st.plotly_chart(build_architecture(sizes, model.depth, model.n_parameters), width="stretch", key="ae_architecture")
    elif current_step == 2:
        with view_slot.container():
            c1, c2 = st.columns([3, 2])
            c1.plotly_chart(build_tour_reconstruction(list(FEATURE_NAMES), model.Z[focus], recon_all[focus], pca_recon_focus), width="stretch", key="ae_tour_reconstruction")
            c2.markdown("**Code dieser Tour**")
            c2.table({
                "Größe": ["Code 1", "Code 2", "Fehler Autoencoder", "Fehler PCA (2 Komp.)"],
                "Wert": [f"{model.embedding[focus, 0]:.2f}", f"{model.embedding[focus, 1]:.2f}", f"{err_all[focus]:.3f}", f"{((model.Z[focus] - pca_recon_focus) ** 2).mean():.3f}"],
            })
    elif current_step == 3:
        ep = snap_ep if epoch is None else epoch
        with view_slot.container():
            c1, c2 = st.columns([2, 3])
            c1.markdown(f"**Einbettung (Code) nach Epoche {ep}**")
            c1.plotly_chart(build_embedding(model.snapshots[ep], z_color, "Code 1", "Code 2"), width="stretch", key="ae_snapshot")
            c2.markdown("**Training**")
            c2.plotly_chart(build_training(model.loss_history, mse_floor, dict(r2_points), analysis.snapshot_far, marker=ep), width="stretch", key="ae_training")
    else:
        with view_slot.container():
            c1, c2 = st.columns(2)
            c1.markdown("**Autoencoder: Code**")
            c1.plotly_chart(build_embedding(model.embedding, z_color, "Code 1", "Code 2"), width="stretch", key="ae_embed_step")
            c2.markdown("**Zum Vergleich: PCA**")
            c2.plotly_chart(build_embedding(analysis.pca_2d, z_color, "PC1", "PC2"), width="stretch", key="pca_embed_step")


if auto_play:
    for s in STEP_LABELS:
        if s == 3:
            for ep in snap_epochs:
                _render(3, ep)
                time.sleep(0.35)
        else:
            _render(s)
            time.sleep(1.0)
    step = 4
else:
    _render(step)

if step == 1:
    kind = "linear (keine versteckte Schicht)" if model.depth == 0 else f"{model.depth} versteckte Schicht(en) je Seite, Breite {model.width}, Aktivierung {model.activation}"
    st.caption(
        f"Das Netz ist {kind}: Encoder 12 → … → **2** (Engpass, linear), Decoder 2 → … → 12 (linear). Blau: Encoder, orange: Engpass, grün: Decoder; gezeichnet werden höchstens 10 Knoten je Schicht, darunter steht die echte Breite. "
        f"{model.n_parameters:,} Parameter werden gelernt - bei {dataset.n} Touren und 12 Kennzahlen sind das {model.n_parameters / (dataset.n * 12):.2f} Parameter je Datenwert.".replace(",", ".")
    )
elif step == 2:
    st.caption(
        f"Eine Tour (nahe der Mitte der Wolke) läuft durch den Encoder auf 2 Werte und durch den Decoder zurück auf 12. Grau: was die PCA mit ebenfalls nur 2 Komponenten aus dieser Tour rekonstruieren könnte. "
        f"Fehler dieser Tour: Autoencoder {err_all[focus]:.3f}, PCA {((model.Z[focus] - pca_recon_focus) ** 2).mean():.3f}. Über alle Touren: Autoencoder {analysis.mse_ae:.3f}, PCA-Untergrenze {analysis.mse_pca:.3f}."
    )
elif step == 3:
    st.caption(
        f"Links: der Code aller Touren zu Beginn (Zufallsgewichte) und im Verlauf des Trainings. Rechts: der Rekonstruktionsfehler fällt (rot gestrichelt: das Beste, was eine lineare Methode mit 2 Komponenten erreichen kann - die PCA-Untergrenze). "
        f"Fehler am Ende: {analysis.mse_ae:.3f}. R² und Abstandstreue der wahren Faktoren steigen nicht unbedingt mit: der Code wird darauf trainiert, zu rekonstruieren, nicht die Faktoren zu treffen."
    )
else:
    st.caption("Farbe = versteckter Faktor 1. Verläuft sie im Code glatt und ohne Überlappung, hat das Netz die Fläche entrollt - die Achsen selbst haben keine feste Bedeutung.")

st.markdown("---")

# --- Ergebnis --------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Was der Autoencoder gefunden hat - und die anderen fünf Verfahren auf denselben Daten")
st.caption(
    f"Vergleichsverfahren mit ihren guten Einstellungen (aus den Demos davor): UMAP n_neighbors {C.UMAP_N_NEIGHBORS}, min_dist {C.UMAP_MIN_DIST:g}; PaCMAP n_neighbors {C.PACMAP_N_NEIGHBORS}; "
    f"t-SNE Perplexity {C.TSNE_PERPLEXITY}; Isomap k = {C.ISOMAP_K}; PCA auf z-Werten."
)
if len(iso_idx) < dataset.n:
    st.warning(f"⚠️ Der Isomap-Graph ist nicht zusammenhängend: nur {len(iso_idx)} von {dataset.n} Touren sind in der Isomap-Einbettung enthalten (siehe Isomap-Demo).")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Rekonstruktionsfehler", f"{analysis.mse_ae:.3f}", delta=f"{analysis.mse_ae - analysis.mse_pca:+.3f} ggü. PCA-Untergrenze", delta_color="inverse",
          help="Mittleres Fehlerquadrat je Kennzahl (z-Werte) aus nur 2 Werten. Die PCA-Untergrenze ist das Beste, was eine lineare Methode mit 2 Komponenten erreichen kann - das nichtlineare Netz kann sie unterschreiten.")
m2.metric("R² der wahren Faktoren", f"{metrics['ae']['r2']:.2f}", delta=f"{metrics['ae']['r2'] - metrics['umap']['r2']:+.2f} ggü. UMAP", delta_color="normal",
          help="Wie gut lassen sich die versteckten Faktoren aus den zwei Werten zurückgewinnen (quadratische Regression). UMAP mit derselben Messung im Delta.")
m3.metric("Abstandstreue ferner Paare", f"{metrics['ae']['far']:.2f}", delta=f"{metrics['ae']['far'] - metrics['umap']['far']:+.2f} ggü. UMAP", delta_color="normal",
          help="Korrelation der Paarabstände in der Einbettung mit den Paarabständen der wahren Faktoren, nur für die obere Hälfte der wahren Abstände - der Test für 'globale Struktur'.")
m4.metric("Trustworthiness", f"{metrics['ae']['trust']:.2f}", delta=f"{metrics['ae']['trust'] - metrics['umap']['trust']:+.2f} ggü. UMAP", delta_color="normal",
          help=f"Nachbarschaft erhalten: Anteil der Nachbarn in der 2-D-Einbettung, die auch im Originalraum Nachbarn sind (k = {C.TRUST_NEIGHBORS}); 1 = perfekt.")

names = (("ae", "Autoencoder", "Code", model.embedding, z_color), ("pca", "PCA", "PC", analysis.pca_2d, z_color), ("isomap", "Isomap (Vergleich)", "Isomap", analysis.iso_2d, z_color[iso_idx]),
         ("tsne", "t-SNE (Vergleich)", "t-SNE", analysis.tsne.embedding, z_color), ("umap", "UMAP (Vergleich)", "UMAP", analysis.umap.embedding, z_color),
         ("pacmap", "PaCMAP (Vergleich)", "PaCMAP", analysis.pacmap.embedding, z_color))
row1 = st.columns(3)
row2 = st.columns(3)
for slot, (key, title, axis, coords, color) in zip(row1 + row2, names):
    with slot:
        st.markdown(f"**{title}**")
        st.plotly_chart(build_embedding(coords, color, f"{axis} 1" if axis in ("Code", "PC") else f"{axis}-Koordinate 1", f"{axis} 2" if axis in ("Code", "PC") else f"{axis}-Koordinate 2"), width="stretch", key=f"{key}_embedding")

order = ("ae", "pca", "isomap", "tsne", "umap", "pacmap")
labels = {"ae": "Autoencoder", "pca": "PCA", "isomap": "Isomap", "tsne": "t-SNE", "umap": "UMAP", "pacmap": "PaCMAP"}
st.table({
    "Verfahren": [labels[k] for k in order],
    "R² der Faktoren": [f"{metrics[k]['r2']:.2f}" for k in order],
    "Abstandstreue (gesamt)": [f"{metrics[k]['fid']:.2f}" for k in order],
    "nahe Paare": [f"{metrics[k]['near']:.2f}" for k in order],
    "ferne Paare": [f"{metrics[k]['far']:.2f}" for k in order],
    "Trustworthiness": [f"{metrics[k]['trust']:.2f}" for k in order],
    "Rekonstruktionsfehler": [f"{analysis.mse_ae:.3f}", f"{analysis.mse_pca:.3f}", "–", "–", "–", "–"],
})

st.markdown("---")

# --- Architektur, Lernrate, Epochen -----------------------------------------------------------------------------------------

st.subheader("📐 Wie stark hängt das Ergebnis von Architektur, Lernrate und Epochen ab?")
st.markdown(
    """
Anders als die Verfahren mit Nachbarschaftsgraph hat das Autoencoder **viele Stellschrauben, und sein Ergebnis hängt zusätzlich vom zufälligen Start ab**. Live für Ihr aktuelles Szenario über **feste Sweep-Datensätze**
(3 Datensätze × 2 Starts, unabhängig vom Demo-Seed) geprüft; die Fehlerbalken zeigen die Streuung, nicht nur den Mittelwert:
"""
)
if code == "lr_high":
    st.warning(
        f"⚠️ **Lernrate zu hoch**: das Training springt über das Minimum - der Rekonstruktionsfehler bleibt bei {vd['mse']:.3f}, ein Referenzlauf mit Lernrate {C.DEFAULT_LR:g} und {C.REFERENCE_EPOCHS} Epochen erreicht {vd['mse_ref']:.3f} "
        f"(PCA-Untergrenze {vd['mse_pca']:.3f}). R² der Faktoren {vd['r2']:.2f} gegen {vd['r2_ref']:.2f}."
    )
elif code == "not_trained":
    st.warning(
        f"⚠️ **Noch nicht trainiert**: nach {vd['n_epochs']} Epochen liegt der Rekonstruktionsfehler bei {vd['mse']:.3f}, ein Referenzlauf mit {C.REFERENCE_EPOCHS} Epochen erreicht {vd['mse_ref']:.3f} "
        f"(PCA-Untergrenze {vd['mse_pca']:.3f}). Das R² der Faktoren ({vd['r2']:.2f} gegen {vd['r2_ref']:.2f}) leidet weniger als der Rekonstruktionsfehler."
    )
elif code == "bottleneck_narrow":
    st.warning(
        f"⚠️ **Zu schmale versteckte Schicht**: bei Breite {vd['width']} passt nicht genug Information durch - der Rekonstruktionsfehler ({vd['mse']:.3f}) ist nicht besser als die PCA-Untergrenze ({vd['mse_pca']:.3f}). "
        "Ab Breite 8 rekonstruiert das Netz deutlich besser."
    )
elif code == "linear_pca":
    angle = f"{vd['angle']:.4f}" if vd["angle"] is not None else "-"
    st.info(
        f"ℹ️ **Lineares Autoencoder = PCA**: der Rekonstruktionsfehler ({vd['mse']:.3f}) erreicht die PCA-Untergrenze ({vd['mse_pca']:.3f}), der Decoder-Unterraum weicht nur um {angle}° vom PCA-Unterraum ab. "
        f"R² der Faktoren {vd['r2']:.2f} - wie die PCA ({vd['r2_pca']:.2f}). Erst versteckte Schichten (Tiefe ≥ 1) machen es nichtlinear."
    )
elif code == "extremes_kept":
    st.success(
        f"✅ **Extreme bleiben erhalten**: mit {vd['outlier_pct']} % Sonderfahrten liegt das R² der Faktoren bei {vd['r2']:.2f} - t-SNE {vd['r2_tsne']:.2f}, UMAP {vd['r2_umap']:.2f} und PaCMAP {vd['r2_pacmap']:.2f} brechen ein. "
        f"PCA ({vd['r2_pca']:.2f}) und Isomap ({vd['r2_iso']:.2f}) bleiben besser. Als glatte parametrische Abbildung staucht das Netz die Extreme nicht an den Rand der Wolke. (Im Test galt das bei 5-10 %, nicht bei 2 %.)"
    )
elif code == "no_advantage":
    st.info(f"ℹ️ **Kein Vorteil vor der PCA**: die Daten sind gerade (Krümmung 0) - R² der Faktoren {vd['r2']:.2f} (Autoencoder) gegen {vd['r2_pca']:.2f} (PCA); der Rekonstruktionsfehler {vd['mse']:.3f} gegen {vd['mse_pca']:.3f}.")
elif code == "ae_wins":
    leads = vd["r2"] >= max(vd["r2_umap"], vd["r2_tsne"], vd["r2_pacmap"]) - 0.02
    tail = (
        f"Auf diesem Datensatz und mit diesem Start liegt es beim R² sogar vorn (UMAP {vd['r2_umap']:.2f}, t-SNE {vd['r2_tsne']:.2f}, PaCMAP {vd['r2_pacmap']:.2f}, Isomap {vd['r2_iso']:.2f}) - im Mittel über vier Datensätze aber nicht "
        "(0.72 gegen UMAP 0.92), und ein anderer Start kann das Bild deutlich verändern (siehe Stabilität unten)."
        if leads else
        f"Die Nachbarschaftsverfahren finden die Faktoren allerdings besser zurück (UMAP {vd['r2_umap']:.2f}, t-SNE {vd['r2_tsne']:.2f}, PaCMAP {vd['r2_pacmap']:.2f}, Isomap {vd['r2_iso']:.2f}) - und das Ergebnis hängt vom Start ab (siehe Stabilität unten)."
    )
    st.success(f"✅ **Das Netz entfaltet die Fläche**: Rekonstruktionsfehler {vd['mse']:.3f} gegen {vd['mse_pca']:.3f} (PCA-Untergrenze), R² der Faktoren {vd['r2']:.2f} gegen {vd['r2_pca']:.2f} bei der PCA. " + tail)
else:
    st.info(f"Das Autoencoder erreicht R² {vd['r2']:.2f} (PCA {vd['r2_pca']:.2f}), Rekonstruktionsfehler {vd['mse']:.3f} (PCA-Untergrenze {vd['mse_pca']:.3f}) - kein klarer Gewinn und kein klarer Bruch.")

for title, rows_, xkey, current, log, key in (("Tiefe", depth_rows, "depth", float(depth), False, "depth"), ("Breite der versteckten Schichten", width_rows, "width", float(width), True, "width"),
                                                ("Lernrate", lr_rows, "lr", float(lr), True, "lr")):
    st.markdown(f"**{title}**")
    st.plotly_chart(build_sweep(rows_, xkey, current, title, analysis.mse_pca, log=log), width="stretch", key=f"{key}_sweep")
st.caption(
    f"Gleiche Daten-Einstellungen (q = {dataset.q}, Krümmung {curvature:.2f}, Rauschen {noise:.2f}, Sonderfahrten {int(outlier_pct)} %), jeweils nur der geprüfte Regler wächst, alles Übrige Standard "
    f"({C.DEFAULT_DEPTH} / {C.DEFAULT_WIDTH} / {C.DEFAULT_ACTIVATION} / {C.DEFAULT_LR:g}), {C.SWEEP_N_EPOCHS} Epochen, {C.SWEEP_N_TOURS} Touren. Im Test: Tiefe 0 = PCA-Niveau, Tiefe 2 am besten, Tiefe 3 beim R² schlechter; "
    "Breite ab 8 rekonstruiert gut, sehr breite Netze finden die Faktoren schlechter; Lernrate 0.3 bricht das Training. Die Streuung des R² (Fehlerbalken) ist groß: **ein einzelner Lauf sagt wenig**."
)

st.markdown("---")

# --- Linear = PCA -----------------------------------------------------------------------------------------------------------

st.markdown("## 🔁 Linear = PCA")
st.caption(
    "Ein Autoencoder ohne versteckte Schichten mit quadratischem Verlust lernt denselben 2-D-**Unterraum** wie die PCA (nicht dieselben Achsen). Live nachgemessen: ein solches lineares Netz wird auf Ihren Daten trainiert, "
    "und der größte **Hauptwinkel** zwischen dem vom Decoder aufgespannten Unterraum und den ersten beiden PCA-Achsen fällt gegen 0."
)
lin = _linear(data_params)
st.plotly_chart(build_angle_curve(lin["angles"]), width="stretch", key="angle_curve")
final_angle = list(lin["angles"].values())[-1]
st.caption(
    f"Nach {max(lin['angles'])} Epochen: größter Hauptwinkel {final_angle:.4f}°, Rekonstruktionsfehler {lin['mse']:.4f} gegen PCA-Untergrenze {lin['mse_pca']:.4f} (Verhältnis {lin['ratio']:.4f}). "
    "Zu Beginn (Zufallsgewichte) liegt der Winkel bei etwa 80°. Über vier Datensätze lag der Winkel nach 3000 Epochen bei höchstens 0.014°, das Fehlerverhältnis bei 1.000."
)

st.markdown("---")

# --- Decoder ----------------------------------------------------------------------------------------------------------------

st.markdown("## 🎨 Der Decoder: aus dem 2-D-Raum zurück in Touren")
st.caption(
    "Das einzige Verfahren der Linie mit echtem Decoder: jeder Punkt des 2-D-Raums lässt sich in eine (synthetische) Tour zurückübersetzen. Unten ein Gitter über den Wertebereich der Trainings-Codes, für jede gewählte Kennzahl als Heatmap "
    "(Originaleinheiten; Zellen weit weg von allen Trainings-Codes sind ausgeblendet, weiße Punkte = die Touren)."
)
shown = st.multiselect("Kennzahlen", list(FEATURE_NAMES), default=["Distanz", "Stopps", "Verspätung", "Fahrzeit je km"], key="decoder_features", max_selections=4)
if shown:
    xs, ys, tours, dist_to_data = decoder_grid(model)
    cell = float(np.median(np.diff(xs))) * 2.5
    panels = [(f"{name} [{C.FEATURES[list(FEATURE_NAMES).index(name)][1]}]", tours[:, list(FEATURE_NAMES).index(name)]) for name in shown]
    st.plotly_chart(build_decoder_grid(xs, ys, panels, model.embedding, dist_to_data, cell), width="stretch", key="decoder_grid")
else:
    st.info("Bitte mindestens eine Kennzahl wählen.")
st.markdown("**Rekonstruktionsfehler je Tour**")
st.plotly_chart(build_error_map(model.embedding, analysis.errors, dataset.outlier), width="stretch", key="error_map")
auc = anomaly_auc(analysis.errors, dataset.outlier)
st.caption(
    "Touren, die der Decoder schlecht nachbauen kann, haben große Fehler (warm gefärbt). "
    + (f"Ihre {int(dataset.outlier.sum())} Sonderfahrten (schwarz umrandet) lassen sich am Fehler mit einer AUC von **{auc:.2f}** erkennen (1 = perfekt, 0.5 = Zufall). Über 4 Datensätze streute die AUC bei 5 % Sonderfahrten zwischen 0.60 und 0.92, "
       "bei 2 % zwischen 0.36 und 0.86, bei 10 % zwischen 0.73 und 0.92 - als Anomalie-Detektor ist das nicht verlässlich." if auc is not None else "Ohne Sonderfahrten gibt es nichts zu erkennen - der Regler links fügt welche hinzu.")
)

st.markdown("---")

# --- Vergleich --------------------------------------------------------------------------------------------------------------

st.markdown("## 🆚 Was der Autoencoder gegenüber den anderen ändert - gemessen")
st.markdown(
    """
| Frage | Ergebnis im Test (300 Touren, 4 feste Datensätze, wenn nicht anders angegeben) |
|---|---|
| **Rekonstruktion / Decoder** | ✅ Fehler **0.021** gegen PCA-Untergrenze 0.456 (mehr als 20-fach kleiner); die anderen Verfahren haben keinen Decoder |
| **Linear = PCA** | ✅ Hauptwinkel ≤ 0.014°, Fehlerverhältnis 1.000 |
| **Neue Touren einbetten** | ✅ parametrisch: 1000 neue Touren in etwa 1 ms, die Trainings-Einbettung bleibt unverändert. ⚠️ Aber: R² der neuen Touren 0.62-0.75 (UMAP `transform` 0.88-0.93); Rekonstruktionsfehler neuer Touren 0.10-0.36 gegen 0.02 im Training - **Datenhunger** |
| **Extreme (Sonderfahrten)** | ✅/⚠️ bei 5 %: R² **0.58** (UMAP 0.36, t-SNE 0.35, PaCMAP 0.30; PCA 0.67, Isomap 0.77), bei 10 %: 0.74 (0.24 / 0.26 / 0.22) - aber bei 2 %: 0.51 (0.62 / 0.66 / 0.58) |
| **Faktoren zurückgewinnen** | ❌ R² **0.72** gegen UMAP 0.92, t-SNE 0.89, PaCMAP 0.89 (PCA 0.51); Streuung ±0.12 über Datensätze und Starts; bei Rauschen 0.8: 0.54 (0.90 / 0.80 / 0.88) |
| **Stabilität** (Start egal) | ❌ mittlere paarweise Abweichung zufälliger Starts (200 Touren, q = 2): Autoencoder 0.42-0.63, t-SNE 0.37-0.53, PaCMAP 0.10-0.43, UMAP 0.01-0.22 |
| **Globale Struktur** | ⚠️ ferne Paare 0.45 (UMAP 0.60, t-SNE 0.58, PaCMAP 0.34, Isomap 0.89) |
| **Rechenzeit** | ⚠️ Training 1.1 s bei n = 600 (PaCMAP 0.4 s, UMAP 1.3 s, t-SNE 4.8 s); dafür danach fast kostenlos |

Die Experimente mit Knopf unten prüfen Out-of-sample, Stabilität und Rechenzeit für Ihre Einstellungen nach.
"""
)

rng = np.random.default_rng(0)
m_iso = len(iso_idx)
pa = rng.integers(0, m_iso, size=min(1500, m_iso * (m_iso - 1) // 2))
pb = rng.integers(0, m_iso, size=len(pa))
keep = pa != pb
pa, pb = pa[keep], pb[keep]
ga, gb = iso_idx[pa], iso_idx[pb]


def _norm(d):
    return d / d.mean()


latent = _norm(np.linalg.norm(dataset.z[ga] - dataset.z[gb], axis=1))
ae_d = _norm(np.linalg.norm(model.embedding[ga] - model.embedding[gb], axis=1))
umap_d = _norm(np.linalg.norm(analysis.umap.embedding[ga] - analysis.umap.embedding[gb], axis=1))
iso_d = _norm(np.linalg.norm(analysis.iso_2d[pa] - analysis.iso_2d[pb], axis=1))
st.markdown("**📏 Globale Struktur: Abstände**")
st.plotly_chart(build_distance_fidelity(latent, [("Autoencoder", ae_d, "#1f77b4"), ("UMAP", umap_d, "#8e5fbf"), ("Isomap", iso_d, "#8c564b")]), width="stretch", key="distance_fidelity")
st.caption(
    f"Jeder Punkt ein Tourenpaar: Abstand in der 2-D-Einbettung gegen den Abstand der wahren Faktoren; auf der gestrichelten Diagonale wäre die Einbettung abstandstreu. Korrelation für **nahe** Paare: "
    f"Autoencoder {metrics['ae']['near']:.2f}, UMAP {metrics['umap']['near']:.2f}, Isomap {metrics['isomap']['near']:.2f} - für **ferne** Paare: Autoencoder {metrics['ae']['far']:.2f}, UMAP {metrics['umap']['far']:.2f}, "
    f"Isomap {metrics['isomap']['far']:.2f}."
)

st.markdown("**🆕 Neue Touren einbetten (Out-of-sample)**")
st.caption(
    "Der Encoder ist eine feste Abbildung: neue Touren werden einfach hindurchgeschickt, ohne Neuberechnung und ohne dass sich die bereits eingebetteten Touren verschieben (bei UMAP, PaCMAP und t-SNE ist das anders). "
    f"Test: die letzten {C.HOLDOUT_FRACTION * 100:.0f} % der Touren zurückhalten, das Netz auf den übrigen trainieren; gemessen wird auch der Rekonstruktionsfehler der neuen Touren gegen den der Trainings-Touren (Generalisierungslücke)."
)
if st.button("🆕 Neue Touren testen", key="oos_start"):
    st.session_state["oos_on"] = True
if st.session_state.get("oos_on"):
    with st.spinner("Trainiere Autoencoder, PaCMAP, UMAP und t-SNE auf 80 % der Touren..."):
        oos = _oos(data_params, settings)
    tc, ec = dataset.z[oos["train"], 0], dataset.z[oos["test"], 0]
    st.plotly_chart(build_out_of_sample([("Autoencoder: encode", oos["model"].embedding, oos["y_ae"]), ("UMAP: transform", oos["umap_train"], oos["y_umap"]),
                                         ("PaCMAP: transform (Behelf)", oos["pacmap_train"], oos["y_pacmap"]), ("t-SNE: Näherung", oos["tsne_train"], oos["y_tsne"])], tc, ec), width="stretch", key="oos_plot")
    st.caption(
        f"Sterne = zurückgehaltene Touren. R² der wahren Faktoren für die neuen Touren: **Autoencoder {oos['r2_ae']:.2f}**, UMAP {oos['r2_umap']:.2f}, PaCMAP {oos['r2_pacmap']:.2f}, t-SNE-Näherung {oos['r2_tsne']:.2f} "
        f"(Trainings-Touren Autoencoder: {oos['r2_train']:.2f}). Rekonstruktionsfehler: Training {oos['mse_train']:.3f}, neue Touren **{oos['mse_test']:.3f}** - das {oos['mse_test'] / max(oos['mse_train'], 1e-9):.1f}-fache: "
        "das Netz kennt die Fläche nur dort gut, wo es Touren gesehen hat."
    )

st.markdown("**🔀 Stabilität: Autoencoder, PaCMAP, UMAP und t-SNE auf denselben Datensätzen**")
st.caption(
    "Wie stark hängt das Bild vom zufälligen Start ab? Je Datensatz vier zufällige Starts (beim Autoencoder: Anfangsgewichte); gemessen wird der **mittlere paarweise Procrustes-Abstand** (nach bester Drehung/Spiegelung; "
    "0 = gleiches Bild, 1 = unabhängig) - für alle Verfahren auf denselben vier festen Datensätzen mit 200 Touren und Ihren Datenreglern; das Autoencoder mit Ihren Einstellungen, die anderen mit ihren Standards. Dauert etwa 25 Sekunden."
)
if st.button("🔀 Stabilität vergleichen", key="stability_start"):
    st.session_state["stability_on"] = True
if st.session_state.get("stability_on"):
    with st.spinner("Rechne 64 Läufe..."):
        stab = _stability(int(q), float(curvature), float(noise), int(outlier_pct), settings)
    st.plotly_chart(build_stability_compare(stab), width="stretch", key="stability_plot")
    st.caption(
        "Mittlere paarweise Abstände (Autoencoder / PaCMAP / UMAP / t-SNE) je Datensatz: " + "; ".join(f"{r['ae']:.2f} / {r['pacmap']:.2f} / {r['umap']:.2f} / {r['tsne']:.2f}" for r in stab)
        + f". Im Mittel: Autoencoder {np.mean([r['ae'] for r in stab]):.2f}, PaCMAP {np.mean([r['pacmap'] for r in stab]):.2f}, UMAP {np.mean([r['umap'] for r in stab]):.2f}, t-SNE {np.mean([r['tsne'] for r in stab]):.2f}."
    )

st.markdown("**⏱️ Rechenzeit**")
st.caption(
    "Das Training braucht viele Durchläufe über alle Touren (Vorwärts- und Rückwärtsrechnung durch jede Schicht), ist aber eine einzige Rechnung ohne n×n-Matrizen; danach kostet jede neue Tour nur einen Vorwärtsdurchlauf. "
    "UMAP baut zusätzlich einen Graphen, t-SNE berechnet je Iteration alle n² Paare."
)
if "timing_rows" not in st.session_state:
    if st.button("⏱️ Rechenzeit messen (ca. 15 s)", key="timing_start", help=f"Misst das Training des Autoencoders und die anderen Verfahren für n = {', '.join(str(n) for n in C.TIMING_NS)} auf diesem Rechner."):
        with st.spinner("Messe..."):
            st.session_state["timing_rows"] = timing_sweep()
        st.rerun()
else:
    rows = st.session_state["timing_rows"]
    st.plotly_chart(build_timing(rows), width="stretch", key="timing_chart")
    st.table({
        "Touren n": [r["n"] for r in rows],
        "Autoencoder (Training)": [f"{r['ae']:.2f} s" for r in rows],
        "PaCMAP": [f"{r['pacmap']:.2f} s" for r in rows],
        "UMAP": [f"{r['umap']:.2f} s" for r in rows],
        "t-SNE": [f"{r['tsne']:.2f} s" for r in rows],
        "Isomap": [f"{r['isomap']:.3f} s" for r in rows],
        "PCA": [f"{r['pca'] * 1000:.2f} ms" for r in rows],
        "1000 neue Touren (Autoencoder)": [f"{r['encode'] * 1000:.2f} ms" for r in rows],
    })
    st.caption("Gemessen auf diesem Rechner (Wandzeit, ein Lauf je n, Standard-Einstellungen aller Verfahren): die Faktoren hängen von Rechner und Zwischenspeichern ab.")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Netz.** Eingabe $x \in \mathbb{R}^{12}$ (z-Werte). Encoder $f$ und Decoder $g$ sind Hintereinanderschaltungen affiner Abbildungen $h \mapsto h W_\ell + b_\ell$ mit Aktivierung $\sigma$ (tanh, ReLU oder sigmoid) nach jeder Schicht
**außer** dem Engpass (2 Neuronen, linear) und der Ausgabe (linear): $c = f(x) \in \mathbb{R}^2$, $\hat{x} = g(c) \in \mathbb{R}^{12}$. Bei Tiefe 0 ist $\hat{x} = (x W_1 + b_1) W_2 + b_2$.

**Verlust.** Mittlerer quadratischer Rekonstruktionsfehler $L = \frac{1}{12\,n} \sum_{i=1}^{n} \lVert x_i - g(f(x_i)) \rVert^2$.

**Backpropagation.** Mit $\delta_L = \frac{2}{12\,n}(\hat{X} - X)$ läuft der Fehler rückwärts: $\partial L/\partial W_\ell = H_{\ell-1}^\top \delta_\ell$, $\delta_{\ell-1} = (\delta_\ell W_\ell^\top) \odot \sigma'(H_{\ell-1})$
(ohne $\sigma'$ an Engpass und Ausgabe). Adam: $m \leftarrow \beta_1 m + (1-\beta_1) g$, $v \leftarrow \beta_2 v + (1-\beta_2) g^2$, $\theta \leftarrow \theta - \eta_t\, m / (\sqrt{v} + \varepsilon)$. Die Gradienten sind im Test per
finite Differenzen für alle Aktivierungen und Tiefen geprüft. Initialisierung: Xavier (tanh, sigmoid, linear) bzw. He (ReLU).

**Linear = PCA.** Für Tiefe 0 und quadratischen Verlust ist jedes Minimum von $L$ ein Paar $(W_1, W_2)$, dessen Bild denselben 2-D-Unterraum aufspannt wie die ersten zwei Hauptachsen $V_2$ (Baldi & Hornik, 1989); der minimale Fehler
ist die Summe der übrigen Eigenwerte, $\frac{1}{12\,n}\sum_{k>2} s_k^2$ (Eckart-Young). **Hauptwinkel** zwischen zwei Unterräumen: $\arccos$ der Singulärwerte von $Q_A^\top Q_B$ ($Q$ = orthonormale Basen).

**Neue Punkte.** $c_\text{neu} = f(x_\text{neu})$ - keine Optimierung, die Trainings-Codes bleiben unverändert.

**Grenzen.** (1) *Kein Nachbarschafts- oder Abstandsziel*: der Code muss nur rekonstruierbar sein, nicht die wahren Faktoren zeigen - R² und Nachbarschaft können schwanken. (2) *Lokale Minima und Zufallsstart*: verschiedene Anfangsgewichte liefern
verschiedene Einbettungen (Demo: Stabilität). (3) *Datenhunger*: viele Parameter, wenige Touren - der Fehler neuer Touren liegt ein Mehrfaches über dem der Trainings-Touren (Demo: Out-of-sample). (4) *Rauschen und Extreme*: das Netz lernt
beides mit; Sonderfahrten erschweren das Training (Demo). (5) Vollbatch-Training ohne Regularisierung und ohne Validierungsabbruch - für große Daten braucht man Mini-Batches. (6) Achsen des Codes haben keine feste Bedeutung.

**Trustworthiness** (Venna & Kaski, 2001): $T = 1 - \frac{2}{nk(2n - 3k - 1)} \sum_i \sum_{j \in U_i} (r(i,j) - k)$ mit $U_i$ = Nachbarn in der Einbettung, die im Originalraum keine sind, und $r(i,j)$ ihrem Originalrang.
**Abstandstreue** = Pearson-Korrelation der Paarabstände der 2-D-Einbettung mit den Paarabständen der wahren Faktoren (nah/fern: untere/obere Hälfte der wahren Abstände). **Procrustes-Abstand**: $1 - (\sum s_i)^2$ mit $s_i$ den
Singulärwerten von $A^\top B$ nach Zentrierung und Normierung beider Einbettungen (wie `scipy.spatial.procrustes`).

Implementiert in `ae_algorithm.py` (Netz, Backpropagation, Adam, Encode/Decode, Hauptwinkel), `ae_isomap.py` / `ae_tsne.py` / `ae_umap.py` / `ae_pacmap.py` (Vergleichsverfahren, wortgleich aus den Demos davor),
`ae_scenario.py` (Lieferrouten-Generator, wortgleich aus pca-demo) und `ae_evaluation.py` (Kennzahlen, Sweeps, Verdict, Stabilität, Out-of-sample, Zeitmessung).
        """
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
