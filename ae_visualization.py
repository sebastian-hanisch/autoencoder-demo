"""Plotly-Visualisierungen der Autoencoder-Demo: Architektur, eine Tour hindurch, Einbettungen, Trainingsverlauf, Sweeps, Hauptwinkel zur PCA, Decoder-Gitter, Fehlerkarte, Abstandstreue, Stabilitätsvergleich,
Out-of-sample und Rechenzeit. Alle Figuren laufen durch `lock_axes` (Touch-Scrolling-Konvention des Portfolios: keine Zoom-/Pan-Gesten im Chart)."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BLUE, ORANGE, GREEN, RED, GRAY, PURPLE = "#1f77b4", "#d68a2e", "#2ca02c", "#d62728", "#8a8f98", "#8e5fbf"


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _scatter(coords, color, name="Touren", size=7, showscale=False, label="latenter Faktor 1", opacity=1.0):
    return go.Scatter(
        x=coords[:, 0], y=coords[:, 1], mode="markers", name=name, hoverinfo="skip",
        marker=dict(color=color, colorscale="Viridis", size=size, showscale=showscale, opacity=opacity, line=dict(width=0.5, color="white"),
                    colorbar=dict(title=label) if showscale else None),
    )


def build_architecture(sizes, depth, n_parameters):
    """Schichtdiagramm: Encoder (blau), Engpass (orange), Decoder (grün). Je Schicht höchstens 10 Knoten gezeichnet, darunter die echte Breite."""
    n_layers = len(sizes)
    xs = np.arange(n_layers)
    fig = go.Figure()
    shown = [min(s, 10) for s in sizes]
    pos = [(x, np.linspace(-(k - 1) / 2, (k - 1) / 2, k)) for x, k in zip(xs, shown)]
    lx, ly = [], []
    for (x0, y0), (x1, y1) in zip(pos[:-1], pos[1:]):
        for a in y0:
            for b in y1:
                lx += [x0, x1, None]
                ly += [a, b, None]
    fig.add_trace(go.Scatter(x=lx, y=ly, mode="lines", line=dict(color="rgba(120,120,120,0.10)", width=1), hoverinfo="skip", showlegend=False))
    for i, (x, ys) in enumerate(pos):
        color = ORANGE if i == depth + 1 else (BLUE if i <= depth else GREEN)
        fig.add_trace(go.Scatter(x=[x] * len(ys), y=ys, mode="markers", marker=dict(color=color, size=13, line=dict(width=1, color="white")), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=xs, y=[-6.2] * n_layers, mode="text", text=[str(s) for s in sizes], textposition="middle center", hoverinfo="skip", showlegend=False))
    labels = {0: "Eingabe (12 Kennzahlen)", depth + 1: "Engpass (2 Werte)", n_layers - 1: "Rekonstruktion"}
    for i, text in labels.items():
        fig.add_annotation(x=i, y=6.3, text=text, showarrow=False, font=dict(size=11))
    fig.add_annotation(x=(depth) / 2, y=7.3, text="Encoder", showarrow=False, font=dict(size=13, color=BLUE))
    fig.add_annotation(x=depth + 1 + (n_layers - 1 - depth - 1) / 2, y=7.3, text="Decoder", showarrow=False, font=dict(size=13, color=GREEN))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, range=[-7, 8])
    fig.update_layout(template="plotly_white", height=360, margin=dict(l=10, r=10, t=10, b=10), title=dict(text=f"{n_parameters:,} Parameter".replace(",", "."), x=0.99, xanchor="right", y=0.04, font=dict(size=12)))
    return lock_axes(fig)


def build_tour_reconstruction(names, original, reconstructed, pca_reconstructed):
    """Eine Tour: die 12 Kennzahlen (z-Werte) vor und nach dem Durchgang durch Encoder und Decoder - dazu die Rekonstruktion der PCA mit ebenfalls 2 Komponenten."""
    fig = go.Figure()
    fig.add_trace(go.Bar(y=names, x=original, orientation="h", name="Original", marker_color=BLUE, hovertemplate="%{y}: %{x:.2f}<extra>Original</extra>"))
    fig.add_trace(go.Bar(y=names, x=reconstructed, orientation="h", name="Autoencoder", marker_color=ORANGE, hovertemplate="%{y}: %{x:.2f}<extra>Autoencoder</extra>"))
    fig.add_trace(go.Bar(y=names, x=pca_reconstructed, orientation="h", name="PCA (2 Komponenten)", marker_color=GRAY, hovertemplate="%{y}: %{x:.2f}<extra>PCA</extra>"))
    fig.update_xaxes(title="z-Wert")
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(template="plotly_white", barmode="group", height=480, margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=-0.15))
    return lock_axes(fig)


def build_embedding(coords, color, title_x, title_y, label="latenter Faktor 1", height=380):
    fig = go.Figure(_scatter(coords, color, showscale=True, label=label))
    fig.update_xaxes(title=title_x)
    fig.update_yaxes(title=title_y)
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=10, t=20, b=10))
    return lock_axes(fig)


def build_training(loss_history, mse_floor, snapshot_r2, snapshot_far, marker=None):
    """Trainingsverlauf: Rekonstruktionsfehler je Epoche (log) mit der PCA-Untergrenze, R² der wahren Faktoren und Abstandstreue ferner Paare an den Schnappschüssen."""
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Rekonstruktionsfehler (MSE)", "R² der wahren Faktoren", "Abstandstreue ferner Paare"))
    epochs = np.arange(1, len(loss_history) + 1)
    fig.add_trace(go.Scatter(x=epochs, y=loss_history, mode="lines", line=dict(color=BLUE, width=3), showlegend=False, hovertemplate="Epoche %{x}: %{y:.3f}<extra></extra>"), row=1, col=1)
    fig.add_hline(y=mse_floor, line_dash="dash", line_color=RED, row=1, col=1, annotation_text="PCA-Untergrenze", annotation_position="top right")
    r2, far = sorted(snapshot_r2.items()), sorted(snapshot_far.items())
    fig.add_trace(go.Scatter(x=[max(e, 1) for e, _ in r2], y=[v for _, v in r2], mode="lines+markers", line=dict(color=PURPLE, width=3), showlegend=False, hovertemplate="Epoche %{x}: %{y:.2f}<extra></extra>"), row=1, col=2)
    fig.add_trace(go.Scatter(x=[max(e, 1) for e, _ in far], y=[v for _, v in far], mode="lines+markers", line=dict(color=ORANGE, width=3), showlegend=False, hovertemplate="Epoche %{x}: %{y:.2f}<extra></extra>"), row=1, col=3)
    for col in (1, 2, 3):
        if marker is not None:
            fig.add_vline(x=max(marker, 1), line_dash="dot", line_color=GRAY, row=1, col=col)
    fig.update_xaxes(title_text="Epoche", type="log")
    fig.update_yaxes(type="log", col=1)
    fig.update_yaxes(range=[-0.3, 1.02], col=2)
    fig.update_yaxes(range=[-0.3, 1.02], col=3)
    fig.update_layout(template="plotly_white", height=330, margin=dict(l=10, r=10, t=40, b=10))
    return lock_axes(fig)


def build_sweep(rows, xkey, current, xtitle, mse_floor, log=False):
    """Sweep über einen Regler: R² der Faktoren (± Streuung über Datensätze und Starts) und Rekonstruktionsfehler (log, mit PCA-Untergrenze)."""
    xs = [r[xkey] for r in rows]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("R² der wahren Faktoren (± Streuung)", "Rekonstruktionsfehler (log)"))
    fig.add_trace(go.Scatter(x=xs, y=[r["r2"] for r in rows], error_y=dict(type="data", array=[r["r2_std"] for r in rows], visible=True), mode="lines+markers", line=dict(color=PURPLE, width=3), name="R²"), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs, y=[r["mse"] for r in rows], mode="lines+markers", line=dict(color=BLUE, width=3), name="MSE"), row=1, col=2)
    fig.add_hline(y=mse_floor, line_dash="dash", line_color=RED, row=1, col=2, annotation_text="PCA-Untergrenze", annotation_position="top right")
    for col in (1, 2):
        if current is not None:
            fig.add_vline(x=current, line_dash="dot", line_color=GRAY, row=1, col=col)
    fig.update_xaxes(title_text=xtitle, tickvals=xs, ticktext=[f"{x:g}" for x in xs], type="log" if log else "linear")
    fig.update_yaxes(range=[-0.1, 1.1], col=1)
    fig.update_yaxes(type="log", col=2)
    fig.update_layout(template="plotly_white", height=330, margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
    return lock_axes(fig)


def build_angle_curve(angles):
    """Größter Hauptwinkel (Grad) zwischen dem Decoder-Unterraum des linearen Autoencoders und dem PCA-Unterraum gegen die Epoche (log-log): er läuft gegen 0."""
    items = sorted(angles.items())
    fig = go.Figure(go.Scatter(x=[max(e, 1) for e, _ in items], y=[max(a, 1e-4) for _, a in items], mode="lines+markers", line=dict(color=ORANGE, width=3), hovertemplate="Epoche %{x}: %{y:.3f}°<extra></extra>"))
    fig.update_xaxes(title="Epoche", type="log")
    fig.update_yaxes(title="größter Hauptwinkel (Grad)", type="log")
    fig.update_layout(template="plotly_white", height=300, margin=dict(l=10, r=10, t=20, b=10), showlegend=False)
    return lock_axes(fig)


def build_decoder_grid(xs, ys, panels, embedding, distances, max_distance):
    """Decoder auf einem Gitter des 2-D-Raums: je Kennzahl eine Heatmap (Original-Einheiten), Zellen weit entfernt von Trainings-Codes ausgeblendet; darüber die Trainings-Touren. `panels`: [(Titel, Werte [n·n])]."""
    n = len(xs)
    fig = make_subplots(rows=1, cols=len(panels), subplot_titles=[p[0] for p in panels], horizontal_spacing=0.06)
    mask = (distances > max_distance).reshape(n, n)
    for col, (_, values) in enumerate(panels, start=1):
        z = np.where(mask, np.nan, values.reshape(n, n))
        fig.add_trace(go.Heatmap(x=xs, y=ys, z=z, colorscale="Viridis", showscale=False, hoverongaps=False, hovertemplate="(%{x:.2f}, %{y:.2f}): %{z:.2f}<extra></extra>"), row=1, col=col)
        fig.add_trace(go.Scatter(x=embedding[:, 0], y=embedding[:, 1], mode="markers", marker=dict(color="rgba(255,255,255,0.55)", size=3), hoverinfo="skip", showlegend=False), row=1, col=col)
    fig.update_xaxes(title_text="Code 1")
    fig.update_yaxes(title_text="Code 2", col=1)
    fig.update_layout(template="plotly_white", height=330, margin=dict(l=10, r=10, t=40, b=10))
    return lock_axes(fig)


def build_error_map(embedding, errors, outlier_mask):
    """Einbettung, gefärbt nach dem Rekonstruktionsfehler je Tour (log); Sonderfahrten mit Rand."""
    fig = go.Figure(go.Scatter(x=embedding[:, 0], y=embedding[:, 1], mode="markers", hoverinfo="skip",
                               marker=dict(color=np.log10(np.maximum(errors, 1e-6)), colorscale="Turbo", size=7, showscale=True, colorbar=dict(title="log10 MSE"), line=dict(width=0.4, color="white"))))
    if outlier_mask.any():
        fig.add_trace(go.Scatter(x=embedding[outlier_mask, 0], y=embedding[outlier_mask, 1], mode="markers", name="Sonderfahrten", hoverinfo="skip",
                                 marker=dict(color="rgba(0,0,0,0)", size=13, line=dict(width=2, color="#111111"))))
    fig.update_xaxes(title="Code 1")
    fig.update_yaxes(title="Code 2")
    fig.update_layout(template="plotly_white", height=380, margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=-0.2))
    return lock_axes(fig)


def build_distance_fidelity(latent_pairs, panels):
    """Paarabstände der 2-D-Einbettung gegen die Abstände der wahren Faktoren (je auf Mittelwert 1 normiert). `panels`: [(Titel, Abstände, Farbe)]; auf der Diagonalen ist die Einbettung abstandstreu."""
    fig = make_subplots(rows=1, cols=len(panels), subplot_titles=[p[0] for p in panels])
    for col, (_, pairs, color) in enumerate(panels, start=1):
        top = float(max(latent_pairs.max(), pairs.max())) * 1.05
        fig.add_trace(go.Scatter(x=latent_pairs, y=pairs, mode="markers", marker=dict(color=color, size=5, opacity=0.35), hoverinfo="skip", showlegend=False), row=1, col=col)
        fig.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", line=dict(color=GRAY, dash="dash"), hoverinfo="skip", showlegend=False), row=1, col=col)
    fig.update_xaxes(title_text="Abstand der wahren Faktoren (normiert)")
    fig.update_yaxes(title_text="Abstand in der Einbettung (normiert)", col=1)
    fig.update_layout(template="plotly_white", height=360, margin=dict(l=10, r=10, t=40, b=10))
    return lock_axes(fig)


def build_stability_compare(rows):
    """Mittlere paarweise Procrustes-Abstände zufälliger Starts je Datensatz für Autoencoder, PaCMAP, UMAP und t-SNE (niedriger = stabiler)."""
    labels = [f"Datensatz {i + 1}" for i in range(len(rows))]
    fig = go.Figure()
    for key, name, color in (("ae", "Autoencoder", BLUE), ("pacmap", "PaCMAP", ORANGE), ("umap", "UMAP", PURPLE), ("tsne", "t-SNE", RED)):
        fig.add_trace(go.Bar(x=labels, y=[r[key] for r in rows], name=name, marker_color=color, hovertemplate="%{x}: %{y:.2f}<extra>" + name + "</extra>"))
    fig.update_yaxes(title="mittlerer Procrustes-Abstand", range=[0, 1])
    fig.update_layout(template="plotly_white", barmode="group", height=320, margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=-0.25))
    return lock_axes(fig)


def build_out_of_sample(panels, train_color, test_color):
    """Zurückgehaltene Touren (Sterne) in den Einbettungen der Verfahren (2 × 2). `panels`: [(Titel, Training, Neu)]."""
    fig = make_subplots(rows=2, cols=2, subplot_titles=[p[0] for p in panels], vertical_spacing=0.14)
    for i, (_, train, test) in enumerate(panels):
        r, c = i // 2 + 1, i % 2 + 1
        fig.add_trace(go.Scatter(x=train[:, 0], y=train[:, 1], mode="markers", hoverinfo="skip", showlegend=False, marker=dict(color=train_color, colorscale="Viridis", size=5, opacity=0.5)), row=r, col=c)
        fig.add_trace(go.Scatter(x=test[:, 0], y=test[:, 1], mode="markers", hoverinfo="skip", showlegend=False,
                                 marker=dict(color=test_color, colorscale="Viridis", cmin=float(train_color.min()), cmax=float(train_color.max()), size=10, symbol="star",
                                             line=dict(width=1.2, color="#14233B"))), row=r, col=c)
    fig.update_layout(template="plotly_white", height=560, margin=dict(l=10, r=10, t=40, b=10))
    return lock_axes(fig)


def build_timing(rows):
    ns = np.array([r["n"] for r in rows], dtype=float)
    fig = go.Figure()
    for key, label, color in (("ae", "Autoencoder (Training)", BLUE), ("pacmap", "PaCMAP", ORANGE), ("umap", "UMAP", PURPLE), ("tsne", "t-SNE", RED), ("isomap", "Isomap", "#8c564b"), ("pca", "PCA", GREEN),
                              ("encode", "Autoencoder (1000 neue Touren)", "#17becf")):
        fig.add_trace(go.Scatter(x=ns, y=[max(r[key], 1e-6) for r in rows], mode="lines+markers", name=label, line=dict(color=color, width=3, dash="dot" if key == "encode" else "solid")))
    fig.update_xaxes(title="Anzahl Touren n", type="log")
    fig.update_yaxes(title="Rechenzeit (s)", type="log")
    fig.update_layout(template="plotly_white", height=360, margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=-0.3))
    return lock_axes(fig)
