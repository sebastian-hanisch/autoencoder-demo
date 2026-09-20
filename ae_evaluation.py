"""Auswertung: findet der Autoencoder die wahren Faktoren zurück - und was kann er, was die anderen nicht können (rekonstruieren, neue Touren einbetten, Extreme erhalten), und was nicht (stabil sein,
mit wenig Daten auskommen)? Alle Kennzahlen werden am Datensatz gemessen; die wahren latenten Faktoren z sind bekannt (Lieferrouten-Erzeugung, inkl. Sonderfahrten). Autoencoder, PCA, Isomap, t-SNE, UMAP und PaCMAP
werden mit denselben Messungen bewertet.

- **R² der wahren Faktoren**: Rekonstruktion von z aus den zwei Koordinaten per quadratischer Regression (monotone Umparametrisierungen werden nicht bestraft).
- **Abstandstreue** (gesamt / nahe Paare / ferne Paare): Pearson-Korrelation der Paarabstände in der Einbettung mit den Paarabständen der wahren Faktoren; "nah" = untere Hälfte der wahren Paarabstände, "fern" = obere Hälfte.
- **Trustworthiness** (Venna & Kaski): bleiben Nachbarn Nachbarn?
- **Rekonstruktionsfehler**: mittleres Fehlerquadrat je Merkmal in z-Werten - beim Autoencoder aus dem Decoder, bei der PCA die Untergrenze für 2 Komponenten (Eckart-Young).
- **Procrustes-Abstand**: wie stark unterscheiden sich zwei Einbettungen derselben Touren nach bester Drehung/Spiegelung/Skalierung (0 = gleich, 1 = unabhängig)."""

import itertools
import time
from dataclasses import dataclass

import numpy as np

import ae_constants as C
from ae_algorithm import decode, encode, fit_autoencoder, linear_angle_curve, pca_subspace, reconstruct
from ae_isomap import fit_isomap, pairwise_distances, standardize
from ae_pacmap import fit_pacmap
from ae_pacmap import transform as pacmap_transform
from ae_scenario import generate_dataset
from ae_tsne import embed_new_naive, fit_tsne, procrustes_disparity
from ae_umap import fit_umap
from ae_umap import transform as umap_transform


def trustworthiness(X_high, X_low, n_neighbors=C.TRUST_NEIGHBORS):
    """Trustworthiness (Venna & Kaski, 2001): Anteil der Nachbarn im Einbettungsraum, die auch im Originalraum echte Nachbarn sind, mit
    Rang-Strafe für eingeschleppte Fremde. 1 = perfekt. Eigene Implementierung, gegen sklearn geprüft (nur im Test)."""
    n = len(X_high)
    k = n_neighbors
    d_high = np.linalg.norm(X_high[:, None, :] - X_high[None, :, :], axis=-1)
    d_low = np.linalg.norm(X_low[:, None, :] - X_low[None, :, :], axis=-1)
    np.fill_diagonal(d_high, np.inf)
    np.fill_diagonal(d_low, np.inf)
    ranks_high = np.argsort(np.argsort(d_high, axis=1), axis=1) + 1            # Rang 1 = nächster Nachbar
    neighbors_low = np.argsort(d_low, axis=1)[:, :k]
    penalty = 0.0
    for i in range(n):
        r = ranks_high[i, neighbors_low[i]]
        penalty += float(np.maximum(r - k, 0).sum())
    return 1.0 - 2.0 / (n * k * (2 * n - 3 * k - 1)) * penalty


def _quad_features(e):
    return np.column_stack([e[:, 0], e[:, 1], e[:, 0] ** 2, e[:, 0] * e[:, 1], e[:, 1] ** 2, np.ones(len(e))])


def r2_quadratic(coords2, z):
    """R² der Rekonstruktion von z aus zwei Koordinaten (quadratische Regression, Mittel über die Faktoren, gewichtet mit ihrer Varianz)."""
    A = _quad_features(coords2)
    beta, *_ = np.linalg.lstsq(A, z, rcond=None)
    return float(1.0 - (z - A @ beta).var(0).sum() / z.var(0).sum())


def pca_project(X, n_components=2):
    Z = standardize(X)
    _, _, vt = np.linalg.svd(Z, full_matrices=False)
    return Z @ vt[:n_components].T


def distance_fidelity(coords2, z):
    iu = np.triu_indices(len(z), 1)
    return float(np.corrcoef(pairwise_distances(coords2)[iu], pairwise_distances(z)[iu])[0, 1])


def distance_fidelity_split(coords2, z):
    """(nahe Paare, ferne Paare): Abstandstreue getrennt für die untere und die obere Hälfte der wahren Paarabstände."""
    iu = np.triu_indices(len(z), 1)
    dz = pairwise_distances(z)[iu]
    dc = pairwise_distances(coords2)[iu]
    near = dz <= np.median(dz)
    return float(np.corrcoef(dz[near], dc[near])[0, 1]), float(np.corrcoef(dz[~near], dc[~near])[0, 1])


def make_dataset(n_tours, q, curvature, noise, outlier_pct, seed):
    return generate_dataset(n_tours, q, curvature, noise, outlier_pct, seed)


def _metrics(coords, z, Z):
    near, far = distance_fidelity_split(coords, z)
    return {"r2": r2_quadratic(coords, z), "fid": distance_fidelity(coords, z), "near": near, "far": far, "trust": trustworthiness(Z, coords)}


@dataclass(frozen=True)
class Settings:
    depth: int = C.DEFAULT_DEPTH
    width: int = C.DEFAULT_WIDTH
    activation: str = C.DEFAULT_ACTIVATION
    lr: float = C.DEFAULT_LR
    n_epochs: int = C.DEFAULT_N_EPOCHS
    init_start: int = C.DEFAULT_INIT_START           # 1-basiert (Seed = init_start − 1)


def _with(settings, **changes):
    return Settings(**{**settings.__dict__, **changes})


def run_ae(X, s):
    return fit_autoencoder(X, s.depth, s.width, s.activation, s.lr, s.n_epochs, s.init_start - 1)


def _needs_reference(s):
    return s.lr >= C.REFERENCE_FROM_LR or s.n_epochs < C.REFERENCE_BELOW_EPOCHS


@dataclass(frozen=True)
class Analysis:
    ae: object
    ref: object                      # Referenzlauf mit Standard-Lernrate und -Epochen, wenn die Einstellungen davon stark abweichen; sonst None
    pacmap: object
    umap: object
    tsne: object
    isomap: object
    iso_2d: np.ndarray
    pca_2d: np.ndarray
    iso_indices: np.ndarray
    metrics: dict                    # {"ae", "pca", "isomap", "tsne", "umap", "pacmap", "ref"?} -> {"r2","fid","near","far","trust"}
    mse_ae: float                    # Rekonstruktionsfehler des Autoencoders (Trainings-Touren, z-Werte)
    mse_pca: float                   # Untergrenze der PCA mit 2 Komponenten
    errors: np.ndarray               # Rekonstruktionsfehler je Tour
    snapshot_r2: dict                # Epoche -> R² der Einbettung zu diesem Zeitpunkt
    snapshot_far: dict               # Epoche -> Abstandstreue ferner Paare zu diesem Zeitpunkt
    angle: object                    # bei Tiefe 0: größter Hauptwinkel (Grad) zum PCA-Unterraum, sonst None
    angle_curve: dict                # bei Tiefe 0: Epoche -> Hauptwinkel, sonst {}


def analyse(dataset, settings):
    Z = standardize(dataset.X)
    ae = run_ae(dataset.X, settings)
    pac = fit_pacmap(dataset.X, C.PACMAP_N_NEIGHBORS, C.PACMAP_MN_RATIO, C.PACMAP_FP_RATIO, C.PACMAP_N_ITER)
    um = fit_umap(dataset.X, C.UMAP_N_NEIGHBORS, C.UMAP_MIN_DIST, C.UMAP_N_EPOCHS)
    ts = fit_tsne(dataset.X, C.TSNE_PERPLEXITY, C.TSNE_N_ITER)
    iso = fit_isomap(dataset.X, C.ISOMAP_K, 2)
    pca2 = pca_project(dataset.X)
    iso2 = iso.embedding[:, :2]
    metrics = {
        "ae": _metrics(ae.embedding, dataset.z, Z),
        "pca": _metrics(pca2, dataset.z, Z),
        "isomap": _metrics(iso2, dataset.z[iso.indices], Z[iso.indices]),
        "tsne": _metrics(ts.embedding, dataset.z, Z),
        "umap": _metrics(um.embedding, dataset.z, Z),
        "pacmap": _metrics(pac.embedding, dataset.z, Z),
    }
    ref = None
    if _needs_reference(settings):
        ref = run_ae(dataset.X, _with(settings, lr=C.DEFAULT_LR, n_epochs=C.REFERENCE_EPOCHS))
        metrics["ref"] = _metrics(ref.embedding, dataset.z, Z)
    axes, floor = pca_subspace(Z, C.N_CODE)
    _, errors = reconstruct(ae, dataset.X)
    angle_curve = linear_angle_curve(ae, axes) if ae.depth == 0 else {}
    snap = {e: r2_quadratic(y, dataset.z) for e, y in ae.snapshots.items() if e > 0}
    snap_far = {e: distance_fidelity_split(y, dataset.z)[1] for e, y in ae.snapshots.items() if e > 0}
    return Analysis(ae=ae, ref=ref, pacmap=pac, umap=um, tsne=ts, isomap=iso, iso_2d=iso2, pca_2d=pca2, iso_indices=iso.indices, metrics=metrics, mse_ae=ae.loss, mse_pca=floor, errors=errors,
                    snapshot_r2=snap, snapshot_far=snap_far, angle=angle_curve[max(angle_curve)] if angle_curve else None, angle_curve=angle_curve)


def verdict(analysis, dataset, settings):
    """Verdict-Kaskade (Warnungen zuerst) -> (Stufe, Code, Daten). Warnungen stützen sich auf einen Referenzlauf oder auf die PCA-Untergrenze, jeweils mit großem Abstand zur Schwelle."""
    m = analysis.metrics
    a = m["ae"]
    data = {"r2": a["r2"], "r2_pca": m["pca"]["r2"], "r2_iso": m["isomap"]["r2"], "r2_tsne": m["tsne"]["r2"], "r2_umap": m["umap"]["r2"], "r2_pacmap": m["pacmap"]["r2"], "far": a["far"],
            "far_pca": m["pca"]["far"], "far_iso": m["isomap"]["far"], "far_umap": m["umap"]["far"], "far_tsne": m["tsne"]["far"], "far_pacmap": m["pacmap"]["far"], "trust": a["trust"],
            "mse": analysis.mse_ae, "mse_pca": analysis.mse_pca, "depth": settings.depth, "width": settings.width, "lr": settings.lr, "n_epochs": settings.n_epochs, "outlier_pct": dataset.outlier_pct,
            "angle": analysis.angle}
    if analysis.ref is not None:
        data.update({"mse_ref": analysis.ref.loss, "r2_ref": m["ref"]["r2"], "trust_ref": m["ref"]["trust"]})
        if settings.lr >= C.REFERENCE_FROM_LR and analysis.mse_ae > C.LR_HIGH_FACTOR * analysis.ref.loss:
            return "warning", "lr_high", data
        if settings.n_epochs < C.REFERENCE_BELOW_EPOCHS and analysis.mse_ae > C.NOT_TRAINED_FACTOR * analysis.ref.loss:
            return "warning", "not_trained", data
    if settings.depth == 0:
        return "info", "linear_pca", data
    if settings.width <= 2 and analysis.mse_ae > 0.9 * analysis.mse_pca:
        return "warning", "bottleneck_narrow", data
    # Referenz ist das MITTEL der drei Nachbarschaftsverfahren, nicht ihr Maximum: bei Sonderfahrten schwankt das Ergebnis jedes
    # einzelnen Verfahrens (v. a. t-SNE) von Lauf zu Lauf und je nach Rechenumgebung (andere BLAS-Kerne -> anderer Trainingsverlauf),
    # sodass ein einzelner glueckliche Lauf das Urteil kippen wuerde (Linux-CI: "neutral" statt "extremes_kept").
    neighbour_ref = float(np.mean([m["tsne"]["r2"], m["umap"]["r2"], m["pacmap"]["r2"]]))
    if dataset.outlier_pct > 0 and a["r2"] >= 0.2 and a["r2"] - neighbour_ref >= 0.10:
        return "success", "extremes_kept", data
    if dataset.curvature == 0 and a["r2"] - m["pca"]["r2"] < 0.03:
        return "info", "no_advantage", data
    if a["r2"] - m["pca"]["r2"] >= 0.10:
        return "success", "ae_wins", data
    return "info", "neutral", data


def convergence_rows(analysis):
    """(Epoche, R²) an den Schnappschüssen des aktuellen Laufs - ohne Extra-Rechnung."""
    return sorted(analysis.snapshot_r2.items())


def anomaly_auc(errors, outlier_mask):
    """AUC (Rang-Statistik) der Rekonstruktionsfehler für die Erkennung der Sonderfahrten: 1 = alle Sonderfahrten haben größere Fehler als alle anderen Touren, 0.5 = Zufall. None ohne Sonderfahrten."""
    k = int(outlier_mask.sum())
    if k == 0 or k == len(errors):
        return None
    ranks = errors.argsort().argsort() + 1
    return float((ranks[outlier_mask].sum() - k * (k + 1) / 2) / (k * (len(errors) - k)))


def decoder_grid(model, n=C.DECODER_GRID, margin=0.05):
    """Decoder auf einem n × n-Gitter über den Wertebereich der Trainings-Einbettung -> (xs, ys, decodierte Touren [n·n, 12] in ORIGINAL-Einheiten, Abstand jedes Gitterpunkts zum nächsten Trainings-Code)."""
    emb = model.embedding
    lo, hi = emb.min(0), emb.max(0)
    pad = (hi - lo) * margin
    xs, ys = np.linspace(lo[0] - pad[0], hi[0] + pad[0], n), np.linspace(lo[1] - pad[1], hi[1] + pad[1], n)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    tours = decode(model, pts) * model.scale + model.mean
    dist = np.sqrt(((pts[:, None, :] - emb[None, :, :]) ** 2).sum(-1)).min(1)
    return xs, ys, tours, dist


def _run_stats(datasets, fn):
    """fn(dataset, start) -> Kennzahlen-Dict; Mittel und Streuung über Datensätze × Starts."""
    rows = [fn(ds, start) for ds in datasets for start in C.SWEEP_INIT_STARTS]
    keys = rows[0].keys()
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}, {k: float(np.std([r[k] for r in rows])) for k in keys}


def _sweep(values, key, q, curvature, noise, outlier_pct, base, n_tours=C.SWEEP_N_TOURS, n_epochs=C.SWEEP_N_EPOCHS, seeds=C.SWEEP_SEEDS):
    datasets = [(make_dataset(n_tours, q, curvature, noise, outlier_pct, s)) for s in seeds]
    Zs = {id(ds): standardize(ds.X) for ds in datasets}
    rows = []
    for value in values:
        settings = _with(base, n_epochs=n_epochs, **{key: value})

        def one(ds, start):
            model = run_ae(ds.X, _with(settings, init_start=start + 1))
            m = _metrics(model.embedding, ds.z, Zs[id(ds)])
            return {"r2": m["r2"], "far": m["far"], "trust": m["trust"], "mse": model.loss}
        mean, std = _run_stats(datasets, one)
        rows.append({key: value, **mean, **{k + "_std": v for k, v in std.items()}})
    return rows


def depth_sweep(q, curvature, noise, outlier_pct, values=C.SWEEP_DEPTHS):
    """Feste Sweep-Datensätze × Initialisierungen: je Tiefe mittleres R², ferne Paare, Trustworthiness, Rekonstruktionsfehler (mit Streuung); sonst Standard, verkürzte Epochen."""
    return _sweep(values, "depth", q, curvature, noise, outlier_pct, Settings())


def width_sweep(q, curvature, noise, outlier_pct, values=C.SWEEP_WIDTHS):
    return _sweep(values, "width", q, curvature, noise, outlier_pct, Settings())


def lr_sweep(q, curvature, noise, outlier_pct, values=C.SWEEP_LRS):
    return _sweep(values, "lr", q, curvature, noise, outlier_pct, Settings())


def linear_equals_pca(dataset, n_epochs=3000, lr=C.DEFAULT_LR, seed=0):
    """Lineares Autoencoder (Tiefe 0) trainieren und mit der PCA vergleichen: Hauptwinkel-Kurve über die Epochen (Grad) und Verhältnis des Rekonstruktionsfehlers zur PCA-Untergrenze."""
    model = fit_autoencoder(dataset.X, 0, C.DEFAULT_WIDTH, C.DEFAULT_ACTIVATION, lr, n_epochs, seed)
    axes, floor = pca_subspace(model.Z, C.N_CODE)
    return {"angles": linear_angle_curve(model, axes), "mse": model.loss, "mse_pca": floor, "ratio": model.loss / floor}


def _median_pairwise(embeddings):
    return float(np.median([procrustes_disparity(a, b) for a, b in itertools.combinations(embeddings, 2)]))


def stability(q, curvature, noise, outlier_pct, settings=None, n_tours=200, seeds=C.STABILITY_SEEDS, starts=C.STABILITY_STARTS):
    """Fairer Vergleich auf denselben Datensätzen: je Datensatz vier zufällige Starts, mittlere paarweise Procrustes-Abstände für Autoencoder (Gewichts-Initialisierung), PaCMAP, UMAP und t-SNE."""
    base = settings or Settings()
    out = []
    for seed in seeds:
        ds = make_dataset(n_tours, q, curvature, noise, outlier_pct, seed)
        ae, pac, um, ts = [], [], [], []
        for s in starts:
            ae.append(run_ae(ds.X, _with(base, init_start=s + 1)).embedding)
            pac.append(fit_pacmap(ds.X, C.PACMAP_N_NEIGHBORS, C.PACMAP_MN_RATIO, C.PACMAP_FP_RATIO, C.PACMAP_N_ITER, "random", s).embedding)
            um.append(fit_umap(ds.X, C.UMAP_N_NEIGHBORS, C.UMAP_MIN_DIST, C.UMAP_N_EPOCHS, 5, "random", s).embedding)
            ts.append(fit_tsne(ds.X, C.TSNE_PERPLEXITY, C.TSNE_N_ITER, init="random", seed=s).embedding)
        out.append({"seed": seed, "ae": _median_pairwise(ae), "pacmap": _median_pairwise(pac), "umap": _median_pairwise(um), "tsne": _median_pairwise(ts), "embeddings": ae})
    return out


def out_of_sample(dataset, settings, fraction=C.HOLDOUT_FRACTION):
    """Die letzten `fraction` der Touren zurückhalten. Autoencoder: einfach `encode` (Trainings-Einbettung bleibt unverändert); dazu der Rekonstruktionsfehler der neuen Touren gegen den der Trainings-Touren
    (Generalisierungslücke) und die Vergleichsverfahren (UMAP-`transform`, PaCMAP-Behelf, t-SNE-Näherung)."""
    n = dataset.n
    n_test = max(10, int(round(fraction * n)))
    train, test = np.arange(n - n_test), np.arange(n - n_test, n)
    z_test = dataset.z[test]

    def r2_new(train_emb, y):
        beta, *_ = np.linalg.lstsq(_quad_features(train_emb), dataset.z[train], rcond=None)
        return float(1 - (z_test - _quad_features(y) @ beta).var(0).sum() / z_test.var(0).sum())
    ae = run_ae(dataset.X[train], settings)
    y_ae = encode(ae, dataset.X[test])
    _, err_test = reconstruct(ae, dataset.X[test])
    pac = fit_pacmap(dataset.X[train], C.PACMAP_N_NEIGHBORS, C.PACMAP_MN_RATIO, C.PACMAP_FP_RATIO, C.PACMAP_N_ITER)
    y_pac = pacmap_transform(pac, dataset.X[test])
    um = fit_umap(dataset.X[train], C.UMAP_N_NEIGHBORS, C.UMAP_MIN_DIST, C.UMAP_N_EPOCHS)
    y_um = umap_transform(um, dataset.X[test])
    ts = fit_tsne(dataset.X[train], C.TSNE_PERPLEXITY, C.TSNE_N_ITER)
    y_ts = embed_new_naive(ts, dataset.X[test], 10)
    return {"train": train, "test": test, "model": ae, "pacmap_train": pac.embedding, "umap_train": um.embedding, "tsne_train": ts.embedding, "y_ae": y_ae, "y_pacmap": y_pac, "y_umap": y_um, "y_tsne": y_ts,
            "r2_ae": r2_new(ae.embedding, y_ae), "r2_pacmap": r2_new(pac.embedding, y_pac), "r2_umap": r2_new(um.embedding, y_um), "r2_tsne": r2_new(ts.embedding, y_ts),
            "r2_train": r2_quadratic(ae.embedding, dataset.z[train]), "mse_train": ae.loss, "mse_test": float(err_test.mean())}


def timing_sweep(ns=C.TIMING_NS, seed=100_000):
    """Gemessene Rechenzeit (Sekunden) von Autoencoder-Training, PaCMAP, UMAP, t-SNE, Isomap und PCA für wachsende n, plus die Zeit, 1000 neue Touren durch den fertigen Encoder zu schicken (eigene Messung, Rechner-abhängig)."""
    rows = []
    for n in ns:
        ds = generate_dataset(n, 2, 0.0, 0.1, 0, seed)
        out = {"n": int(n)}
        model = None
        for name, fn in (("ae", lambda: run_ae(ds.X, Settings())), ("pacmap", lambda: fit_pacmap(ds.X, C.PACMAP_N_NEIGHBORS, C.PACMAP_MN_RATIO, C.PACMAP_FP_RATIO, C.PACMAP_N_ITER)),
                         ("umap", lambda: fit_umap(ds.X, C.UMAP_N_NEIGHBORS, C.UMAP_MIN_DIST, C.UMAP_N_EPOCHS)), ("tsne", lambda: fit_tsne(ds.X, C.TSNE_PERPLEXITY, C.TSNE_N_ITER)),
                         ("isomap", lambda: fit_isomap(ds.X, C.ISOMAP_K, 2)), ("pca", lambda: pca_project(ds.X))):
            t = time.perf_counter()
            result = fn()
            out[name] = time.perf_counter() - t
            if name == "ae":
                model = result
        new = np.tile(ds.X, (max(1, 1000 // n + 1), 1))[:1000]
        t = time.perf_counter()
        encode(model, new)
        out["encode"] = time.perf_counter() - t
        rows.append(out)
    return rows
