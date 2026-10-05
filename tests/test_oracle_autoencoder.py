"""Orakel-Tests (unabhängiger Rechenweg): Gradient per Complex-Step über alle Parameter, Adam in Lehrbuchform, Eckart-Young per Eigenzerlegung,
Hauptwinkel gegen scipy, quadratisches R² per QR, AUC mit Gleichständen gegen die Paar-Definition."""

import itertools

import numpy as np
import pytest

import ae_algorithm as A
import ae_evaluation as E
import ae_scenario as S


def _act(name, x):
    if name == "tanh":
        return np.tanh(x)
    if name == "sigmoid":
        return 1 / (1 + np.exp(-x))
    return np.where(x.real > 0, x, 0 * x)


def _loss_complex(ws, bs, X, name, depth):
    n_layers, total = len(ws), 0
    for row in X:                                           # eine Tour nach der anderen, komplexe Arithmetik
        h = row.astype(complex)
        for i in range(n_layers):
            h = h @ ws[i] + bs[i]
            if i != depth and i != n_layers - 1:
                h = _act(name, h)
        total = total + ((h - row) ** 2).sum()
    return total / X.size


@pytest.mark.parametrize("name", A.ACTIVATIONS)
@pytest.mark.parametrize("depth", [0, 1, 2, 3])
def test_every_gradient_entry_matches_the_complex_step_derivative(name, depth):
    rng = np.random.default_rng(depth)
    X = rng.normal(size=(6, 5))
    w, b = A.init_parameters(A.layer_sizes(5, depth, 3), name, 3)
    b = [bi + rng.normal(scale=0.3, size=bi.shape) for bi in b]
    loss, gw, gb = A.loss_and_gradients(w, b, X, name, depth)
    wc, bc = [x.astype(complex) for x in w], [x.astype(complex) for x in b]
    assert loss == pytest.approx(_loss_complex(wc, bc, X, name, depth).real, abs=1e-12)
    h = 1e-30
    for params, grads in ((wc, gw), (bc, gb)):
        for k in range(len(params)):
            for idx in np.ndindex(params[k].shape):
                params[k][idx] += 1j * h
                derivative = _loss_complex(wc, bc, X, name, depth).imag / h
                params[k][idx] -= 1j * h
                assert grads[k][idx] == pytest.approx(derivative, rel=1e-9, abs=1e-12)


def test_training_equals_textbook_adam_with_bias_corrected_moments():
    """Die Demo nutzt die effiziente Form (Kingma & Ba, Alg. 1, letzter Absatz): lr_t = lr·√(1−β2^t)/(1−β1^t) mit ε auf dem unkorrigierten √v - gleich der
    Lehrbuchform mit ε_t = ε/√(1−β2^t)."""
    ds = S.generate_dataset(60, 2, 0.8, 0.2, 2, 5)
    for depth, name in ((1, "tanh"), (2, "sigmoid"), (0, "relu")):
        model = A.fit_autoencoder(ds.X, depth, 5, name, 0.03, 25, 1)
        Z = (ds.X - ds.X.mean(0)) / ds.X.std(0, ddof=1)
        w, b = A.init_parameters(A.layer_sizes(12, depth, 5), name, 1)
        P = w + b
        nl = len(w)
        m = [np.zeros_like(p) for p in P]
        v = [np.zeros_like(p) for p in P]
        for t in range(1, 26):
            _, gw, gb = A.loss_and_gradients(P[:nl], P[nl:], Z, name, depth)
            for k, g in enumerate(gw + gb):
                m[k] = 0.9 * m[k] + 0.1 * g
                v[k] = 0.999 * v[k] + 0.001 * g * g
                P[k] = P[k] - 0.03 * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v[k] / (1 - 0.999 ** t)) + 1e-8 / np.sqrt(1 - 0.999 ** t))
        assert max(np.abs(x - y).max() for x, y in zip(model.weights + model.biases, P)) < 1e-6


def test_pca_floor_and_axes_equal_the_eigendecomposition_and_principal_angles_equal_scipy():
    linalg = pytest.importorskip("scipy.linalg")
    rng = np.random.default_rng(1)
    for _ in range(30):
        n, d, nc = int(rng.integers(10, 40)), int(rng.integers(3, 10)), 2
        Zr = rng.normal(size=(n, d)) @ rng.normal(size=(d, d))
        Zc = Zr - Zr.mean(0)
        axes, floor = A.pca_subspace(Zr, nc)
        top = np.linalg.eigh(Zc.T @ Zc)[1][:, ::-1][:, :nc]
        assert floor == pytest.approx(((Zc - Zc @ top @ top.T) ** 2).mean(), rel=1e-9)
        assert np.degrees(linalg.subspace_angles(axes.T, top)).max() < 1e-4
        a, b2 = rng.normal(size=(nc, d)), rng.normal(size=(nc, d))
        assert np.allclose(np.sort(A.principal_angles(a, b2)), np.sort(np.degrees(linalg.subspace_angles(a.T, b2.T))), atol=1e-5)


def test_quadratic_r2_equals_a_qr_based_fit_and_fidelity_equals_pearson_of_pair_distances():
    rng = np.random.default_rng(2)
    for _ in range(30):
        n, q = int(rng.integers(20, 60)), int(rng.integers(1, 4))
        coords = rng.normal(size=(n, 2))
        z = rng.normal(size=(n, q)) + 0.5 * coords[:, :1] ** 2
        x, y = coords[:, 0], coords[:, 1]
        Q = np.linalg.qr(np.stack([np.ones(n), x, y, x * x, x * y, y * y], 1))[0]
        r2 = 1 - ((z - Q @ (Q.T @ z)) ** 2).sum() / ((z - z.mean(0)) ** 2).sum()
        assert E.r2_quadratic(coords, z) == pytest.approx(r2, abs=1e-9)
        iu = np.triu_indices(n, 1)
        dz = np.array([np.linalg.norm(z[i] - z[j]) for i, j in zip(*iu)])
        dc = np.array([np.linalg.norm(coords[i] - coords[j]) for i, j in zip(*iu)])
        assert E.distance_fidelity(coords, z) == pytest.approx(np.corrcoef(dc, dz)[0, 1], abs=1e-9)
        near = dz <= np.median(dz)
        assert E.distance_fidelity_split(coords, z)[1] == pytest.approx(np.corrcoef(dc[~near], dz[~near])[0, 1], abs=1e-9)


def test_trustworthiness_equals_the_rank_definition_by_hand():
    rng = np.random.default_rng(3)
    for _ in range(15):
        n, k = int(rng.integers(15, 40)), int(rng.integers(1, 6))
        Xh = rng.normal(size=(n, 5))
        Xl = Xh[:, :2] + rng.normal(scale=1.0, size=(n, 2))
        dh = np.linalg.norm(Xh[:, None] - Xh[None], axis=-1)
        dl = np.linalg.norm(Xl[:, None] - Xl[None], axis=-1)
        penalty = 0
        for i in range(n):
            rank = {j: r + 1 for r, j in enumerate(j for j in np.argsort(dh[i], kind="stable") if j != i)}
            penalty += sum(max(rank[j] - k, 0) for j in [j for j in np.argsort(dl[i], kind="stable") if j != i][:k])
        assert E.trustworthiness(Xh, Xl, k) == pytest.approx(1 - 2 / (n * k * (2 * n - 3 * k - 1)) * penalty, abs=1e-9)


def test_anomaly_auc_counts_ties_as_half_a_pair():
    """AUC = Anteil (Sonderfahrt, normale Tour)-Paare mit größerem Fehler der Sonderfahrt, Gleichstand zählt halb."""
    rng = np.random.default_rng(4)
    for _ in range(50):
        n = int(rng.integers(6, 30))
        errors = rng.integers(0, 5, size=n).astype(float)                  # viele Gleichstände
        mask = rng.random(n) < 0.3
        if mask.sum() in (0, n):
            continue
        pairs = [(1.0 if a > b else 0.5 if a == b else 0.0) for a, b in itertools.product(errors[mask], errors[~mask])]
        assert E.anomaly_auc(errors, mask) == pytest.approx(np.mean(pairs), abs=1e-12)


def test_out_of_sample_r2_uses_uncentred_residuals():
    """Held-out-R² = 1 - SSE/SST mit unzentrierten Residuen (zentrierte würden einen konstanten Versatz der Vorhersage verzeihen)."""
    ds = S.generate_dataset(100, 2, 1.0, 0.25, 0, 11)
    out = E.out_of_sample(ds, E.Settings(n_epochs=150))
    zt = ds.z[out["test"]]
    for name, train_emb, y in (("ae", out["model"].embedding, out["y_ae"]), ("pacmap", out["pacmap_train"], out["y_pacmap"]), ("umap", out["umap_train"], out["y_umap"]),
                               ("tsne", out["tsne_train"], out["y_tsne"])):
        beta = np.linalg.pinv(E._quad_features(train_emb)) @ ds.z[out["train"]]
        resid = zt - E._quad_features(y) @ beta
        assert out["r2_" + name] == pytest.approx(1 - (resid ** 2).sum() / ((zt - zt.mean(0)) ** 2).sum(), abs=1e-6)
