import numpy as np
import pytest

from ae_algorithm import (
    ACTIVATIONS, activate, activation_grad, decode, encode, fit_autoencoder, forward, init_parameters, layer_sizes, linear_angle_curve, loss_and_gradients, n_parameters, pca_subspace,
    principal_angles, reconstruct, snapshot_epochs,
)
from ae_isomap import fit_isomap, standardize
from ae_pacmap import fit_pacmap, find_weight
from ae_scenario import generate_dataset
from ae_tsne import fit_tsne, procrustes_disparity
from ae_umap import fit_umap


def _data(n=300, q=2, curvature=1.0, noise=0.25, seed=7):
    return generate_dataset(n, q, curvature, noise, 0, seed)


@pytest.mark.parametrize("activation", ACTIVATIONS)
@pytest.mark.parametrize("depth", [0, 1, 2, 3])
def test_backpropagation_matches_finite_differences(activation, depth):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 12))
    w, b = init_parameters(layer_sizes(12, depth, 6), activation, 1)
    b = [bias + rng.normal(scale=0.1, size=bias.shape) for bias in b]                # Biases 0 + tote ReLU-Einheiten ergäben Vorzeichenwechsel exakt bei 0 (nicht differenzierbar)
    _, gw, gb = loss_and_gradients(w, b, X, activation, depth)
    for params, grads in ((w, gw), (b, gb)):
        for k in range(len(params)):
            for idx in ((0, 0), (1, 1)) if params[k].ndim == 2 else ((0,), (1,)):
                idx = tuple(min(i, s - 1) for i, s in zip(idx, params[k].shape))
                e, tol = (1e-9, 1e-5) if activation == "relu" else (1e-6, 1e-7)                 # ReLU: Knick - kleiner Schritt, damit kein Vorzeichenwechsel überschritten wird
                params[k][idx] += e
                lp = loss_and_gradients(w, b, X, activation, depth)[0]
                params[k][idx] -= 2 * e
                lm = loss_and_gradients(w, b, X, activation, depth)[0]
                params[k][idx] += e
                assert abs((lp - lm) / (2 * e) - grads[k][idx]) < tol


def test_activation_derivatives_match_finite_differences():
    x = np.linspace(-2.5, 2.5, 21) + 0.013                                          # nicht exakt 0: ReLU-Knick
    for name in ACTIVATIONS:
        numeric = (activate(name, x + 1e-6) - activate(name, x - 1e-6)) / 2e-6
        assert np.allclose(activation_grad(name, activate(name, x)), numeric, atol=1e-6)
    with pytest.raises(ValueError):
        activate("swish", x)


def test_layer_sizes_and_parameter_count():
    assert layer_sizes(12, 0, 16) == [12, 2, 12] and layer_sizes(12, 2, 16) == [12, 16, 16, 2, 16, 16, 12]
    assert n_parameters(layer_sizes(12, 0, 16)) == 12 * 2 + 2 + 2 * 12 + 12 and n_parameters([3, 4, 2]) == 3 * 4 + 4 + 4 * 2 + 2


def test_bottleneck_and_output_are_linear_and_hidden_layers_use_the_activation():
    w, b = init_parameters(layer_sizes(12, 1, 8), "relu", 0)
    outs = forward(w, b, np.random.default_rng(1).normal(size=(50, 12)), "relu", 1)
    assert (outs[1] >= 0).all()                                                     # versteckte Schicht: ReLU
    assert (outs[2] < 0).any() and (outs[-1] < 0).any()                             # Engpass und Ausgabe: linear


def test_training_is_deterministic_records_snapshots_and_the_loss_falls():
    ds = _data(150)
    m1, m2 = fit_autoencoder(ds.X, 1, 8, "tanh", 0.03, 200, 0), fit_autoencoder(ds.X, 1, 8, "tanh", 0.03, 200, 0)
    assert np.array_equal(m1.embedding, m2.embedding) and not np.array_equal(m1.embedding, fit_autoencoder(ds.X, 1, 8, "tanh", 0.03, 200, 1).embedding)
    assert set(m1.snapshots) == set(snapshot_epochs(200)) == set(m1.weight_snapshots) and np.array_equal(m1.snapshots[200], m1.embedding)
    assert m1.loss_history.shape == (200,) and m1.loss_history[-1] < 0.3 * m1.loss_history[0] and m1.n_parameters == n_parameters(layer_sizes(12, 1, 8))


def test_encode_of_the_training_tours_equals_the_training_embedding_and_reconstruct_equals_decode_of_encode():
    ds = _data(120)
    m = fit_autoencoder(ds.X, 2, 8, "tanh", 0.03, 300, 0)
    assert np.allclose(encode(m, ds.X), m.embedding, atol=1e-12)
    recon, err = reconstruct(m, ds.X)
    assert np.allclose(decode(m, encode(m, ds.X)), recon, atol=1e-12) and np.isclose(err.mean(), loss_and_gradients(m.weights, m.biases, m.Z, m.activation, m.depth)[0], atol=1e-12)


def test_invalid_options_are_rejected():
    X = _data(60).X
    with pytest.raises(ValueError):
        fit_autoencoder(X, 1, 8, "swish", 0.03, 10, 0)
    with pytest.raises(ValueError):
        fit_autoencoder(X, -1, 8, "tanh", 0.03, 10, 0)


def test_principal_angles_hand_instances():
    a = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
    assert np.allclose(principal_angles(a, a), 0, atol=1e-6)
    assert np.allclose(principal_angles(a, np.array([[0, 0, 1.0, 0], [0, 0, 0, 1.0]])), 90, atol=1e-6)
    assert np.allclose(principal_angles(a, np.array([[np.cos(0.3), np.sin(0.3), 0, 0], [-np.sin(0.3), np.cos(0.3), 0, 0]])), 0, atol=1e-6)      # gedrehte Basis desselben Raums
    tilted = np.array([[1.0, 0, 0, 0], [0, np.cos(0.5), np.sin(0.5), 0]])
    assert abs(principal_angles(a, tilted).max() - np.degrees(0.5)) < 1e-6


def test_pca_subspace_gives_the_eckart_young_error_floor():
    Z = standardize(_data(200).X)
    axes, floor = pca_subspace(Z, 2)
    recon = (Z - Z.mean(0)) @ axes.T @ axes
    assert np.allclose(np.abs(axes @ axes.T), np.eye(2), atol=1e-12) and abs(((Z - Z.mean(0) - recon) ** 2).mean() - floor) < 1e-12


def test_linear_autoencoder_learns_the_pca_subspace_on_several_datasets():
    """Belegt 'Linear = PCA': nach 3000 Epochen Hauptwinkel < 0.5 Grad und Fehler = PCA-Untergrenze (gemessen: ≤ 0.014 Grad, Verhältnis 1.000)."""
    for seed in (100_000, 100_001, 100_002):
        ds = _data(seed=seed)
        m = fit_autoencoder(ds.X, 0, 16, "tanh", 0.03, 3000, 0)
        axes, floor = pca_subspace(m.Z, 2)
        angles = linear_angle_curve(m, axes)
        assert angles[3000] < 0.5 and abs(m.loss / floor - 1.0) < 1e-3 and angles[0] > 30            # Start: Zufallsraum, weit weg


def test_linear_angle_curve_needs_a_linear_network():
    m = fit_autoencoder(_data(80).X, 1, 8, "tanh", 0.03, 20, 0)
    with pytest.raises(ValueError):
        linear_angle_curve(m, np.eye(12)[:2])


def test_nonlinear_autoencoder_reconstructs_far_better_than_the_pca_floor_on_the_curved_surface():
    """Belegt 'Rekonstruktion': mittlerer Fehler 0.021 gegen PCA-Untergrenze 0.456 (mehr als 20-fach) - hier mit Sicherheitsabstand: unter 20 % der Untergrenze, über drei Datensätze."""
    for seed in (100_000, 100_001, 100_002):
        ds = _data(seed=seed)
        m = fit_autoencoder(ds.X, 2, 16, "tanh", 0.03, 1500, 0)
        assert m.loss < 0.2 * pca_subspace(m.Z, 2)[1]


def test_copied_isomap_tsne_umap_pacmap_and_scenario_match_their_reference_values():
    ds = _data()
    r = fit_isomap(ds.X, 8, 2)
    assert abs(float(np.abs(r.embedding).sum()) - 1786.6726227517283) < 1e-6 and abs(float(r.geodesic.sum()) - 621595.593373937) < 1e-3
    assert abs(fit_tsne(ds.X, 30, 750).kl - 0.3098433168465976) < 2e-3            # chaotisch: auf CI andere BLAS
    um = fit_umap(ds.X, 15, 0.1, 100)
    assert um.connected and abs(float(um.graph.sum()) - 1809.4579300820042) < 1e-3 and abs(um.a - 1.5769) < 1e-3
    pac = fit_pacmap(ds.X, 10, 0.5, 2.0, 90)
    assert (pac.n_neighbors, pac.n_mn, pac.n_fp) == (10, 5, 20) and np.allclose(find_weight(0, (100, 100, 250)), (1000.0, 2.0, 1.0))
    assert procrustes_disparity(np.eye(2), np.eye(2) * 3) < 1e-12
    assert abs(float(ds.X.sum()) - 14553337.310875032) < 1e-6 and abs(float(generate_dataset(200, 3, 0.4, 0.3, 5, 42).X.sum()) - 10763498.969287368) < 1e-6
