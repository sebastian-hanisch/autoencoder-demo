"""Autoencoder von Grund auf (numpy, Backpropagation von Hand, Adam, Vollbatch-Training).

Ein Netz aus **Encoder** und **Decoder**, das die 12 z-Werte je Tour durch einen **linearen 2-D-Engpass** zwängt und daraus wieder rekonstruiert; gelernt wird durch Minimierung des mittleren quadratischen
Rekonstruktionsfehlers. Die 2 Engpass-Werte sind die Einbettung.

- **Tiefe 0**: kein versteckter Layer - Encoder 12 → 2, Decoder 2 → 12, alles linear. Ein solches **lineare Autoencoder** mit MSE-Verlust spannt (bei ausreichendem Training) denselben Unterraum auf wie die
  ersten zwei Hauptkomponenten der PCA (Baldi & Hornik 1989; Eckart-Young) - nicht dieselben Achsen, sondern denselben Raum.
- **Tiefe ≥ 1**: versteckte Schichten der Breite w mit einer nichtlinearen Aktivierung (tanh / ReLU / sigmoid) vor und nach dem Engpass. Der Encoder ist dann eine gelernte, glatte, **parametrische** Abbildung
  (neue Touren: einfach durch das Netz schicken), der Decoder bildet Punkte des 2-D-Raums wieder in Touren ab.
- Initialisierung: Xavier (tanh, sigmoid, lineare Schichten) bzw. He (ReLU); der Zufalls-Seed ist vom Datensatz-Seed entkoppelt."""

from dataclasses import dataclass

import numpy as np

ACTIVATIONS = ("tanh", "relu", "sigmoid")
BETA1, BETA2, ADAM_EPS = 0.9, 0.999, 1e-8


def activate(name, x):
    if name == "tanh":
        return np.tanh(x)
    if name == "relu":
        return np.maximum(x, 0.0)
    if name == "sigmoid":
        return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))
    raise ValueError(f"unbekannte Aktivierung {name!r}")


def activation_grad(name, out):
    """Ableitung der Aktivierung, ausgedrückt durch ihren AUSGABEwert."""
    if name == "tanh":
        return 1.0 - out ** 2
    if name == "relu":
        return (out > 0).astype(float)
    if name == "sigmoid":
        return out * (1.0 - out)
    raise ValueError(f"unbekannte Aktivierung {name!r}")


def layer_sizes(n_features, depth, width, n_code=2):
    return [n_features] + [width] * depth + [n_code] + [width] * depth + [n_features]


def n_parameters(sizes):
    return int(sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:])))


def _has_activation(layer, depth, n_layers):
    """Aktivierung nach jeder Schicht außer dem Engpass (Index `depth`) und der Ausgabeschicht."""
    return layer != depth and layer != n_layers - 1


def init_parameters(sizes, activation, seed):
    rng = np.random.default_rng(seed)
    weights, biases = [], []
    for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
        if activation == "relu":
            weights.append(rng.normal(size=(fan_in, fan_out)) * np.sqrt(2.0 / fan_in))
        else:
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            weights.append(rng.uniform(-limit, limit, size=(fan_in, fan_out)))
        biases.append(np.zeros(fan_out))
    return weights, biases


def forward(weights, biases, X, activation, depth):
    """Vorwärtsdurchlauf; gibt alle Ausgaben der Schichten zurück (Index 0 = Eingabe). Der Engpass liegt bei Index depth + 1."""
    n_layers = len(weights)
    outs = [X]
    for i in range(n_layers):
        z = outs[-1] @ weights[i] + biases[i]
        outs.append(activate(activation, z) if _has_activation(i, depth, n_layers) else z)
    return outs


def loss_and_gradients(weights, biases, X, activation, depth):
    """Mittlerer quadratischer Rekonstruktionsfehler (je Element) und seine Gradienten (Backpropagation von Hand)."""
    n, d = X.shape
    outs = forward(weights, biases, X, activation, depth)
    diff = outs[-1] - X
    loss = float((diff ** 2).mean())
    delta = 2.0 * diff / (n * d)
    n_layers = len(weights)
    gw, gb = [None] * n_layers, [None] * n_layers
    for i in reversed(range(n_layers)):
        gw[i] = outs[i].T @ delta
        gb[i] = delta.sum(0)
        if i > 0:
            delta = delta @ weights[i].T
            if _has_activation(i - 1, depth, n_layers):
                delta = delta * activation_grad(activation, outs[i])
    return loss, gw, gb


def snapshot_epochs(n_epochs, count=12):
    grid = np.unique(np.round(np.geomspace(1, max(n_epochs, 2), count)).astype(int))
    return sorted(set(int(e) for e in grid) | {0, int(n_epochs)})


@dataclass(frozen=True)
class AEModel:
    weights: list
    biases: list
    activation: str
    depth: int
    width: int
    embedding: np.ndarray            # [n, 2] Engpass-Werte der Trainings-Touren
    snapshots: dict                  # Epoche -> Einbettung (Kopie), inkl. 0 und n_epochs
    weight_snapshots: dict           # Epoche -> (Gewichte, Biases) (Kopien)
    loss_history: np.ndarray         # [n_epochs] Trainingsverlust
    mean: np.ndarray
    scale: np.ndarray
    Z: np.ndarray
    lr: float
    n_epochs: int
    seed: int

    @property
    def n(self):
        return len(self.Z)

    @property
    def loss(self):
        return float(self.loss_history[-1])

    @property
    def n_parameters(self):
        return n_parameters(layer_sizes(self.Z.shape[1], self.depth, self.width))


def fit_autoencoder(X, depth=1, width=16, activation="tanh", lr=0.01, n_epochs=1000, seed=0):
    if activation not in ACTIVATIONS:
        raise ValueError(f"activation in {ACTIVATIONS}")
    if depth < 0:
        raise ValueError("depth >= 0")
    X = np.asarray(X, dtype=float)
    mean = X.mean(0)
    scale = X.std(0, ddof=1)
    scale = np.where(scale > 0, scale, 1.0)
    Z = (X - mean) / scale
    sizes = layer_sizes(Z.shape[1], depth, width)
    weights, biases = init_parameters(sizes, activation, seed)
    params = weights + biases
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    n_layers = len(weights)
    wanted = set(snapshot_epochs(n_epochs))
    loss_history = np.zeros(n_epochs)

    def code(ws, bs):
        return forward(ws, bs, Z, activation, depth)[depth + 1]
    snapshots = {0: code(weights, biases).copy()} if 0 in wanted else {}
    weight_snapshots = {0: ([w.copy() for w in weights], [b.copy() for b in biases])} if 0 in wanted else {}
    for epoch in range(1, n_epochs + 1):
        loss, gw, gb = loss_and_gradients(weights, biases, Z, activation, depth)
        loss_history[epoch - 1] = loss
        lr_t = lr * np.sqrt(1.0 - BETA2 ** epoch) / (1.0 - BETA1 ** epoch)
        for k, g in enumerate(gw + gb):
            m[k] = BETA1 * m[k] + (1 - BETA1) * g
            v[k] = BETA2 * v[k] + (1 - BETA2) * g ** 2
            params[k] -= lr_t * m[k] / (np.sqrt(v[k]) + ADAM_EPS)
        if epoch in wanted:
            snapshots[epoch] = code(weights, biases).copy()
            weight_snapshots[epoch] = ([w.copy() for w in weights], [b.copy() for b in biases])
    return AEModel(weights=weights, biases=biases, activation=activation, depth=depth, width=width, embedding=code(weights, biases), snapshots=snapshots, weight_snapshots=weight_snapshots,
                   loss_history=loss_history, mean=mean, scale=scale, Z=Z, lr=float(lr), n_epochs=int(n_epochs), seed=int(seed))


def encode(model, X_new):
    """Neue Touren einbetten: durch den Encoder schicken (parametrisch - keine Neuberechnung, die Trainings-Einbettung bleibt unverändert)."""
    Zn = (np.asarray(X_new, dtype=float) - model.mean) / model.scale
    return forward(model.weights, model.biases, Zn, model.activation, model.depth)[model.depth + 1]


def decode(model, Y):
    """Punkte des 2-D-Raums in (standardisierte) Touren zurückabbilden."""
    out = np.asarray(Y, dtype=float)
    n_layers = len(model.weights)
    for i in range(model.depth + 1, n_layers):
        z = out @ model.weights[i] + model.biases[i]
        out = activate(model.activation, z) if _has_activation(i, model.depth, n_layers) else z
    return out


def reconstruct(model, X):
    """Rekonstruktion (in z-Werten) und ihr Fehler je Tour (mittleres Fehlerquadrat über die Merkmale)."""
    Zn = (np.asarray(X, dtype=float) - model.mean) / model.scale
    outs = forward(model.weights, model.biases, Zn, model.activation, model.depth)
    return outs[-1], ((outs[-1] - Zn) ** 2).mean(1)


def pca_subspace(Z, n_components=2):
    """Erste Hauptachsen (Zeilen) der z-Werte und der kleinste mögliche mittlere Rekonstruktionsfehler bei n_components (Eckart-Young)."""
    Zc = Z - Z.mean(0)
    _, s, vt = np.linalg.svd(Zc, full_matrices=False)
    floor = float((s[n_components:] ** 2).sum() / Zc.size)
    return vt[:n_components], floor


def principal_angles(A, B):
    """Hauptwinkel (Grad) zwischen den von den ZEILEN von A und B aufgespannten Unterräumen; 0 = gleicher Unterraum."""
    qa, _ = np.linalg.qr(A.T)
    qb, _ = np.linalg.qr(B.T)
    s = np.clip(np.linalg.svd(qa.T @ qb, compute_uv=False), -1.0, 1.0)
    return np.degrees(np.arccos(s))


def decoder_subspace(weights, biases, depth):
    """Zeilenraum des linearen Decoders (nur für Tiefe 0 sinnvoll): [2, 12]."""
    return weights[depth + 1]


def linear_angle_curve(model, pca_axes):
    """Größter Hauptwinkel (Grad) zwischen dem Decoder-Unterraum eines linearen Autoencoders und dem PCA-Unterraum, an den Schnappschüssen. Nur Tiefe 0."""
    if model.depth != 0:
        raise ValueError("nur für das lineare Autoencoder (Tiefe 0)")
    return {e: float(principal_angles(decoder_subspace(ws, bs, 0), pca_axes).max()) for e, (ws, bs) in sorted(model.weight_snapshots.items())}
