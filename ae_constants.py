"""Defaults, Slider-Grenzen und Presets für die Autoencoder-Demo. Merkmale und Erzeugungs-Konstanten sind wortgleich aus pca-demo übernommen
(dieselben Lieferrouten - dieselbe gekrümmte Fläche, an der PCA scheiterte); alles Übrige ist neu."""

# --- Merkmale: 12 Kennzahlen je Tour in 4 Gruppen zu je 3 (Name, Einheit, Mittelwert, typische Streuung in Einheiten) ------------
FEATURES = (
    ("Distanz", "m", 45000.0, 15000.0),
    ("Stopps", "Anzahl", 60.0, 20.0),
    ("Ladegewicht", "kg", 1200.0, 400.0),
    ("Zeitfenster-Enge", "min", 90.0, 30.0),
    ("Verspätung", "min", 12.0, 8.0),
    ("Überstunden", "min", 25.0, 15.0),
    ("Fahrzeit je km", "s", 90.0, 25.0),
    ("Stop-and-go-Anteil", "%", 22.0, 10.0),
    ("Parkzeit", "min", 35.0, 12.0),
    ("Retourenquote", "Anteil", 0.06, 0.02),
    ("Sonderwünsche", "Anzahl", 4.0, 2.0),
    ("Zustellversuche", "Anzahl", 1.3, 0.5),
)
N_FEATURES = len(FEATURES)
FEATURE_NAMES = tuple(f[0] for f in FEATURES)
FEATURE_LABELS = tuple(f"{f[0]} [{f[1]}]" for f in FEATURES)
GROUPS = ("Größe", "Zeitdruck", "Verkehr", "Sonderfälle")     # je 3 aufeinanderfolgende Merkmale
GROUP_OF_FEATURE = tuple(i // 3 for i in range(N_FEATURES))

# --- Regler ------------------------------------------------------------------------------------------------------------
DEFAULT_N_TOURS = 300
N_TOURS_MIN, N_TOURS_MAX = 100, 600
DEFAULT_Q = 2
Q_MIN, Q_MAX = 1, 4
DEFAULT_CURVATURE = 1.0
CURVATURE_MIN, CURVATURE_MAX = 0.0, 1.0
DEFAULT_NOISE = 0.25
NOISE_MIN, NOISE_MAX = 0.0, 1.0
DEFAULT_OUTLIER_PCT = 0
OUTLIER_PCT_MIN, OUTLIER_PCT_MAX = 0, 10
DEFAULT_DEPTH = 2
DEPTH_MIN, DEPTH_MAX = 0, 3                        # 0 = lineares Autoencoder (nur Engpass)
WIDTH_CHOICES = (2, 4, 8, 16, 32, 64)
DEFAULT_WIDTH = 16
ACTIVATIONS = ("tanh", "relu", "sigmoid")
DEFAULT_ACTIVATION = "tanh"
LR_CHOICES = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3)
DEFAULT_LR = 0.03
DEFAULT_N_EPOCHS = 2000
N_EPOCHS_MIN, N_EPOCHS_MAX = 50, 4000
INIT_STARTS = (1, 2, 3, 4, 5)                      # Start-Nummer der Gewichts-Initialisierung (Seed = Nummer − 1)
DEFAULT_INIT_START = 1
DEFAULT_SEED = 7
N_CODE = 2

# --- Erzeugung ---------------------------------------------------------------------------------------------------------
OUTLIER_SCALE = 10.0                   # Sonderfahrten: latenter Faktor um diesen Faktor vergrößert
CROSS_LOADING = 0.15                   # kleine Querladungen zwischen Merkmalsgruppen
WITHIN_LOADINGS = (0.95, 0.9, 0.85)    # Ladung der drei Merkmale einer Gruppe auf ihren Faktor
CURVATURE_FREQUENCY = 1.6              # Frequenz der sin/cos-Terme der Krümmung
CURVATURE_AMPLITUDE = 2.0              # Länge jeder Spalte der Krümmungsmatrix (in z-Einheiten bei Krümmung 1)
LAYOUT_SEED = 20240915                 # feste Ladungs- und Krümmungsmatrizen (unabhängig vom Seed der Touren)


# --- Auswertung --------------------------------------------------------------------------------------------------------
TRUST_NEIGHBORS = 10
UMAP_N_NEIGHBORS, UMAP_MIN_DIST, UMAP_N_EPOCHS = 15, 0.1, 500      # Vergleichsverfahren mit ihren guten Einstellungen (siehe die Demos davor)
PACMAP_N_NEIGHBORS, PACMAP_MN_RATIO, PACMAP_FP_RATIO, PACMAP_N_ITER = 10, 0.5, 2.0, 450
TSNE_PERPLEXITY, TSNE_N_ITER = 30, 500
ISOMAP_K = 10
SWEEP_SEEDS = tuple(100_000 + i for i in range(3))                 # feste Sweep-Datensätze, unabhängig vom Demo-Seed
SWEEP_INIT_STARTS = (0, 1)
SWEEP_N_TOURS = 200
SWEEP_N_EPOCHS = 800
SWEEP_DEPTHS = (0, 1, 2, 3)
SWEEP_WIDTHS = (4, 8, 16, 32, 64)
SWEEP_LRS = (0.003, 0.01, 0.03, 0.1, 0.3)
STABILITY_SEEDS = tuple(100_000 + i for i in range(4))             # feste Datensätze für den fairen Stabilitätsvergleich
STABILITY_STARTS = (0, 1, 2, 3)
HOLDOUT_FRACTION = 0.2
TIMING_NS = (100, 200, 400, 600)
REFERENCE_EPOCHS = 2000                            # Referenzlauf (Standard-Lernrate und -Epochen) für "nicht trainiert" / "Lernrate zu hoch"
REFERENCE_FROM_LR = 0.1
REFERENCE_BELOW_EPOCHS = 500
NOT_TRAINED_FACTOR = 2.0
LR_HIGH_FACTOR = 3.0
DECODER_GRID = 25

_BASE = {"n_tours": DEFAULT_N_TOURS, "q": DEFAULT_Q, "curvature": DEFAULT_CURVATURE, "noise": DEFAULT_NOISE, "outlier_pct": DEFAULT_OUTLIER_PCT, "depth": DEFAULT_DEPTH, "width": DEFAULT_WIDTH,
         "activation": DEFAULT_ACTIVATION, "lr": DEFAULT_LR, "n_epochs": DEFAULT_N_EPOCHS, "init_start": DEFAULT_INIT_START, "seed": DEFAULT_SEED}
PRESETS = {
    "Gekrümmte Fläche: Autoencoder rekonstruiert": {**_BASE},
    "Linear = PCA": {**_BASE, "depth": 0},
    "Zu wenige Epochen": {**_BASE, "n_epochs": 100},
    "Lernrate zu hoch": {**_BASE, "lr": 0.3},
    "Sonderfahrten: Extreme bleiben erhalten": {**_BASE, "outlier_pct": 5},
    "Gerade Daten: kein Vorteil": {**_BASE, "curvature": 0.0},
}
PRESET_HELP = {
    "Gekrümmte Fläche: Autoencoder rekonstruiert": "Dieselbe gebogene Fläche wie in den Demos davor: das Netz (2 versteckte Schichten je Seite, Breite 16) rekonstruiert die 12 Kennzahlen aus nur 2 Werten mit einem Fehler von 0.023 - die PCA schafft mit 2 Komponenten bestenfalls 0.46. "
                                                "Die versteckten Faktoren findet es je nach Datensatz und Start unterschiedlich gut zurück (R² hier 0.95; im Mittel über 4 Datensätze nur 0.72, einzelne Läufe 0.6-0.95 - UMAP kommt auf 0.92).",
    "Linear = PCA": "Ohne versteckte Schichten (Tiefe 0) ist das Autoencoder linear: sein Rekonstruktionsfehler erreicht die PCA-Untergrenze (0.461 gegen 0.461) und der von seinem Decoder aufgespannte Unterraum weicht nach dem Training nur noch um Bruchteile eines Grades von der PCA ab - dasselbe Verfahren, nur per Gradientenabstieg gelernt.",
    "Zu wenige Epochen": "Nach nur 100 Epochen ist das Netz noch nicht trainiert: der Rekonstruktionsfehler liegt bei etwa 0.10 statt 0.023 (über 4 Datensätze das 4-9-fache). Die Demo rechnet einen Referenzlauf mit 2000 Epochen daneben.",
    "Lernrate zu hoch": "Mit Lernrate 0.3 (statt 0.03) springt das Training über das Minimum: der Rekonstruktionsfehler bleibt bei etwa 0.5 - so schlecht wie die PCA - statt 0.023, und die Einbettung verliert Nachbarschaft und Abstände.",
    "Sonderfahrten: Extreme bleiben erhalten": "5 % Sonderfahrten mit extremen Werten: anders als t-SNE, UMAP und PaCMAP (R² etwa 0.1-0.35) bricht das Autoencoder weniger ein (R² etwa 0.3-0.8, je nach Rechenumgebung; der Abstand zu den drei Verfahren bleibt) - als glatte parametrische Abbildung behält es die Größenordnung der Extreme; PCA (0.76) und Isomap (0.75) bleiben aber besser.",
    "Gerade Daten: kein Vorteil": "Krümmung 0: die Kennzahlen hängen linear von den Faktoren ab, die PCA ist optimal (R² 0.98) - das nichtlineare Netz erreicht dasselbe (0.98), aber mit einem Vielfachen des Aufwands.",
}
PRESET_EXPECTED_BANDS = {
    "Gekrümmte Fläche: Autoencoder rekonstruiert": {"verdict": "ae_wins", "mse": (0.008, 0.07), "mse_pca": (0.4, 0.5), "r2": (0.4, 1.0), "r2_pca": (0.4, 0.6), "r2_umap": (0.75, 0.97), "r2_iso": (0.94, 1.0)},
    "Linear = PCA": {"verdict": "linear_pca", "mse": (0.4, 0.5), "mse_pca": (0.4, 0.5), "angle": (0.0, 2.0), "r2": (0.35, 0.65), "r2_pca": (0.4, 0.6)},
    "Zu wenige Epochen": {"verdict": "not_trained", "mse": (0.04, 0.35), "mse_ref": (0.008, 0.07)},
    "Lernrate zu hoch": {"verdict": "lr_high", "mse": (0.2, 1.0), "mse_ref": (0.008, 0.07)},
    "Sonderfahrten: Extreme bleiben erhalten": {"verdict": "extremes_kept", "r2": (0.2, 1.0), "r2_umap": (-0.3, 0.6), "r2_tsne": (-0.3, 0.6), "r2_pacmap": (-0.3, 0.6), "r2_pca": (0.55, 0.9), "r2_iso": (0.6, 0.9)},
    "Gerade Daten: kein Vorteil": {"verdict": "no_advantage", "r2": (0.85, 1.0), "r2_pca": (0.94, 1.0)},
}
