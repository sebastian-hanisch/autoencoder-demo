# Autoencoder an Lieferrouten-Kennzahlen – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-autoencoder-demo.streamlit.app/)**

Siebtes und **letztes** Stück der **Dimensionsreduktion-Linie** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning":
anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo **ein** Verfahren – den **Autoencoder** – an einem wachsenden Beispiel und ist das **erste neuronale Netz** im Portfolio.
Vehikel: **dieselben 12 Lieferrouten-Kennzahlen wie in [pca-demo](../pca-demo), [isomap-demo](../isomap-demo), [lle-demo](../lle-demo), [tsne-demo](../tsne-demo), [umap-demo](../umap-demo) und [pacmap-demo](../pacmap-demo)**
(der Generator ist wortgleich kopiert und per Test gegen dessen Ausgabe eingefroren), erzeugt aus wenigen versteckten Faktoren – dieselbe gekrümmte Fläche, an der PCA scheiterte. PCA, Isomap, t-SNE, UMAP und PaCMAP stehen als Vergleich daneben.

**Einordnung in die Reihe (die Kanten des Graphen):** der Autoencoder ist ein **unabhängiger Ast direkt nach der PCA** (nicht nach UMAP/PaCMAP): ohne versteckte Schichten *ist* er die PCA (Baldi & Hornik 1989 – hier live nachgemessen), mit ihnen ein
gelernter, vollständig **parametrischer** Encoder **plus Decoder** – kein Nachbarschaftsgraph, keine Paare. Die Demo **misst**, was das bringt und kostet.
```
pca-demo → autoencoder-demo   (unabhängiger Ast; die Linie ist damit vollständig)
pca-demo → isomap-demo | lle-demo | tsne-demo → umap-demo → pacmap-demo
```

| Frage | Ergebnis (300 Touren, 4 feste Datensätze, wenn nicht anders angegeben) |
|---|---|
| Rekonstruktion / Decoder | ✅ Fehler **0.021** gegen PCA-Untergrenze 0.456 (mehr als 20-fach kleiner); die anderen Verfahren haben keinen Decoder |
| Linear = PCA | ✅ größter Hauptwinkel nach 3000 Epochen ≤ **0.014°**, Fehlerverhältnis 1.000 |
| Neue Touren einbetten | ✅ parametrisch (1000 Touren in ≈ 1 ms, Trainings-Einbettung bleibt unverändert). ⚠️ Aber R² der neuen Touren 0.62–0.75 (UMAP `transform` 0.88–0.93) und Rekonstruktionsfehler 0.10–0.36 gegen 0.02 im Training – **Datenhunger** |
| Extreme (Sonderfahrten) | ✅/⚠️ 5 %: R² **0.58** (UMAP 0.36, t-SNE 0.35, PaCMAP 0.30; PCA 0.67, Isomap 0.77); 10 %: 0.74 (0.24 / 0.26 / 0.22) – aber 2 %: 0.51 (0.62 / 0.66 / 0.58). Lokal gemessen; siehe „Rechenumgebung“ unten |
| Faktoren zurückgewinnen | ❌ R² **0.72** (UMAP 0.92, t-SNE 0.89, PaCMAP 0.89, PCA 0.51), Streuung ±0.12 über Datensätze und Starts; Rauschen 0.8: 0.54 (0.90 / 0.80 / 0.88) |
| Stabilität (Start egal) | ❌ mittlere paarweise Abweichung zufälliger Starts (200 Touren, q = 2): Autoencoder 0.42–0.63, t-SNE 0.37–0.53, PaCMAP 0.10–0.43, UMAP 0.01–0.22 |
| Globale Struktur | ⚠️ ferne Paare 0.45 (UMAP 0.60, t-SNE 0.58, PaCMAP 0.34, Isomap 0.89) |
| Rechenzeit | ⚠️ Training 1.1 s bei n = 600 (PaCMAP 0.4 s, UMAP 1.3 s, t-SNE 4.8 s); danach fast kostenlos |

## Was die Demo zeigt

1. **Autoencoder in Aktion** (Schritt-Slider + Abspielen): **Architektur** (Schichtdiagramm, Parameterzahl) → **eine Tour hindurch** (Original, Rekonstruktion des Netzes und der PCA mit ebenfalls 2 Komponenten, Code, Fehler) →
   **Training** (Schnappschüsse des Codes, Rekonstruktionsfehler mit der PCA-Untergrenze, R² und Abstandstreue ferner Paare je Schnappschuss) → Ergebnis neben der PCA.
2. **Was der Autoencoder gefunden hat – und die anderen fünf Verfahren auf denselben Daten:** sechs Einbettungen, Rekonstruktionsfehler, R² der wahren Faktoren, Abstandstreue (gesamt/nah/fern), Trustworthiness.
3. **📐 Wie stark hängt das Ergebnis von Architektur, Lernrate und Epochen ab?** (live über feste Sweep-Datensätze ab 100000 × zwei Starts, **mit Fehlerbalken**), mit Verdict (Lernrate zu hoch → nicht trainiert → zu schmale Schicht →
   linear = PCA → Extreme bleiben erhalten → kein Vorteil → Netz entfaltet die Fläche).
4. **🔁 Linear = PCA:** ein lineares Netz wird auf Ihren Daten trainiert, der größte Hauptwinkel zum PCA-Unterraum fällt gegen 0. **🎨 Der Decoder:** ein Gitter des 2-D-Raums wird in Touren zurückübersetzt (Heatmaps gewählter Kennzahlen),
   dazu die **Fehlerkarte** je Tour und der Anomalie-Blick auf die Sonderfahrten.
5. **🆚 Gemessener Vergleich:** Abstände nah/fern; Experimente auf Abruf (Knopf): Out-of-sample (Encoder gegen UMAP-`transform`, PaCMAP-Behelf, t-SNE-Näherung, mit Generalisierungslücke), **fairer Stabilitätsvergleich**
   (Autoencoder/PaCMAP/UMAP/t-SNE auf denselben Datensätzen), Rechenzeit.

Regler: Touren, wahre Faktoren q, Krümmung, Rauschen, **Sonderfahrten**, Tiefe (0 = linear), Breite (bei Tiefe 0 ausgeblendet, Auswahl bleibt erhalten), Aktivierung, Lernrate, Epochen, Start der Gewichte.

Messwerte (Seed 7, 300 Touren, q = 2, Tiefe 2, Breite 16, tanh, Lernrate 0.03, 2000 Epochen, Start 1, wenn nicht anders angegeben; die Presets prüfen sie mit weiten Bändern):

| Situation | Messung |
|---|---|
| Gekrümmte Fläche | Rekonstruktionsfehler **0.023** gegen 0.461 (PCA-Untergrenze); R² 0.95 (**glücklicher Datensatz/Start** – im Mittel über 4 Datensätze 0.72), UMAP 0.88, t-SNE 0.93, PaCMAP 0.80, Isomap 0.98, PCA 0.50 |
| Tiefe 0 | Fehler 0.461 = Untergrenze, Hauptwinkel 0.0002°, R² 0.50 = PCA |
| 100 Epochen | Fehler **0.099** statt 0.023 (über 4 Datensätze das 4–9-fache); R² leidet kaum |
| Lernrate 0.3 | Fehler **0.48** (drei weitere Datensätze: 0.62–0.64) – so schlecht wie die PCA, R² 0.55, ferne Paare 0.01 |
| 5 % Sonderfahrten | R² **0.61**, UMAP 0.14, t-SNE 0.12, PaCMAP 0.14, PCA 0.76, Isomap 0.75 |
| Gerade Daten | R² 0.98 = PCA 0.98: **kein Vorteil** |
| Breite 2 | Fehler 0.462 – nicht besser als die PCA (zu schmale versteckte Schicht) |

**Fairer Vergleich über 4 Datensätze** (300 Touren, feste Seeds 100000–100003, Start 1; Mittel von R² / nahe / ferne Paare / Trustworthiness): Standard – Autoencoder 0.72 / 0.57 / 0.45 / 0.99, UMAP 0.92 / 0.72 / 0.60 / 0.98, t-SNE 0.89 / 0.68 / 0.58 / 0.99,
PaCMAP 0.89 / 0.53 / 0.34 / 0.98, Isomap 0.98 / 0.91 / 0.89 / 0.99, PCA 0.51 / 0.61 / 0.38 / 0.86. q = 3: R² 0.32 (UMAP 0.55, t-SNE 0.38, PaCMAP 0.41). Rauschen 0.8: 0.54 (0.90 / 0.80 / 0.88). Gerade Daten: 0.98 (UMAP 0.92, PCA 0.99).
Sonderfahrten (R² Autoencoder / UMAP / t-SNE / PaCMAP; PCA / Isomap): 2 % 0.51 / 0.62 / 0.66 / 0.58 (0.66 / 0.78); 5 % 0.58 / 0.36 / 0.35 / 0.30 (0.67 / 0.77); 10 % 0.74 / 0.24 / 0.26 / 0.22 (0.77 / 0.81).
Rekonstruktionsfehler Autoencoder / PCA-Untergrenze: Standard 0.021 / 0.456; q = 3: 0.100 / 0.52; Rauschen 0.8: 0.136 / 0.51; 2 %: 0.068 / 0.24; **5 %: 0.116 / 0.062** – mit Extremen rekonstruiert das Netz *schlechter* als die PCA-Untergrenze;
10 %: 0.021 / 0.042.

**Hyperparameter** (300 Touren, 5 Datensätze × 3 Starts; Mittel R² / Fehler): Epochen 50: 0.71 / 0.314, 100: 0.77 / 0.131, 300: 0.77 / 0.039, 500: 0.78 / 0.030, 1000: 0.79 / 0.026, 2000: 0.79 / 0.022, 4000: 0.80 / 0.021. Lernrate 0.001: 0.76 / 0.033, 0.003: 0.71 / 0.023,
0.01: 0.77 / 0.022, 0.03: 0.79 / 0.022, 0.1: 0.75 / 0.039 (Bereich 0.022–0.089), 0.3: 0.39 / 0.605. Tiefe 0: 0.50 / 0.458, 1: 0.74 / 0.038, 2: 0.79 / 0.022, 3: 0.62 / 0.024. Breite 2: 0.51 / 0.462, 4: 0.80 / 0.189, 8: 0.75 / 0.045, 16: 0.79 / 0.022,
32: 0.74 / 0.020, 64: 0.67 / 0.018. Aktivierung tanh 0.79 / 0.022, sigmoid 0.79 / 0.022, ReLU 0.63 / 0.049. Das R² streut mit ±0.12–0.18 über Datensätze und Starts; die **Live-Sweeps der App** (200 Touren, 800 Epochen, 3 × 2 Läufe) zeigen die Streuung als Fehlerbalken.
Tiefer und breiter rekonstruiert besser, findet die Faktoren aber nicht besser.

**Linear = PCA** (4 Datensätze, 3000 Epochen): größter Hauptwinkel 0.0067° / 0.0093° / 0.0136° / 0.0°, Fehlerverhältnis 1.0000; nach 79 Epochen noch 1.3°–27°, nach 338 Epochen 0.12°–0.19°.

**Anomalie-Blick** (AUC der Rekonstruktionsfehler für die Erkennung der Sonderfahrten, 4 Datensätze): 2 %: 0.36–0.86, 5 %: 0.60–0.92, 10 %: 0.73–0.92 – kein verlässlicher Detektor.

**Stabilität** (Median der sechs paarweisen Procrustes-Abstände von vier zufälligen Starts, 200 Touren, 4 feste Datensätze; Autoencoder / PaCMAP / UMAP / t-SNE): q = 2: 0.54 / 0.11 / 0.01 / 0.48, 0.42 / 0.10 / 0.01 / 0.53, 0.63 / 0.30 / 0.22 / 0.37,
0.57 / 0.43 / 0.11 / 0.43; q = 3: 0.44 / 0.51 / 0.16 / 0.58, 0.60 / 0.37 / 0.35 / 0.76, 0.48 / 0.10 / 0.05 / 0.68, 0.79 / 0.80 / 0.25 / 0.72. Verschiedene Anfangsgewichte finden verschiedene Abbildungen derselben Fläche.

**Out-of-sample** (letzte 20 % zurückgehalten, 4 feste Seeds; Autoencoder / UMAP / PaCMAP-Behelf / t-SNE-Näherung, R² der neuen Touren): 0.66 / 0.88 / 0.79 / 0.79, 0.75 / 0.93 / 0.91 / 0.69, 0.62 / 0.88 / 0.84 / 0.74, 0.69 / 0.93 / 0.82 / 0.85; Rekonstruktionsfehler
Training / neue Touren des Autoencoders: 0.018 / 0.36, 0.025 / 0.10, 0.022 / 0.30, 0.023 / 0.15. Der Encoder verschiebt die Trainings-Einbettung nicht (bei UMAP, PaCMAP und t-SNE ändert sich beim Neu-Rechnen das Bild).

**Rechenzeit** (lokale Messung, Standard-Einstellungen; Autoencoder-Training / PaCMAP / UMAP / t-SNE; Encoder für 1000 neue Touren ≈ 1 ms): n = 100: 0.36 s / 0.07 s / 0.32 s / 0.08 s; n = 200: 0.52 / 0.16 / 0.49 / 0.24; n = 400: 0.77 / 0.26 / 0.93 / 2.3; n = 600: **1.08 / 0.37 / 1.34 / 4.8**.

## Modell und Verfahren

- **Generator** (`ae_scenario.py`): wortgleich aus pca-demo; latente Faktoren, 12 Merkmale in 4 Gruppen, Krümmung `κ·B·h(z)`, Rauschen, Sonderfahrten. Das Netz arbeitet auf z-Werten.
- **Autoencoder** (`ae_algorithm.py`, numpy, **Backpropagation von Hand**, ohne Deep-Learning-Bibliothek): Encoder 12 → [versteckte Schichten] → **2 (linearer Engpass)** → Decoder spiegelbildlich → 12 (linear); Aktivierung tanh, ReLU oder sigmoid nach jeder
  Schicht außer Engpass und Ausgabe; Verlust MSE; **Adam**, Vollbatch; Xavier- (tanh, sigmoid, linear) bzw. He-Initialisierung mit eigenem Seed (entkoppelt vom Datensatz-Seed). `encode` schickt neue Touren durch den Encoder, `decode` bildet Punkte des 2-D-Raums zurück.
  Für Tiefe 0: Hauptwinkel zwischen Decoder-Unterraum und PCA-Unterraum, Rekonstruktions-Untergrenze der PCA (Eckart-Young).
- **PCA / Isomap / t-SNE / UMAP / PaCMAP** (`ae_isomap.py`, `ae_tsne.py`, `ae_umap.py`, `ae_pacmap.py`): wortgleich aus den Demos davor kopiert (nur Vergleichsverfahren, feste gute Einstellungen).
- **Auswertung** (`ae_evaluation.py`): R² der wahren Faktoren aus den zwei Koordinaten per quadratischer Regression; Abstandstreue (nah/fern); Trustworthiness; Rekonstruktionsfehler; Sweeps mit Streuung; Referenzläufe für "nicht trainiert" und "Lernrate zu hoch";
  Stabilitäts- und Out-of-sample-Vergleich auf denselben Datensätzen; Zeitmessung.

## Was nicht funktioniert hat / Grenzen

- **Der Default-Seed 7 ist ein glücklicher Fall** (R² 0.95 gegen im Mittel 0.72 über vier Datensätze): die App sagt im Erfolgs-Urteil ausdrücklich, dass das Netz hier vorn liegt, im Mittel aber nicht, und der Preset-Hilfetext nennt den Mittelwert.
- **Die Faktoren werden nicht besser zurückgewonnen, wenn das Netz größer wird:** Tiefe 3 und Breite 64 rekonstruieren am besten, das R² fällt aber ab (0.62 bzw. 0.67). Der Code muss nur rekonstruierbar sein, nicht die wahren Faktoren zeigen.
- **Sonderfahrten: nur eine Teilantwort.** Bei 5–10 % bleibt das Netz deutlich vor t-SNE/UMAP/PaCMAP, bei 2 % nicht verlässlich (Vorsprung je nach Rechenumgebung −0.11 bis +0.11), und es bleibt unter PCA und Isomap; mit Extremen rekonstruiert es teils schlechter als die PCA-Untergrenze (5 %: 0.116 gegen 0.062).
  Die Fehlerkarte taugt nicht als Anomalie-Detektor (AUC 0.36–0.92).
- **Rechenumgebung:** Die Trainingsverläufe (und bei den Sonderfahrten auch t-SNE) hängen vom BLAS-Kern ab: gleiche Seeds auf anderer CPU/anderem Betriebssystem liefern andere Zahlen. Gemessen mit `OPENBLAS_CORETYPE` (Seed 7, 5 % Sonderfahrten): R² des Autoencoders 0.61–0.79, t-SNE 0.10–0.34. In GitHub Actions (Linux) lag der mittlere Vorsprung des Autoencoders über vier Datensätze bei 5 % Sonderfahrten bei +0.36 (lokal +0.24), bei 2 % bei +0.07 (lokal −0.11; mit dem BLAS-Kern SANDYBRIDGE +0.25 bzw. +0.11). Deshalb vergleicht das Urteil „Extreme bleiben erhalten“ gegen das **Mittel** der drei Nachbarschaftsverfahren (nicht gegen das beste), und die Tests prüfen nur, dass der Vorsprung mit dem Anteil der Extreme wächst, statt eines festen Wertes. Die Tabellen oben stammen aus einem lokalen Windows-Lauf.
- **Stabilität ist die größte Schwäche:** verschiedene Anfangsgewichte liefern ähnlich verschiedene Bilder wie t-SNE, deutlich mehr als UMAP oder PaCMAP.
- **Datenhunger:** 1038 Parameter bei 300 Touren; der Rekonstruktionsfehler neuer Touren ist ein Mehrfaches des Trainingsfehlers (5–16-fach). Kein Weight-Decay, kein Validierungsabbruch, kein Mini-Batch – für die Demo bewusst schlicht.
- **Vollbatch-Adam auf ≤ 600 Touren:** Training in 0.4–1.1 s; größere Datensätze brauchen Mini-Batches. Ein totes ReLU-Netz (kleine Breite, wenige Epochen) kann einen konstanten Code liefern – dann sind Kennzahlen bedeutungslos (Warnungen beim Normieren der Abstände).
- **Grenzen (Text):** Achsen des Codes haben keine feste Bedeutung; der Engpass ist fest 2-dimensional (Einbettung für die Darstellung).

## Verifikation

- **Backpropagation** für alle drei Aktivierungen und Tiefen 0–3 gegen finite Differenzen (1e-7; ReLU mit kleinerem Schritt, Biases ungleich 0 wegen des Knicks bei exakt 0); Aktivierungs-Ableitungen; Schichtaufbau (Aktivierung nur in versteckten Schichten, Engpass und Ausgabe linear);
  Training deterministisch bei festem Seed, Verlust fällt; `encode` der Trainings-Touren = Trainings-Einbettung, `reconstruct` = `decode(encode)`; Hauptwinkel per Handinstanz; PCA-Untergrenze = Eckart-Young.
- **Linear = PCA** über drei Datensätze (Hauptwinkel < 0.5°, Fehlerverhältnis < 1.001, Startwinkel > 30°); nichtlineares Netz rekonstruiert unter 20 % der PCA-Untergrenze.
- Generator bit-identisch zu pca-demo; Isomap-/t-SNE-/UMAP-/PaCMAP-Kopien gegen eingefrorene Referenzwerte (mit Toleranzen für chaotische Verfahren); Trustworthiness gegen `sklearn.manifold.trustworthiness` (1e-9).
- Die Tabelle "Ergebnis im Test" ist als Tests hinterlegt (Decoder-Vorteil, Faktoren schlechter als UMAP, Sonderfahrten bei 5 % vs 2 %, Stabilität, Generalisierungslücke, Rechenzeit-Reihenfolge, Linear = PCA); Verdict-Codes; alle 6 Presets in weiten Bändern;
  AppTest-Rauchtests (Default, jedes Preset, jeder Schritt, Randgrößen, Breite bei Tiefe 0 ausgeblendet und wiederhergestellt, Experimente auf Abruf), Achsensperre aller Figuren.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App: Schritte, Ergebnis, 📐 Sweeps, 🔁 Linear = PCA, 🎨 Decoder, 🆚 Vergleich, Mathe |
| `ae_algorithm.py` | Autoencoder von Grund auf (Backpropagation, Adam, Encode/Decode, Hauptwinkel) |
| `ae_isomap.py`, `ae_tsne.py`, `ae_umap.py`, `ae_pacmap.py` | Vergleichsverfahren (wortgleich aus den Demos davor) |
| `ae_scenario.py`, `ae_constants.py` | Lieferrouten-Generator (wortgleich aus pca-demo), Konstanten, Presets |
| `ae_evaluation.py` | Kennzahlen, Sweeps, Verdict, Stabilität, Out-of-sample, Zeitmessung, Decoder-Gitter |
| `ae_presets.py`, `ae_visualization.py` | Permalink/Presets, Plotly-Figuren (achsengesperrt) |
| `tests/` | Gradientenprüfung, sklearn-Kreuzvergleich, Generator-Referenz, Auswertung, Presets, AppTest |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

Teil des [Operations-Research-Demo-Portfolios](https://sebastianhanisch.net/demos.html) von
[Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
Interesse an einer maßgeschneiderten Lösung? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html).
