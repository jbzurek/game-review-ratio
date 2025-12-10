# **Game Review Ratio**

Predykcja odsetka pozytywnych recenzji gier na podstawie metadanych Steam.

---

# **SPIS TREŚCI**

* [Cel i zakres](#cel-i-zakres)
* [Architektura](#architektura-high-level)
* [Dane](#dane)
* [Potok (Kedro)](#potok-kedro)
* [Eksperymenty i wyniki (W&B)](#eksperymenty-i-wyniki-wb)
* [Model i Model Card](#model-i-model-card)
* [Środowisko i instalacja](#środowisko-i-instalacja)
* [Uruchomienie lokalne (API + UI)](#uruchomienie-lokalne-bez-dockera)
* [Docker i docker-compose](#docker-i-docker-compose)
* [Wdrożenie w chmurze (GCP Cloud Run)](#wdrożenie-w-chmurze-gcp-cloud-run)
* [Konfiguracja: ENV i sekrety](#konfiguracja-env-i-sekrety)
* [API (FastAPI)](#api-fastapi)
* [UI (Streamlit)](#ui-streamlit)
* [Baza danych](#baza-danych)
* [Monitoring i diagnostyka](#monitoring-i-diagnostyka)
* [Testy i jakość](#testy-i-jakość)
* [Struktura repozytorium](#struktura-repozytorium)
* [Required Columns (API)](#required-columns-api)
* [Załączniki / linki](#załączniki--linki)

---

# **CEL I ZAKRES**

Celem projektu jest trenowanie i porównywanie modeli przewidujących `pct_pos_total` (odsetek pozytywnych recenzji gier Steam).
Projekt przeznaczony jest dla zespołów data science i developerów chcących przeprowadzać automatyczne eksperymenty ML.

**Ograniczenia:**

* dane pochodzą z jednego źródła (Kaggle – Steam Games Dataset)

---

# **ARCHITEKTURA (HIGH-LEVEL)**

```
Steam Dataset
      ↓
Kedro Pipeline
      ↓
Artefakty modeli i metryk
      ↓
W&B (eksperymenty, wersjonowanie, alias "production")
      ↓
FastAPI (endpoint: /predict)
      ↓
Streamlit UI [TODO]
      ↓
GCP Cloud Run [TODO]
```

Eksperymenty śledzone są w
**W&B Dashboard:** [W&B GameReviewRatio](https://wandb.ai/zurek-jakub-polsko-japo-ska-akademia-technik-komputerowych/gamereviewratio)

---

# **DANE**

**Źródło:** [Steam Games Dataset (Kaggle)](https://www.kaggle.com/datasets/artermiloff/steam-games-dataset)

**Licencja:** MIT
**Data pobrania:** 07.10.2025

**Rozmiar:** próbka 100 wierszy (`data/01_raw/sample_100.csv`)

**Target:** `pct_pos_total`
**Cechy:** metadane gier: gatunki, tagi, platformy, języki, opisy, daty, wydawcy itd.

**PII:** brak (zbiór publiczny, opis produktów, nie użytkowników)

---

# **POTOK (KEDRO)**

Uruchomienie potoku:

```
kedro run
```

### Główne nody:

* `load_raw` – ładuje raw CSV
* `basic_clean` – czyszczenie danych (daty, NA, binarizacja, one-hot, MLB)
* `split_data` – podział na train/test
* `train_autogluon` – trening AutoGluon (eksperymenty)
* `evaluate_autogluon` – liczenie RMSE
* `train_baseline` – RandomForest, baseline
* `evaluate` – RMSE baseline’u
* `choose_best_model` – wybór najlepszego modelu

### Kluczowe pliki konfiguracji:

* `conf/base/catalog.yml`
* `conf/base/parameters.yml`

### Artefakty:

* `data/03_processed/` – przetworzone dane
* `data/06_models/`

  * `baseline_model.pkl`
  * `ag_model.pkl`
  * `production_model.pkl`
  * `required_columns.json`
* `data/09_tracking/`

  * `baseline_metrics.json`
  * `ag_metrics.json`

Diagram (Kedro-Viz):

<p align="center">
  <img src="images/kedro-pipeline.svg" width="70%" />
</p>

---

# **EKSPERYMENTY I WYNIKI (W&B)**

Wszystkie eksperymenty dostępne są w panelu **W&B Dashboard:**
[W&B GameReviewRatio](https://wandb.ai/zurek-jakub-polsko-japo-ska-akademia-technik-komputerowych/gamereviewratio)

Trzy konfiguracje AutoGluon zostały uruchomione.

## **Metryki porównawcze (RMSE / MAE / R²)**

| Model / Eksperyment         | Parametry                                                  | RMSE       | MAE    | R²      |
| --------------------------- | ---------------------------------------------------------- | ---------- | ------ | ------- |
| **Baseline (RandomForest)** | `n_estimators=200`, `random_state=42`                      | **≈ 7.05** | –      | –       |
| **AutoGluon – Exp 1**       | `time_limit=30`, `medium_quality_faster_train`             | **7.4308** | 5.6989 | 0.2012  |
| **AutoGluon – Exp 2**       | `time_limit=60`, `medium_quality`                          | **7.4308** | 5.6989 | 0.2012  |
| **AutoGluon – Exp 3**       | `time_limit=120`, `high_quality_fast_inference_only_refit` | **9.0734** | 7.6147 | −0.1909 |

## **Wnioski**

* Baseline ma najlepsze metryki.
* AutoGluon nie pokazuje przewagi na małym zbiorze.
* Exp1 i Exp2 identyczne.
* Exp3 przeuczony.

## **Zapis artefaktów**

Każdy run loguje:

* `ag_model.pkl`
* `ag_metrics.json`
* `baseline_model.pkl`
* `baseline_metrics.json`
* `required_columns.json`
* `train_time_s`

Najlepszy model (alias `production`) zapisany jest jako:

```
data/06_models/production_model.pkl
```

---

# **MODEL I MODEL CARD**

Model produkcyjny:

```
data/06_models/production_model.pkl
```

Model Card:

```
docs/model_card.md
```

---

# **ŚRODOWISKO I INSTALACJA**

Wymagania:

* Python 3.11

Instalacja:

```
conda env create -f environment.yml
conda activate asi-ml
python -m ipykernel install --user --name asi-ml --display-name "Python (asi-ml)"
```

---

# **URUCHOMIENIE LOKALNE (BEZ DOCKERA)**

API:

```
uvicorn src.api.main:app --reload --port 8000
```

---

# **DOCKER I DOCKER-COMPOSE**

---

# **WDROŻENIE W CHMURZE (GCP CLOUD RUN)**

---

# **KONFIGURACJA: ENV I SEKRETY**

---

# **API (FASTAPI)**

---

# **UI (STREAMLIT)**

---

# **BAZA DANYCH**

W repozytorium znajduje się plik:

```
predictions.db
```

SQLite – lokalne logowanie predykcji z API.

---

# **MONITORING I DIAGNOSTYKA**

---

# **TESTY I JAKOŚĆ**

Uruchamianie:

```
pytest -q
pre-commit run -a
```

---

# **STRUKTURA REPOZYTORIUM**

```
GameReviewRatio/
│
├── conf/
│   ├── base/
│   │   ├── catalog.yml
│   │   └── parameters.yml
│   └── local/
│
├── data/
│   ├── 01_raw/
│   ├── 02_interim/
│   ├── 03_processed/
│   ├── 06_models/
│   │   ├── AutogluonModels/
│   │   ├── ag_model.pkl
│   │   ├── baseline_model.pkl
│   │   ├── production_model.pkl
│   │   └── required_columns.json
│   └── 09_tracking/
│       ├── baseline_metrics.json
│       └── ag_metrics.json
│
├── docs/
│   └── model_card.md
│
├── images/
│   └── kedro-pipeline.svg
│
├── src/
│   ├── api/
│       └── main.py
│   └── gamereviewratio/
│       └── pipelines/
│           └── evaluation/
│               ├── nodes.py
│               └── pipeline.py
│
├── tests/
│   └── pipelines/
│       └── evaluation/
│
├── .gitignore
├── .pre-commit-config.yaml
├── .telemetry
├── environment.yml
├── predictions.db
├── pyproject.toml
└── README.md
```

---

# **REQUIRED COLUMNS (API)**

Plik:

```
data/06_models/required_columns.json
```

zawiera listę kolumn, które muszą zostać przekazane do modelu przy inferencji.

Model nie przyjmie danych:

* z brakującymi kolumnami,
* z dodatkowymi kolumnami,
* w innej kolejności,
* z innym typem danych niż trenowane.

Przykład zawartości:

```json
[
  "release_date",
  "languages_en",
  "platform_windows",
  "tag_multiplayer",
  "genre_action"
]
```

W API odbywa się automatyczna walidacja:

1. sprawdzanie brakujących kolumn,
2. wypełnianie pustymi wartościami, jeśli pipeline to wspiera,
3. reranking kolumn do zgodnego z treningiem porządku.

Jeśli cokolwiek się nie zgadza, to API zwróci błąd:

```
400 - InvalidInputError: Missing or unexpected columns
```

---

# **ZAŁĄCZNIKI / LINKI**

* **W&B Project:** [W&B GameReviewRatio](https://wandb.ai/zurek-jakub-polsko-japo-ska-akademia-technik-komputerowych/gamereviewratio)
* **Artefakty modelu**: dostępne w W&B
* **Model Card:** `docs/model_card.md`
* **Diagram potoku:** `images/kedro-pipeline.svg`

---
