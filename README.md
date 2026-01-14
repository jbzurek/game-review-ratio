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
* [Wymagane kolumny (API)](#required-columns-api)
* [Załączniki / linki](#załączniki--linki)

---

# **CEL I ZAKRES**

Celem projektu jest trenowanie i porównywanie modeli przewidujących `pct_pos_total` (odsetek pozytywnych recenzji gier Steam).
Projekt przeznaczony jest dla zespołów data science i developerów chcących przeprowadzać automatyczne eksperymenty ML.

**Ograniczenia:**

* dane pochodzą z jednego źródła (Kaggle – Steam Games Dataset)

---

# **ARCHITEKTURA (POZIOM HIGH-LEVEL)**

```

Steam Dataset
    |
Kedro Pipeline
    |
Artefakty modeli i metryk
    |
W&B (eksperymenty, wersjonowanie, alias "production")
    |
FastAPI (endpoint: /predict)
    |
Streamlit UI
    |
GCP Cloud Run

```

Eksperymenty śledzone są w **Panelu W&B:** [W&B GameReviewRatio](https://wandb.ai/zurek-jakub-polsko-japo-ska-akademia-technik-komputerowych/gamereviewratio)

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

---

# **EKSPERYMENTY I WYNIKI (W&B)**

Wszystkie eksperymenty dostępne są w panelu [W&B GameReviewRatio](https://wandb.ai/zurek-jakub-polsko-japo-ska-akademia-technik-komputerowych/gamereviewratio).

Zostały uruchomione trzy konfiguracje AutoGluon.

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

# **MODEL I KARTA MODELU**

Model produkcyjny:

```
data/06_models/production_model.pkl
```

Karta modelu:

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

# **KONFIGURACJA: ZMIENNE ŚRODOWISKOWE I SEKRETY**

---

# **API (FastAPI)**

---

## Szybki start: uruchomienie API

### 1. Uruchom API

W katalogu głównym repozytorium:

```
uvicorn src.api.main:app --reload --port 8000
```

---

### 2. Sprawdź status serwisu

Endpoint `GET /healthz`:

```
curl http://127.0.0.1:8000/healthz
```

Oczekiwany wynik:

```
{"status": "ok"}
```

---

### 3. Wykonaj predykcję

Endpoint `POST /predict`

#### Windows (curl.exe)

```
curl.exe --% -X POST "http://127.0.0.1:8000/predict" ^
  -H "Content-Type: application/json" ^
  -d "{\"data\":{\"appid\":12345,\"price\":0.0,\"user_score\":7.5}}"
```

#### PowerShell

```
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict `
  -Method POST `
  -Headers @{ "Content-Type" = "application/json" } `
  -Body '{"data": {"appid": 12345, "price": 0.0, "user_score": 7.5}}'
```

---

## Jak podejrzeć zapisaną bazę (SQLite)

## Jak podejrzeć zapisy w bazie (SQLite)

* jeśli masz narzędzie `sqlite3`:

```
sqlite3 predictions.db "select id, ts, payload, prediction, model_version from predictions order by id desc limit 5;"
```
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

---

# **WYMAGANE KOLUMNY (API)**

Plik:

```
data/06_models/required_columns.json
```

---

# **ZAŁĄCZNIKI / LINKI**

* **W&B Project**
* **Artefakty modelu**
* **Model Card**
* **Diagram potoku**
