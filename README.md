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

**Panel W&B (eksperymenty):** https://wandb.ai/zurek-jakub-polsko-japo-ska-akademia-technik-komputerowych/gamereviewratio

---

# **DANE**

**Źródło (Kaggle):** https://www.kaggle.com/datasets/artermiloff/steam-games-dataset

**Licencja:** MIT
**Data pobrania:** 07.10.2025

**Target:** `pct_pos_total`

**Cechy:** metadane gier (gatunki, tagi, platformy, języki, daty wydania, wydawcy itd.)

**PII:** brak

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

---

# **EKSPERYMENTY I WYNIKI (W&B)**

**Panel W&B (wszystkie runy):** https://wandb.ai/zurek-jakub-polsko-japo-ska-akademia-technik-komputerowych/gamereviewratio

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

```
conda env create -f environment.yml
conda activate asi-ml
python -m ipykernel install --user --name asi-ml --display-name "Python (asi-ml)"
```

---

# **URUCHOMIENIE LOKALNE (BEZ DOCKERA)**

```
uvicorn src.api.main:app --reload --port 8000
```

---

# **DOCKER I DOCKER-COMPOSE**

```
docker compose up --build
```

Healthcheck:
```
curl http://localhost:8000/healthz
```

---

# **API (FastAPI)**

Uruchomienie:
```
uvicorn src.api.main:app --reload --port 8000
```

Healthcheck:
```
curl http://localhost:8000/healthz
```

---

# **UI (STREAMLIT)**

```
http://localhost:8501
```

---

# **BAZA DANYCH**

SQLite – lokalne logowanie predykcji z API (tryb lokalny, bez Dockera).
W docker-compose wykorzystywana jest baza PostgreSQL w kontenerze.

---

# **TESTY I JAKOŚĆ**

```
pytest -q
pre-commit run -a
```

---

# **WYMAGANE KOLUMNY (API)**

```
data/06_models/required_columns.json
```

---

# **ZAŁĄCZNIKI / LINKI**

**W&B Project:** https://wandb.ai/zurek-jakub-polsko-japo-ska-akademia-technik-komputerowych/gamereviewratio

**Dataset (Kaggle):** https://www.kaggle.com/datasets/artermiloff/steam-games-dataset

**Model Card:**
docs/model_card.md

**Artefakty modelu:**
data/06_models/
