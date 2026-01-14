🇵🇱 [Polski](model_card.md)

# **Model Card – Game Review Ratio (Production Model)**

## **1. Problem and Intended Use**

The model is designed to **predict the percentage of positive game reviews (`pct_pos_total`)** based on public metadata of games available on the Steam platform.

### Intended use cases

* quick quality assessment of a game without reading thousands of reviews,
* support for game developers in market analysis and benchmarking,
* data analysis: identifying which game attributes correlate with positive reception,
* exploration of trends across genres, tags, platforms, and other metadata.

### Out-of-scope / not recommended uses

* financial, business, or investment decisions without human oversight,
* using the model to evaluate individual users (the dataset does not contain user data).

The model is **analytical and decision-supporting**, not a decision-making system.

---

## **2. Data (source, license, size, PII)**

### Data source

Data: [Steam Games Dataset (Kaggle)](https://www.kaggle.com/datasets/artermiloff/steam-games-dataset)

### License

MIT (according to the dataset description).

### Dataset size

* Target full dataset: approx. 5,000 games.

### Data content

* genres, tags, categories,
* supported languages,
* platforms (Windows / Mac / Linux),
* publisher and developer,
* release date,
* number of reviews and positive review percentage,
* textual descriptions and additional metadata.

### Personally identifiable information (PII)

None.
The dataset contains **no user data**. It only includes product-level information about games.

---

## **3. Metrics**

### Primary metric

**RMSE (Root Mean Squared Error)**
Chosen because it penalizes large prediction errors and is a standard regression metric.

### Validation

* data split: **80% train / 20% test**, random,
* reproducibility: `random_state = 42`.

### Results (production model – baseline)

* **RMSE (test): ~7.05**
* **MAE:** –
* **R²:** –
  *(the baseline model logs only RMSE)*

### Benchmark models (AutoGluon)

| Experiment     | Parameters                                                 | RMSE  | MAE   | R²    |
| -------------- | ---------------------------------------------------------- | ----- | ----- | ----- |
| **AG – Exp 1** | `time_limit=30`, `medium_quality_faster_train`             | ~7.43 | ~5.69 | ~0.20 |
| **AG – Exp 2** | `time_limit=60`, `medium_quality`                          | ~7.43 | ~5.69 | ~0.20 |
| **AG – Exp 3** | `time_limit=120`, `high_quality_fast_inference_only_refit` | ~9.07 | ~7.61 | −0.19 |

### Conclusions

* **The baseline model is the best (lowest RMSE)** and was selected as the production model.
* AutoGluon did not outperform the baseline on the available data sample.
* Experiments AG 1 and AG 2 produce identical results — longer training time does not improve quality.
* The `high_quality_fast_inference_only_refit` preset (Exp 3) leads to overfitting.

---

## **4. Limitations and Risks**

### Limitations

* the model relies exclusively on metadata,
* large differences between AAA and indie games may introduce bias,
* niche and older titles are under-represented.

### Risks

* misinterpretation of predictions without domain context,
* false confidence in game quality,
* using the model for decisions it was not designed for.

### Risk mitigation

* regular retraining and monitoring in Weights & Biases,
* error analysis and rejection of high-uncertainty predictions.

---

## **5. Versioning**

### W&B run (production model)

[W&B Dashboard — GameReviewRatio](https://wandb.ai/zurek-jakub-polsko-japo-ska-akademia-technik-komputerowych/gamereviewratio)

### Model artifact

`gamereviewratio/baseline_model:production`

*(The AutoGluon model was kept as a candidate but not selected for production.)*

### Code version

Commit: `575d69d`
(`kedro run` executed from this commit)

### Data version

File: `data/01_raw/sample_100.csv`
Source: Steam Games Dataset (Kaggle)

### Environment

* Python 3.11
* AutoGluon 1.x (benchmark models)
* scikit-learn 1.5
* Kedro 1.0
* wandb
* pandas, numpy, pyarrow
