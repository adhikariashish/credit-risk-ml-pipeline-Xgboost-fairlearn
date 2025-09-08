# Credit Risk Machine Learning Pipeline 

A modular, configurable pipeline to build, evaluate, and serve a credit risk model with fairness constraints and explainability.

---

## 🧭 Executive Summary
The repository delivers a credit-risk modeling pipeline that trains, scores, and explains default risk on quarterly datasets. It emphasizes schema stability, fairness, and explainability while keeping operations simple for batch scoring.

* **What it does**: Ingests quarterly features, engineers signals, fits an XGBoost-based classifier, and produces calibrated risk scores and binary decisions; also generates SHAP explanations for global & local interpretability.
* **Why it matters**: Provides transparent, defensible risk estimates for underwriting/portfolio monitoring with built-in fairness checks and repeatable results suitable for model risk management.
* **Key design choices**:

  * **Single preprocessor** (impute + One-Hot) fit **once on train** to eliminate column-mismatch drift.
  * **Top-K + “Other” bucketing** done in feature engineering (no early OHE) to control cardinality.
  * **SMOTE on train only** (post-preprocessing) for class imbalance.
  * **Calibrated XGBoost** (optional) for probability quality.
  * **Feature names persisted** and enforced on the booster so SHAP shows human-readable labels.
* **Inputs**: `processed/<QUARTER>_features.csv` for scoring; optional `processed/<QUARTER>_features_bg.csv` for SHAP background; configuration in `pipeline_config/config.yaml`.
* **Outputs**: Versioned model bundles `models/final_model_<SERIAL>.joblib` (preprocessor, model, threshold, `feature_names`), batch predictions `predictions/<QUARTER>_preds.csv`, and SHAP plots.
* **Operational UX**: CLI for training/prediction plus bash helpers (flags-first with prompt fallbacks) that validate serials/quarters and check required files before running.
* **Governance & fairness**: Sensitive features are retained from raw/bucketed data for slice metrics; pipeline supports fairness evaluation and documentation.

*In short: a dependable, explainable, and maintainable batch credit-risk ML pipeline that risk, compliance, and engineering teams can all live with.*


## 🏗️ Architecture Overview	
flowchart LR
    subgraph Ingestion & FE
      A[Raw quarterly data\n(YYYYQn)] --> B[build_features.py\nTop-K bucketing (no OHE)]
      B --> C[processed/<QUARTER>_features.csv]
      B --> Cbg[processed/<QUARTER>_features_bg.csv\n(SHAP background)]
    end

    subgraph Training Path
      C --> D[Split\n(train/val/test)]
      D --> E[Preprocessor (fit on *train*)\nImputer + OneHotEncoder(ignore)]
      E --> F[Transform splits]
      F --> G[SMOTE (train only)]
      G --> H[XGBoost / Calibrated model]
      H --> I[[Bundle (.joblib)\npreprocessor + model + threshold + feature_names]]
    end

    subgraph Scoring Path
      C --> J[Preprocessor (transform)]
      J --> K[Predict proba + label]
      K --> L[predictions/<QUARTER>_preds.csv]
    end

    subgraph Explainability & Fairness
      Cbg --> M[Preprocessor (transform) → BG DF]
      J --> N[Sample rows to explain]
      I --> O[Set booster.feature_names]
      O --> P[SHAP TreeExplainer\n(model_output=probability)]
      M --> P
      N --> P
      P --> Q[SHAP summary / local plots]

      B --> R[Sensitive features from raw/bucketed]
      K --> S[Metrics by group (Fairlearn)]
      R --> S
    end

### Components & Responsibilities

* **Feature Engineering (`src/features/build_features.py`)**

  * Business transforms, ratios, date parts.
  * **Top-K + “Other” bucketing in place** for high-cardinality categoricals.
  * *No OHE here*; keep categoricals as `object`/`category`.

* **Preprocessor (built during training)**

  * `SimpleImputer` (numeric `median`; categorical `most_frequent`).
  * `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`.
  * **Fit once on train**, reuse via `.transform()` for val/predict/SHAP.

* **Modeling (`src/models/train_model.py`)**

  * XGBoost (optionally calibrated), threshold selection.
  * **SMOTE train-only** *after* preprocessing.
  * Persist **bundle**: `preprocessor`, `model`, `threshold`, `feature_names`.

* **Scoring (`src/models/predict_model.py`)**

  * Load bundle → drop `default_flag` and sensitive column → `preprocessor.transform` → `predict_proba` + label → CSV.

* **Explainability (SHAP)**

  * Preprocess background set to a BG DataFrame with the same columns as model input.
  * Sample from the **preprocessed** scoring DataFrame for explanations.
  * Set `model.get_booster().feature_names = feature_names` so plots show human-readable names.

* **Fairness (optional)**

  * Compute metrics by **sensitive features from raw/bucketed** columns (aligned to predictions).
  
## 📁 Repository Layout

```text
project_root/
├─ pipeline_config/          # configuration + absolute path resolution
│  ├─ __init__.py
│  └─ config.yaml            # all hyperparams, paths, templates, fairness keys
├─ data/
│  ├─ raw/                   # pristine source files by quarter (e.g., 2024Q4.csv, 2025Q1.csv)
│  ├─ interim/               # cleaned/stitched but not model-ready
│  └─ processed/             # model-ready artifacts
│     ├─ <QUARTER>_features.csv         # e.g., 2024Q4_features.csv
│     └─ <QUARTER>_features_bg.csv      # SHAP background (same schema pre-preprocessor)
├─ src/
│  ├─ __init__.py
│  ├─ data/
│  │  └─ make_dataset.py     # raw → interim/processed recipes & loaders
│  ├─ features/
│  │  └─ build_features.py   # feature engineering; Top-K bucketing IN PLACE (no OHE)
│  └─ models/
│     ├─ train_model.py      # split → fit preprocessor once → SMOTE train-only → train → bundle
│     └─ predict_model.py    # scoring CLI + SHAP (uses saved preprocessor/model/feature_names)
├─ models/                   # serialized bundles (joblib)
│  └─ final_model_<SERIAL>.joblib
├─ predictions/              # generated prediction CSVs
│  └─ <QUARTER>_preds.csv
├─ scripts/
│  └─ run_full_pipeline.sh   # flags-first; prompts if --serial/--data not supplied
├─ notebooks/                # optional exploratory work (EDA, experiments)
│  ├─ 00_define_problem.ipynb
│  ├─ 01_EDA.ipynb
│  └─ 02_model_experiments.ipynb
├─ requirements.txt
└─ README.md
```

## 🛠️  Setup & Installation 

**1) Clone the repo**
```bash
git clone <YOUR-REPO-URL>.git
cd <YOUR-REPO-FOLDER>
```
**2) Install requirements**
```bash
python -m venv .venv && source .venv/bin/activate   
# (Windows PowerShell: .\.venv\Scripts\Activate.ps1)
pip install -r requirements.txt
```
**3) Add your quarter(s) to data/raw/**
```bash 
data/raw/
└─ 2024Q4.csv           # example 
# (add more: 2025Q1.csv, etc.)
```
**4) Run the pipeline**
```bash
bash scripts/run_full_pipeline.sh --quarters 2024Q4
# or multiple:
bash scripts/run_full_pipeline.sh --quarters 2024Q4,2025Q1
```

## ⚙️ Configuration (`pipeline_config/config.yaml`)
## Configuration (Quick)

All settings live in **`pipeline_config/config.yaml`**.  
Edit these minimal keys before running the pipeline:

- **Paths**
  - `data.processed_dir` → where `<QUARTER>_features.csv` live
  - `model.output_dir` → where model bundles are saved
  - `model.prediction_dir` → where predictions are written

- **Templates**
  - `templates.processed` → processed features filename (per quarter)
  - `templates.processed_bg` → SHAP background filename (per quarter)
  - `model.output_template` → model bundle filename

- **Model**
  - `model.random_seed` → reproducibility
  - `model.resample` → `smote` or `none`
  - `model.fairlearn.sensitive_feature` → column name to exclude from features & use for slices

### Minimal example (`pipeline_config/config.yaml`)
```yaml
data:
  processed_dir: data/processed

templates:
  processed: "{quarter}_features.csv"         # e.g., 2024Q4_features.csv
  processed_bg: "{quarter}_features_bg.csv"   # e.g., 2024Q4_features_bg.csv

model:
  output_dir: models
  prediction_dir: predictions
  output_template: "final_model_{quarter}.joblib"
  random_seed: 42
  resample: smote                             # or: none
  fairlearn:
    sensitive_feature: property_state         # name present in processed CSV
```

**Naming conventions**

- <QUARTER> = YYYYQn (e.g., 2024Q4, 2025Q1)

- <SERIAL> = one or more quarters joined by _ (e.g., 2024Q4_2025Q1)

- Files expected:

	- data/processed/<QUARTER>_features.csv

	- data/processed/<QUARTER>_features_bg.csv (optional, for SHAP)

	- models/final_model_<SERIAL>.joblib (created by training)


## 🧹 Data Management (schemas & splits)
## Data Management

### Data Contracts & Schemas
- **Unit of observation:** one row per entity–time (e.g., loan per `YYYYQn`).
- **Minimum required columns (examples):**
  - Identifier(s): `record_id` (unique), `as_of_quarter` (`YYYYQn`)
  - Features: numeric & categorical fields used by the model (no label leakage)
  - Label (for training only): `default_flag` (0/1)
  - Sensitive feature (kept for fairness, **not** fed to model): e.g., `property_state`  
- **Types:**  
  - Numeric → `int`/`float` (no strings in numeric columns)  
  - Categorical → `string`/`category` (not already one-hot encoded)  
  - Dates → ISO strings or pandas `datetime64[ns]` (consistent timezone policy)

### Files & Locations
- **Processed features (per quarter):** `data/processed/<QUARTER>_features.csv`
- **SHAP background (per quarter):** `data/processed/<QUARTER>_features_bg.csv`
- **Predictions (outputs):** `predictions/<QUARTER>_preds.csv`

**Conventions**
- `<QUARTER>` = `YYYYQn` (e.g., `2024Q4`)
- Model bundle name uses **serials** joined by `_` (e.g., `final_model_2024Q4_2025Q1.joblib`)

### Partitions & Splits
- **Partitioning:** by quarter (`YYYYQn`) to preserve temporal ordering.
- **Training splits:** temporal split into **train/val/test**; avoid random mixing across quarters unless justified.

### Validation (recommended checks)
- **Schema:** required columns present; no unexpected extra columns at inference.
- **Dtypes:** numeric columns are numeric; categoricals are not one-hot encoded.
- **Ranges/nulls:** no impossible values; missing rates within expected bounds.
- **Uniqueness:** `record_id` unique within a quarter (or defined composite key).
- **Drift sanity:** basic stats vs. prior quarters (mean/std/category share deltas).

### Sensitive Features (fairness)
- Keep the **raw/bucketed** sensitive attribute (e.g., `property_state`) in the dataset for slicing metrics.
- **Do not** feed sensitive features to the model input; drop before preprocessing/inference.

### SHAP Background Guidance
- Build `<QUARTER>_features_bg.csv` as a **representative sample** of the quarter(s) used for explanation (e.g., stratified by label or key segments).
- At explanation time, **preprocess** the background with the saved preprocessor so its columns match the model’s feature space exactly.

### Data Hygiene & Governance
- **No PII** beyond agreed contract; mask or hash identifiers if needed.
- Track data provenance (source, extract date, filters).
- Version datasets by quarter; do not overwrite historical artifacts.

## 📚 Dataset & Compliance Checklist (Public GSE Data)

### Source we reference
- **Freddie Mac — Single-Family Loan-Level Dataset** (SF LLD)  
  - Dataset page & terms summary (non-commercial/research use; redistribution requires license). :contentReference[oaicite:0]{index=0}  
  - **Terms / Additional Terms**: redistribution or *derived products* require a separate license; otherwise prohibited. :contentReference[oaicite:1]{index=1}  
  - **User Guide** (fields, definitions, data coverage) & **Release Notes** (field changes, e.g., Property Valuation Method values). :contentReference[oaicite:2]{index=2}

> If you are instead using the **Fannie Mae Single-Family Loan Performance Data**, review their dataset page & FAQs and follow similar no-redistribution rules. :contentReference[oaicite:3]{index=3}

---

### What we do (required)
- **No raw data in repo**  
  We **do not commit** SF LLD raw files. Users must obtain data directly from Freddie Mac under their terms. :contentReference[oaicite:4]{index=4}
- **No redistribution of derived products**  
  We **do not publish** repackaged subsets, row-level predictions, or “derived products” that could substitute for the dataset without a redistribution license. We ship **code only**; example outputs (if any) are **aggregated**. :contentReference[oaicite:5]{index=5}
- **Attribution & license notice**  
  We clearly attribute Freddie Mac as the data source and link to their terms. (See “Dataset Attribution” below.) :contentReference[oaicite:6]{index=6}
- **Data placement**  
  Local paths only (e.g., `data/raw/`, `data/processed/`).
- **PII & re-identification**  
  We do not attempt re-identification; we operate on released fields only (see User Guide). :contentReference[oaicite:7]{index=7}
- **Schema drift awareness**  
  We monitor **Release Notes** for field value updates (e.g., `Property Valuation Method` new codes). :contentReference[oaicite:8]{index=8}

---

### How to obtain data (user)
1. Visit the dataset page and **accept terms** to access the files. :contentReference[oaicite:9]{index=9}  
2. Download the desired quarterly files and place them under your local `data/raw/` (or use your own staging).  
3. Run the repository scripts to transform into `data/processed/<QUARTER>_features.csv`.

---

### Dataset Attribution (example)
> **Data**: Freddie Mac **Single-Family Loan-Level Dataset**.  
> Access and use are subject to Freddie Mac’s **Terms of Use** and **Additional Terms**. This repository **does not redistribute** the dataset or any derived product; users must obtain the data directly from Freddie Mac. :contentReference[oaicite:10]{index=10}


## 🧱 Feature Engineering (Top-K bucketing; no OHE)

### Goals

* Create stable, informative predictors while **avoiding target leakage**.
* Keep the **model input schema consistent** across quarters (no surprise columns).

### Inputs → Outputs

* **Input:** raw/interim quarterly tables.
* **Output (per quarter):** `data/processed/<QUARTER>_features.csv` containing:

  * predictor columns (numeric + categorical **not OHE’d**),
  * `default_flag` (training only),
  * configured `sensitive_feature` (kept for fairness, **not** used by the model).

### Transformations (what we do)

* **Categorical (high-cardinality):**

  * **Top-K + “Other” bucketing in place** to cap levels.
  * Keep as `object/category`; **do not** one-hot encode here (OHE happens later in the preprocessor).
  * **Freeze mapping on train**; reuse for validation/test/predict to avoid drift.

* **Categorical (low-cardinality):**

  * Leave values as-is (still no OHE here).

* **Sensitive feature(s):**

  * Preserve a clean copy (raw or bucketed) for **fairness slicing**.
  * Exclude from model features at inference.

* **Numeric features:**

  * Domain transforms (e.g., ratios, log1p where appropriate).
  * Optional clipping/winsorization for extreme outliers (document if applied).
  * No scaling required for tree models (keep simple).

* **Dates/tenure:**

  * Derive safe aggregates (e.g., ages, months-on-book).
  * Respect temporal boundaries (no peeking into the future).

* **Text/IDs:**

  * Drop free text; keep stable identifiers (`record_id`, `as_of_quarter`) for joins and auditing.

### Leakage & Reproducibility

* Fit **any data-dependent mapping on train only** (e.g., Top-K level sets).
* build composite features based on the available features

### Minimal Top-K Bucketing Pattern (in place; no OHE)

```python
def bucket_topk_inplace(df, cols, k=10, other_label="Other", include_missing=True):
    for col in cols:
        s = df[col].astype("string")
        if include_missing:
            s = s.fillna("Missing")
        keep = s.value_counts(dropna=False).nlargest(k).index
        df[col] = s.where(s.isin(keep), other_label)
    return df
```

### Quality Checks (recommended)

* Cardinality after bucketing ≤ K for each targeted column.
* No unexpected **new** categories post-mapping.
* Missingness and basic stats within expected ranges quarter-over-quarter.
* `record_id` uniqueness and consistent `as_of_quarter` tagging.


## 🛠️ Preprocessing (single ColumnTransformer)

### Goals

* Provide a **single source of truth** for imputations & encodings.
* Guarantee **schema stability** across train/val/test/predict.
* Keep inference simple: **fit once on train, transform everywhere else**.

### Design

* Implement a **`ColumnTransformer`** that:

  * **Numeric**: `SimpleImputer(strategy="median")`
  * **Categorical**: `SimpleImputer(strategy="most_frequent")` → `OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype="float32")`
* Treat **boolean** columns as numeric (cast to `0/1`) to avoid accidental OHE.
* **No OHE in feature engineering**; all OHE happens here.

### Common Pitfalls

* **“columns are missing”** → You OHE’d earlier or used a different preprocessor. Use the **train-fitted** preprocessor and raw/bucketed categoricals.
* **XGBoost dtype error (object)** → Ensure you call `preprocessor.transform` and cast to `float32`.
* **SHAP showing numbers** → Persist `feature_names` and set `booster.feature_names` before explaining.


## 🧪 Training Workflow (SMOTE train-only)

### Goals
- Train a robust, explainable model with **schema-stable features**.
- Apply **SMOTE only on the training split** (after preprocessing).
- Persist everything needed for inference: **preprocessor**, **model**, **threshold**, **feature_names**.

### Steps
1. **Load processed features** for one or more quarters (e.g., `2024Q4`, `2025Q1`).
2. **Separate columns**:
   - Features (no label, no sensitive attribute)
   - Label: `default_flag` (train/val/test only)
   - Sensitive attribute (kept for fairness metrics; **not** fed to model)
3. **Split** into **train / val / test** (temporal split preferred).
4. **Build preprocessor** on **train** only:
   - Numeric: `SimpleImputer(median)`
   - Categorical: `SimpleImputer(most_frequent)` → `OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype="float32")`
5. **Transform** train/val/test with the **fitted** preprocessor (no refits).
6. **SMOTE** (if enabled): apply **after preprocessing**, **train only**.
7. **Train model** (e.g., XGBoost or calibrated variant) on `X_train_pre[,_smoted]`.
8. **Select threshold** on **validation** (optimize for your business metric).
9. **Evaluate** on **test** (AUROC/AUPRC, calibration, fairness slices using the sensitive attribute).
10. **Persist bundle**: `{"preprocessor","model","threshold","feature_names"}` to `models/final_model_<SERIAL>.joblib`  
    - `<SERIAL>` is one or more quarters joined by `_` (e.g., `2024Q4_2025Q1`).

### Quick CLI (example)

```bash
python src/models/train_model.py --quarters 2024Q4,2025Q1
# -> models/final_model_2024Q4_2025Q1.joblib
```

### Notes & Gotchas

* **Do not** one-hot encode in feature engineering; all OHE lives in the **preprocessor**.
* **Never** refit the preprocessor on val/test; always **transform** with the train-fitted one.
* Keep dtypes numeric at model input (`float32` recommended for XGBoost).
* Save **feature\_names** in the bundle; they’re needed for SHAP and auditing.

## 📊 Evaluation (metrics & slices)

### What we report
- **Discrimination:** AUROC, AUPRC (practical for imbalanced data)
- **At chosen threshold:** Precision, Recall, F1, Specificity, FPR/TPR, Confusion matrix
- **Calibration:** Brier score (lower is better); optional calibration plot
- **Slices (fairness):** Metrics **by sensitive group** (e.g., `property_state` bucketed Top-K+Other)
  - Disparities: **difference** (max–min) and **ratio** (min/max) for selection rate, TPR, FPR, etc.

> Evaluate on a **temporal holdout** (test quarter) after picking the threshold on **validation**.

### Suggested table (example)

| Metric          | Value       |
| --------------- | ----------- |
| AUROC           | 0.81        |
| AUPRC           | 0.42        |
| Brier           | 0.16        |
| Precision @ thr | 0.31        |
| Recall @ thr    | 0.62        |
| F1 @ thr        | 0.41        |
| TPR / FPR       | 0.62 / 0.18 |

## 🏆 Results & Metrics (achieved)

**Model version (SERIAL):** `2024Q4_2025Q1`  
**Saved bundle:** `models/final_model_2024Q4_2025Q1.joblib`  
**Decision policy:** choose threshold under a **top-percentile budget ≤ 5%** of scored accounts.

---

### Validation (after tuning & calibration)
| Metric | Tuned | Calibrated |
|---|---:|---:|
| ROC-AUC | **0.9102** | **0.9118** |
| Average Precision (AUPRC) | **0.1287** | **0.1235** |

**Threshold selection (policy)**  
Candidates with `top_pct ≤ 0.05`:
- `top_pct=0.01` → `thr=0.0509`, **P**=0.1019, **R**=0.3223  
- `top_pct=0.02` → `thr=0.0336`, **P**=0.0780, **R**=0.3858 ✅ *(chosen)*  
- `top_pct=0.05` → `thr=0.0107`, **P**=0.0391, **R**=0.5533  

**Chosen threshold:** `0.0336` (≈ **2%** selection rate on validation)

---

### Fairness (validation)
Fairness reduction: **Demographic Parity** with ε = **0.02** (Fairlearn reductions).

| Metric (val) | Base | Fair |
|---|---:|---:|
| Accuracy | 0.9728 | **0.9742** |
| ROC-AUC | **0.9118** | 0.9074 |
| Log-loss | **0.0207** | 0.0230 |
| Reported parity gap (internal) | 0.0169 | ~0.0194 |

> The *fair* model is selected to comply with the DP constraint; it trades a small AUC for parity.

---

### Final Test Evaluation (using threshold ≈ **0.03**)
| Metric | Value |
|---|---:|
| ROC-AUC | **0.9146** |
| Average Precision (AUPRC) | **0.1243** |
| Precision @ thr | **0.0868** |
| Recall @ thr | **0.3799** |
| Accuracy | **0.9794** |

*(Class-1 report excerpt @ thr ≈ 0.03: P=0.0668–0.0868, R≈0.38, F1≈0.11; support ≈ 787.)*

---

### Explainability (SHAP) — top global drivers
1. `num__loan_age`  
2. `cat__first_time_home_buyer_indicator_{N,Y}`  
3. `num__upb_ratio`  
4. `num__days_to_maturity`  
5. `cat__seller_name_Other`  
6. `num__original_upb`  
7. `num__months_since_origination`  
8. `cat__servicer_name_Other`  
9. `num__number_of_borrowers`  
10. `num__total_principal_current_imputed`


## 🗣️ What these results mean (Stakeholder summary)

- **Operating point chosen:** review the **top ~2%** highest-risk loans each quarter.
- **Effectiveness at this point:** we capture **~38% of all future defaults** while keeping the review workload small.

### What that looks like in practice (per 10,000 loans)
- **Flagged for review:** ~**200** loans (2%)
- **Expected true defaults among flags (precision ≈ 8.7%)**: **~17**
- **Expected non-defaults to review:** **~183**

> In other words: with ~200 case reviews, we surface ~17 likely defaults.  
> Pushing beyond 2% increases work a lot but adds relatively few extra defaults:
> - **1% workload:** ~100 reviews → **~10** true defaults, **~90** non-defaults  
> - **5% workload:** ~500 reviews → **~20** true defaults, **~480** non-defaults

### Why we picked 2%
- It’s near the best trade-off between **defaults caught** and **operational cost**:
  - **1%** misses too many defaults (recall ~32%) even though precision is slightly higher.
  - **5%** catches more defaults (recall ~55%) but **more than doubles** workload while precision halves (~3.9%), so each additional review yields fewer true positives.

### Quality & fairness
- Overall discrimination is strong (**AUC ≈ 0.915** on test).
- We apply a **demographic parity** fairness constraint (ε≈0.02), incurring only a small AUC trade-off to keep selection rates more even across groups.


## ⚖️ Fairness (sensitive features & checks)

### Goals
- Evaluate model behavior **across groups** defined by a sensitive attribute.
- Keep sensitive data **out of the model features** but **in the dataset** for slicing and reporting.
- Provide clear **gap/ratio** summaries and an auditable report.

### Sensitive Feature Handling
- **Source:** use a raw or **Top-K+Other bucketed** column (e.g., `property_state` → top 10 + `Other`).
- **Training:** keep the sensitive column in the dataframe but **exclude it** from the feature set passed to the preprocessor/model.
- **Inference:** keep the same column alongside predictions (index-aligned) for slicing.
- **Stability:** if you bucket (Top-K), **freeze the mapping on train** and reuse it for all future scoring to avoid shifting groups.

### Evaluation Procedure
1. Choose a **global decision threshold** on the validation set (business-aligned).
2. On the **test** (or scoring) set, compute overall metrics and **by-group** metrics.
3. Summarize **disparities** as both **difference (max–min)** and **ratio (min/max)**.
4. Save an artifact: CSV/JSON with overall, per-group, and disparity summaries.

## 🔍 Explainability (SHAP)

**Purpose:** quantify global and local feature influence on the model’s predictions in the **same feature space the model sees** (post-preprocessing).

### Inputs
- Background sample (per quarter): `data/processed/<QUARTER>_features_bg.csv` (representative rows)
- Scoring features: `data/processed/<QUARTER>_features.csv`
- Saved bundle: `models/final_model_<SERIAL>.joblib` (must include `preprocessor`, `model`, `feature_names`)

### How it works
1. **Load bundle** → get `preprocessor`, `model`, `feature_names`.
2. **Set booster names** (for XGBoost) so plots show real labels:
   ```python
   try: model.get_booster().feature_names = list(feature_names)
   except Exception: pass

### CLI usage
``` bash
python src/models/predict_model.py \
  --serial 2024Q4_2025Q1 \
  --data 2024Q4 \
  --explain --shap-bg 2024Q4 --shap-frac 0.10
```

## 📦 Packaging & Artifacts (bundle contents + naming)

### What gets packaged
A single **joblib bundle** per model version (**SERIAL**) containing everything needed for deterministic inference:

```text
models/final_model_<SERIAL>.joblib
```
**Bundle contents (dict):**

* `preprocessor` → fitted `ColumnTransformer` (impute + OHE once)
* `model` → trained estimator (e.g., XGBoost or calibrated wrapper)
* `threshold` → decision cutoff chosen on validation
* `feature_names` → `list[str]` from `preprocessor.get_feature_names_out()`

> **SERIAL** = one or more quarters joined by `_` (e.g., `2024Q4_2025Q1`).

---

### Naming conventions

* **Model bundle:** `models/final_model_<SERIAL>.joblib`
* **Predictions:** `predictions/<QUARTER>_preds.csv`
* **Processed features (input):** `data/processed/<QUARTER>_features.csv`
* **SHAP background:** `data/processed/<QUARTER>_features_bg.csv`

---

### Artifact checklist per run

* ✅ **Model bundle**: `models/final_model_<SERIAL>.joblib`
* ✅ **Predictions CSV** (per scored quarter): `predictions/<QUARTER>_preds.csv`

---

### Versioning tips

* Treat `<SERIAL>` as a **model version** (data window encoded in the name).
* Keep older bundles; never overwrite—enable easy rollback & A/B comparisons.

---

## 🖥️ Inference / Serving (batch scoring CLI)

### Prereqs
- Trained bundle: `models/final_model_<SERIAL>.joblib`
- Features to score: `data/processed/<QUARTER>_features.csv`
- *(Optional)* SHAP background: `data/processed/<QUARTER>_features_bg.csv`

### Python CLI
```bash
python src/models/predict_model.py \
  --serial 2024Q4_2025Q1 \
  --data 2024Q4 \
  --output-csv predictions/2024Q4_preds.csv
````

**Flags**

* `--serial` : model version (e.g., `2024Q4` or `2024Q4_2025Q1`)
* `--data`   : quarter to score (e.g., `2024Q4`)
* `--output-csv` : path to save predictions (defaults to config)
* `--explain` : also run SHAP explanations
* `--shap-bg` : background quarter for SHAP (e.g., `2024Q4`)
* `--shap-frac` : fraction of rows to explain (e.g., `0.10`)

**With SHAP**

```bash
python src/models/predict_model.py \
  --serial 2024Q4_2025Q1 \
  --data 2024Q4 \
  --explain --shap-bg 2024Q4 --shap-frac 0.10
```

### Bash helper (flags-first, then prompts if missing)

```bash
bash scripts/run_full_pipeline.sh \
  --serial 2024Q4_2025Q1 \
  --data 2024Q4 \
  --explain --shap-bg 2024Q4 --shap-frac 0.2
```

### What the scorer does

1. **Load bundle:** `preprocessor`, `model`, `threshold`, `feature_names`.
2. **Load features:** `processed/<QUARTER>_features.csv`; drop `default_flag` and the configured `sensitive_feature`.
3. **Transform:** `preprocessor.transform(...)` → numeric matrix (`float32`).
4. **Predict:** probabilities + binary label (using saved `threshold`).
5. **Save CSV:** `predictions/<QUARTER>_preds.csv`.
6. *(If `--explain`)* Preprocess SHAP BG, sample rows, build `TreeExplainer(model_output="probability")`, plot summary.

### Outputs

* `predictions/<QUARTER>_preds.csv` with:
* SHAP plots (displayed; save manually if needed)

### Validation & guards

* Serial format: `YYYYQn` or `YYYYQn_YYYYQm` (e.g., `2024Q4_2025Q1`)
* Files exist: bundle + `<QUARTER>_features.csv` (+ BG if using SHAP)
* Schema: transformed columns count == `len(feature_names)`

### Common errors (quick fixes)

* **Invalid serial/quarter** → use `YYYYQn`; for multiple quarters in `--serial`, join with `_`.
* **File not found** → ensure paths match `pipeline_config/config.yaml`.
* **“columns are missing”** → don’t pass pre-OHE’d data; let the saved **preprocessor** do OHE once.
* **XGBoost dtype error (object)** → you skipped transform; always use `preprocessor.transform` → `float32`.
* **SHAP shows numbers** → set `model.get_booster().feature_names = feature_names` and plot with the **Explanation** + **DataFrame** (no `.values`).


## 🔁 Retraining Strategy (rolling vs. once-and-score)
### Scope
- **Batch-only** training & scoring
- **Manual window selection** via `--quarters` (one or multiple)
- **Single train-fitted preprocessor** (impute + OHE once)
- **SMOTE on train only** (after preprocessing)
- **Threshold chosen on validation**
- **Test evaluation + (optional) fairness slices**
- **Bundle saved** as `models/final_model_<SERIAL>.joblib`
- **Optional SHAP** on demand during scoring

### Simple procedure
1. **Pick training quarters** (e.g., `2024Q4,2025Q1`) → this becomes the **SERIAL** `2024Q4_2025Q1`.
2. **Train**  
   - Temporal split into **train/val/test**  
   - Fit **preprocessor on train**; transform val/test  
   - Apply **SMOTE train-only**  
   - Train model; **select threshold** on validation  
   - Evaluate on test (and optional fairness slices)  
   - **Save bundle** with `preprocessor`, `model`, `threshold`, `feature_names`
3. **Score a quarter** with the saved bundle (optionally run SHAP).

### CLI examples
```bash
# Train on multiple quarters (creates bundle)
python src/models/train_model.py --quarters 2024Q4,2025Q1
# -> models/final_model_2024Q4_2025Q1.joblib

# Score a specific quarter
python src/models/predict_model.py --serial 2024Q4_2025Q1 --data 2024Q4

# Score with SHAP explanations
python src/models/predict_model.py --serial 2024Q4_2025Q1 --data 2024Q4 \
  --explain --shap-bg 2024Q4 --shap-frac 0.10
```
### Minimal checklist (per retrain)

* [ ] Choose quarters → **SERIAL** (e.g., `2024Q4_2025Q1`)
* [ ] Fit preprocessor on **train only**; transform val/test
* [ ] **SMOTE** on train (post-preprocessing)
* [ ] Pick **threshold** on validation
* [ ] Evaluate on **test** (+ optional fairness slices)
* [ ] Save **bundle** (`preprocessor`, `model`, `threshold`, `feature_names`)
* [ ] (Optional) Document quick metrics & SHAP screenshot


## 🧯 Troubleshooting & FAQ

### Model & Preprocessing

**Q: “ValueError: columns are missing: {...}” during `preprocessor.transform`**  
**A:** You likely passed already OHE’d data or used a different preprocessor than the one saved in the bundle.  
- Use **raw/bucketed** categoricals (no OHE in features).  
- Always load the **train-fitted** preprocessor from the bundle and call `.transform(...)`.  


## 📈 Results & Artifacts (where outputs land)

### Primary outputs
- **Model bundle** → `models/final_model_<SERIAL>.joblib`  
  Contains: `preprocessor`, `model`, `threshold`, `feature_names`
- **Predictions (per quarter)** → `predictions/<QUARTER>_preds.csv`  
  Columns include: `proba`, `pred_label` (and optionally feature columns if you keep them)

### Inputs kept for reproducibility
- **Processed features (per quarter)** → `data/processed/<QUARTER>_features.csv`
- **SHAP background (per quarter, optional)** → `data/processed/<QUARTER>_features_bg.csv`

### Example after a train + score run


models/
└─ final_model_2024Q4_2025Q1.joblib

predictions/
└─ 2024Q4_preds.csv

data/processed/
├─ 2024Q4_features.csv
├─ 2024Q4_features_bg.csv
└─ 2025Q1_features.csv


### Naming conventions
- `<QUARTER>` = `YYYYQn` (e.g., `2024Q4`)  
- `<SERIAL>` = one or more quarters joined by `_` (e.g., `2024Q4_2025Q1`)

> Tip: Never overwrite bundles. Keep all versions to enable rollbacks and side-by-side comparisons.

## 🗺️ Roadmap / Future Work
- Add automated data/schema checks before train/score.
- Export overall + by-group metrics to `reports/` (CSV/JSON).
- Save SHAP plots alongside predictions.
- Improve CLI UX: `--help`, `--dry-run`, clearer errors.
- Add unit tests (transformers) + smoke test (tiny end-to-end score).
- Track bundles and metrics in a simple registry (e.g., MLflow).
- Add basic drift monitors (data/prediction) with alert thresholds.
