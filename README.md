# Ukrainian War Restoration Projects — End-to-End Binary Classification

[![CI](https://github.com/VKK-00/dream-ml-project/actions/workflows/ci.yml/badge.svg)](../../actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository contains a reproducible ML pipeline to analyse restoration projects from the **DREAM** platform and predict project outcomes (e.g., *complete* vs *unsuccessful*). It’s designed to be interview-ready, leakage-aware, and easy to extend as new data arrives.

## Contents

- **Notebook:** `notebooks/code.ipynb` — exploratory work and a full runnable workflow.
- **Library code:** `src/utils.py` — helpers for EDA, preprocessing, models, CV, diagnostics, importance, cleaning/merging, and artefact saving.
- **Script:** `scripts/train.py` — trains models end-to-end and writes artifacts.
- **Tests:** `tests/` — unit tests for critical utilities.
- **Config:** `config/config.yaml` — feature lists and training knobs.
- **Sample data:** `sample_minimal.xlsx` — tiny example file to try the pipeline locally.
- **CI:** `.github/workflows/ci.yml` — lint, format, notebook clean, and tests via GitHub Actions.
- **Dev tooling:** `.pre-commit-config.yaml`, `pyproject.toml`, `requirements.txt`, `Dockerfile`, `Makefile`.

## Data sources (Excel inputs)

Place Excel files in the project root or in `./data/` (preferred). The notebook/script expects four inputs:

1) **Extraction.xlsx** — project catalogue 
 Source: `https://bi.dream.gov.ua/archive/?qlikTicket=gt_P.thZLa6PNqQ7&qlikTicket=e1oo25HlYN_JKtAg#/projectDetails`

2) **Project completeness.xlsx** — project data availability
 Source: `https://bi.dream.gov.ua/archive/?qlikTicket=gt_P.thZLa6PNqQ7&qlikTicket=e1oo25HlYN_JKtAg#/dataAvailability`

3) **Digital index.xlsx** — regional digitalization index
 _(Normalize column **“Region Name”** to **`Region`** before merging.)_

4) **Codificated IRI + Transparency.xlsx** — institutional capacity / transparency
 _(Normalize column **“Oblast”** to **`Region`** before merging.)_

> ⚠️ **Do not commit production data.** Keep filenames consistent or adjust patterns in config/notebook.

## Quick start

### 1) Environment
```bash
python -m venv .venv
# Windows
. .venv/Scripts/activate
# macOS/Linux
# source .venv/bin/activate

python -m pip install -U pip
pip install -r requirements.txt
```

### 2) Put data
- Copy real Excel files into `./data/` (recommended) **or** keep them in the root next to the repo.
- For a quick smoke test, you can use `sample_minimal.xlsx` (very small).

### 3) Run the notebook
```bash
jupyter lab  # or: jupyter notebook
```
Open `notebooks/code.ipynb` and run cells top-to-bottom. Artefacts will be saved in `./artifacts/` (model, metrics, feature importances, model card).

### 4) Or run the training script
```bash
python scripts/train.py
```
Outputs (in `./artifacts/`):
- `ua_projects_best_<MODEL>_<TS>.joblib` — trained pipeline 
- `metrics.json` — holdout metrics 
- `feature_importance_top25.csv` — top permutation importances 
- `model_card.md` — short model card

## Configuration
Edit **`config/config.yaml`** to manage:
- feature lists (numeric/categorical),
- target column and mapping,
- CV settings, calibration, SMOTE, random seeds,
- region column mapping for Digital Index and IRI.

The code also provides a **strict** cleaner that avoids mode imputation on raw tables; imputation happens safely inside the scikit-learn pipeline when building the modelling matrix.

## What’s inside the pipeline
- **Preprocessing:** median imputation for numeric; rare-category grouping + one-hot for categorical; standard scaling for numeric.
- **Models:** Logistic Regression, Random Forest, Gradient Boosting (XGBoost optional).
- **Model selection:** repeated stratified CV; **PR-AUC** primary, ROC-AUC secondary.
- **Calibration:** optional (sigmoid or isotonic) for reliable probabilities.
- **Diagnostics:** ROC/PR/calibration plots, residual checks, permutation importance.
- **Stability:** adversarial train↔test AUC to flag drift.
- **Fairness slices:** simple group metrics and disparity helpers.

## Development

### Lint, format, notebooks
This repo uses **pre-commit** with:
- **ruff** (lint),
- **black** (format),
- **nbstripout** (clean Jupyter outputs),
- EOF/trailing-whitespace fixers.

Install and run locally:
```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

### Tests
```bash
pytest -q
```
> If `pytest` is not installed, add it locally or via `requirements-dev.txt`.

## Docker
```bash
docker build -t dream-ml .
docker run --rm -v "$PWD:/app" dream-ml python scripts/train.py
```
Mount the repo so your `data/` and `artifacts/` are visible inside the container.

## Repository layout
```
.
├─ .github/workflows/ci.yml          # CI: lint/format/nbstripout/tests
├─ config/config.yaml                # configuration for features & training
├─ data/                             # (optional) place Excel inputs here
├─ sample_minimal.xlsx               # tiny example dataset
├─ notebooks/code.ipynb              # EDA + full workflow
├─ scripts/train.py                  # CLI training pipeline
├─ src/
│  ├─ __init__.py
│  └─ utils.py                       # helpers: EDA, models, CV, cleaning, artifacts
├─ tests/
│  ├─ conftest.py
│  └─ test_utils_basic.py
├─ .pre-commit-config.yaml
├─ Dockerfile
├─ Makefile
├─ pyproject.toml
├─ requirements.txt
└─ README.md
```

## License & Data Use
Code is released under the MIT License (`LICENSE`). 
Data originates from the **DREAM** platform; comply with any relevant terms and governance requirements.

## Acknowledgments
- **DREAM — Digital Restoration Ecosystem for Accountable Management** (archive endpoints referenced above).
- Thanks to open-source contributors behind NumPy, pandas, scikit-learn, ruff, black, and friends.
