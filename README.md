# Ukrainian War Restoration Projects вЂ” End-to-End Binary Classification

[![CI](https://github.com/VKK-00/dream-ml-project/actions/workflows/ci.yml/badge.svg)](../../actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository contains a reproducible ML pipeline to analyse restoration projects from the **DREAM** platform and predict project outcomes (e.g., *complete* vs *unsuccessful*). ItвЂ™s designed to be interview-ready, leakage-aware, and easy to extend as new data arrives.

## Contents

- **Notebook:** `notebooks/code.ipynb` вЂ” exploratory work and a full runnable workflow.
- **Library code:** `src/utils.py` вЂ” helpers for EDA, preprocessing, models, CV, diagnostics, importance, cleaning/merging, and artefact saving.
- **Script:** `scripts/train.py` вЂ” trains models end-to-end and writes artifacts.
- **Tests:** `tests/` вЂ” unit tests for critical utilities.
- **Config:** `config/config.yaml` вЂ” feature lists and training knobs.
- **Sample data:** `sample_minimal.xlsx` вЂ” tiny example file to try the pipeline locally.
- **CI:** `.github/workflows/ci.yml` вЂ” lint, format, notebook clean, and tests via GitHub Actions.
- **Dev tooling:** `.pre-commit-config.yaml`, `pyproject.toml`, `requirements.txt`, `Dockerfile`, `Makefile`.

## Data sources (Excel inputs)

Place Excel files in the project root or in `./data/` (preferred). The notebook/script expects four inputs:

1) **Extraction.xlsx** вЂ” project catalogue
 Source: `https://bi.dream.gov.ua/archive/?qlikTicket=gt_P.thZLa6PNqQ7&qlikTicket=e1oo25HlYN_JKtAg#/projectDetails`

2) **Project completeness.xlsx** вЂ” project data availability
 Source: `https://bi.dream.gov.ua/archive/?qlikTicket=gt_P.thZLa6PNqQ7&qlikTicket=e1oo25HlYN_JKtAg#/dataAvailability`

3) **Digital index.xlsx** вЂ” regional digitalization index
 _(Normalize column **вЂњRegion NameвЂќ** to **`Region`** before merging.)_

4) **Codificated IRI + Transparency.xlsx** вЂ” institutional capacity / transparency
 _(Normalize column **вЂњOblastвЂќ** to **`Region`** before merging.)_

> вљ пёЏ **Do not commit production data.** Keep filenames consistent or adjust patterns in config/notebook.

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
- `ua_projects_best_<MODEL>_<TS>.joblib` вЂ” trained pipeline
- `metrics.json` вЂ” holdout metrics
- `feature_importance_top25.csv` вЂ” top permutation importances
- `model_card.md` вЂ” short model card

## Configuration
Edit **`config/config.yaml`** to manage:
- feature lists (numeric/categorical),
- target column and mapping,
- CV settings, calibration, SMOTE, random seeds,
- region column mapping for Digital Index and IRI.

The code also provides a **strict** cleaner that avoids mode imputation on raw tables; imputation happens safely inside the scikit-learn pipeline when building the modelling matrix.

## WhatвЂ™s inside the pipeline
- **Preprocessing:** median imputation for numeric; rare-category grouping + one-hot for categorical; standard scaling for numeric.
- **Models:** Logistic Regression, Random Forest, Gradient Boosting (XGBoost optional).
- **Model selection:** repeated stratified CV; **PR-AUC** primary, ROC-AUC secondary.
- **Calibration:** optional (sigmoid or isotonic) for reliable probabilities.
- **Diagnostics:** ROC/PR/calibration plots, residual checks, permutation importance.
- **Stability:** adversarial trainв†”test AUC to flag drift.
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
в”њв”Ђ .github/workflows/ci.yml          # CI: lint/format/nbstripout/tests
в”њв”Ђ config/config.yaml                # configuration for features & training
в”њв”Ђ data/                             # (optional) place Excel inputs here
в”њв”Ђ sample_minimal.xlsx               # tiny example dataset
в”њв”Ђ notebooks/code.ipynb              # EDA + full workflow
в”њв”Ђ scripts/train.py                  # CLI training pipeline
в”њв”Ђ src/
в”‚  в”њв”Ђ __init__.py
в”‚  в””в”Ђ utils.py                       # helpers: EDA, models, CV, cleaning, artifacts
в”њв”Ђ tests/
в”‚  в”њв”Ђ conftest.py
в”‚  в””в”Ђ test_utils_basic.py
в”њв”Ђ .pre-commit-config.yaml
в”њв”Ђ Dockerfile
в”њв”Ђ Makefile
в”њв”Ђ pyproject.toml
в”њв”Ђ requirements.txt
в””в”Ђ README.md
```

## License & Data Use
Code is released under the MIT License (`LICENSE`).
Data originates from the **DREAM** platform; comply with any relevant terms and governance requirements.

## Acknowledgments
- **DREAM вЂ” Digital Restoration Ecosystem for Accountable Management** (archive endpoints referenced above).
- Thanks to open-source contributors behind NumPy, pandas, scikit-learn, ruff, black, and friends.
