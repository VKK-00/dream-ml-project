# Ukrainian War Restoration Projects — End-to-End Binary Classification (Excel → ML)

[![CI](https://github.com/VKK-00/dream-ml-project/actions/workflows/ci.yml/badge.svg)](../../actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository contains a reproducible, leakage-aware ML pipeline to analyse restoration projects for war-damaged assets in Ukraine using data from the **DREAM** platform. The code is reusable and scales as the number of projects grows year over year. As more data arrives, retraining and recalibration typically improve accuracy and reveal subtler relationships.

---

## Highlights

- **Leakage-aware preprocessing**: numeric median imputation; rare-category grouping → one-hot; **no mode imputation** on raw categoricals (strict cleaner).
- **Robust evaluation**: repeated CV (PR-AUC primary), **calibrated probabilities** (sigmoid or isotonic).
- **Structured EDA**: numeric summaries, cardinality, missingness, boxplots, target-by-group scans.
- **Diagnostics**: residual checks, fairness slices (min-support guards).
- **Explainability**: permutation importance.
- **Sanity checks**: adversarial train↔test drift AUC.
- **Reproducible artifacts**: model (`.joblib`), metrics, importances, model card → `./artifacts`.

---

## Repository structure

```
.
├─ .github/workflows/ci.yml           # CI: pre-commit + pytest
├─ config/config.yaml                 # model/data config (paths, features, CV, etc.)
├─ data/                              # place real Excel files here (gitignored)
├─ sample_minimal.xlsx                # tiny demo file
├─ notebooks/code.ipynb               # E2E demo notebook
├─ scripts/train.py                   # training + CV + artifacts
├─ src/
│  ├─ __init__.py
│  └─ utils.py                        # cleaning, preprocessing, CV, metrics, importances, saving
├─ tests/
│  ├─ conftest.py
│  └─ test_utils_basic.py             # smoke tests for utils
├─ .pre-commit-config.yaml            # ruff, black, nbstripout, whitespace hooks
├─ Dockerfile                         # optional container build
├─ Makefile                           # optional helpers
├─ pyproject.toml                     # ruff/black config
├─ requirements.txt                   # runtime deps
├─ LICENSE
└─ README.md
```

---

## Data sources (Excel inputs)

Place **four** inputs in `./data/`:

1. **`Extraction.xlsx`** — project catalogue  
 Source (DREAM archive):  
 `https://bi.dream.gov.ua/archive/?qlikTicket=gt_P.thZLa6PNqQ7&qlikTicket=e1oo25HlYN_JKtAg#/projectDetails`

2. **`Project completeness.xlsx`** — project data availability  
 Source (DREAM archive):  
 `https://bi.dream.gov.ua/archive/?qlikTicket=gt_P.thZLa6PNqQ7&qlikTicket=e1oo25HlYN_JKtAg#/dataAvailability`

3. **`Digital index.xlsx`** — regional digitalization index  
 _(Normalize column **“Region Name”** to **`Region`** before merging.)_

4. **`Codificated IRI + Transparency.xlsx`** — institutional capacity / transparency  
 _(Normalize column **“Oblast”** to **`Region`** before merging.)_

> ⚠️ **Do not commit raw production data.** Keep filenames consistent or adjust patterns in config/notebook.

---

## Getting started

```bash
# 1) Create & activate a venv
python -m venv .venv
# Windows
. .venv/Scripts/activate
# macOS/Linux
# source .venv/bin/activate

# 2) Install dependencies
python -m pip install -U pip
pip install -r requirements.txt
```

### Option A — Notebook

```bash
jupyter lab   # or: jupyter notebook
```
Open `notebooks/code.ipynb` and run all cells. Artifacts will appear in `./artifacts/`:
- `ua_projects_best_<MODEL>_<TS>.joblib`
- `metrics.json`
- `feature_importance_top25.csv`
- `model_card.md`

### Option B — Script

```bash
python scripts/train.py --config config/config.yaml
```

---

## Configuration

- Centralized in `config/config.yaml` (scripted runs) and the `cfg` block in the notebook.
- Defaults:
  - CV: **5 splits × 2 repeats**
  - Calibration: **sigmoid**
  - Rare-category min freq: **30**
  - Test size: **20%**

The strict cleaner avoids mode-imputing object columns in the raw table; imputation happens in the scikit-learn pipeline when building features.

---

## Development

### Pre-commit (lint/format/notebooks)

This repo uses **ruff**, **black**, **nbstripout**, EOF and whitespace fixers.

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

### Tests

```bash
pytest -q
```

If `pytest` is missing in CI, add it to your workflow or a `requirements-dev.txt` and install there.

---

## Docker

```bash
docker build -t dream-ml .
docker run --rm -v "$PWD:/app" dream-ml python scripts/train.py
```

Mount the repo so your `data/` and `artifacts/` are visible inside the container.

---

## Ethics & responsible use

This code supports Ukraine’s reconstruction analysis. Use predictions with care and in concert with governance: input validation, fairness review, calibration checks, and human oversight.

---

## License

MIT — see [LICENSE](LICENSE).
