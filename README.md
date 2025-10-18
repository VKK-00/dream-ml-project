# Ukrainian War Restoration Projects — End-to-End Binary Classification (Excel → ML)

[![CI](https://github.com/VKK-00/dream-ml-project/actions/workflows/ci.yml/badge.svg)](../../actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository contains a leakage-aware, interview-ready ML pipeline to analyse restoration projects for war-damaged assets in Ukraine using data from the DREAM platform. The code is reusable and scales as the number of projects grows year over year. As more data arrives, retraining and recalibration typically improve accuracy and reveal subtler relationships.

---

## Highlights

- **Leakage-aware preprocessing**: median imputation for numerics; rare-category grouping → One-Hot; no mode imputation for categoricals in model features.
- **Robust evaluation**: repeated CV (PR-AUC primary), **calibrated probabilities** (Platt/sigmoid).
- **Structured EDA**: numeric summaries, cardinality, missingness, boxplots, target-by-group scans.
- **Diagnostics**: residual checks (probabilistic), fairness slices (min-support guards).
- **Explainability**: permutation importance aggregated across seeds.
- **Sanity checks**: adversarial train↔test drift AUC.
- **Reproducible artefacts**: model `.joblib`, metrics, importances, model card → `./artifacts`.

---

## Repository structure

```
.
├─ .github/workflows/
│  └─ ci.yml                     # CI for linting, tests
├─ config/
│  └─ config.yaml                # central config (paths, ML options)
├─ data/
│  ├─ README.md                  # what to put in /data
│  └─ sample_minimal.xlsx        # tiny toy dataset (sheets: extraction, completeness, digital, iri)
├─ notebooks/
│  └─ code.ipynb                 # main analysis & training notebook
├─ scripts/
│  └─ train.py                   # scriptable training entrypoint
├─ src/
│  ├─ __init__.py
│  └─ utils.py                   # helpers (EDA, CV, residuals, drift, save_artifacts, etc.)
├─ tests/
│  └─ test_utils_basic.py        # minimal unit tests
├─ .editorconfig
├─ .gitignore
├─ .pre-commit-config.yaml
├─ Dockerfile
├─ LICENSE
├─ Makefile
├─ README.md
└─ requirements.txt
```

---

## Data sources (Excel inputs)

The notebook expects **four inputs** in `./data/`:

1. **`Extraction.xlsx`** — project catalogue  
   Source (DREAM archive):  
   `https://bi.dream.gov.ua/archive/?qlikTicket=gt_P.thZLa6PNqQ7&qlikTicket=e1oo25HlYN_JKtAg#/projectDetails`

2. **`Project completeness.xlsx`** — project data availability  
   Source (DREAM archive):  
   `https://bi.dream.gov.ua/archive/?qlikTicket=gt_P.thZLa6PNqQ7&qlikTicket=e1oo25HlYN_JKtAg#/dataAvailability`

3. **`Digital index.xlsx`** — regional digitalization index  
   _(Normalize “Region Name” → `Region`.)_

4. **`Codificated IRI + Transparency.xlsx`** — IRI/Transparency  
   _(Normalize “Oblast” → `Region`.)_

> ⚠️ **Do not commit raw production data** unless governance permits it. Keep file names consistent, or adjust file patterns in the notebook/config.

---

## Getting started

```bash
# 1) Create & activate a venv
python -m venv .venv
# Windows:
. .venv/Scripts/activate
# macOS/Linux:
# source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) (Option A) Run via notebook
jupyter lab  # or: jupyter notebook
# open notebooks/code.ipynb and Run All

# 3) (Option B) Run via script
python scripts/train.py --config config/config.yaml
```

Place your Excel files into `./data/`. After a successful run, find artifacts in `./artifacts/`:

- `ua_projects_best_<MODEL>_<TS>.joblib`
- `metrics.json`
- `feature_importance_top25.csv`
- `model_card.md`

---

## Configuration

Centralised in:
- `config/config.yaml` (scripted runs), and/or
- the `cfg` block at the top of `notebooks/code.ipynb`.

Defaults:
- CV: **5 splits × 2 repeats**
- Calibration: **sigmoid**
- Rare-category min freq: **30**
- Test size: **20%**

---

## Why this scales well

- **PR-AUC focus** helps with imbalanced labels; **ROC-AUC** as secondary.
- **Calibration** keeps probabilities meaningful as the dataset evolves.
- **Rare-category grouping** contains exploding cardinality as new entities appear.
- **Permutation importance (multi-seed)** reduces variance on small test sets.
- **Fairness and drift checks** remain valid and more stable with more data.

As DREAM accumulates more projects annually, retraining + recalibration can improve accuracy and reveal finer-grained patterns.

---

## Ethics & responsible use

This code supports Ukraine’s reconstruction analysis. Predictions should be paired with governance: input validation, fairness review, calibration checks, and human oversight.

---

## License

MIT — see [LICENSE](LICENSE).
