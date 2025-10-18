"""
utils.py — Reusable helpers for the project
-------------------------------------------

This module provides:
- Deterministic seeding across common libraries
- OS-agnostic Excel discovery and loading with wildcard fallback
- Safe numeric parsing for mixed-format Excel columns
- EDA utilities (overview, top-k categories, numeric histograms, robust correlation heatmap)
- Rare category grouping transformer for categorical stability
- Leakage-aware preprocessing pipeline (impute → rare-group → OHE → scale)
- Model builders with optional SMOTE and probability calibration (version-safe)
- Repeated CV summary and best-model selection
- Holdout evaluation and diagnostic plots (ROC, PR, reliability)
- Fairness slices by group and disparity helpers
- Permutation importance with robust feature-name extraction
- Adversarial train↔test AUC for drift checking
- Domain cleaning and merging for the four input Excel files
- Artifact saving (pipeline, metrics, importances, model card)

All functions are pure and avoid side effects on import.
"""

from __future__ import annotations

import inspect
import json
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Optional dependencies with guards
try:
    import xgboost as xgb

    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    HAS_IMB = True
except Exception:
    HAS_IMB = False


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------


def set_global_seed(seed: int = 42) -> None:
    """
    Make randomness reproducible for Python, NumPy, and (if present) PyTorch.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


# ---------------------------------------------------------------------
# Data discovery and loading
# ---------------------------------------------------------------------


def _candidate_roots(extra_env_var: str = "PROJECT_DATA_DIR") -> List[Path]:
    """
    Produce a small ordered list of plausible directories where raw files may live.
    """
    roots: List[Path] = []
    if extra_env_var in os.environ:
        roots.append(Path(os.environ[extra_env_var]))
    roots.extend(
        [
            Path.cwd(),
            Path.cwd() / "data",
            Path.cwd().parent / "data",
            Path("/mnt/data"),
            Path.home() / "Downloads",
        ]
    )
    seen, uniq = set(), []
    for r in roots:
        if r not in seen:
            uniq.append(r)
            seen.add(r)
    return uniq


def _list_xlsx(roots: Iterable[Path]) -> List[Path]:
    """
    List all .xlsx files reachable under given roots (recursive).
    """
    out: List[Path] = []
    for root in roots:
        if root.exists():
            out.extend(root.rglob("*.xlsx"))
    return out


def resolve_file(
    candidates: Sequence[str], roots: Optional[Iterable[Path]] = None
) -> Path:
    """
    Resolve a file by attempting:
      1) direct path or join under each root,
      2) case-insensitive exact filename,
      3) wildcard patterns like '*Extraction*.xlsx'.

    Raises FileNotFoundError with a helpful diagnostic if not found.
    """
    roots = list(roots) if roots is not None else _candidate_roots()
    cands = list(candidates)

    # 1) Direct paths or joined paths
    for name in cands:
        p = Path(name)
        if p.is_absolute() and p.exists():
            return p
        for root in roots:
            q = root / name
            if q.exists():
                return q

    # 2) Case-insensitive exact match
    exact_names = {n.lower() for n in cands if all(ch not in n for ch in "*?[]")}
    for root in roots:
        if not root.exists():
            continue
        for x in root.rglob("*.xlsx"):
            if x.name.lower() in exact_names:
                return x

    # 3) Wildcard patterns
    for root in roots:
        if not root.exists():
            continue
        for name in cands:
            if any(ch in name for ch in "*?[]"):
                hits = list(root.rglob(name))
                if hits:
                    return hits[0]

    nearby = _list_xlsx(roots)
    msg = [
        "Could not resolve any of the candidate Excel files:",
        *[f"  - {n}" for n in cands],
        "",
        "Searched under these roots:",
        *[f"  - {r.resolve()}" for r in roots],
        "",
        f"Found {len(nearby)} .xlsx nearby; first few:",
        *[f"  - {p}" for p in nearby[:20]],
    ]
    raise FileNotFoundError("\n".join(msg))


def mirror_to_dir(candidates: Sequence[str], dest: Path) -> Path:
    """
    Copy the resolved file into `dest` to keep the repo self-contained.
    Returns the destination path, skipping if already present.
    """
    dest.mkdir(parents=True, exist_ok=True)
    src = resolve_file(candidates)
    dst = dest / src.name
    if not dst.exists():
        import shutil

        shutil.copy2(src, dst)
        print(f"[COPIED] {src} -> {dst}")
    else:
        print(f"[SKIP] Already present: {dst}")
    return dst


def load_xlsx(candidates: Sequence[str], **read_kwargs) -> pd.DataFrame:
    """
    Load the first resolvable Excel among `candidates` using openpyxl.
    """
    path = resolve_file(candidates)
    try:
        return pd.read_excel(path, engine="openpyxl", **read_kwargs)
    except Exception as e:
        raise RuntimeError(f"Failed to read '{path}': {e}") from e


# ---------------------------------------------------------------------
# EDA helpers
# ---------------------------------------------------------------------


def to_float_safe(val) -> float | np.nan:
    """
    Parse numbers stored as messy strings (spaces, commas, non-breaking spaces).
    Returns NaN for non-parsable values.
    """
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float, np.number)):
        return float(val)
    s = str(val).strip()
    if s in {"", "-", "nan", "None", "N/A"}:
        return np.nan
    s = s.replace("\u00a0", "").replace(" ", "").replace(",", "")
    try:
        return float(s)
    except Exception:
        return np.nan


def eda_overview(
    df: pd.DataFrame, target_col: str | None = None, top_k_missing: int = 10
) -> None:
    """
    Print shape, dtypes, and top missingness report. Designed for quick notebook inspection.
    """
    print("Shape:", df.shape)
    print("\nDtypes:\n", df.dtypes.head(20))
    miss = df.isna().mean().sort_values(ascending=False).head(top_k_missing)
    print("\nTop missingness:\n", miss)


def plot_topk_categories(df: pd.DataFrame, col: str, k: int = 12) -> None:
    """
    Plot the top-k frequent categories for a column.
    """
    if col not in df.columns:
        print(f"Column {col!r} not found")
        return
    vc = df[col].astype(str).value_counts().head(k)
    plt.figure(figsize=(7, 4))
    plt.bar(vc.index, vc.values)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Top {k} categories — {col}")
    plt.tight_layout()
    plt.show()


def _as_numeric_series(s: pd.Series) -> pd.Series:
    """
    Coerce a series to float using `to_float_safe` for robustness on Excel strings.
    """
    if s.dtype == "O":
        return s.map(to_float_safe)
    return pd.to_numeric(s, errors="coerce")


def plot_numeric_histograms(
    df: pd.DataFrame, numeric_cols: List[str], bins: int = 30, max_cols: int = 8
) -> None:
    """
    Plot histograms for up to `max_cols` numeric columns, robust to non-numeric tokens.
    """
    cols = [c for c in numeric_cols if c in df.columns][:max_cols]
    for c in cols:
        x = _as_numeric_series(df[c]).dropna()
        if x.empty:
            continue
        plt.figure(figsize=(5, 3.5))
        plt.hist(x.values, bins=bins)
        plt.title(f"Histogram — {c}")
        plt.tight_layout()
        plt.show()


def plot_corr_heatmap(
    df: pd.DataFrame, numeric_cols: List[str], max_cols: int = 15
) -> None:
    """
    Correlation heatmap computed after safe numeric coercion.
    Keeps columns with at least 10 non-null values and non-zero variance.
    """
    cols_present = [c for c in numeric_cols if c in df.columns]
    if not cols_present:
        print("No candidate numeric columns present.")
        return

    X_num = pd.DataFrame({c: _as_numeric_series(df[c]) for c in cols_present})

    valid = []
    for c in X_num.columns:
        s = X_num[c]
        if s.notna().sum() >= 10 and s.std(skipna=True) > 0:
            valid.append(c)

    cols = valid[:max_cols]
    if len(cols) < 2:
        print("Not enough valid numeric columns for correlation heatmap.")
        return

    corr = X_num[cols].corr(min_periods=10)

    plt.figure(figsize=(0.55 * len(cols) + 3, 0.55 * len(cols) + 3))
    im = plt.imshow(corr.values, interpolation="nearest", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(cols)), cols, rotation=45, ha="right")
    plt.yticks(range(len(cols)), cols)
    plt.title("Correlation heatmap")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# Feature engineering: rare category grouping
# ---------------------------------------------------------------------


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """
    Group infrequent categorical levels into a sentinel token.

    Parameters
    ----------
    min_freq : int
        Minimum count for a level to be preserved.
    token : str
        Replacement for all rare categories.
    columns : Optional[List[str]]
        If provided, only these columns are considered for grouping.
    """

    def __init__(
        self,
        min_freq: int = 30,
        token: str = "__OTHER__",
        columns: Optional[List[str]] = None,
    ):
        self.min_freq = int(min_freq)
        self.token = token
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        X = pd.DataFrame(X).copy()
        self.columns_ = (
            self.columns or X.select_dtypes(include="object").columns.tolist()
        )
        self.levels_: Dict[str, set] = {}
        for c in self.columns_:
            vc = X[c].astype("object").value_counts(dropna=False)
            self.levels_[c] = set(vc[vc >= self.min_freq].index)
        return self

    def transform(self, X):
        from sklearn.utils.validation import check_is_fitted

        check_is_fitted(self, "levels_")
        X = pd.DataFrame(X).copy()
        for c in self.columns_:
            if c in X.columns:
                X[c] = (
                    X[c]
                    .astype("object")
                    .where(X[c].isin(self.levels_[c]), other=self.token)
                )
        return X


# ---------------------------------------------------------------------
# Preprocessor and models
# ---------------------------------------------------------------------


def build_preprocessor(
    numeric: List[str], categorical: List[str], rare_min_freq: int = 30
) -> ColumnTransformer:
    """
    Create a leakage-safe preprocessor:
      numeric: median imputation → standard scaling
      categorical: mode imputation → rare-category grouping → one-hot encoding
    """
    enc_sig = inspect.signature(OneHotEncoder.__init__)
    ohe_kwargs = (
        dict(handle_unknown="ignore", sparse_output=False)
        if "sparse_output" in enc_sig.parameters
        else dict(handle_unknown="ignore", sparse=False)
    )

    numeric_tf = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_tf = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("rare", RareCategoryGrouper(min_freq=rare_min_freq)),
            ("onehot", OneHotEncoder(**ohe_kwargs)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_tf, numeric),
            ("cat", categorical_tf, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _wrap_with_calibration(model, method: str = "isotonic", cv: int = 3):
    """
    Wrap a classifier with CalibratedClassifierCV in a version-safe way.
    """
    sig = inspect.signature(CalibratedClassifierCV.__init__)
    kwargs = {"method": method, "cv": cv}
    if "estimator" in sig.parameters:  # sklearn >= 1.5
        kwargs["estimator"] = model
    else:  # older sklearn
        kwargs["base_estimator"] = model
    return CalibratedClassifierCV(**kwargs)


def build_models(
    preprocessor: ColumnTransformer,
    seed: int = 42,
    use_smote: bool = True,
    smote_k_neighbors: int = 5,
    calibrate: bool = True,
    calibration: str = "isotonic",
) -> Dict[str, Pipeline]:
    """
    Define a compact suite of strong baselines. If imbalanced-learn is available and
    `use_smote` is True, insert SMOTE between preprocessing and classifier.
    Optionally wrap each classifier with probability calibration.
    """

    def wrap(model):
        steps: List[Tuple[str, object]] = [("preprocessor", preprocessor)]
        if HAS_IMB and use_smote:
            steps.append(
                ("smote", SMOTE(k_neighbors=smote_k_neighbors, random_state=seed))
            )
        final_estimator = (
            _wrap_with_calibration(model, method=calibration, cv=3)
            if calibrate
            else model
        )
        steps.append(("model", final_estimator))
        return ImbPipeline(steps) if (HAS_IMB and use_smote) else Pipeline(steps)

    models: Dict[str, Pipeline] = {
        "LogReg": wrap(
            LogisticRegression(C=1.5, max_iter=800, class_weight="balanced")
        ),
        "RF": wrap(
            RandomForestClassifier(
                n_estimators=400,
                max_depth=None,
                min_samples_split=5,
                n_jobs=-1,
                class_weight="balanced_subsample",
                random_state=seed,
            )
        ),
        "GB": wrap(GradientBoostingClassifier(random_state=seed)),
    }
    if HAS_XGB:
        models["XGB"] = wrap(
            xgb.XGBClassifier(
                n_estimators=600,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                tree_method="hist",
                n_jobs=-1,
                random_state=seed,
                eval_metric="logloss",
            )
        )
    return models


# ---------------------------------------------------------------------
# Evaluation and selection
# ---------------------------------------------------------------------


def cv_summary_for_models(
    models: Dict[str, Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    n_repeats: int = 2,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Cross-validate all models with ROC-AUC, PR-AUC, F1 and Brier.
    Returns a sorted summary with simple percentile-based confidence intervals.
    """
    scorers = {
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "f1": "f1",
        "brier": "neg_brier_score",
    }
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=seed
    )
    rows = []
    for name, pipe in models.items():
        cvres = cross_validate(pipe, X, y, scoring=scorers, cv=rskf, n_jobs=-1)

        def ci(a):
            a = np.asarray(a)
            mean = float(np.mean(a))
            lo, hi = np.percentile(a, [2.5, 97.5])
            return mean, float(lo), float(hi)

        roc_m, roc_lo, roc_hi = ci(cvres["test_roc_auc"])
        pr_m, pr_lo, pr_hi = ci(cvres["test_pr_auc"])
        f1_m, f1_lo, f1_hi = ci(cvres["test_f1"])
        brier_m = -float(np.mean(cvres["test_brier"]))
        rows.append(
            {
                "model": name,
                "roc_auc_mean": roc_m,
                "roc_auc_lo": roc_lo,
                "roc_auc_hi": roc_hi,
                "pr_auc_mean": pr_m,
                "pr_auc_lo": pr_lo,
                "pr_auc_hi": pr_hi,
                "f1_mean": f1_m,
                "f1_lo": f1_lo,
                "f1_hi": f1_hi,
                "brier_mean": brier_m,
            }
        )
    df = (
        pd.DataFrame(rows)
        .sort_values(["pr_auc_mean", "roc_auc_mean"], ascending=False)
        .reset_index(drop=True)
    )
    return df


def choose_best_name(cv_df: pd.DataFrame) -> str:
    """
    Select the best model primarily by PR-AUC, then by ROC-AUC.
    """
    return cv_df.sort_values(["pr_auc_mean", "roc_auc_mean"], ascending=False).iloc[0][
        "model"
    ]


def holdout_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute standard binary metrics on the holdout set.
    """
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "brier": brier_score_loss(y_true, y_prob),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def plot_curves(y_true: np.ndarray, y_prob: np.ndarray, title_prefix: str = "") -> None:
    """
    Plot ROC, PR, and calibration (reliability) curves for probability outputs.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    from sklearn.calibration import calibration_curve

    frac_pos, mean_pred = calibration_curve(
        y_true, y_prob, n_bins=10, strategy="quantile"
    )

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, label=f"AUC={roc_auc_score(y_true, y_prob):.3f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{title_prefix}ROC")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, lw=2, label=f"AP={average_precision_score(y_true, y_prob):.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{title_prefix}Precision–Recall")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 5))
    plt.plot(mean_pred, frac_pos, marker="o")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(f"{title_prefix}Calibration (Reliability)")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# Permutation importance (robust names)
# ---------------------------------------------------------------------


def _feature_names_from_preprocessor(pre) -> List[str]:
    """
    Attempt to extract output feature names from a fitted ColumnTransformer `pre`.
    Works across sklearn versions and nested Pipelines.
    """
    try:
        names = pre.get_feature_names_out()
        return [str(x) for x in names]
    except Exception:
        pass

    names: List[str] = []
    for name, trans, cols in pre.transformers_:
        if name == "remainder":
            continue
        try:
            if hasattr(trans, "get_feature_names_out"):
                nn = trans.get_feature_names_out(cols)
                names.extend([str(x) for x in nn])
                continue
            from sklearn.pipeline import Pipeline as SkPipeline

            if isinstance(trans, SkPipeline):
                last = trans.steps[-1][1]
                if hasattr(last, "get_feature_names_out"):
                    nn = last.get_feature_names_out(cols)
                    names.extend([str(x) for x in nn])
                    continue
        except Exception:
            pass
        names.extend([str(c) for c in cols])
    return names


def permutation_importance_df(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
    n_repeats: int = 10,
    top_k: int = 25,
) -> pd.DataFrame:
    """
    Compute permutation importances for a fitted pipeline and return the top K entries.
    """
    from sklearn.inspection import permutation_importance

    pre = pipeline.named_steps["preprocessor"]
    feature_names = _feature_names_from_preprocessor(pre)

    rng = np.random.RandomState(random_state)
    idx = rng.choice(len(X), size=min(2000, len(X)), replace=False)
    X_s, y_s = X.iloc[idx], y.iloc[idx]

    perm = permutation_importance(
        pipeline, X_s, y_s, n_repeats=n_repeats, random_state=random_state, n_jobs=-1
    )

    n_imp = perm.importances_mean.shape[0]
    if n_imp != len(feature_names):
        import warnings

        warnings.warn(
            f"Feature name count ({len(feature_names)}) != importances ({n_imp}); aligning by min length.",
            RuntimeWarning,
        )
        m = min(n_imp, len(feature_names))
        feat = feature_names[:m]
        imp_mean = perm.importances_mean[:m]
    else:
        feat = feature_names
        imp_mean = perm.importances_mean

    imp_df = (
        pd.DataFrame({"feature": feat, "importance": imp_mean})
        .sort_values("importance", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )
    return imp_df


# ---------------------------------------------------------------------
# Fairness slices and disparity
# ---------------------------------------------------------------------


def slice_binary_metrics(
    X: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    col: str,
    threshold: float = 0.5,
    min_n: int = 30,
) -> pd.DataFrame:
    """Lightweight group slice metrics used in tests."""
    if col not in X.columns:
        return pd.DataFrame(columns=[col, "n", "pos", "neg", "tpr", "fpr", "roc_auc"])

    X_local = X.reset_index(drop=True)
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    rows = []
    for g, idx in X_local.groupby(col).groups.items():
        idx = np.fromiter(idx, dtype=int)
        if len(idx) < min_n:
            continue

        yt, yp, pr = y_true[idx], y_pred[idx], y_prob[idx]
        n_pos = int((yt == 1).sum())
        n_neg = int((yt == 0).sum())
        if n_pos == 0 or n_neg == 0:
            continue

        tpr = float(((yt == 1) & (yp == 1)).sum() / n_pos)
        fpr = float(((yt == 0) & (yp == 1)).sum() / n_neg)
        roc = float("nan") if len(np.unique(yt)) < 2 else float(roc_auc_score(yt, pr))
        rows.append(
            {
                col: g,
                "n": int(len(idx)),
                "pos": n_pos,
                "neg": n_neg,
                "tpr": tpr,
                "fpr": fpr,
                "roc_auc": roc,
            }
        )

    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def disparity(df: pd.DataFrame, metric: str) -> float:
    """
    Simple disparity measure as max minus min for a given metric column.
    """
    return float(df[metric].max() - df[metric].min())


# ---------------------------------------------------------------------
# Residual diagnostics and adversarial drift
# ---------------------------------------------------------------------


def residual_diagnostics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title_prefix: str = "",
    show_examples: bool = False,
    X_test: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    """
    Plot residual-like diagnostics for probabilistic binary classifiers:
      - histogram of p
      - residuals (y − p) distribution
      - residuals vs. predicted probability
    Returns simple summary stats for logging.
    """
    p = np.asarray(y_prob, float)
    y = np.asarray(y_true, float)
    resid = y - p

    plt.figure(figsize=(5.5, 3.5))
    plt.hist(p, bins=30)
    plt.title(f"{title_prefix}Predicted probability distribution")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(5.5, 3.5))
    plt.hist(resid, bins=30)
    plt.title(f"{title_prefix}Residuals (y − p)")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(5.5, 3.5))
    plt.scatter(p, resid, s=12, alpha=0.35)
    plt.axhline(0.0, lw=1)
    plt.xlabel("Predicted probability")
    plt.ylabel("Residual (y − p)")
    plt.title(f"{title_prefix}Residuals vs probability")
    plt.tight_layout()
    plt.show()

    if show_examples and X_test is not None:
        # Intentionally off by default to avoid exposing rows in public repos.
        worst = np.argsort(np.abs(resid))[::-1][:10]
        print("Examples with largest |residual| (top 10 indices):", worst.tolist())
        try:
            print(X_test.iloc[worst].head(10).to_string())
        except Exception:
            pass

    return {
        "residual_mean": float(np.mean(resid)),
        "residual_std": float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.0,
        "p_mean": float(np.mean(p)),
        "p_pos_mean": float(np.mean(p[y == 1])) if (y == 1).any() else float("nan"),
        "p_neg_mean": float(np.mean(p[y == 0])) if (y == 0).any() else float("nan"),
    }


def adversarial_auc(preprocessor, X_train, X_test, seed: int = 42) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import Pipeline

    adv_X = pd.concat([X_train, X_test], ignore_index=True)
    adv_y = np.r_[np.zeros(len(X_train), dtype=int), np.ones(len(X_test), dtype=int)]

    pipe = Pipeline(
        [("preprocessor", preprocessor), ("clf", LogisticRegression(max_iter=1000))]
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    prob = cross_val_predict(
        pipe, adv_X, adv_y, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    return float(roc_auc_score(adv_y, prob))


# ---------------------------------------------------------------------
# Domain cleaning and merging
# ---------------------------------------------------------------------


def _coerce_numeric_series(s: pd.Series) -> pd.Series:
    """Coerce mixed-format Excel numerics to float without filling missing."""
    if s.dtype.kind in "biufc":
        return s.astype(float)
    # remove NBSP, spaces, and commas; coerce
    return (
        s.astype(str)
        .str.replace("\u00a0", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(",", "", regex=False)
        .replace({"": np.nan, "nan": np.nan, "NaN": np.nan, "N/A": np.nan, "-": np.nan})
        .pipe(pd.to_numeric, errors="coerce")
    )


def clean_and_merge_strict(
    df_ext: pd.DataFrame,
    df_comp: pd.DataFrame,
    df_dig: pd.DataFrame,
    df_iri: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    """
    Strict, domain-aware cleaning with NO mode imputation for categoricals.
    Leaves object/geography as NaN; ML pipeline will handle imputation safely.
    """
    d = df_ext.copy()

    # 1) Normalize whitespace and "fake missing" tokens to real NaN for *all* object columns
    _missing_tokens = {"", "nan", "NaN", "N/A", "None", "-"}
    for c in d.columns:
        if d[c].dtype == "O":
            s = d[c].astype(str).str.strip()
            s = s.where(~s.str.lower().isin(_missing_tokens))
            d[c] = s

    # 2) Domain: Region normalization & multi-region marker
    if "Region" in d.columns:
        d["Region"] = d["Region"].astype("object")
        d["MultipleRegions"] = d["Region"].astype(str).str.contains(",", na=False)
        d.loc[d["MultipleRegions"], "Region"] = "Multiple"

    # 3) Drop narrative/high-cardinality leakage-prone columns (safe if absent)
    drop_cols = [
        "Link to Portal",
        "Project author",
        "Project title",
        "Initiator",
        "Management body",
        "Implementer",
        "Balancer",
        "Object title",
        "Community",
        "Funding source",
        "Strategic documents",
        "Legal basis",
        "IFI project",
        "State budget program",
        "Local budget program",
        "Shelter type",
        "Canteen type",
        "MOS Facility",
        "MOH Facility",
        "RDDP code",
        "CPV-code of an investment object",
        "Project update date",
        "Date of design documents approval",
    ]
    d.drop(
        columns=[c for c in drop_cols if c in d.columns], inplace=True, errors="ignore"
    )

    # 4) Numeric coercion ONLY (no filling here — imputation is in the sklearn pipeline)
    numeric_cols = [
        "Forseen duration, months",
        "Overall estimated budget, ₴",
        "Committed funding, ₴",
        "Contract value, ₴",
        "Number of objects",
        "Number of contracting processes",
        "Number of contracts",
        "Number of investment objects requiring design documents",
    ]
    for c in numeric_cols:
        if c in d.columns:
            d[c] = _coerce_numeric_series(d[c]).replace([np.inf, -np.inf], np.nan)

    # 5) DO NOT mode-impute ANY object columns (including geography)
    #    Leave NaN — later, the modeling cell turns NaN into "Unknown" *only* for X, not the raw table.

    # 6) Deduplicate by Project code if present
    if "Project code" in d.columns:
        d = d.drop_duplicates("Project code")

    # 7) Merge completeness on Project code (columns trimmed if needed)
    if ("Project code" in d.columns) and ("Project code" in df_comp.columns):
        comp_trim = df_comp.drop(
            columns=[
                c for c in ["Initiator", "Link to Portal"] if c in df_comp.columns
            ],
            errors="ignore",
        )
        d = d.merge(comp_trim, on="Project code", how="left")

    # 8) Merge Digital Index by region (rename to canonical)
    dig_region = cfg["features"]["region_cols"].get("digital", "Region Name")
    canon = cfg["features"]["canonical_region"]
    if dig_region in df_dig.columns and "Region" in d.columns:
        dig2 = df_dig.rename(columns={dig_region: canon}).copy()
        dig2[canon] = dig2[canon].astype(str).str.strip()
        d["Region"] = d["Region"].astype(str).str.strip()
        d = d.merge(dig2, left_on="Region", right_on=canon, how="left")
        if canon in d.columns and canon != "Region":
            d.drop(columns=[canon], inplace=True, errors="ignore")

    # 9) Merge IRI by region (rename to canonical)
    iri_region = cfg["features"]["region_cols"].get("iri", "Oblast")
    if iri_region in df_iri.columns and "Region" in d.columns:
        iri2 = df_iri.rename(columns={iri_region: canon}).copy()
        iri2[canon] = iri2[canon].astype(str).str.strip()
        d = d.merge(iri2, left_on="Region", right_on=canon, how="left")
        if canon in d.columns and canon != "Region":
            d.drop(columns=[canon], inplace=True, errors="ignore")

    # 10) Feature: Financial_Efficiency = committed / budget (no fill; leave NaN if denominator 0 or missing)
    if {"Committed funding, ₴", "Overall estimated budget, ₴"}.issubset(d.columns):
        denom = d["Overall estimated budget, ₴"].replace({0.0: np.nan})
        d["Financial_Efficiency"] = (d["Committed funding, ₴"] / denom).replace(
            [np.inf, -np.inf], np.nan
        )

    # 11) Final trim of trailing whitespace in strings; keep NaNs as NaN
    for c in d.columns:
        if d[c].dtype == "O":
            s = d[c].astype(str)
            d[c] = s.where(s.notna(), np.nan).str.strip()

    return d


# ---------------------------------------------------------------------
# Artifact saving
# ---------------------------------------------------------------------


def save_artifacts(
    artifacts_dir: Path,
    pipeline: Pipeline,
    metrics: Dict[str, float],
    importances: pd.DataFrame,
    model_name: str,
    data_note: str,
    calibration_method: str | None = None,
) -> Dict[str, Path]:
    """
    Persist the trained pipeline, metrics, permutation importances and a compact model card.
    Returns a dict of artifact paths.
    """
    import time

    import joblib

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())

    model_path = artifacts_dir / f"ua_projects_best_{model_name}_{ts}.joblib"
    metrics_path = artifacts_dir / "metrics.json"
    imp_path = artifacts_dir / "feature_importance_top25.csv"
    card_path = artifacts_dir / "model_card.md"

    # Save model + tables
    joblib.dump(pipeline, model_path)
    importances.to_csv(imp_path, index=False, encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    cal_text = calibration_method or "unspecified"
    card = f"""
# Model Card — Ukrainian Projects Outcome Classifier

## Intended Use
Binary classification of project outcomes (complete vs unsuccessful) for analysis and demonstration purposes.

## Data Sources
{data_note}

## Preprocessing
Numeric: median imputation and standard scaling.
Categorical: most-frequent imputation, rare-category grouping, one-hot encoding.

## Model and Calibration
Best model selected via repeated stratified cross-validation (primary PR-AUC, secondary ROC-AUC).
Probabilities calibrated with **{cal_text}**.

## Holdout Metrics
ROC-AUC: {metrics.get('roc_auc', float('nan')):.3f}
PR-AUC: {metrics.get('pr_auc', float('nan')):.3f}
Brier: {metrics.get('brier', float('nan')):.3f}
F1: {metrics.get('f1', float('nan')):.3f}

## Limitations
Excel inputs are assumed consistent with the provided schemas; residual sampling bias may remain.
Not intended for production without additional hardening and monitoring.
""".strip()

    card_path.write_text(card, encoding="utf-8")

    return {
        "model": model_path,
        "metrics": metrics_path,
        "importances": imp_path,
        "model_card": card_path,
    }
