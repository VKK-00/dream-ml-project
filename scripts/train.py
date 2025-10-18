# scripts/train.py
import json
from pathlib import Path

from sklearn.model_selection import train_test_split

from src.utils import (
    adversarial_auc,
    build_models,
    build_preprocessor,
    choose_best_name,
    clean_and_merge,
    cv_summary_for_models,
    holdout_metrics,
    load_xlsx,
    mirror_to_dir,
    permutation_importance_df,
    set_global_seed,
)


def main():
    set_global_seed(42)
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    artifacts = root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    files = {
        "extraction": ["Extraction.xlsx", "*Extraction*.xlsx"],
        "completeness": ["Project completeness.xlsx", "*Project*completeness*.xlsx"],
        "digital": ["Digital index.xlsx", "*Digital*index*.xlsx"],
        "iri": [
            "Codificated IRI + Transparency.xlsx",
            "*Codificated*IRI*Transparency*.xlsx",
        ],
    }
    paths = {k: mirror_to_dir(v, data_dir) for k, v in files.items()}
    df_ext = load_xlsx([paths["extraction"].name])
    df_comp = load_xlsx([paths["completeness"].name])
    df_dig = load_xlsx([paths["digital"].name])
    df_iri = load_xlsx([paths["iri"].name])

    cfg = {...}  # (reuse your cfg dict here or load from config/config.yaml)

    df = clean_and_merge(df_ext, df_comp, df_dig, df_iri, cfg)

    mask = df[cfg["features"]["target_col"]].isin(
        cfg["features"]["target_mapping"].keys()
    )
    df_f = df.loc[mask].copy()
    df_f["Project_Outcome"] = (
        df_f[cfg["features"]["target_col"]]
        .map(cfg["features"]["target_mapping"])
        .astype(int)
    )

    X = df_f.drop(
        columns=[
            "Project code",
            cfg["features"]["target_col"],
            "Date",
            "Project_Outcome",
        ],
        errors="ignore",
    )
    y = df_f["Project_Outcome"]

    numeric = [c for c in cfg["features"]["numeric"] if c in X.columns]
    categorical = [c for c in X.columns if c not in numeric and X[c].dtype == "object"]

    pre = build_preprocessor(
        numeric, categorical, rare_min_freq=cfg["ml"]["rare_min_freq"]
    )
    models = build_models(
        pre,
        seed=cfg["ml"]["random_state"],
        use_smote=cfg["ml"].get("use_smote", False),
        calibrate=cfg["ml"]["calibrate"],
        calibration=cfg["ml"]["calibration"],
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg["ml"]["test_size"],
        stratify=y,
        random_state=cfg["ml"]["random_state"],
    )

    cv_df = cv_summary_for_models(
        models,
        X_train,
        y_train,
        n_splits=cfg["ml"]["cv_n_splits"],
        n_repeats=cfg["ml"]["cv_n_repeats"],
        seed=cfg["ml"]["random_state"],
    )
    best = choose_best_name(cv_df)
    best_pipe = models[best].fit(X_train, y_train)

    y_prob = best_pipe.predict_proba(X_test)[:, 1]
    metrics = holdout_metrics(y_test.to_numpy(), y_prob, threshold=0.5)

    adv = adversarial_auc(
        best_pipe.named_steps["preprocessor"],
        X_train,
        X_test,
        seed=cfg["ml"]["random_state"],
    )
    imp = permutation_importance_df(best_pipe, X_test, y_test, top_k=25)
    (artifacts / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    imp.to_csv(
        artifacts / "feature_importance_top25.csv", index=False, encoding="utf-8"
    )
    print(f"Best: {best} | Metrics: {metrics} | Adv AUC: {round(adv, 3)}")


if __name__ == "__main__":
    main()
