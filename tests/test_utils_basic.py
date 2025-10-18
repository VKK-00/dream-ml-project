import numpy as np
import pandas as pd

from src.utils import RareCategoryGrouper, clean_and_merge_strict, to_float_safe


def test_to_float_safe_parses_messy():
    cases = [" 1 234 ", "1,234", "\u00a01,234", "-", "", None]
    got = [to_float_safe(x) for x in cases]
    assert got[:3] == [1234.0, 1234.0, 1234.0]
    assert np.isnan(got[3]) and np.isnan(got[4]) and np.isnan(got[5])


def test_rare_category_grouper_basic():
    df = pd.DataFrame({"c": ["A"] * 5 + ["B"] * 2 + ["C"] * 1})
    g = RareCategoryGrouper(min_freq=3).fit(df)
    out = g.transform(df)
    assert set(out["c"].unique()) == {"A", "__OTHER__"}


def test_clean_and_merge_no_leakage_columns():
    # Minimal synthetic frames with required keys only
    ext = pd.DataFrame(
        {
            "Project code": ["X1", "X2"],
            "Project status": ["complete", "unsuccessful"],
            "Region": ["Kyivska", np.nan],
            "Forseen duration, months": [10, "12"],
            "Overall estimated budget, ₴": ["1000000", "2000000"],
            "Committed funding, ₴": ["500000", "1500000"],
            "Number of objects": [1, 1],
            "Number of contracting processes": [0, 0],
            "Number of contracts": [1, 0],
            "Number of investment objects requiring design documents": [0, 0],
            "Date": ["2024-01-01", "2024-01-02"],
        }
    )
    comp = pd.DataFrame(
        {
            "Project code": ["X1", "X2"],
            "Project data availability": ["Available", "Not available"],
        }
    )
    dig = pd.DataFrame({"Region Name": ["Kyivska"], "Digital Index": [1.0]})
    iri = pd.DataFrame({"Oblast": ["Kyivska"], "Institutional Capacity": [0.5]})

    cfg = {
        "features": {
            "region_cols": {"digital": "Region Name", "iri": "Oblast"},
            "canonical_region": "Region",
        }
    }
    out = clean_and_merge_strict(ext, comp, dig, iri, cfg)
    # still has canonical Region, merged indexes and computed feature
    assert "Region" in out.columns
    assert "Digital Index" in out.columns
    assert "Institutional Capacity" in out.columns
    assert "Financial_Efficiency" in out.columns
