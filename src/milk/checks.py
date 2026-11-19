"""Detectors for the seven data problems. Each function only READS the data."""
import numpy as np
import pandas as pd

NUMERIC = ["cows", "buffaloes", "fodder_price_rs", "temp_c", "vet_visits", "feed_kg"]


def check_leaking_column(train, holdout, target="litres_per_day"):
    """Columns missing from holdout that also track the target closely."""
    missing = [c for c in train.columns if c not in holdout.columns and c != target]
    for c in missing:
        ratio = (train[c] / train[target]).median()
        if abs(ratio - 1) < 0.02:  # almost the same number as the target
            return {"present": True, "column": c}
    return {"present": False, "column": None}


def check_unit_mixup(train, group_col="village_cluster", target="litres_per_day"):
    """Find a (column, group) whose median is a round multiple of the others."""
    round_factors = [10, 12, 60, 100, 1000]
    for col in NUMERIC:
        med = train.groupby(group_col)[col].median()
        for grp, m in med.items():
            others = med.drop(grp).median()
            if others <= 0:
                continue
            ratio = m / others
            for f in round_factors:
                if abs(ratio - f) / f < 0.15:
                    return {"present": True, "column": col, "group_value": grp, "factor": f}
                if abs(1 / ratio - f) / f < 0.15:
                    return {"present": True, "column": col, "group_value": grp, "factor": 1 / f}
    return {"present": False, "column": None, "group_value": None, "factor": None}


def check_sentinel_value(train, candidates=(-1, -999, 999, 9999, -9999)):
    for col in NUMERIC + ["litres_paid_in_passbook", "litres_per_day"]:
        for v in candidates:
            n = int((train[col] == v).sum())
            if n >= 3:
                return {"present": True, "column": col, "value": v, "count": n}
    return {"present": False, "column": None, "value": None, "count": None}


def check_duplicate_rows(train):
    n = int(train.duplicated().sum())
    return {"present": n > 0, "count": n if n > 0 else None}


def check_stuck_sensor(train, min_run=5):
    """Same value on every row for a long run of consecutive rows (sorted by date)."""
    d = train.drop_duplicates().copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values(["date", "record_id"]).reset_index(drop=True)
    for col in ["fodder_price_rs", "temp_c", "feed_kg"]:  # continuous readings
        run_id = d[col].ne(d[col].shift()).cumsum()
        runs = d.groupby(run_id).agg(n=(col, "size"), a=("date", "min"), b=("date", "max"))
        best = runs.sort_values("n", ascending=False).iloc[0]
        if best["n"] >= min_run:
            return {"present": True, "column": col,
                    "start_date": str(best["a"].date()), "end_date": str(best["b"].date())}
    return {"present": False, "column": None, "start_date": None, "end_date": None}


def check_target_typos(train, target="litres_per_day", factor=10):
    """Values far above a big empty gap in the sorted target."""
    s = train[target].sort_values().reset_index(drop=True)
    cut = s.quantile(0.5) * 3  # anything above 3x the median is suspect
    big = s[s > cut]
    # confirm: after dividing by factor they land inside the normal range
    normal_max = s[s <= cut].max()
    ok = (big / factor).between(s[s <= cut].min(), normal_max).all()
    if len(big) and ok:
        return {"present": True, "count": int(len(big))}
    return {"present": False, "count": None}


def check_spelling_variants(train, later):
    """Compare text columns with the clean spellings in later.csv."""
    for col in train.select_dtypes(exclude="number").columns:
        if col in ("record_id", "date"):
            continue
        clean = set(later[col].unique())
        if set(train[col].unique()) != clean:
            return {"present": True, "column": col, "true_levels": len(clean)}
    return {"present": False, "column": None, "true_levels": None}
