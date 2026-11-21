import pandas as pd
from milk.clean import clean_train

def test_clean_train_drops_duplicate_rows():
    train = pd.DataFrame([
        {"record_id": "r1", "village_cluster": "Cluster A", "litres_per_day": 5},
        {"record_id": "r1", "village_cluster": "Cluster A", "litres_per_day": 5},
    ])

    findings = {
        "unit_mixup": {"present": False},
        "target_typos": {"present": False},
        "leaking_column": {"present": False},
    }

    cleaned, log = clean_train(train, findings)

    assert len(cleaned) == 1


def test_clean_train_corrects_only_mixed_unit_group():
    train = pd.DataFrame([
        {"record_id": "r1", "village_cluster": "Cluster A", "fodder_price_rs": 5, "litres_per_day": 10},
        {"record_id": "r2", "village_cluster": "Cluster B", "fodder_price_rs": 500, "litres_per_day": 12},
    ])

    findings = {
        "unit_mixup": {
            "present": True,
            "column": "fodder_price_rs",
            "group_value": "Cluster B",
            "factor": 100,
        },
        "target_typos": {"present": False},
        "leaking_column": {"present": False},
    }

    cleaned, log = clean_train(train, findings)

    price_a = cleaned.loc[cleaned["village_cluster"] == "Cluster A", "fodder_price_rs"].iloc[0]
    price_b = cleaned.loc[cleaned["village_cluster"] == "Cluster B", "fodder_price_rs"].iloc[0]
    assert price_a == 5
    assert price_b == 5
