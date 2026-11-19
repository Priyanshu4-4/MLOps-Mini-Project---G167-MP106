"""Cleaning steps. Raw train.csv is never modified; a new DataFrame is returned."""
import pandas as pd


def clean_train(train, findings, target="litres_per_day", group_col="village_cluster"):
    df = train.copy()
    log = []

    n0 = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    log.append(f"dropped {n0 - len(df)} duplicate rows")

    um = findings["unit_mixup"]
    if um["present"]:
        mask = df[group_col] == um["group_value"]
        df.loc[mask, um["column"]] = df.loc[mask, um["column"]] / um["factor"]
        log.append(f"{um['column']} in {um['group_value']} divided by {um['factor']}")

    if findings["target_typos"]["present"]:
        cut = df[target].median() * 3
        mask = df[target] > cut
        df.loc[mask, target] = df.loc[mask, target] / 10
        log.append(f"divided {int(mask.sum())} target typos by 10")

    leak = findings["leaking_column"]
    if leak["present"]:
        df = df.drop(columns=[leak["column"]])
        log.append(f"dropped leaking column {leak['column']}")

    return df, log
