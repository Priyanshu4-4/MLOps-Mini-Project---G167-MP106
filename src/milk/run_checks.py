"""Run all seven checks, print results, write cleaned data and update findings."""
import hashlib
import yaml
import pandas as pd
from pathlib import Path
from milk import checks
from milk.clean import clean_train


def main(params_path="params.yaml"):
    p = yaml.safe_load(open(params_path))
    d = p["data"]
    train = pd.read_csv(d["train_path"])
    later = pd.read_csv(d["later_path"])
    holdout = pd.read_csv(d["holdout_path"])
    print("train.csv sha256:", hashlib.sha256(open(d["train_path"], "rb").read()).hexdigest())

    f = {
        "leaking_column": checks.check_leaking_column(train, holdout, p["target"]),
        "unit_mixup": checks.check_unit_mixup(train, p["group_col"]),
        "sentinel_value": checks.check_sentinel_value(train, p["checks"]["sentinel_candidates"]),
        "duplicate_rows": checks.check_duplicate_rows(train),
        "stuck_sensor": checks.check_stuck_sensor(train, p["checks"]["stuck_min_run"]),
        "target_typos": checks.check_target_typos(train, p["target"], p["checks"]["typo_factor"]),
        "spelling_variants": checks.check_spelling_variants(train, later),
    }
    for k, v in f.items():
        print(k, v)

    clean, log = clean_train(train, f, p["target"], p["group_col"])
    Path(d["clean_path"]).parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(d["clean_path"], index=False)
    print("\n".join(log), "\ncleaned shape:", clean.shape)

    form = yaml.safe_load(open(d["findings_path"]))
    form["data_issues"].update(f)
    yaml.safe_dump(form, open(d["findings_path"], "w"), sort_keys=False)


if __name__ == "__main__":
    main()
