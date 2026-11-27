"""Measure feature drift and v1 errors by village cluster on later.csv."""
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from mlflow import MlflowClient
from sklearn.metrics import mean_absolute_error


def population_stability_index(reference, current, bins=10):
    """Calculate PSI using quantile bins defined by the reference data."""
    reference = pd.to_numeric(reference, errors="raise").dropna()
    current = pd.to_numeric(current, errors="raise").dropna()
    if reference.empty or current.empty:
        raise ValueError("PSI needs non-empty reference and current values")

    edges = np.unique(reference.quantile(np.linspace(0, 1, bins + 1)).to_numpy())
    if len(edges) < 2:
        # A constant reference feature can only have one reference bucket.
        # The epsilon keeps the logarithm finite if current data differs.
        ref_share = np.array([1.0])
        current_share = np.array([float((current == reference.iloc[0]).mean())])
        current_share = np.append(current_share, 1.0 - current_share[0])
        ref_share = np.append(ref_share, 0.0)
    else:
        edges[0], edges[-1] = -np.inf, np.inf
        ref_bins = pd.cut(reference, bins=edges, include_lowest=True)
        current_bins = pd.cut(current, bins=edges, include_lowest=True)
        ref_share = ref_bins.value_counts(sort=False, normalize=True).to_numpy()
        current_share = (
            current_bins.value_counts(sort=False, normalize=True).to_numpy()
        )

    epsilon = 1e-6
    ref_share = np.clip(ref_share, epsilon, None)
    current_share = np.clip(current_share, epsilon, None)
    return float(np.sum((current_share - ref_share) * np.log(current_share / ref_share)))


def main(params_path="params.yaml"):
    params = yaml.safe_load(Path(params_path).read_text(encoding="utf-8"))
    data_config = params["data"]
    model_config = params["model"]
    findings_path = Path(data_config["findings_path"])

    clean_train = pd.read_csv(data_config["clean_path"])
    later = pd.read_csv(data_config["later_path"])
    numeric_features = model_config["numeric_features"]
    group_col = params["group_col"]
    target = params["target"]

    missing = set(numeric_features + [group_col, target]).difference(later.columns)
    missing.update(set(numeric_features + [target]).difference(clean_train.columns))
    if missing:
        raise ValueError(f"Required monitoring columns are missing: {sorted(missing)}")

    psi = {
        feature: round(
            population_stability_index(clean_train[feature], later[feature]), 6
        )
        for feature in numeric_features
    }

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()
    model_name = model_config["registered_model_name"]
    v1_versions = [
        version
        for version in client.search_model_versions(f"name='{model_name}'")
        if int(version.version) == 1
    ]
    if not v1_versions:
        raise RuntimeError(
            f"{model_name} version 1 is not registered; cannot measure v1 group MAE."
        )
    v1_model = mlflow.sklearn.load_model(f"models:/{model_name}/1")
    feature_columns = numeric_features + model_config["categorical_features"]
    predictions = v1_model.predict(later[feature_columns])
    errors = pd.DataFrame(
        {group_col: later[group_col], "absolute_error": (later[target] - predictions).abs()}
    )
    mae_by_group = errors.groupby(group_col)["absolute_error"].mean().to_dict()
    mae_by_group = {str(group): round(float(mae), 6) for group, mae in mae_by_group.items()}

    findings = yaml.safe_load(findings_path.read_text(encoding="utf-8"))
    monitoring = findings["monitoring"]
    monitoring["psi"] = psi
    monitoring["most_drifted_feature"] = max(psi, key=psi.get)
    monitoring["mae_by_group"] = mae_by_group
    monitoring["worst_served_group"] = max(mae_by_group, key=mae_by_group.get)
    findings_path.write_text(
        yaml.safe_dump(findings, sort_keys=False), encoding="utf-8"
    )

    print("PSI (clean train vs later):")
    for feature, value in psi.items():
        print(f"  {feature}: {value:.6f}")
    print("v1 MAE by cluster on later.csv:")
    for group, value in mae_by_group.items():
        print(f"  {group}: {value:.6f}")
    print("Most drifted feature:", monitoring["most_drifted_feature"])
    print("Worst-served cluster:", monitoring["worst_served_group"])
    print(f"Updated {findings_path}")


if __name__ == "__main__":
    main()
