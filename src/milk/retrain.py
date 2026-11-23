"""Retrain v2, compare it with v1, and apply the promotion rule."""
import hashlib
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from mlflow import MlflowClient
from sklearn.metrics import mean_absolute_error

from milk.train import make_pipeline


def read_yaml(path):
    with open(path, encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def main(params_path="params.yaml"):
    params = read_yaml(params_path)
    data_config = params["data"]
    model_config = params["model"]
    train_path = Path(data_config["train_path"])
    later_path = Path(data_config["later_path"])
    clean_path = Path(data_config["clean_path"])
    findings_path = Path(data_config["findings_path"])
    model_path = Path("models/model.joblib")

    if not clean_path.exists():
        raise FileNotFoundError(
            f"{clean_path} is missing; run `python -m milk.run_checks` first."
        )

    cleaned_train = pd.read_csv(clean_path)
    later = pd.read_csv(later_path).sort_values("date").reset_index(drop=True)
    check_start = int(len(later) * 0.70)
    later_train = later.iloc[:check_start]
    later_check = later.iloc[check_start:]
    if later_train.empty or later_check.empty:
        raise ValueError("later.csv needs rows in both the first 70% and last 30%")

    findings = read_yaml(findings_path)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(model_config["experiment_name"])
    client = MlflowClient()
    model_name = model_config["registered_model_name"]
    existing_versions = client.search_model_versions(f"name='{model_name}'")
    existing_v2 = next(
        (version for version in existing_versions if int(version.version) == 2),
        None,
    )
    if existing_v2 is not None:
        current_champion = client.get_model_version_by_alias(model_name, "champion")
        monitoring = findings["monitoring"]
        print("v2 is already registered; no new model version was created.")
        print(f"v1 MAE on check: {monitoring['mae_v1_on_check']:.3f}")
        print(f"v2 MAE on check: {monitoring['mae_v2_on_check']:.3f}")
        print(f"Decision: {monitoring['decision']}")
        print(f"Champion version: {current_champion.version}")
        print("The existing v2 run is in the Default MLflow experiment.")
        return
    if any(int(version.version) > 2 for version in existing_versions):
        raise RuntimeError(
            f"{model_name} already has a version newer than v2; refusing to register another."
        )

    v1_model = joblib.load(model_path)
    feature_columns = (
        model_config["numeric_features"] + model_config["categorical_features"]
    )
    target = params["target"]
    mae_v1 = float(
        mean_absolute_error(
            later_check[target], v1_model.predict(later_check[feature_columns])
        )
    )

    best_run = client.get_run(findings["tracking"]["best_run_id"])
    best_params = best_run.data.params
    model_params = {
        "n_estimators": int(best_params["n_estimators"]),
        "max_depth": (
            None if best_params["max_depth"] == "none" else int(best_params["max_depth"])
        ),
    }

    combined = pd.concat([cleaned_train, later_train], ignore_index=True)
    v2_model = make_pipeline(model_config, model_params)
    v2_model.fit(combined[feature_columns], combined[target])
    mae_v2 = float(
        mean_absolute_error(
            later_check[target], v2_model.predict(later_check[feature_columns])
        )
    )
    decision = "promote" if mae_v2 <= mae_v1 * 0.92 else "keep"

    with train_path.open("rb") as train_file:
        data_sha256 = hashlib.sha256(train_file.read()).hexdigest()

    with mlflow.start_run(run_name="v2_retraining") as run:
        mlflow.log_params(
            {
                "data_sha256": data_sha256,
                "model_version": "v2",
                "model_type": "RandomForestRegressor",
                "n_estimators": model_params["n_estimators"],
                "max_depth": (
                    "none"
                    if model_params["max_depth"] is None
                    else model_params["max_depth"]
                ),
                "training_rows": len(combined),
                "later_train_rows": len(later_train),
                "check_rows": len(later_check),
            }
        )
        mlflow.log_metrics(
            {"mae_v1_on_check": mae_v1, "mae_v2_on_check": mae_v2}
        )
        mlflow.sklearn.log_model(
            sk_model=v2_model,
            name="model",
            input_example=combined[feature_columns].head(5),
            serialization_format="cloudpickle",
        )
        v2_run_id = run.info.run_id

    v2_version = mlflow.register_model(
        model_uri=f"runs:/{v2_run_id}/model",
        name=model_name,
    )
    if decision == "promote":
        client.set_registered_model_alias(model_name, "champion", v2_version.version)
        champion_version = int(v2_version.version)
        joblib.dump(v2_model, model_path)
    else:
        client.set_registered_model_alias(model_name, "champion", "1")
        champion_version = 1

    findings["tracking"]["champion_version"] = champion_version
    findings["monitoring"].update(
        {
            "mae_v1_on_check": round(mae_v1, 6),
            "mae_v2_on_check": round(mae_v2, 6),
            "decision": decision,
        }
    )
    with findings_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(findings, stream, sort_keys=False)

    print(f"v1 MAE on final 30% of later.csv: {mae_v1:.3f}")
    print(f"v2 MAE on final 30% of later.csv: {mae_v2:.3f}")
    print("Decision:", decision)
    print(f"Registered v2 as version {v2_version.version}")
    print(f"Champion version: {champion_version}")


if __name__ == "__main__":
    main()
