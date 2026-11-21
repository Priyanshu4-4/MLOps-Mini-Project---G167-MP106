"""Train and register the first milk-collection model with MLflow."""
import hashlib
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from mlflow import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def make_pipeline(config, model_params):
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", config["numeric_features"]),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                config["categorical_features"],
            ),
        ]
    )
    regressor = RandomForestRegressor(
        **model_params,
        random_state=config["random_state"],
        n_jobs=-1,
    )
    return Pipeline(
        steps=[("preprocessing", preprocessing), ("regressor", regressor)]
    )


def main(params_path="params.yaml"):
    with open(params_path, encoding="utf-8") as params_file:
        params = yaml.safe_load(params_file)

    data_config = params["data"]
    model_config = params["model"]
    clean_path = Path(data_config["clean_path"])
    train_path = Path(data_config["train_path"])
    findings_path = Path(data_config["findings_path"])

    if not clean_path.exists():
        raise FileNotFoundError(
            f"{clean_path} is missing; run `python -m milk.run_checks` first."
        )

    data = pd.read_csv(clean_path)
    feature_columns = (
        model_config["numeric_features"] + model_config["categorical_features"]
    )
    x = data[feature_columns]
    y = data[params["target"]]
    x_train, x_validation, y_train, y_validation = train_test_split(
        x,
        y,
        test_size=model_config["validation_size"],
        random_state=model_config["random_state"],
    )

    with train_path.open("rb") as train_file:
        data_sha256 = hashlib.sha256(train_file.read()).hexdigest()

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(model_config["experiment_name"])

    results = []
    for index, candidate in enumerate(model_config["candidates"], start=1):
        pipeline = make_pipeline(model_config, candidate)
        run_name = f"v1_random_forest_{index}"

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params(
                {
                    "data_sha256": data_sha256,
                    "model_version": "v1",
                    "model_type": "RandomForestRegressor",
                    "n_estimators": candidate["n_estimators"],
                    "max_depth": candidate["max_depth"] or "none",
                    "validation_size": model_config["validation_size"],
                    "random_state": model_config["random_state"],
                }
            )
            pipeline.fit(x_train, y_train)
            predictions = pipeline.predict(x_validation)
            mae = float(mean_absolute_error(y_validation, predictions))
            mlflow.log_metric("mae_validation", mae)
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="model",
                input_example=x_train.head(5),
                serialization_format="cloudpickle",
            )
            results.append(
                {
                    "run_id": run.info.run_id,
                    "mae": mae,
                    "pipeline": pipeline,
                }
            )
            print(f"{run_name}: validation MAE = {mae:.3f}")

    best = min(results, key=lambda result: result["mae"])
    model_name = model_config["registered_model_name"]
    model_uri = f"runs:/{best['run_id']}/model"
    registered = mlflow.register_model(model_uri=model_uri, name=model_name)

    client = MlflowClient()
    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=registered.version,
    )

    model_path = Path("models/model.joblib")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["pipeline"], model_path)

    with findings_path.open(encoding="utf-8") as findings_file:
        findings = yaml.safe_load(findings_file)
    findings["tracking"].update(
        {
            "experiment_name": model_config["experiment_name"],
            "best_run_id": best["run_id"],
            "registered_model_name": model_name,
            "champion_version": int(registered.version),
        }
    )
    with findings_path.open("w", encoding="utf-8") as findings_file:
        yaml.safe_dump(findings, findings_file, sort_keys=False)

    print(f"Best v1 validation MAE: {best['mae']:.3f}")
    print(f"Best run id: {best['run_id']}")
    print(f"Registered {model_name} version {registered.version} as @champion")
    print(f"Saved champion model to {model_path}")


if __name__ == "__main__":
    main()
