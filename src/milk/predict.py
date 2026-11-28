
from pathlib import Path

import joblib
import pandas as pd
import yaml


def main(params_path="params.yaml"):
    with open(params_path, encoding="utf-8") as params_file:
        params = yaml.safe_load(params_file)

    model_config = params["model"]
    feature_columns = (
        model_config["numeric_features"] + model_config["categorical_features"]
    )

    # models/model.joblib is the @champion (v2), saved by retrain.py on promotion
    model = joblib.load("models/model.joblib")

    holdout = pd.read_csv(params["data"]["holdout_path"])
    predictions = model.predict(holdout[feature_columns])

    output = pd.DataFrame(
        {"record_id": holdout["record_id"], "prediction": predictions}
    )
    output_path = Path("predictions/holdout_predictions.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    print(f"Saved {len(output)} predictions to {output_path}")


if __name__ == "__main__":
    main()