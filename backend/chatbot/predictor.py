import joblib
import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]

stress_model = joblib.load(
    BASE_DIR / "artifacts/stress_classification/stress_bundle.joblib"
)

anxiety_model = joblib.load(
    BASE_DIR / "artifacts/anxiety_classification/anxiety_pipeline.joblib"
)


def predict(condition: str, feature_answers: dict):

    df = pd.DataFrame([feature_answers])

    if condition == "anxiety":
        return anxiety_model.predict(df)[0]

    if condition == "stress":
        return stress_model.predict(df)[0]

    return None