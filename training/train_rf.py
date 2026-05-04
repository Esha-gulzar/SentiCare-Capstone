import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from preprocessing.pipeline import TherapeuticPreprocessor

# ---------------------------
# 1. Load data
# ---------------------------
df = pd.read_csv("data/processed/anxiety_clean.csv")

TARGET = "Anxiety Level (1-10)"

X = df.drop(columns=[TARGET])
y = df[TARGET]

# ---------------------------
# 2. Split
# ---------------------------
X_train_df, X_val_df, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# 3. Preprocessing
# ---------------------------
column_map = {
    "age": ["Age"],
    "gender": ["Gender"],
    "sleep_hours": ["Sleep Hours"],
    "heart_rate": ["Heart Rate (bpm)"]
}

preprocessor = TherapeuticPreprocessor(column_map)
preprocessor.fit(X_train_df)

X_train = preprocessor.transform(X_train_df)
X_val   = preprocessor.transform(X_val_df)

# ---------------------------
# 4. Train model
# ---------------------------
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ---------------------------
# 5. Evaluate
# ---------------------------
preds = model.predict(X_val)

print("MSE:", mean_squared_error(y_val, preds))
print("MAE:", mean_absolute_error(y_val, preds))
print("R2 :", r2_score(y_val, preds))

# ---------------------------
# 6. Save artifacts
# ---------------------------
joblib.dump(preprocessor, "artifacts/preprocessor.joblib")
joblib.dump(model, "artifacts/anxiety_model.joblib")

print("Saved preprocessor and model.")
