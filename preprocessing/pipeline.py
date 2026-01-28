import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class TherapeuticPreprocessor:
    """
    Single-source preprocessing pipeline.
    - Column harmonization
    - Missing value handling
    - Categorical encoding
    - Scaling (fit on train only)
    """

    def __init__(self, column_map: dict):
        """
        column_map:
        {
            "heart_rate": ["Heart Rate (bpm)", "HR"],
            "sleep_hours": ["Sleep Hours"],
            ...
        }
        """
        self.column_map = column_map
        self.scaler = StandardScaler()

        # learned during fit
        self.numeric_cols = None
        self.categorical_cols = None
        self.feature_cols_ = None

    # -------------------------
    # internal helpers
    # -------------------------
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        for canonical, variants in self.column_map.items():
            for v in variants:
                if v in df.columns:
                    df[canonical] = df[v]
                    break

        return df

    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        # one-hot encode categoricals
        return pd.get_dummies(df, columns=self.categorical_cols, drop_first=True)

    # -------------------------
    # public API
    # -------------------------
    def fit(self, df: pd.DataFrame):
        """
        Fit ONLY on training data
        """
        df = df.copy()
        df = self._standardize_columns(df)

        # infer column types
        self.numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        self.categorical_cols = df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        # fill missing values (train statistics only)
        for col in self.numeric_cols:
            df[col] = df[col].fillna(df[col].median())

        for col in self.categorical_cols:
            df[col] = df[col].fillna("Unknown")

        # encode
        df_encoded = self._encode(df)

        # store final feature space
        self.feature_cols_ = df_encoded.columns.tolist()

        # fit scaler on numeric columns only
        self.scaler.fit(df_encoded[self.numeric_cols])

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform train / val / test / inference data
        """
        if self.feature_cols_ is None:
            raise RuntimeError("Preprocessor must be fitted before transform().")

        df = df.copy()
        df = self._standardize_columns(df)

        # fill missing values using SAME logic
        for col in self.numeric_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = 0

        for col in self.categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna("Unknown")
            else:
                df[col] = "Unknown"

        # encode
        df_encoded = self._encode(df)

        # align columns EXACTLY to training space
        df_encoded = df_encoded.reindex(
            columns=self.feature_cols_, fill_value=0
        )

        # scale numeric columns
        df_encoded[self.numeric_cols] = self.scaler.transform(
            df_encoded[self.numeric_cols]
        )

        return df_encoded
