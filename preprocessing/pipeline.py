import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


class TherapeuticPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, column_map: dict):
        self.column_map = column_map
        self.column_transformer = None
        self.numeric_cols_ = None
        self.categorical_cols_ = None
        self.feature_names_ = None

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        new_df = pd.DataFrame()

        for canonical, variants in self.column_map.items():
            for v in variants:
                if v in df.columns:
                    new_df[canonical] = df[v]
                    break
            else:
                raise ValueError(
                    f"Missing required column for canonical feature: '{canonical}'"
                )

        return new_df

    def fit(self, X, y=None):
        X = self._standardize_columns(X)

        self.numeric_cols_ = X.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()

        self.categorical_cols_ = X.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        self.column_transformer = ColumnTransformer([
            ("num", numeric_pipeline, self.numeric_cols_),
            ("cat", categorical_pipeline, self.categorical_cols_)
        ])

        self.column_transformer.fit(X)
        self.feature_names_ = self.column_transformer.get_feature_names_out()

        return self

    def transform(self, X):
        if self.column_transformer is None:
            raise RuntimeError("Preprocessor must be fitted first.")

        X = self._standardize_columns(X)
        transformed = self.column_transformer.transform(X)

        return pd.DataFrame(
            transformed,
            columns=self.feature_names_
        )