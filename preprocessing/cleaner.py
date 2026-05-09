import os

import numpy as np
import pandas as pd

from utils.config import COLUMNS, PATHS, PREPROCESSING


class DataCleaner:
    def __init__(self) -> None:
        self.raw_path     = PATHS["raw_data"]
        self.cleaned_path = PATHS["cleaned_data"]
        self.strategy     = PREPROCESSING["missing_value_strategy"]
        self.std_threshold = PREPROCESSING["outlier_std_threshold"]

        self._df: pd.DataFrame = pd.DataFrame()
        self._all_feature_columns = list(set(
            list(COLUMNS.values())
        ))

    # Public Interface

    def load(self) -> "DataCleaner":
        if not os.path.exists(self.raw_path):
            raise FileNotFoundError(f"Raw dataset not found at: {self.raw_path}")

        self._df = pd.read_csv(self.raw_path, comment="#")

        missing = [c for c in self._all_feature_columns if c not in self._df.columns]
        if missing:
            raise ValueError(f"Dataset is missing expected columns: {missing}")

        return self

    def clean(self) -> "DataCleaner":
        self._df = self._df[self._all_feature_columns].copy()
        self._drop_fully_empty_rows()
        self._fill_missing_values()
        self._remove_outliers()
        self._drop_invalid_physical_values()
        self._reset_index()
        return self

    def save(self) -> "DataCleaner":
        os.makedirs(os.path.dirname(self.cleaned_path), exist_ok=True)
        self._df.to_csv(self.cleaned_path, index=False)
        return self

    def get_dataframe(self) -> pd.DataFrame:
        if self._df.empty:
            raise RuntimeError("No data loaded. Call load() and clean() first.")
        return self._df.copy()

    def summary(self) -> dict:
        return {
            "total_rows"      : len(self._df),
            "total_columns"   : len(self._df.columns),
            "missing_values"  : int(self._df.isnull().sum().sum()),
            "duplicate_rows"  : int(self._df.duplicated().sum()),
            "columns"         : list(self._df.columns),
        }

    # Private Steps

    def _drop_fully_empty_rows(self) -> None:
        before = len(self._df)
        self._df.dropna(how="all", inplace=True)
        dropped = before - len(self._df)
        if dropped:
            print(f"[Cleaner] Dropped {dropped} fully empty rows.")

    def _fill_missing_values(self) -> None:
        numeric_cols = self._df.select_dtypes(include=[np.number]).columns.tolist()

        for col in numeric_cols:
            if self._df[col].isnull().any():
                if self.strategy == "median":
                    fill_value = self._df[col].median()
                elif self.strategy == "mean":
                    fill_value = self._df[col].mean()
                elif self.strategy == "drop":
                    self._df.dropna(subset=[col], inplace=True)
                    continue
                else:
                    raise ValueError(
                        f"Unknown missing_value_strategy: '{self.strategy}'. "
                        f"Use 'median', 'mean', or 'drop'."
                    )
                self._df[col].fillna(fill_value, inplace=True)

        cat_cols = self._df.select_dtypes(include=["object", "category"]).columns.tolist()
        for col in cat_cols:
            if self._df[col].isnull().any():
                self._df[col].fillna("UNKNOWN", inplace=True)

    def _remove_outliers(self) -> None:
        numeric_cols = self._df.select_dtypes(include=[np.number]).columns.tolist()
        before = len(self._df)

        for col in numeric_cols:
            mean = self._df[col].mean()
            std  = self._df[col].std()

            if std == 0:
                continue

            lower = mean - self.std_threshold * std
            upper = mean + self.std_threshold * std
            self._df = self._df[(self._df[col] >= lower) & (self._df[col] <= upper)]

        removed = before - len(self._df)
        if removed:
            print(f"[Cleaner] Removed {removed} outlier rows (>{self.std_threshold}σ).")

    def _drop_invalid_physical_values(self) -> None:
        before = len(self._df)

        radius_col = COLUMNS["planet_radius"]
        temp_col   = COLUMNS["temperature"]

        if radius_col in self._df.columns:
            self._df = self._df[self._df[radius_col] > 0]

        if temp_col in self._df.columns:
            self._df = self._df[self._df[temp_col] > 0]

        removed = before - len(self._df)
        if removed:
            print(f"[Cleaner] Dropped {removed} rows with physically invalid values.")

    def _reset_index(self) -> None:
        self._df.reset_index(drop=True, inplace=True)