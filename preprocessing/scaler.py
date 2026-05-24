import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from typing import Optional

from utils.config import FEATURES, PREPROCESSING


class FeatureScaler:
    """
    Scales feature columns before they are fed into KNN models.
    Operates on habitability and planet_type feature sets independently.

    Labels ('habitable', 'planet_type') are never touched.
    """

    _SCALER_MAP = {
        "standard": StandardScaler,
        "minmax"  : MinMaxScaler,
        "robust"  : RobustScaler,
    }

    def __init__(self) -> None:
        method = PREPROCESSING["scaling_method"]

        if method not in self._SCALER_MAP:
            raise ValueError(
                f"Unknown scaling_method '{method}' in config. "
                f"Choose from: {list(self._SCALER_MAP.keys())}"
            )

        self._method                      = method
        self._habitability_scaler         = self._SCALER_MAP[method]()
        self._planet_type_scaler          = self._SCALER_MAP[method]()
        self._habitability_fitted: bool   = False
        self._planet_type_fitted: bool    = False

    # ─── Public Interface ─────────────────────────────────────────────────────

    def fit_habitability(self, df: pd.DataFrame) -> "FeatureScaler":
        cols = self._get_cols(df, "habitability")
        self._habitability_scaler.fit(df[cols])
        self._habitability_fitted = True
        return self

    def fit_planet_type(self, df: pd.DataFrame) -> "FeatureScaler":
        cols = self._get_cols(df, "planet_type")
        self._planet_type_scaler.fit(df[cols])
        self._planet_type_fitted = True
        return self

    def fit(self, df: pd.DataFrame) -> "FeatureScaler":
        self.fit_habitability(df)
        self.fit_planet_type(df)
        return self

    def transform_habitability(self, df: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted("habitability")
        return self._apply_transform(df, "habitability", self._habitability_scaler)

    def transform_planet_type(self, df: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted("planet_type")
        return self._apply_transform(df, "planet_type", self._planet_type_scaler)

    def fit_transform_habitability(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit_habitability(df).transform_habitability(df)

    def fit_transform_planet_type(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit_planet_type(df).transform_planet_type(df)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fits and transforms both feature sets.
        Returns a single DataFrame with all scaled columns merged,
        plus any non-feature columns (labels, etc.) preserved.
        """
        hab_df  = self.fit_transform_habitability(df)
        type_df = self.fit_transform_planet_type(df)

        type_only_cols = [
            c for c in FEATURES["planet_type"]
            if c not in FEATURES["habitability"]
        ]

        result = hab_df.copy()
        for col in type_only_cols:
            result[col] = type_df[col]

        return result

    def get_feature_means(self, model: str) -> Optional[np.ndarray]:
        """Returns per-feature means learned during fit (StandardScaler only)."""
        scaler = self._get_scaler(model)
        return getattr(scaler, "mean_", None)

    def get_feature_scales(self, model: str) -> Optional[np.ndarray]:
        """Returns per-feature scales learned during fit."""
        scaler = self._get_scaler(model)
        return getattr(scaler, "scale_", None)

    def summary(self, model: str) -> dict:
        scaler = self._get_scaler(model)
        cols   = FEATURES[model]
        return {
            "model"         : model,
            "method"        : self._method,
            "features"      : cols,
            "feature_means" : self.get_feature_means(model),
            "feature_scales": self.get_feature_scales(model),
        }

    # ─── Private Helpers ──────────────────────────────────────────────────────

    def _apply_transform(
        self,
        df    : pd.DataFrame,
        model : str,
        scaler,
    ) -> pd.DataFrame:
        cols   = self._get_cols(df, model)
        result = df.copy()

        scaled_values       = scaler.transform(df[cols])
        result[cols]        = scaled_values

        return result

    def _get_cols(self, df: pd.DataFrame, model: str) -> list:
        cols    = FEATURES[model]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"DataFrame is missing columns for '{model}' model: {missing}. "
                f"Make sure FeatureEngineer has been run first."
            )
        return cols

    def _check_fitted(self, model: str) -> None:
        fitted = (
            self._habitability_fitted if model == "habitability"
            else self._planet_type_fitted
        )
        if not fitted:
            raise RuntimeError(
                f"Scaler for '{model}' has not been fitted yet. "
                f"Call fit_{model}() or fit() before transforming."
            )

    def _get_scaler(self, model: str):
        if model == "habitability":
            return self._habitability_scaler
        elif model == "planet_type":
            return self._planet_type_scaler
        else:
            raise ValueError(f"Unknown model '{model}'. Use 'habitability' or 'planet_type'.")