import numpy as np
import pandas as pd

from utils.config import (
    COLUMNS,
    HABITABILITY_THRESHOLDS,
    LABELS,
    PLANET_TYPE_THRESHOLDS,
)


class FeatureEngineer:
    """
    Consumes the cleaned DataFrame from DataCleaner and produces:
      - New calculated features (ESI, flux ratio, density proxy, etc.)
      - Derived labels: 'habitable' and 'planet_type'

    These engineered columns are what both KNN models train on.
    """

    # Earth reference constants (used for similarity calculations)
    EARTH_RADIUS_REF    = 1.0       # Earth radii
    EARTH_TEMP_REF      = 288.0     # Kelvin (mean surface temperature)
    EARTH_FLUX_REF      = 1.0       # S_earth
    EARTH_GRAVITY_REF   = 1.0       # relative surface gravity
    EARTH_PERIOD_REF    = 365.25    # days

    def __init__(self, df: pd.DataFrame) -> None:
        if df.empty:
            raise ValueError("Input DataFrame is empty. Run DataCleaner first.")
        self._df = df.copy()

    # ─── Public Interface ─────────────────────────────────────────────────────

    def engineer(self) -> "FeatureEngineer":
        self._add_earth_similarity_index()
        self._add_flux_ratio()
        self._add_thermal_habitability_score()
        self._add_size_category_score()
        self._add_orbital_zone_flag()
        self._add_density_proxy()
        self._add_habitable_label()
        self._add_planet_type_label()
        return self

    def get_dataframe(self) -> pd.DataFrame:
        return self._df.copy()

    def feature_summary(self) -> dict:
        new_cols = [
            "earth_similarity_index",
            "flux_ratio",
            "thermal_habitability_score",
            "size_category_score",
            "in_habitable_zone",
            "density_proxy",
            "habitable",
            "planet_type",
        ]
        available = [c for c in new_cols if c in self._df.columns]
        return {
            "engineered_features" : available,
            "total_rows"          : len(self._df),
            "habitable_count"     : int(self._df["habitable"].sum()) if "habitable" in self._df.columns else None,
            "planet_type_counts"  : self._df["planet_type"].value_counts().to_dict() if "planet_type" in self._df.columns else None,
        }

    # ─── Feature Calculations ─────────────────────────────────────────────────

    def _add_earth_similarity_index(self) -> None:
        """
        ESI measures how physically similar a planet is to Earth.
        Score ranges from 0.0 (completely different) to 1.0 (Earth twin).
        Formula is adapted from the Planetary Habitability Laboratory (PHL) ESI.
        """
        radius_col = COLUMNS["planet_radius"]
        temp_col   = COLUMNS["temperature"]
        flux_col   = COLUMNS["stellar_flux"]

        radius_sim = 1 - np.abs(
            (self._df[radius_col] - self.EARTH_RADIUS_REF) /
            (self._df[radius_col] + self.EARTH_RADIUS_REF)
        )
        temp_sim = 1 - np.abs(
            (self._df[temp_col] - self.EARTH_TEMP_REF) /
            (self._df[temp_col] + self.EARTH_TEMP_REF)
        )
        flux_sim = 1 - np.abs(
            (self._df[flux_col] - self.EARTH_FLUX_REF) /
            (self._df[flux_col] + self.EARTH_FLUX_REF)
        )

        # Geometric mean of all similarity components
        self._df["earth_similarity_index"] = np.cbrt(radius_sim * temp_sim * flux_sim).clip(0.0, 1.0)

    def _add_flux_ratio(self) -> None:
        """
        Ratio of stellar flux received to Earth's flux.
        flux_ratio = 1.0 means exact same energy as Earth receives.
        > 1.75 → too hot, < 0.25 → too cold.
        """
        flux_col = COLUMNS["stellar_flux"]
        self._df["flux_ratio"] = self._df[flux_col] / self.EARTH_FLUX_REF

    def _add_thermal_habitability_score(self) -> None:
        """
        Gaussian-shaped score centered on Earth's temperature (288K).
        Score of 1.0 = exactly Earth temperature, drops off on both sides.
        """
        temp_col = COLUMNS["temperature"]
        sigma    = 50.0   # tolerance in Kelvin

        self._df["thermal_habitability_score"] = np.exp(
            -0.5 * ((self._df[temp_col] - self.EARTH_TEMP_REF) / sigma) ** 2
        ).clip(0.0, 1.0)

    def _add_size_category_score(self) -> None:
        """
        Gaussian score centered on Earth radius (1.0 R_earth).
        Smaller or larger planets score lower.
        """
        radius_col = COLUMNS["planet_radius"]
        sigma      = 0.8   # tolerance in Earth radii

        self._df["size_category_score"] = np.exp(
            -0.5 * ((self._df[radius_col] - self.EARTH_RADIUS_REF) / sigma) ** 2
        ).clip(0.0, 1.0)

    def _add_orbital_zone_flag(self) -> None:
        """
        Boolean flag: is this planet inside the classical habitable zone?
        Based on stellar flux thresholds from config.
        """
        flux_col  = COLUMNS["stellar_flux"]
        min_flux  = HABITABILITY_THRESHOLDS["min_stellar_flux"]
        max_flux  = HABITABILITY_THRESHOLDS["max_stellar_flux"]

        self._df["in_habitable_zone"] = (
            (self._df[flux_col] >= min_flux) &
            (self._df[flux_col] <= max_flux)
        ).astype(int)

    def _add_density_proxy(self) -> None:
        """
        Rough density proxy using mass and radius.
        Density ∝ mass / radius³  (no actual units — used as a relative metric)
        Helps distinguish rocky planets (high density) from gas giants (low density).
        """
        mass_col   = COLUMNS["planet_mass"]
        radius_col = COLUMNS["planet_radius"]

        radius_cubed = self._df[radius_col] ** 3
        radius_cubed = radius_cubed.replace(0, np.nan)

        self._df["density_proxy"] = (self._df[mass_col] / radius_cubed).fillna(0.0)

    # ─── Label Generation ─────────────────────────────────────────────────────

    def _add_habitable_label(self) -> None:
        """
        Derives binary habitable label using scientific thresholds from config.
        A planet is labelled habitable only if ALL conditions are met.
        """
        t     = HABITABILITY_THRESHOLDS
        r_col = COLUMNS["planet_radius"]
        t_col = COLUMNS["temperature"]
        f_col = COLUMNS["stellar_flux"]

        is_habitable = (
            (self._df[t_col] >= t["min_temperature"])   &
            (self._df[t_col] <= t["max_temperature"])   &
            (self._df[f_col] >= t["min_stellar_flux"])  &
            (self._df[f_col] <= t["max_stellar_flux"])  &
            (self._df[r_col] >= t["min_radius"])        &
            (self._df[r_col] <= t["max_radius"])
        )

        self._df["habitable"] = is_habitable.astype(int)

    def _add_planet_type_label(self) -> None:
        """
        Classifies each planet into one of four categories based on radius:
            Rocky       →  radius < 1.5 R_earth
            Earth-like  →  1.5 ≤ radius < 2.0 R_earth
            Ice Giant   →  2.0 ≤ radius < 6.0 R_earth
            Gas Giant   →  radius ≥ 6.0 R_earth
        """
        thresholds = PLANET_TYPE_THRESHOLDS
        radius_col = COLUMNS["planet_radius"]
        types      = LABELS["planet_types"]    # ["Rocky", "Gas Giant", "Ice Giant", "Earth-like"]

        def classify(radius: float) -> str:
            if radius < thresholds["rocky_max_radius"]:
                return "Rocky"
            elif radius < thresholds["earth_like_max_radius"]:
                return "Earth-like"
            elif radius < thresholds["ice_giant_max_radius"]:
                return "Ice Giant"
            else:
                return "Gas Giant"

        self._df["planet_type"] = self._df[radius_col].apply(classify)

        unexpected = set(self._df["planet_type"].unique()) - set(types)
        if unexpected:
            raise ValueError(f"Unexpected planet types generated: {unexpected}")