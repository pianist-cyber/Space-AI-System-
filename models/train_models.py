import os
from typing import Optional

import pandas as pd

from models.habitability_knn import HabitabilityKNN
from models.planet_type_knn import PlanetTypeKNN
from preprocessing.cleaner import DataCleaner
from preprocessing.feature_engineering import FeatureEngineer
from preprocessing.scaler import FeatureScaler
from utils.config import PATHS
from utils.helpers import (
    check_class_imbalance,
    class_distribution,
    save_csv,
    timer,
    validate_dataframe,
)
from utils.logger import get_logger, log_dict, log_section

logger = get_logger(__name__)


class ModelPipeline:
    """
    Orchestrates the full Space AI pipeline from raw CSV to final predictions.

    Two modes:
        TRAIN   → clean → engineer → scale → train both models → predict → save
        PREDICT → load cleaned data → load saved models → predict → save

    Typical usage:
        pipeline = ModelPipeline()
        pipeline.run()                          # full pipeline
        pipeline.run(predict_only=True)         # skip training, use saved models
    """

    def __init__(self) -> None:
        self._cleaner    : Optional[DataCleaner]      = None
        self._engineer   : Optional[FeatureEngineer]  = None
        self._scaler     : Optional[FeatureScaler]    = None
        self._hab_model  : Optional[HabitabilityKNN]  = None
        self._type_model : Optional[PlanetTypeKNN]    = None
        self._raw_df     : Optional[pd.DataFrame]     = None
        self._clean_df   : Optional[pd.DataFrame]     = None
        self._engineered_df: Optional[pd.DataFrame]   = None
        self._scaled_df  : Optional[pd.DataFrame]     = None
        self._results_df : Optional[pd.DataFrame]     = None
        self._eval_hab   : Optional[dict]             = None
        self._eval_type  : Optional[dict]             = None

        logger.info("ModelPipeline initialised")

    # ─── Public Interface ─────────────────────────────────────────────────────

    @timer
    def run(self, predict_only: bool = False) -> pd.DataFrame:
        """
        Runs the complete pipeline end to end.

        Args:
            predict_only: If True, skips training and loads saved models from disk.
                          Use this after the first successful training run.

        Returns:
            Final results DataFrame with all predictions merged.
        """
        log_section(logger, "Space AI System — Pipeline Start")

        self.run_preprocessing()

        if predict_only:
            self.load_trained_models()
        else:
            self.run_training()

        self.run_prediction()
        self.save_results()

        assert self._results_df is not None, "Results DataFrame should not be None at this point"

        log_section(logger, "Space AI System — Pipeline Complete")
        return self._results_df

    # ─── Stage 1 — Preprocessing ──────────────────────────────────────────────

    @timer
    def run_preprocessing(self) -> pd.DataFrame:
        """
        Runs all three preprocessing stages:
            DataCleaner → FeatureEngineer → FeatureScaler
        Returns the fully scaled and engineered DataFrame.
        """
        log_section(logger, "Stage 1 — Preprocessing")

        # ── Step 1: Clean ─────────────────────────────────────────────────────
        logger.info("Step 1/3 — Cleaning raw data...")
        self._cleaner  = DataCleaner()
        self._cleaner.load().clean().save()
        self._clean_df = self._cleaner.get_dataframe()

        clean_summary  = self._cleaner.summary()
        log_dict(logger, clean_summary, label="Cleaner Summary")

        # ── Step 2: Feature Engineering ───────────────────────────────────────
        logger.info("Step 2/3 — Engineering features...")
        self._engineer     = FeatureEngineer(self._clean_df)
        self._engineer.engineer()
        self._engineered_df = self._engineer.get_dataframe()

        eng_summary = self._engineer.feature_summary()
        log_dict(logger, eng_summary, label="Engineer Summary")

        # ── Step 3: Scaling ───────────────────────────────────────────────────
        logger.info("Step 3/3 — Scaling features...")
        self._scaler    = FeatureScaler()
        self._scaled_df = self._scaler.fit_transform(self._engineered_df)

        logger.info(
            f"Preprocessing complete — "
            f"{len(self._scaled_df)} planets ready for training"
        )

        return self._scaled_df

    # ─── Stage 2 — Training ───────────────────────────────────────────────────

    @timer
    def run_training(self) -> dict:
        """
        Trains both KNN models on the scaled DataFrame.
        Evaluates each model immediately after training.
        Returns a dictionary of evaluation metrics for both models.
        """
        log_section(logger, "Stage 2 — Training Models")
        self._check_preprocessing_done()

        # ── Habitability Model ────────────────────────────────────────────────
        logger.info("Training HabitabilityKNN...")
        self._hab_model  = HabitabilityKNN()
        self._hab_model.train(self._scaled_df)
        self._eval_hab   = self._hab_model.evaluate()
        self._hab_model.save_model()

        # ── Planet Type Model ─────────────────────────────────────────────────
        logger.info("Training PlanetTypeKNN...")
        self._type_model = PlanetTypeKNN()
        self._type_model.train(self._scaled_df)
        self._eval_type  = self._type_model.evaluate()
        self._type_model.save_model()

        logger.info("Both models trained and saved successfully")
        return self.get_evaluation_summary()

    # ─── Stage 3 — Load Saved Models ─────────────────────────────────────────

    def load_trained_models(self) -> "ModelPipeline":
        """
        Loads both previously saved KNN models from disk.
        Use this to skip retraining when models are already trained.
        """
        log_section(logger, "Stage 2 — Loading Saved Models")

        self._hab_model  = HabitabilityKNN()
        self._hab_model.load_model()

        self._type_model = PlanetTypeKNN()
        self._type_model.load_model()

        logger.info("Both models loaded from disk successfully")
        return self

    # ─── Stage 4 — Prediction ─────────────────────────────────────────────────

    @timer
    def run_prediction(self) -> pd.DataFrame:
        """
        Runs both models on the full scaled DataFrame.
        Merges all predictions, probabilities, and engineered features
        into a single unified results DataFrame.
        """
        log_section(logger, "Stage 3 — Running Predictions")
        self._check_preprocessing_done()
        self._check_models_ready()

        base_df = self._engineered_df.copy() if self._engineered_df is not None else pd.DataFrame()

        self._hab_model  = self._hab_model  if self._hab_model  else HabitabilityKNN()
        self._type_model = self._type_model if self._type_model else PlanetTypeKNN()
        assert self._hab_model is not None and self._type_model is not None, "Models must be loaded or trained before prediction"
        assert self._scaled_df is not None, "Scaled DataFrame must be available for prediction"
        
        logger.info("Predicting habitability...")
        hab_predictions = self._hab_model.predict(self._scaled_df)
        hab_probas      = self._hab_model.predict_proba(self._scaled_df)

        base_df["habitable_predicted"]    = hab_predictions.values
        base_df["prob_not_habitable"]     = hab_probas["prob_not_habitable"].values
        base_df["prob_habitable"]         = hab_probas["prob_habitable"].values

        # ── Planet Type Predictions ───────────────────────────────────────────
        logger.info("Predicting planet types...")
        type_predictions = self._type_model.predict(self._scaled_df)
        type_probas      = self._type_model.predict_proba(self._scaled_df)

        base_df["planet_type_predicted"]  = type_predictions.values
        for col in type_probas.columns:
            base_df[col] = type_probas[col].values

        self._results_df = base_df

        # ── Log Final Breakdown ───────────────────────────────────────────────
        self._log_prediction_summary()

        return self._results_df

    # ─── Stage 5 — Save Results ───────────────────────────────────────────────

    def save_results(self, path: Optional[str] = None) -> None:
        """
        Saves the final prediction results DataFrame to CSV.
        Uses PATHS["output"] from config.py by default.
        """
        if self._results_df is None:
            raise RuntimeError(
                "No results to save. Run run_prediction() first."
            )
        output_path = path or PATHS["output"]
        save_csv(self._results_df, output_path)
        logger.info(f"Results saved → {output_path}")
        logger.info(f"Total planets in output : {len(self._results_df)}")

    # ─── Evaluation & Summaries ───────────────────────────────────────────────

    def get_evaluation_summary(self) -> dict:
        """
        Returns evaluation metrics for both models in a single dictionary.
        """
        return {
            "habitability_model": self._eval_hab,
            "planet_type_model" : self._eval_type,
        }

    def get_pipeline_summary(self) -> dict:
        """
        Returns a complete summary of every pipeline stage.
        Useful for report_generator.py and final output.
        """
        summary = {
            "preprocessing": {
                "cleaner" : self._cleaner.summary()   if self._cleaner   else None,
                "engineer": self._engineer.feature_summary() if self._engineer else None,
                "scaler_habitability": self._scaler.summary("habitability") if self._scaler else None,
                "scaler_planet_type" : self._scaler.summary("planet_type")  if self._scaler else None,
            },
            "models": {
                "habitability": self._hab_model.summary()  if self._hab_model  else None,
                "planet_type" : self._type_model.summary() if self._type_model else None,
            },
            "evaluation": self.get_evaluation_summary(),
            "results": {
                "total_planets"         : len(self._results_df) if self._results_df is not None else None,
                "habitable_count"       : int(self._results_df["habitable_predicted"].sum()) if self._results_df is not None else None,
                "planet_type_breakdown" : self._results_df["planet_type_predicted"].value_counts().to_dict() if self._results_df is not None else None,
            },
        }
        log_section(logger, "Pipeline Summary")
        log_dict(logger, summary["results"], label="Results")
        return summary

    def get_results(self) -> pd.DataFrame:
        """Returns the final results DataFrame after prediction."""
        if self._results_df is None:
            raise RuntimeError(
                "No results available. Run run_prediction() first."
            )
        return self._results_df.copy()

    # ─── Private Helpers ──────────────────────────────────────────────────────

    def _check_preprocessing_done(self) -> None:
        if self._scaled_df is None:
            raise RuntimeError(
                "Preprocessing has not been run yet. "
                "Call run_preprocessing() first."
            )

    def _check_models_ready(self) -> None:
        if self._hab_model is None or self._type_model is None:
            raise RuntimeError(
                "Models are not ready. "
                "Call run_training() or load_trained_models() first."
            )

    def _log_prediction_summary(self) -> None:
        if self._results_df is None:
            return

        total     = len(self._results_df)
        hab_count = int(self._results_df["habitable_predicted"].sum())
        hab_pct   = (hab_count / total) * 100 if total > 0 else 0

        logger.info(f"Total planets analysed  : {total}")
        logger.info(f"Habitable planets found : {hab_count} ({hab_pct:.1f}%)")
        logger.info("Planet type breakdown   :")

        type_counts = self._results_df["planet_type_predicted"].value_counts()
        for ptype, count in type_counts.items():
            pct = (count / total) * 100
            logger.info(f"  {ptype:<15} : {count} ({pct:.1f}%)")


# ─── Module-Level Convenience Function ───────────────────────────────────────

def run_pipeline(predict_only: bool = False) -> pd.DataFrame:
    """
    Top-level convenience function called by main.py.

    Args:
        predict_only: Skip training and use saved models.

    Returns:
        Final results DataFrame.

    Usage:
        from models.train_models import run_pipeline
        results = run_pipeline()
        results = run_pipeline(predict_only=True)
    """
    pipeline = ModelPipeline()
    return pipeline.run(predict_only=predict_only)