import os
import pickle
from typing import Optional
from typing import cast
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier

from utils.config import FEATURES, KNN, LABELS, PATHS
from utils.helpers import (
    check_class_imbalance,
    class_distribution,
    timer,
    validate_dataframe,
)
from utils.logger import get_logger, log_section

logger = get_logger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

TARGET_COL   = "planet_type"
MODEL_SAVE   = os.path.join(PATHS["models_dir"], "planet_type_knn.pkl")
FEATURE_COLS = FEATURES["planet_type"]
PLANET_TYPES = LABELS["planet_types"]   # ["Rocky", "Gas Giant", "Ice Giant", "Earth-like"]


class PlanetTypeKNN:
    """
    KNN classifier that predicts the structural type of a planet.

    Labels:
        Rocky       → small, dense, solid surface  (radius < 1.5 R_earth)
        Earth-like  → moderate size, rocky-ish     (1.5 – 2.0 R_earth)
        Ice Giant   → medium, icy composition      (2.0 – 6.0 R_earth)
        Gas Giant   → large, gaseous               (radius > 6.0 R_earth)

    Typical usage:
        model = PlanetTypeKNN()
        model.train(df)
        predictions = model.predict(df)
        report      = model.evaluate()
    """

    def __init__(self) -> None:
        self._k           : int                            = KNN["planet_type_k"]
        self._metric      : str                            = KNN["metric"]
        self._weights     : Literal["uniform", "distance"] = KNN["weights"]
        self._test_size   : float                          = KNN["test_size"]
        self._cv_folds    : int                            = KNN["cross_validation_folds"]
        self._random_state: int                            = KNN["random_state"]
        self._model       : Optional[KNeighborsClassifier] = None
        self._is_trained  : bool                           = False
        self._X_train     : Optional[np.ndarray]           = None
        self._X_test      : Optional[np.ndarray]           = None
        self._y_train     : Optional[np.ndarray]           = None
        self._y_test      : Optional[np.ndarray]           = None

        logger.info(
            f"PlanetTypeKNN initialised — "
            f"K={self._k}, metric={self._metric}, weights={self._weights}"
        )

    # ─── Public Interface ─────────────────────────────────────────────────────

    @timer
    def train(self, df: pd.DataFrame) -> "PlanetTypeKNN":
        """
        Trains the KNN classifier on the provided DataFrame.
        Stratifies the split to ensure all 4 planet types appear in train/test.
        """
        log_section(logger, "Planet Type KNN — Training")
        self._validate(df)

        X = df[FEATURE_COLS].to_numpy()
        y = df[TARGET_COL].to_numpy()

        self._check_labels(y)

        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(
            X, y,
            test_size    = self._test_size,
            random_state = self._random_state,
            stratify     = y,
        )

        assert self._X_train is not None and self._y_train is not None, "Training split failed."
        assert self._X_test is not None and self._y_test is not None, "Test split failed."

        logger.info(f"Train size : {len(self._X_train)} samples")
        logger.info(f"Test  size : {len(self._X_test)} samples")
        logger.info(f"Features   : {FEATURE_COLS}")

        self._model = KNeighborsClassifier(
            n_neighbors = self._k,
            metric      = self._metric,
            weights     = self._weights,
        )
        self._model.fit(self._X_train, self._y_train)
        self._is_trained = True

        train_acc = accuracy_score(self._y_train, self._model.predict(self._X_train))
        logger.info(f"Training accuracy : {train_acc * 100:.2f}%")

        return self

    @timer
    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Predicts the planet type for each row in the DataFrame.
        Returns a pandas Series of strings: Rocky / Gas Giant / Ice Giant / Earth-like
        """
        self._check_trained()
        self._validate(df, require_target=False)

        assert self._model is not None, "Model must be trained before predicting."
        X           = df[FEATURE_COLS].to_numpy()
        predictions = self._model.predict(X)

        logger.info("Planet type prediction breakdown:")
        counts = pd.Series(predictions).value_counts()
        for ptype, count in counts.items():
            logger.info(f"  {ptype:<15} → {count} planets")

        return pd.Series(predictions, index=df.index, name="planet_type_predicted")

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns probability scores for all 4 planet type classes.
        Columns match the order from self._model.classes_
        Useful for confidence-based ranking or visualization.
        """
        self._check_trained()
        self._validate(df, require_target=False)

        X      = df[FEATURE_COLS].values
        assert self._model is not None, "Model must be trained before predicting probabilities."
        probas = self._model.predict_proba(X)
        cols   = [f"prob_{cls.lower().replace(' ', '_')}" for cls in self._model.classes_]

        return pd.DataFrame(probas, index=df.index, columns=cols)

    @timer
    def evaluate(self, df: Optional[pd.DataFrame] = None) -> dict:
        """
        Evaluates the model on the internal test split (default) or a given DataFrame.
        Uses weighted averaging for multi-class metrics to handle class imbalance.
        Returns accuracy, precision, recall, weighted F1, and confusion matrix.
        """
        self._check_trained()
        log_section(logger, "Planet Type KNN — Evaluation")

        if df is not None:
            self._validate(df)
            X_eval = df[FEATURE_COLS].values
            y_eval = df[TARGET_COL].values
        else:
            X_eval = self._X_test
            y_eval = self._y_test.ravel() if self._y_test is not None else None

        assert self._model is not None, "Model must be trained before evaluation"
        assert X_eval is not None, "Test features (X_eval) cannot be None"
        assert y_eval is not None, "Test labels (y_eval) cannot be None "

        X_eval = cast(np.ndarray, X_eval)
        y_eval = cast(np.ndarray, y_eval)

        y_pred = self._model.predict(X_eval)

        # weighted → accounts for class imbalance across 4 categories
        acc       = accuracy_score(y_eval, y_pred)
        precision = precision_score(y_eval, y_pred, average="weighted", zero_division=0)
        recall    = recall_score(y_eval, y_pred, average="weighted", zero_division=0)
        f1        = f1_score(y_eval, y_pred, average="weighted", zero_division=0)
        cm        = confusion_matrix(y_eval, y_pred, labels=PLANET_TYPES)
        report    = classification_report(
            y_eval, y_pred,
            labels       = PLANET_TYPES,
            target_names = PLANET_TYPES,
            zero_division = 0,
        )

        logger.info(f"Accuracy           : {acc * 100:.2f}%")
        logger.info(f"Precision (weighted): {precision * 100:.2f}%")
        logger.info(f"Recall    (weighted): {recall * 100:.2f}%")
        logger.info(f"F1        (weighted): {f1 * 100:.2f}%")
        logger.info(f"Confusion Matrix:\n{cm}")
        logger.info(f"Full Report:\n{report}")

        return {
            "accuracy"        : round(acc, 4),
            "precision"       : np.round(precision, 4),
            "recall"          : np.round(recall, 4),
            "f1_score"        : np.round(f1, 4),
            "confusion_matrix": cm.tolist(),
            "labels"          : PLANET_TYPES,
            "report"          : report,
        }

    @timer
    def find_best_k(
        self,
        df     : pd.DataFrame,
        k_range: range = range(1, 21),
    ) -> int:
        """
        Tests every K in k_range using cross-validation with weighted F1.
        Logs per-K scores and returns the best K found.

        Tip: Run this inside knn_experiments.ipynb, then update config.py
        """
        log_section(logger, "Planet Type KNN — Finding Best K")
        self._validate(df)

        X_raw = df[FEATURE_COLS].values
        y_raw = df[TARGET_COL].values

        X = cast(np.ndarray, X_raw)
        y = cast(np.ndarray, y_raw)

        best_k      = self._k
        best_score  = 0.0
        results     = {}

        for k in k_range:
            knn = KNeighborsClassifier(
                n_neighbors = k,
                metric      = self._metric,
                weights     = self._weights,
            )
            scores     = cross_val_score(
                knn, X, y,
                cv      = self._cv_folds,
                scoring = "f1_weighted",
            )
            mean_score = scores.mean()
            logger.info(f"  K={k:>2}  →  Weighted F1={mean_score * 100:.2f}%")

            if mean_score > best_score:
                best_score = mean_score
                best_k     = k

        logger.info(f"Best K = {best_k} with Weighted F1 = {best_score * 100:.2f}%")
        logger.info(f"→ Update KNN['planet_type_k'] = {best_k} in config.py")

        return best_k

    def get_neighbours(
        self,
        df          : pd.DataFrame,
        n_neighbours: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns distances and indices of K nearest neighbours for each planet.
        Useful for explaining predictions: which known planets is this similar to?
        """
        self._check_trained()
        self._validate(df, require_target=False)

        k              = n_neighbours or self._k
        X              = df[FEATURE_COLS].values

        assert self._model is not None, "Model must be trained before finding neighbours."
        X = cast(np.ndarray, X)

        distances, idx = self._model.kneighbors(X, n_neighbors=k)

        logger.info(f"Neighbours computed for {len(df)} planets (K={k})")
        return distances, idx

    def get_per_class_accuracy(self, df: Optional[pd.DataFrame] = None) -> dict:
        """
        Returns individual accuracy per planet type.
        Useful for spotting which category the model struggles with most.

        Example output:
            { "Rocky": 0.91, "Gas Giant": 0.88, "Ice Giant": 0.74, "Earth-like": 0.62 }
        """
        self._check_trained()

        if df is not None:
            self._validate(df)
            X_raw = df[FEATURE_COLS].values
            y_raw = df[TARGET_COL].values
        else:
            X_raw = self._X_test
            y_raw = self._y_test
    
        assert self._model is not None, "Model must be trained before computing per-class accuracy."
        assert X_raw is not None, "Evaluation features (X_eval) cannot be None."
        assert y_raw is not None, "Evaluation labels (y_eval) cannot be None."
    
        X_eval = cast(np.ndarray, X_raw)
        y_eval = cast(np.ndarray, y_raw)
    
        y_pred = self._model.predict(X_eval)
        per_class = {}

        for ptype in PLANET_TYPES:
            mask = y_eval == ptype
            if mask.sum() == 0:
                per_class[ptype] = None
                continue
            acc              = accuracy_score(y_eval[mask], y_pred[mask])
            per_class[ptype] = round(acc, 4)
            logger.info(f"  {ptype:<15} accuracy: {acc * 100:.2f}%")

        return per_class

    def get_feature_importance_proxy(self, df: pd.DataFrame) -> dict:
        """
        Estimates feature importance using permutation — shuffles one feature
        at a time and measures the weighted F1 drop.
        Higher drop = more important that feature is for planet type prediction.
        """
        self._check_trained()
        self._validate(df)

        assert self._model is not None, "Model must be trained before calculating feature importance."

        X_raw    = df[FEATURE_COLS].values
        y_raw    = df[TARGET_COL].values

        X = cast(np.ndarray, X_raw)
        y = cast(np.ndarray, y_raw)
        base = float(f1_score(y, self._model.predict(X), average="weighted", zero_division=0))

        importance = {}
        for i, col in enumerate(FEATURE_COLS):
            X_shuffled       = X.copy()
            X_shuffled[:, i] = np.random.permutation(X_shuffled[:, i])
            shuffled_f1      = float(f1_score(
                y, self._model.predict(X_shuffled),
                average="weighted", zero_division=0
            ))
            drop             = base - shuffled_f1
            importance[col]  = round(drop, 4)
            logger.info(f"  {col:<30} → F1 drop: {drop * 100:.2f}%")

        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    def save_model(self, path: str = MODEL_SAVE) -> None:
        """Saves the trained model to disk using pickle."""
        self._check_trained()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._model, f)
        logger.info(f"Model saved to: {path}")

    def load_model(self, path: str = MODEL_SAVE) -> "PlanetTypeKNN":
        """Loads a previously saved model from disk."""
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"No saved model found at: {path}\n"
                f"Train the model first using .train(df)"
            )
        with open(path, "rb") as f:
            self._model = pickle.load(f)
        self._is_trained = True
        logger.info(f"Model loaded from: {path}")
        return self

    def summary(self) -> dict:
        """Returns a summary of the model's current configuration."""
        return {
            "model"       : "PlanetTypeKNN",
            "k"           : self._k,
            "metric"      : self._metric,
            "weights"     : self._weights,
            "features"    : FEATURE_COLS,
            "target"      : TARGET_COL,
            "planet_types": PLANET_TYPES,
            "is_trained"  : self._is_trained,
            "test_size"   : self._test_size,
            "cv_folds"    : self._cv_folds,
        }

    # ─── Private Helpers ──────────────────────────────────────────────────────

    def _validate(self, df: pd.DataFrame, require_target: bool = True) -> None:
        required = FEATURE_COLS + ([TARGET_COL] if require_target else [])
        validate_dataframe(df, required, label="PlanetTypeKNN")

    def _check_trained(self) -> None:
        if not self._is_trained or self._model is None:
            raise RuntimeError(
                "Model has not been trained yet. "
                "Call .train(df) before predict() or evaluate()."
            )

    def _check_labels(self, y: np.ndarray) -> None:
        unique   = set(np.unique(y))
        expected = set(PLANET_TYPES)
        unexpected = unique - expected
        if unexpected:
            raise ValueError(
                f"Unexpected planet types found in '{TARGET_COL}': {unexpected}\n"
                f"Expected values: {expected}\n"
                f"Make sure FeatureEngineer has been run first."
            )
        class_distribution(
            pd.Series(y, name=TARGET_COL),
            label="Planet Type Labels"
        )
        check_class_imbalance(
            pd.Series(y),
            threshold = 0.05,
            label     = "Planet Type"
        )