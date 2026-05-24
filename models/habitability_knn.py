import os
import pickle
from typing import Optional
from typing import Literal
from typing import cast

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

TARGET_COL    = "habitable"
MODEL_SAVE    = os.path.join(PATHS["models_dir"], "habitability_knn.pkl")
FEATURE_COLS  = FEATURES["habitability"]


class HabitabilityKNN:
    """
    KNN classifier that predicts whether a planet is habitable or not.

    Labels:
        1 → Habitable
        0 → Not Habitable

    Typical usage:
        model = HabitabilityKNN()
        model.train(df)
        predictions = model.predict(df)
        report      = model.evaluate(df)
    """

    def __init__(self) -> None:
        self._k          : int                         = KNN["habitability_k"]
        self._metric     : str                         = KNN["metric"]
        self._weights    : Literal['uniform', 'distance'] = 'uniform'
        self._test_size  : float                       = KNN["test_size"]
        self._cv_folds   : int                         = KNN["cross_validation_folds"]
        self._random_state: int                        = KNN["random_state"]
        self._model      : Optional[KNeighborsClassifier] = None
        self._is_trained : bool                        = False
        self._X_train    : Optional[np.ndarray]        = None
        self._X_test     : Optional[np.ndarray]        = None
        self._y_train    : Optional[np.ndarray]        = None
        self._y_test     : Optional[np.ndarray]        = None

        logger.info(
            f"HabitabilityKNN initialised — "
            f"K={self._k}, metric={self._metric}, weights={self._weights}"
        )

    # ─── Public Interface ─────────────────────────────────────────────────────

    @timer
    def train(self, df: pd.DataFrame) -> "HabitabilityKNN":
        """
        Trains the KNN model on the provided DataFrame.
        Splits into train/test sets and logs class distribution.
        """
        log_section(logger, "Habitability KNN — Training")
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

        assert self._X_train is not None
        assert self._X_test is not None
        assert self._y_train is not None

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
        Predicts habitability for each planet in the DataFrame.
        Returns a pandas Series of 0s and 1s aligned with the input index.
        """
        self._check_trained()
        self._validate(df, require_target=False)

        X           = df[FEATURE_COLS].values
        assert self._model is not None, "Model must be trained before predicting."
        predictions = self._model.predict(X)

        habitable_count = int(predictions.sum())
        logger.info(
            f"Predicted {habitable_count} habitable / "
            f"{len(predictions) - habitable_count} not habitable "
            f"out of {len(predictions)} planets"
        )

        return pd.Series(predictions, index=df.index, name="habitable_predicted")

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns probability scores for each class instead of hard labels.
        Columns: ['prob_not_habitable', 'prob_habitable']
        Useful for ranking planets by habitability confidence.
        """
        self._check_trained()
        self._validate(df, require_target=False)

        X      = df[FEATURE_COLS].values
        assert self._model is not None, "Model must be trained before predicting probabilities."
        probas = self._model.predict_proba(X)

        return pd.DataFrame(
            probas,
            index   = df.index,
            columns = ["prob_not_habitable", "prob_habitable"],
        )

    @timer
    def evaluate(self, df: Optional[pd.DataFrame] = None) -> dict:
        """
        Evaluates the model on test data (default) or a provided DataFrame.
        Returns a dictionary with accuracy, precision, recall, F1, and confusion matrix.
        """
        self._check_trained()
        log_section(logger, "Habitability KNN — Evaluation")

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

        acc       = accuracy_score(y_eval, y_pred)
        precision = precision_score(y_eval, y_pred, zero_division=0)
        recall    = recall_score(y_eval, y_pred, zero_division=0)
        f1        = f1_score(y_eval, y_pred, zero_division=0)
        cm        = confusion_matrix(y_eval, y_pred)
        report    = classification_report(
            y_eval, y_pred,
            target_names = ["Not Habitable", "Habitable"],
            zero_division = 0,
        )

        logger.info(f"Accuracy  : {acc * 100:.2f}%")
        logger.info(f"Precision : {precision * 100:.2f}%")
        logger.info(f"Recall    : {recall * 100:.2f}%")
        logger.info(f"F1 Score  : {f1 * 100:.2f}%")
        logger.info(f"Confusion Matrix:\n{cm}")
        logger.info(f"Full Report:\n{report}")

        return {
            "accuracy"        : round(acc, 4),
            "precision"       : round(precision, 4),
            "recall"          : round(recall, 4),
            "f1_score"        : round(f1, 4),
            "confusion_matrix": cm.tolist(),
            "report"          : report,
        }

    @timer
    def find_best_k(
        self,
        df       : pd.DataFrame,
        k_range  : range = range(1, 21),
    ) -> int:
        """
        Tests every K in k_range using cross-validation.
        Logs accuracy for each K, returns the best K found.

        Tip: Run this in knn_experiments.ipynb, then update config.py
        """
        log_section(logger, "Habitability KNN — Finding Best K")
        self._validate(df)

        X = df[FEATURE_COLS].values
        y = df[TARGET_COL].values

        best_k      = self._k
        best_score  = 0.0
        results     = {}

        for k in k_range:
            knn    = KNeighborsClassifier(
                n_neighbors = k,
                metric      = self._metric,
                weights     = self._weights,
            )
            scores      = cross_val_score(knn, X, y, cv=self._cv_folds, scoring="f1")
            mean_score  = scores.mean()
            results[k]  = round(mean_score, 4)
            logger.info(f"  K={k:>2}  →  F1={mean_score * 100:.2f}%")

            if mean_score > best_score:
                best_score = mean_score
                best_k     = k

        logger.info(f"Best K = {best_k} with F1 = {best_score * 100:.2f}%")
        logger.info(f"→ Update KNN['habitability_k'] = {best_k} in config.py")

        return best_k

    def get_neighbours(
        self,
        df      : pd.DataFrame,
        n_neighbours: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the distances and indices of nearest neighbours for each planet.
        Useful for understanding which known planets are most similar to unknowns.
        """
        self._check_trained()
        self._validate(df, require_target=False)

        k = n_neighbours or self._k
        X = df[FEATURE_COLS].values

        distances, indices = self._model.kneighbors(X, n_neighbors=k)
        logger.info(f"Neighbours found for {len(df)} planets (K={k})")
        return distances, indices

    def get_feature_importance_proxy(self, df: pd.DataFrame) -> dict:
        """
        Estimates feature importance by measuring accuracy drop
        when each feature is individually shuffled (permutation method).
        Higher drop = more important feature.
        """
        self._check_trained()
        self._validate(df)

        X     = df[FEATURE_COLS].values
        y     = df[TARGET_COL].values
        base  = accuracy_score(y, self._model.predict(X))

        importance = {}
        for i, col in enumerate(FEATURE_COLS):
            X_shuffled        = X.copy()
            X_shuffled[:, i]  = np.random.permutation(X_shuffled[:, i])
            shuffled_acc      = accuracy_score(y, self._model.predict(X_shuffled))
            drop              = base - shuffled_acc
            importance[col]   = round(drop, 4)
            logger.info(f"  {col:<30} → accuracy drop: {drop * 100:.2f}%")

        sorted_importance = dict(
            sorted(importance.items(), key=lambda x: x[1], reverse=True)
        )
        return sorted_importance

    def save_model(self, path: str = MODEL_SAVE) -> None:
        """Saves the trained model to disk using pickle."""
        self._check_trained()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._model, f)
        logger.info(f"Model saved to: {path}")

    def load_model(self, path: str = MODEL_SAVE) -> "HabitabilityKNN":
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
            "model"       : "HabitabilityKNN",
            "k"           : self._k,
            "metric"      : self._metric,
            "weights"     : self._weights,
            "features"    : FEATURE_COLS,
            "target"      : TARGET_COL,
            "labels"      : LABELS["habitability"],
            "is_trained"  : self._is_trained,
            "test_size"   : self._test_size,
            "cv_folds"    : self._cv_folds,
        }

    # ─── Private Helpers ──────────────────────────────────────────────────────

    def _validate(self, df: pd.DataFrame, require_target: bool = True) -> None:
        required = FEATURE_COLS + ([TARGET_COL] if require_target else [])
        validate_dataframe(df, required, label="HabitabilityKNN")

    def _check_trained(self) -> None:
        if not self._is_trained or self._model is None:
            raise RuntimeError(
                "Model has not been trained yet. "
                "Call .train(df) before predict() or evaluate()."
            )

    def _check_labels(self, y: np.ndarray) -> None:
        unique = set(np.unique(y))
        expected = {0, 1}
        if not unique.issubset(expected):
            raise ValueError(
                f"Unexpected label values in '{TARGET_COL}': {unique}. "
                f"Expected: {expected}. "
                f"Make sure FeatureEngineer has been run first."
            )
        class_distribution(
            pd.Series(y, name=TARGET_COL),
            label="Habitability Labels"
        )
        check_class_imbalance(
            pd.Series(y),
            threshold = 0.10,
            label     = "Habitability"
        )