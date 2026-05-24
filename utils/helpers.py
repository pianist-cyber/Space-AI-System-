import functools
import os
import time
from datetime import datetime
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)


# ─── Directory & File Helpers ─────────────────────────────────────────────────

def ensure_dir(path: str) -> str:
    """
    Creates the directory for a given file path or folder path if it doesn't exist.
    Returns the directory path.

    Usage:
        ensure_dir(PATHS["cleaned_data"])   # creates data/processed/
        ensure_dir(PATHS["logs_dir"])        # creates logs/
    """
    dir_path = path if os.path.splitext(path)[1] == "" else os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
        logger.debug(f"Directory ensured: {dir_path}")
    return dir_path


def file_exists(path: str) -> bool:
    """Returns True if a file exists at the given path, False otherwise."""
    exists = os.path.isfile(path)
    if not exists:
        logger.warning(f"File not found: {path}")
    return exists


def get_file_size_mb(path: str) -> float:
    """Returns the size of a file in megabytes."""
    if not file_exists(path):
        return 0.0
    size = os.path.getsize(path) / (1024 * 1024)
    logger.debug(f"File size of '{path}': {size:.2f} MB")
    return round(size, 2)


# ─── CSV Helpers ──────────────────────────────────────────────────────────────

def load_csv(path: str, comment: str = "#", **kwargs) -> pd.DataFrame:
    """
    Safely loads a CSV file into a DataFrame.
    Skips comment lines (e.g. NASA dataset headers starting with #).
    Raises FileNotFoundError with a clean message if path doesn't exist.
    """
    if not file_exists(path):
        raise FileNotFoundError(
            f"CSV not found at: {path}\n"
            f"Make sure the dataset is downloaded and placed in data/raw/"
        )
    try:
        df = pd.read_csv(path, comment=comment, **kwargs)
        logger.info(f"Loaded CSV: {path} → {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as exc:
        logger.error(f"Failed to load CSV at '{path}': {exc}")
        raise


def save_csv(df: pd.DataFrame, path: str, index: bool = False) -> None:
    """
    Saves a DataFrame to a CSV file.
    Automatically creates parent directories if they don't exist.
    """
    try:
        ensure_dir(path)
        df.to_csv(path, index=index)
        logger.info(f"Saved CSV: {path} → {df.shape[0]} rows, {df.shape[1]} columns")
    except Exception as exc:
        logger.error(f"Failed to save CSV to '{path}': {exc}")
        raise


# ─── DataFrame Validation ─────────────────────────────────────────────────────

def validate_dataframe(
    df      : pd.DataFrame,
    required: list[str],
    label   : str = "DataFrame",
) -> None:
    """
    Validates that a DataFrame contains all required columns.
    Raises ValueError with a clear message listing the missing ones.

    Usage:
        validate_dataframe(df, FEATURES["habitability"], label="Habitability Input")
    """
    if df is None or df.empty:
        raise ValueError(f"{label} is None or empty.")

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"{label} is missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )
    logger.debug(f"{label} passed validation. Rows: {len(df)}")


def check_no_nulls(df: pd.DataFrame, cols: list[str], label: str = "") -> bool:
    """
    Checks if any of the specified columns contain null values.
    Logs a warning per column if nulls are found. Returns True if clean.
    """
    clean  = True
    prefix = f"[{label}] " if label else ""
    for col in cols:
        if col not in df.columns:
            logger.warning(f"{prefix}Column '{col}' not found in DataFrame.")
            clean = False
            continue
        null_count = df[col].isnull().sum()
        if null_count > 0:
            logger.warning(f"{prefix}Column '{col}' has {null_count} null values.")
            clean = False
    return clean


def safe_drop_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Drops columns from a DataFrame only if they exist.
    Silently skips columns that are not present.
    """
    existing = [c for c in cols if c in df.columns]
    skipped  = [c for c in cols if c not in df.columns]
    if skipped:
        logger.debug(f"Skipped dropping non-existent columns: {skipped}")
    return df.drop(columns=existing)


# ─── Timestamp & Naming Helpers ───────────────────────────────────────────────

def get_timestamp(fmt: str = "%Y%m%d_%H%M%S") -> str:
    """
    Returns the current datetime as a formatted string.
    Useful for naming output files uniquely.

    Usage:
        filename = f"prediction_results_{get_timestamp()}.csv"
        → "prediction_results_20250114_103201.csv"
    """
    return datetime.now().strftime(fmt)


def make_versioned_path(base_path: str) -> str:
    """
    Appends a timestamp to a file path to avoid overwriting existing files.

    Usage:
        make_versioned_path("data/outputs/prediction_results.csv")
        → "data/outputs/prediction_results_20250114_103201.csv"
    """
    root, ext      = os.path.splitext(base_path)
    versioned_path = f"{root}_{get_timestamp()}{ext}"
    logger.debug(f"Versioned path: {versioned_path}")
    return versioned_path


# ─── Numeric & Array Helpers ──────────────────────────────────────────────────

def safe_divide(
    numerator  : Any,
    denominator: Any,
    fallback   : float = 0.0,
) -> Any:
    """
    Divides numerator by denominator safely.
    Returns fallback value where denominator is zero instead of raising ZeroDivisionError.
    Works with scalars and numpy arrays.
    """
    try:
        if isinstance(denominator, (pd.Series, np.ndarray)):
            result = np.where(denominator == 0, fallback, numerator / denominator)
            return result
        return fallback if denominator == 0 else numerator / denominator
    except Exception as exc:
        logger.error(f"safe_divide failed: {exc}")
        return fallback


def clip_values(
    series : pd.Series,
    low    : float,
    high   : float,
    label  : str = "",
) -> pd.Series:
    """
    Clips a pandas Series to [low, high] and logs how many values were clipped.
    """
    out_of_range = ((series < low) | (series > high)).sum()
    if out_of_range > 0:
        prefix = f"[{label}] " if label else ""
        logger.warning(f"{prefix}Clipped {out_of_range} values to range [{low}, {high}].")
    return series.clip(lower=low, upper=high)


# ─── Performance Decorator ────────────────────────────────────────────────────

def timer(func: Callable) -> Callable:
    """
    Decorator that logs how long a function takes to execute.

    Usage:
        @timer
        def train_model():
            ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start  = time.perf_counter()
        result = func(*args, **kwargs)
        end    = time.perf_counter()
        elapsed = end - start
        logger.info(f"{func.__qualname__}() completed in {elapsed:.3f}s")
        return result
    return wrapper


# ─── Label & Class Helpers ────────────────────────────────────────────────────

def class_distribution(series: pd.Series, label: str = "") -> dict:
    """
    Returns and logs the value count distribution of a label column.
    Useful for checking class imbalance before training.

    Usage:
        class_distribution(df["habitable"], label="Habitability")
        class_distribution(df["planet_type"], label="Planet Type")
    """
    counts = series.value_counts().to_dict()
    total  = len(series)
    prefix = f"[{label}] " if label else ""
    logger.info(f"{prefix}Class distribution (total={total}):")
    for cls, count in counts.items():
        pct = (count / total) * 100
        logger.info(f"{prefix}  {cls}: {count} ({pct:.1f}%)")
    return counts


def check_class_imbalance(
    series      : pd.Series,
    threshold   : float = 0.15,
    label       : str   = "",
) -> bool:
    """
    Warns if any class represents less than `threshold` fraction of the data.
    Returns True if balanced, False if imbalance is detected.
    """
    counts  = series.value_counts(normalize=True)
    prefix  = f"[{label}] " if label else ""
    balanced = True
    for cls, frac in counts.items():
        if frac < threshold:
            logger.warning(
                f"{prefix}Class imbalance detected — '{cls}' is only "
                f"{frac*100:.1f}% of the data (threshold: {threshold*100:.1f}%)"
            )
            balanced = False
    return balanced