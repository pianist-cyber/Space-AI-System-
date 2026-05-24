import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional

from utils.config import LOGGING, PATHS


# ─── Internal Registry ────────────────────────────────────────────────────────
# Keeps one logger per name so we never create duplicate handlers
_logger_registry: dict[str, logging.Logger] = {}


# ─── ANSI Color Codes (console only) ─────────────────────────────────────────
class _Colors:
    RESET   = "\033[0m"
    GREY    = "\033[38;5;240m"
    BLUE    = "\033[38;5;39m"
    YELLOW  = "\033[38;5;220m"
    RED     = "\033[38;5;196m"
    BOLD    = "\033[1m"


class _ColorFormatter(logging.Formatter):
    """Applies color to console output based on log level."""

    LEVEL_COLORS = {
        logging.DEBUG   : _Colors.GREY,
        logging.INFO    : _Colors.BLUE,
        logging.WARNING : _Colors.YELLOW,
        logging.ERROR   : _Colors.RED,
        logging.CRITICAL: _Colors.BOLD + _Colors.RED,
    }

    def format(self, record: logging.LogRecord) -> str:
        color        = self.LEVEL_COLORS.get(record.levelno, _Colors.RESET)
        record.msg   = f"{color}{record.msg}{_Colors.RESET}"
        return super().format(record)


# ─── Public Factory ───────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """
    Returns a named logger with:
      - colored console output
      - rotating file output  (max 5 MB per file, keeps last 3 files)

    Usage:
        from utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("DataCleaner started")
    """
    if name in _logger_registry:
        return _logger_registry[name]

    logger = logging.getLogger(name)

    # Prevent log messages from bubbling up to root logger
    logger.propagate = False

    level_str = LOGGING.get("level", "INFO").upper()
    level     = getattr(logging, level_str, logging.INFO)
    logger.setLevel(level)

    fmt = LOGGING.get(
        "format",
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    date_fmt = "%Y-%m-%d %H:%M:%S"

    # ── Console Handler ───────────────────────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(_ColorFormatter(fmt=fmt, datefmt=date_fmt))
    logger.addHandler(console_handler)

    # ── File Handler ──────────────────────────────────────────────────────────
    log_path = LOGGING.get("logfile", os.path.join(PATHS["logs_dir"], "space_ai.log"))
    _ensure_log_dir(log_path)

    file_handler = RotatingFileHandler(
        filename    = log_path,
        maxBytes    = 5 * 1024 * 1024,   # 5 MB
        backupCount = 3,
        encoding    = "utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(
        logging.Formatter(fmt=fmt, datefmt=date_fmt)
    )
    logger.addHandler(file_handler)

    _logger_registry[name] = logger
    return logger


# ─── Convenience Helpers ──────────────────────────────────────────────────────

def log_section(logger: logging.Logger, title: str) -> None:
    """Prints a visible section divider in the logs."""
    border = "─" * 60
    logger.info(border)
    logger.info(f"  {title.upper()}")
    logger.info(border)


def log_dataframe_info(logger: logging.Logger, df, label: str = "") -> None:
    """Logs shape, column count, and missing value summary of a DataFrame."""
    try:
        prefix = f"[{label}] " if label else ""
        logger.info(f"{prefix}Shape        : {df.shape[0]} rows × {df.shape[1]} columns")
        logger.info(f"{prefix}Columns      : {list(df.columns)}")
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if missing.empty:
            logger.info(f"{prefix}Missing vals : None")
        else:
            logger.warning(f"{prefix}Missing vals : {missing.to_dict()}")
    except Exception as exc:
        logger.error(f"log_dataframe_info failed: {exc}")


def log_dict(logger: logging.Logger, data: dict, label: str = "") -> None:
    """Logs every key-value pair in a dictionary."""
    prefix = f"[{label}] " if label else ""
    for key, val in data.items():
        logger.info(f"{prefix}{key}: {val}")


def set_log_level(level: str) -> None:
    """
    Dynamically changes the log level for all registered loggers.
    Example: set_log_level("DEBUG")
    """
    numeric = getattr(logging, level.upper(), None)
    if numeric is None:
        raise ValueError(f"Invalid log level: '{level}'. Use DEBUG, INFO, WARNING, ERROR.")
    for lgr in _logger_registry.values():
        lgr.setLevel(numeric)
        for handler in lgr.handlers:
            handler.setLevel(numeric)


# ─── Private ──────────────────────────────────────────────────────────────────

def _ensure_log_dir(log_path: str) -> None:
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)