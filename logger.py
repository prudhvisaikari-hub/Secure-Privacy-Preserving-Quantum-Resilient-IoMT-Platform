"""
utils/logger.py
===============
Centralised logging configuration for SPQR-IoMT.
All modules import get_logger() from here to ensure consistent
formatting, file rotation, and level control across the project.

Usage:
    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Starting experiment...")
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

_LOG_DIR = Path("logs")
_LOG_DIR.mkdir(exist_ok=True)

_FMT      = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"
_ROOT_CONFIGURED = False


def configure_root(level: int = logging.INFO,
                   log_file: Optional[str] = "logs/spqr_iomt.log",
                   max_bytes: int = 5_000_000,
                   backup_count: int = 3):
    """
    Configure the root logger once. Subsequent calls are no-ops.
    Sets up:
      - StreamHandler  → stdout (coloured if rich is available)
      - RotatingFileHandler → logs/spqr_iomt.log
    """
    global _ROOT_CONFIGURED
    if _ROOT_CONFIGURED:
        return
    _ROOT_CONFIGURED = True

    root = logging.getLogger()
    root.setLevel(level)

    formatter = logging.Formatter(_FMT, datefmt=_DATE_FMT)

    # Console handler
    try:
        from rich.logging import RichHandler
        ch = RichHandler(rich_tracebacks=True, show_path=False)
        ch.setFormatter(logging.Formatter("%(message)s", datefmt="[%H:%M:%S]"))
    except ImportError:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
    ch.setLevel(level)
    root.addHandler(ch)

    # Rotating file handler
    if log_file:
        fh = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        fh.setFormatter(formatter)
        fh.setLevel(logging.DEBUG)  # file always captures DEBUG
        root.addHandler(fh)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a named logger. Configures the root logger on first call.

    Args:
        name:  Module name — use __name__ for automatic naming.
        level: Log level for this logger (default INFO).

    Returns:
        logging.Logger instance.
    """
    configure_root()
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


def set_global_level(level: int):
    """Change the log level for all SPQR-IoMT loggers at runtime."""
    logging.getLogger().setLevel(level)
    for name, lgr in logging.Logger.manager.loggerDict.items():
        if isinstance(lgr, logging.Logger):
            lgr.setLevel(level)


# Convenience aliases
DEBUG    = logging.DEBUG
INFO     = logging.INFO
WARNING  = logging.WARNING
ERROR    = logging.ERROR
CRITICAL = logging.CRITICAL


if __name__ == "__main__":
    log = get_logger("utils.logger.demo")
    log.debug("Debug message")
    log.info("Info message")
    log.warning("Warning message")
    log.error("Error message")
    print("Logger configured. Check logs/spqr_iomt.log")
