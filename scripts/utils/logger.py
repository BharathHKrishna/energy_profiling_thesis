import logging
import os

_LEVEL = getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO)

_FMT = logging.Formatter("%(asctime)s | %(name)-24s | %(levelname)s | %(message)s")


def _console_handler():
    h = logging.StreamHandler()
    h.setLevel(_LEVEL)
    h.setFormatter(_FMT)
    return h


def get_logger(name: str) -> logging.Logger:
    """Console-only (removed the RotatingFileHandler to logs/pipeline.log
    2026-08-27): that file was never process-safe -- run_pipeline.py's
    ProcessPoolExecutor spawns 16 worker PROCESSES, each independently
    opening its own RotatingFileHandler on the SAME shared path, so once the
    file hit its 10MB rotation threshold, two processes could race to
    rename it at the same instant and the loser threw a real, live
    FileNotFoundError ("--- Logging error ---" in the log, found during the
    real 10k run). Confirmed nothing in the codebase ever read pipeline.log
    or web.log back -- both were write-only and fully redundant with the
    console handler already captured by run_pipeline.py's own
    `> logs/run_10k_*.log` shell redirect, which every process's stdout
    correctly inherits. Removing the file handler removes the race
    entirely, with zero information loss."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(_LEVEL)
    logger.addHandler(_console_handler())
    return logger
