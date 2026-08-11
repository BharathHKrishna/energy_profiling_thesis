import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_DIR  = Path(os.environ.get("LOG_DIR", "logs"))
_LOG_FILE = _LOG_DIR / "pipeline.log"
_WEB_FILE = _LOG_DIR / "web.log"
_LEVEL    = getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO)

_FMT = logging.Formatter("%(asctime)s | %(name)-24s | %(levelname)s | %(message)s")


def _file_handler(path: Path, max_bytes: int = 10 * 1024 * 1024, backup_count: int = 5):
    path.parent.mkdir(parents=True, exist_ok=True)
    h = RotatingFileHandler(path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
    h.setLevel(_LEVEL)
    h.setFormatter(_FMT)
    return h


def _console_handler():
    h = logging.StreamHandler()
    h.setLevel(_LEVEL)
    h.setFormatter(_FMT)
    return h


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(_LEVEL)

    # Web app loggers → web.log; everything else → pipeline.log
    log_path = _WEB_FILE if name in ("geoenergy", "web") else _LOG_FILE

    logger.addHandler(_console_handler())
    logger.addHandler(_file_handler(log_path))
    return logger
