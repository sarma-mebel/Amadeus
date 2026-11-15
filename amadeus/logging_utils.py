"""Logging helpers for Amadeus."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .config import Config


def setup_logging(config: Config) -> None:
    """Configure Python logging according to the configuration."""

    level = getattr(logging, config.logging.level.upper(), logging.INFO)
    handlers = [logging.StreamHandler()]

    if config.logging.logfile:
        logfile = Path(config.logging.logfile)
        logfile.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(logfile, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
    )


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a module-level logger."""

    return logging.getLogger(name or "amadeus")
