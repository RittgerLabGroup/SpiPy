"""Lightweight structured logging helpers for SPIRES workflows."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


def _serialize_log_value(value: Any) -> str:
    """Serialize a log field into a stable plain-text representation."""
    if isinstance(value, Path):
        value = str(value)
    return json.dumps(value, sort_keys=True)


def format_log_event(event: str, **fields: Any) -> str:
    """Format a structured log event as a single plain-text line."""
    parts = [f"event={_serialize_log_value(event)}"]
    for key in sorted(fields):
        parts.append(f"{key}={_serialize_log_value(fields[key])}")
    return " ".join(parts)


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, **fields: Any) -> None:
    """Emit a structured event on the provided logger."""
    logger.log(level, format_log_event(event, **fields))


def configure_spires_file_logger(
    log_path: str | Path,
    *,
    logger_name: str = "spires",
    level: int = logging.INFO,
    log_to_stdout: bool = True,
) -> logging.Logger:
    """
    Configure a plain-text SPIRES logger suitable for `.log` or `.txt` files.

    The logger writes timestamped lines to ``log_path`` and can optionally also
    stream the same messages to stdout so Slurm captures them.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False

    resolved_log_path = Path(log_path).expanduser().resolve()
    resolved_log_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")

    logger.handlers.clear()

    file_handler = logging.FileHandler(resolved_log_path)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if log_to_stdout:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger
