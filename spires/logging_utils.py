"""Lightweight structured logging helpers for SPIRES workflows."""

from __future__ import annotations

from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any


_LOG_FIELD_PRIORITY = (
    "event",
    "stage",
    "event_type",
    "status",
    "scene_name",
    "input_path",
    "source_type",
    "product",
    "platform",
    "tile",
    "scenes_requested",
    "scenes_prepared",
    "time_count",
    "requested_time_coverage_start",
    "requested_time_coverage_end",
    "time_coverage_start",
    "time_coverage_end",
    "selected_bands",
    "bands_500m",
    "bands_1km",
    "band_selection_source",
    "lut_name",
    "lut_file",
    "output_shape",
    "elapsed_seconds",
)

_VISUAL_LOG_PREFIX_BY_EVENT_TYPE = {
    "start": "====== START ======",
    "summary": "====== SUMMARY ======",
}


def _serialize_log_value(value: Any) -> str:
    """Serialize a log field into a stable plain-text representation."""
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
        return json.dumps(",".join(value))
    return json.dumps(value, sort_keys=True)


def format_log_event(event: str, **fields: Any) -> str:
    """Format a structured log event as a single plain-text line."""
    ordered_keys = [key for key in _LOG_FIELD_PRIORITY if key in fields]
    ordered_keys.extend(sorted(key for key in fields if key not in _LOG_FIELD_PRIORITY))
    parts = []
    visual_prefix = _VISUAL_LOG_PREFIX_BY_EVENT_TYPE.get(fields.get("event_type"))
    if visual_prefix:
        parts.append(visual_prefix)
    parts.append(f"event={_serialize_log_value(event)}")
    for key in ordered_keys:
        parts.append(f"{key}={_serialize_log_value(fields[key])}")
    return " ".join(parts)


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, **fields: Any) -> None:
    """Emit a structured event on the provided logger."""
    logger.log(level, format_log_event(event, **fields))


class _SPIRESLogFormatter(logging.Formatter):
    """Formatter that adds a bare separator line before highlighted records."""

    def format(self, record: logging.LogRecord) -> str:
        formatted = super().format(record)
        visual_prefix = next(
            (prefix for prefix in _VISUAL_LOG_PREFIX_BY_EVENT_TYPE.values() if record.getMessage().startswith(prefix)),
            None,
        )
        if visual_prefix is None:
            return formatted

        timestamp = self.formatTime(record, self.datefmt)
        prefix_width = len(f"{timestamp} {record.levelname} {record.name}")
        separator = "=" * prefix_width
        return f"{separator}\n{formatted}"


def configure_spires_file_logger(
    log_path: str | Path,
    *,
    logger_name: str = "spires",
    level: int = logging.INFO,
    log_to_stdout: bool = True,
    mode: str = "w",
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

    formatter = _SPIRESLogFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")

    logger.handlers.clear()

    file_handler = logging.FileHandler(resolved_log_path, mode=mode)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if log_to_stdout:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


def make_spires_log_path(
    log_dir: str | Path,
    *,
    prefix: str = "spires",
    tile: str | None = None,
    sensor: str | None = None,
    label: str | None = None,
    extension: str = ".log",
    timestamp: str | None = None,
) -> Path:
    """Create a timestamped per-run log path for notebook or batch workflows."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    parts = [prefix]
    if sensor:
        parts.append(sensor)
    if tile:
        parts.append(tile)
    if label:
        parts.append(label)
    parts.append(timestamp)

    log_dir = Path(log_dir).expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / ("_".join(parts) + extension)
