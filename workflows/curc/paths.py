"""Path conventions for CURC-specific SpiPy workflow inputs, outputs, and logs."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from workflows.curc.config import CurcWorkflowConfig


def input_platform_root(config: CurcWorkflowConfig, platform: str) -> Path:
    return config.scratch_root / "input" / config.sensor / platform


def reflectance_dir(config: CurcWorkflowConfig, platform: str) -> Path:
    return input_platform_root(config, platform) / "reflectance"


def ancillary_dir(config: CurcWorkflowConfig, platform: str) -> Path:
    return input_platform_root(config, platform) / "ancillary"


def r0_dir(config: CurcWorkflowConfig, platform: str) -> Path:
    return ancillary_dir(config, platform) / "r0"


def r0_dataset_path(config: CurcWorkflowConfig, platform: str, tile: str, year: int) -> Path:
    return r0_dir(config, platform) / tile / str(year) / f"{platform}_r0_{tile}_{year}.nc"


def output_tile_root(config: CurcWorkflowConfig, platform: str, tile: str) -> Path:
    return config.scratch_root / "output" / config.sensor / platform / tile


def output_raw_water_year_root(config: CurcWorkflowConfig, platform: str, tile: str, water_year: int) -> Path:
    return output_tile_root(config, platform, tile) / "raw" / f"wy{water_year}"


def log_root(config: CurcWorkflowConfig) -> Path:
    return config.scratch_root / "logs"


def job_log_dir(config: CurcWorkflowConfig, platform: str, tile: str, year: int) -> Path:
    return log_root(config)


def timestamped_log_dir(config: CurcWorkflowConfig, *, timestamp: str | None = None) -> Path:
    """Return a top-level timestamped log directory for one workflow execution."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return log_root(config) / timestamp
