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


def _scope_token(
    *,
    water_year: int,
    target_dates: tuple[str, ...] | list[str] = (),
) -> str:
    normalized_dates = tuple(sorted(str(date) for date in target_dates))
    if not normalized_dates:
        return f"wy{water_year}_full"
    if len(normalized_dates) == 1:
        return f"wy{water_year}_{normalized_dates[0]}_single_date"
    return f"wy{water_year}_{normalized_dates[0]}_{normalized_dates[-1]}_date_subset"


def build_run_group_id(
    *,
    sensor: str,
    platform: str,
    water_year: int,
    target_dates: tuple[str, ...] | list[str] = (),
    timestamp: str | None = None,
) -> str:
    """Return a stable run-group identifier for one CURC launch."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{sensor}_{platform}_{_scope_token(water_year=water_year, target_dates=target_dates)}"


def run_group_log_dir(config: CurcWorkflowConfig, *, run_group_id: str) -> Path:
    return log_root(config) / run_group_id


def tile_run_dir(config: CurcWorkflowConfig, *, run_group_id: str, tile: str) -> Path:
    return run_group_log_dir(config, run_group_id=run_group_id) / tile


def tile_detailed_log_dir(config: CurcWorkflowConfig, *, run_group_id: str, tile: str) -> Path:
    return tile_run_dir(config, run_group_id=run_group_id, tile=tile) / "detailed_logs"


def tile_summary_csv_path(
    config: CurcWorkflowConfig,
    *,
    run_group_id: str,
    tile: str,
    water_year: int,
) -> Path:
    return tile_run_dir(config, run_group_id=run_group_id, tile=tile) / f"run_inversion_{tile}_wy{water_year}_summary.csv"


def tile_summary_txt_path(
    config: CurcWorkflowConfig,
    *,
    run_group_id: str,
    tile: str,
    water_year: int,
) -> Path:
    return tile_run_dir(config, run_group_id=run_group_id, tile=tile) / f"run_inversion_{tile}_wy{water_year}_summary.txt"


def run_group_summary_csv_path(
    config: CurcWorkflowConfig,
    *,
    run_group_id: str,
    water_year: int,
) -> Path:
    return run_group_log_dir(config, run_group_id=run_group_id) / f"run_inversion_wy{water_year}_summary.csv"


def run_group_summary_txt_path(
    config: CurcWorkflowConfig,
    *,
    run_group_id: str,
    water_year: int,
) -> Path:
    return run_group_log_dir(config, run_group_id=run_group_id) / f"run_inversion_wy{water_year}_summary.txt"


def job_log_dir(config: CurcWorkflowConfig, platform: str, tile: str, year: int) -> Path:
    return tile_detailed_log_dir(
        config,
        run_group_id=build_run_group_id(sensor=config.sensor, platform=platform, water_year=year),
        tile=tile,
    )


def detailed_log_dir(path: str | Path) -> Path:
    """Return the tile-local detailed log subdirectory for a CURC artifact path."""
    resolved = Path(path).expanduser().resolve()
    if resolved.name == "detailed_logs":
        return resolved
    if resolved.parent.name == "detailed_logs":
        return resolved.parent
    return resolved / "detailed_logs"


def top_level_log_dir(path: str | Path) -> Path:
    """Return the run-group root directory for a CURC log artifact path."""
    resolved = Path(path).expanduser().resolve()
    if resolved.name == "detailed_logs":
        return resolved.parent.parent
    if resolved.parent.name == "detailed_logs":
        return resolved.parent.parent.parent
    if resolved.parent.parent.name == "detailed_logs":
        return resolved.parent.parent.parent.parent
    return resolved
