"""File discovery helpers for CURC-specific SpiPy workflows."""

from __future__ import annotations

from pathlib import Path

from workflows.curc.config import CurcWorkflowConfig
from workflows.curc.dates import iter_dates, parse_iso_date, water_year_bounds
from workflows.curc.manifest import PlannedJob
from spires.sensors.viirs.hdf import parse_viirs_surface_reflectance_filename


def describe_discovery_scope(config: CurcWorkflowConfig, job: PlannedJob) -> dict[str, object]:
    """Return the future discovery scope for a planned CURC job.

    This remains side-effect free for now so the surrounding workflow can be
    developed before wiring in actual filesystem discovery rules.
    """
    canonical = config.canonicalized()
    return {
        "sensor": canonical.sensor,
        "platform": job.platform,
        "tile": job.tile,
        "year": job.year,
        "input_source_root": str(canonical.input_source_root) if canonical.input_source_root is not None else None,
        "date_glob": canonical.date_glob,
    }


def viirs_snpp_source_root(config: CurcWorkflowConfig) -> Path:
    """Return the configured source root for VIIRS SNPP VNP09GA files."""
    canonical = config.canonicalized()
    if canonical.sensor != "viirs":
        raise ValueError("VIIRS SNPP discovery requires sensor='viirs'")
    if canonical.input_source_root is None:
        raise ValueError("VIIRS SNPP discovery requires input_source_root to be configured")
    return Path(canonical.input_source_root).expanduser().resolve()


def viirs_snpp_source_tile_year_root(config: CurcWorkflowConfig, *, tile: str, year: int) -> Path:
    """Return the calendar-year source directory for one VIIRS SNPP tile."""
    return viirs_snpp_source_root(config) / "input" / tile / str(year)


def _discover_vnp09ga_files_for_year(config: CurcWorkflowConfig, *, tile: str, year: int) -> list[Path]:
    root = viirs_snpp_source_tile_year_root(config, tile=tile, year=year)
    if not root.exists():
        return []
    return sorted(root.glob("VNP09GA*.h5"))


def discover_viirs_snpp_reflectance_files(
    config: CurcWorkflowConfig,
    *,
    tile: str,
    water_year: int | None = None,
    target_dates: tuple[str, ...] | list[str] = (),
) -> list[Path]:
    """Discover VIIRS SNPP reflectance files for a water year or explicit dates."""
    canonical = config.canonicalized()
    if canonical.sensor != "viirs" or "snpp" not in canonical.platforms:
        raise ValueError("This discovery helper is currently implemented only for VIIRS SNPP")
    if water_year is None and not target_dates:
        raise ValueError("Provide water_year or target_dates")

    selected_dates = {parse_iso_date(text) for text in target_dates}
    if water_year is not None:
        start, end = water_year_bounds(water_year)
        water_year_dates = {parse_iso_date(text) for text in iter_dates(start, end)}
        selected_dates |= water_year_dates

    candidate_years = sorted({day.year for day in selected_dates})
    files: list[Path] = []
    for year in candidate_years:
        files.extend(_discover_vnp09ga_files_for_year(canonical, tile=tile, year=year))

    selected_files: list[Path] = []
    for path in files:
        metadata = parse_viirs_surface_reflectance_filename(path)
        acquisition_date = parse_iso_date(metadata.acquisition_date)
        if acquisition_date in selected_dates:
            selected_files.append(path)
    return selected_files


def discover_viirs_snpp_water_year_reflectance_files(
    config: CurcWorkflowConfig,
    *,
    tile: str,
    water_year: int,
) -> list[Path]:
    """Discover VIIRS SNPP reflectance files for one water year."""
    return discover_viirs_snpp_reflectance_files(config, tile=tile, water_year=water_year)
