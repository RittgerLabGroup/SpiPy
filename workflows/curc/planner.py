"""Step-oriented planning for CURC workflows."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from workflows.curc.config import CurcWorkflowConfig
from workflows.curc.dates import default_r0_year_for_water_year, iter_dates, r0_source_bounds_for_year
from workflows.curc.discovery import discover_viirs_snpp_reflectance_files
from workflows.curc.paths import (
    ancillary_dir,
    log_root,
    output_raw_water_year_root,
    r0_dir,
    reflectance_dir,
    timestamped_log_dir,
)
from workflows.curc.steps import InversionTaskPlan, SlurmArrayPlan, WorkflowStepPlan
from spires.sensors.viirs.hdf import parse_viirs_surface_reflectance_filename


def _annual_r0_path(config: CurcWorkflowConfig, *, tile: str, r0_year: int) -> Path:
    return r0_dir(config, "snpp") / tile / str(r0_year)


def _group_source_paths_by_date(discovered: list[Path]) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    for path in discovered:
        acquisition_date = parse_viirs_surface_reflectance_filename(path).acquisition_date
        grouped.setdefault(acquisition_date, []).append(path)
    return {date: sorted(paths) for date, paths in sorted(grouped.items())}


def _discover_r0_source_paths(
    config: CurcWorkflowConfig,
    *,
    tile: str,
    r0_year: int,
) -> list[Path]:
    start, end = r0_source_bounds_for_year(r0_year)
    return discover_viirs_snpp_reflectance_files(
        config,
        tile=tile,
        target_dates=tuple(iter_dates(start, end)),
    )

def plan_viirs_snpp_workflow_steps(
    config: CurcWorkflowConfig,
    *,
    tile: str,
    water_year: int,
    target_dates: tuple[str, ...] | list[str] = (),
    r0_year: int | None = None,
) -> list[WorkflowStepPlan]:
    """Plan explicit VIIRS SNPP workflow steps for a tile and water year."""
    canonical = config.canonicalized()
    if canonical.sensor != "viirs" or canonical.platforms != ("snpp",):
        raise ValueError("This planner currently supports only sensor='viirs' and platforms=('snpp',)")

    discovered = discover_viirs_snpp_reflectance_files(
        canonical,
        tile=tile,
        water_year=water_year if not target_dates else None,
        target_dates=tuple(target_dates),
    )
    acquisition_dates = tuple(
        sorted({parse_viirs_surface_reflectance_filename(path).acquisition_date for path in discovered})
    )
    selected_dates = acquisition_dates if acquisition_dates else tuple(target_dates)
    inversion_destination = output_raw_water_year_root(canonical, "snpp", tile, water_year)
    resolved_r0_year = default_r0_year_for_water_year(water_year) if r0_year is None else r0_year
    r0_discovered = _discover_r0_source_paths(canonical, tile=tile, r0_year=resolved_r0_year)
    r0_dates = tuple(
        sorted({parse_viirs_surface_reflectance_filename(path).acquisition_date for path in r0_discovered})
    )

    return [
        WorkflowStepPlan(
            step="stage_reflectance",
            sensor="viirs",
            platform="snpp",
            tile=tile,
            water_year=water_year,
            date_count=len(selected_dates),
            dates=selected_dates,
            source_paths=tuple(str(path) for path in discovered),
            destination_path=str(reflectance_dir(canonical, "snpp") / tile / str(water_year)),
            notes=(
                "Copy or rsync the discovered VNP09GA files from /pl to /scratch/alpine.",
                "This step can target a full water year or a single acquisition date.",
            ),
        ),
        WorkflowStepPlan(
            step="stage_ancillary",
            sensor="viirs",
            platform="snpp",
            tile=tile,
            water_year=water_year,
            date_count=len(selected_dates),
            dates=selected_dates,
            source_paths=(),
            destination_path=str(ancillary_dir(canonical, "snpp") / tile),
            notes=(
                "Ensure static ancillary inputs such as canopy, terrain, landcover, and water masks are staged.",
                "This is independent of a specific scene date and should be rerunnable on its own.",
            ),
        ),
        WorkflowStepPlan(
            step="build_r0",
            sensor="viirs",
            platform="snpp",
            tile=tile,
            water_year=water_year,
            date_count=len(r0_dates),
            dates=r0_dates,
            source_paths=tuple(str(path) for path in r0_discovered),
            destination_path=str(_annual_r0_path(canonical, tile=tile, r0_year=resolved_r0_year)),
            notes=(
                f"Build or refresh the annual R0 inputs needed for inversion using summer scenes from {resolved_r0_year}-06-01 through {resolved_r0_year}-09-30.",
                "This step is independent of the requested inversion date subset within the water year.",
            ),
            r0_year=resolved_r0_year,
        ),
        WorkflowStepPlan(
            step="run_inversion",
            sensor="viirs",
            platform="snpp",
            tile=tile,
            water_year=water_year,
            date_count=len(selected_dates),
            dates=selected_dates,
            source_paths=tuple(str(path) for path in discovered),
            destination_path=str(inversion_destination),
            notes=(
                "Run inversion for the discovered scenes using the staged reflectance, ancillary inputs, and R0 product.",
                f"Per-job logs should land under a timestamped subdirectory of {log_root(canonical)}.",
            ),
            r0_year=resolved_r0_year,
        ),
    ]


def plan_viirs_snpp_inversion_array(
    config: CurcWorkflowConfig,
    *,
    tile: str,
    water_year: int,
    target_dates: tuple[str, ...] | list[str] = (),
    r0_year: int | None = None,
    max_concurrent_tasks: int | None = None,
) -> SlurmArrayPlan:
    """Plan a Slurm array for VIIRS SNPP inversion with one logical task per date."""
    canonical = config.canonicalized()
    if canonical.sensor != "viirs" or canonical.platforms != ("snpp",):
        raise ValueError("This planner currently supports only sensor='viirs' and platforms=('snpp',)")

    discovered = discover_viirs_snpp_reflectance_files(
        canonical,
        tile=tile,
        water_year=water_year if not target_dates else None,
        target_dates=tuple(target_dates),
    )
    grouped = _group_source_paths_by_date(discovered)
    if not grouped and target_dates:
        grouped = {date: [] for date in target_dates}

    resolved_r0_year = default_r0_year_for_water_year(water_year) if r0_year is None else r0_year
    log_dir = timestamped_log_dir(canonical, timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"))
    tasks = []
    for task_index, (acquisition_date, paths) in enumerate(grouped.items()):
        tasks.append(
            InversionTaskPlan(
                task_index=task_index,
                sensor="viirs",
                platform="snpp",
                tile=tile,
                water_year=water_year,
                date=acquisition_date,
                source_paths=tuple(str(path) for path in paths),
                output_path=str(output_raw_water_year_root(canonical, "snpp", tile, water_year)),
                log_path=str(log_dir / f"run_inversion_{acquisition_date}.log"),
                r0_year=resolved_r0_year,
                retry_count=0,
            )
        )

    return SlurmArrayPlan(
        step="run_inversion",
        job_name=f"spipy-viirs-snpp-{tile}-wy{water_year}",
        sensor="viirs",
        platform="snpp",
        tile=tile,
        water_year=water_year,
        task_count=len(tasks),
        array_indices=tuple(task.task_index for task in tasks),
        max_concurrent_tasks=max_concurrent_tasks,
        max_auto_retry_count=canonical.max_auto_retry_count,
        apply_valid_inversion_mask=canonical.apply_valid_inversion_mask,
        use_grouping=canonical.use_grouping,
        grouping_method=canonical.grouping_method,
        tasks=tuple(tasks),
        slurm_profile=canonical.slurm_profile,
        notes=(
            "Logical inversion unit is one acquisition date.",
            "Submission unit is one Slurm array spanning those dates.",
        ),
        r0_year=resolved_r0_year,
    )
