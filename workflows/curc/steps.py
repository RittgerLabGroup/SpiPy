"""Workflow step definitions for CURC orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from workflows.curc.config import SlurmProfile


WorkflowStepName = Literal[
    "stage_reflectance",
    "stage_ancillary",
    "build_r0",
    "run_inversion",
]


@dataclass(frozen=True)
class WorkflowStepPlan:
    """A single step in a planned CURC workflow."""

    step: WorkflowStepName
    sensor: str
    platform: str
    tile: str
    water_year: int
    date_count: int
    dates: tuple[str, ...]
    source_paths: tuple[str, ...]
    destination_path: str
    notes: tuple[str, ...] = ()
    r0_year: int | None = None


@dataclass(frozen=True)
class InversionTaskPlan:
    """One logical inversion task, typically one acquisition date."""

    task_index: int
    sensor: str
    platform: str
    tile: str
    water_year: int
    date: str
    source_paths: tuple[str, ...]
    output_path: str
    log_path: str
    r0_year: int
    retry_count: int = 0


@dataclass(frozen=True)
class SlurmArrayPlan:
    """A Slurm array submission that bundles many logical date tasks."""

    step: WorkflowStepName
    job_name: str
    sensor: str
    platform: str
    tile: str
    water_year: int
    task_count: int
    array_indices: tuple[int, ...]
    max_concurrent_tasks: int | None
    max_auto_retry_count: int
    apply_valid_inversion_mask: bool
    mask_low_reflectance_for_inversion: bool
    low_reflectance_threshold: float
    include_grouped_reflectance_rmse: bool
    use_grouping: bool
    grouping_method: str
    tasks: tuple[InversionTaskPlan, ...]
    slurm_profile: SlurmProfile
    notes: tuple[str, ...] = ()
    r0_year: int | None = None
    include_albedo: bool = True
    include_radiative_forcing: bool = True
    include_delta_vis: bool = True
    albedo_lut_path: str | None = None
    radiative_forcing_lut_path: str | None = None
    terrain_ancillary_root: str | None = None
