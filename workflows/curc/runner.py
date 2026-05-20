"""High-level CURC workflow entrypoints built on job manifests."""

from __future__ import annotations

from workflows.curc.config import CurcWorkflowConfig
from workflows.curc.execution import (
    execute_viirs_snpp_workflow_step,
    preview_viirs_snpp_workflow_step_execution,
    resolve_viirs_snpp_workflow_step,
)
from workflows.curc.manifest import build_job_manifest
from workflows.curc.planner import plan_viirs_snpp_inversion_array, plan_viirs_snpp_workflow_steps
from workflows.curc.runtime import (
    build_viirs_snpp_inversion_runtime_context,
    execute_viirs_snpp_inversion_task,
)
from workflows.curc.slurm import render_array_submission_payload, render_submission_payload
from workflows.curc.slurm import (
    render_array_submission_payload_from_manifest,
    render_sbatch_command_for_array_payload,
)
from workflows.curc.steps import SlurmArrayPlan, WorkflowStepPlan
from workflows.curc.task_manifest import (
    resolve_inversion_task_from_manifest,
    write_inversion_array_manifest,
)
from workflows.curc.status import (
    scan_inversion_array_status,
    should_auto_retry,
    write_retry_manifest,
    write_status_summary_artifacts,
)


def plan_submissions(config: CurcWorkflowConfig) -> list[dict[str, object]]:
    """Return the rendered submission payloads for a CURC workflow config."""
    return [render_submission_payload(job) for job in build_job_manifest(config)]


def plan_viirs_snpp_steps(
    config: CurcWorkflowConfig,
    *,
    tile: str,
    water_year: int,
    target_dates: tuple[str, ...] | list[str] = (),
    r0_year: int | None = None,
) -> list[WorkflowStepPlan]:
    """Return explicit VIIRS SNPP step plans for notebook or script use."""
    return plan_viirs_snpp_workflow_steps(
        config,
        tile=tile,
        water_year=water_year,
        target_dates=target_dates,
        r0_year=r0_year,
    )


def plan_viirs_snpp_inversion_array_submission(
    config: CurcWorkflowConfig,
    *,
    tile: str,
    water_year: int,
    target_dates: tuple[str, ...] | list[str] = (),
    r0_year: int | None = None,
    max_concurrent_tasks: int | None = None,
    manifest_path: str | None = None,
) -> dict[str, object]:
    """Return a rendered Slurm-array submission payload for VIIRS SNPP inversion dates."""
    plan = plan_viirs_snpp_inversion_array(
        config,
        tile=tile,
        water_year=water_year,
        target_dates=target_dates,
        r0_year=r0_year,
        max_concurrent_tasks=max_concurrent_tasks,
    )
    resolved_manifest_path = write_inversion_array_manifest(plan, manifest_path=manifest_path)
    return render_array_submission_payload(plan, manifest_path=resolved_manifest_path)


def plan_viirs_snpp_inversion_array_jobs(
    config: CurcWorkflowConfig,
    *,
    tile: str,
    water_year: int,
    target_dates: tuple[str, ...] | list[str] = (),
    r0_year: int | None = None,
    max_concurrent_tasks: int | None = None,
) -> SlurmArrayPlan:
    """Return the structured Slurm-array plan for VIIRS SNPP inversion dates."""
    return plan_viirs_snpp_inversion_array(
        config,
        tile=tile,
        water_year=water_year,
        target_dates=target_dates,
        r0_year=r0_year,
        max_concurrent_tasks=max_concurrent_tasks,
    )


def preview_viirs_snpp_step_execution(
    config: CurcWorkflowConfig,
    *,
    tile: str,
    water_year: int,
    step: str,
    target_dates: tuple[str, ...] | list[str] = (),
    r0_year: int | None = None,
    rsync_executable: str = "rsync",
    lut_file: str | None = None,
    ndvi_tie_epsilon: float = 0.02,
    zarr_path: str | None = None,
    chunks: dict[str, int] | None = None,
    overwrite: bool = False,
    show_progress: bool = False,
):
    """Preview how one planned VIIRS SNPP workflow step would execute."""
    step_plan = resolve_viirs_snpp_workflow_step(
        config,
        tile=tile,
        water_year=water_year,
        step=step,
        target_dates=target_dates,
        r0_year=r0_year,
    )
    return preview_viirs_snpp_workflow_step_execution(
        config,
        step_plan,
        rsync_executable=rsync_executable,
        lut_file=lut_file,
        ndvi_tie_epsilon=ndvi_tie_epsilon,
        zarr_path=zarr_path,
        chunks=chunks,
        overwrite=overwrite,
        show_progress=show_progress,
    )


def run_viirs_snpp_step(
    config: CurcWorkflowConfig,
    *,
    tile: str,
    water_year: int,
    step: str,
    target_dates: tuple[str, ...] | list[str] = (),
    r0_year: int | None = None,
    execute: bool = False,
    rsync_executable: str = "rsync",
    lut_file: str | None = None,
    ndvi_tie_epsilon: float = 0.02,
    zarr_path: str | None = None,
    chunks: dict[str, int] | None = None,
    overwrite: bool = False,
    show_progress: bool = False,
):
    """Preview or execute one planned VIIRS SNPP workflow step."""
    step_plan = resolve_viirs_snpp_workflow_step(
        config,
        tile=tile,
        water_year=water_year,
        step=step,
        target_dates=target_dates,
        r0_year=r0_year,
    )
    return execute_viirs_snpp_workflow_step(
        config,
        step_plan,
        execute=execute,
        rsync_executable=rsync_executable,
        lut_file=lut_file,
        ndvi_tie_epsilon=ndvi_tie_epsilon,
        zarr_path=zarr_path,
        chunks=chunks,
        overwrite=overwrite,
        show_progress=show_progress,
    )


def resolve_viirs_snpp_inversion_array_task(
    manifest_path: str,
    *,
    task_index: int,
):
    """Resolve one logical inversion task from a manifest-backed array plan."""
    return resolve_inversion_task_from_manifest(manifest_path, task_index)


def resolve_viirs_snpp_inversion_runtime(
    manifest_path: str,
    *,
    task_index: int | None = None,
):
    """Resolve one manifest-backed array task into concrete runtime paths."""
    return build_viirs_snpp_inversion_runtime_context(manifest_path, task_index=task_index)


def run_viirs_snpp_inversion_array_task(
    manifest_path: str,
    *,
    task_index: int | None = None,
    lut_file: str | None = None,
    execution_profile: str = "cluster",
    overwrite: bool = False,
    dry_run: bool = True,
):
    """Execute one manifest-backed VIIRS SNPP inversion task."""
    return execute_viirs_snpp_inversion_task(
        manifest_path,
        task_index=task_index,
        lut_file=lut_file,
        execution_profile=execution_profile,
        overwrite=overwrite,
        dry_run=dry_run,
    )


def scan_viirs_snpp_inversion_array(manifest_path: str):
    """Scan a manifest-backed inversion array for per-date outcomes."""
    report = scan_inversion_array_status(manifest_path)
    write_status_summary_artifacts(manifest_path, report=report)
    return report


def write_viirs_snpp_retry_manifest(
    manifest_path: str,
    *,
    retry_only: bool = True,
):
    """Write a retry manifest for failed VIIRS SNPP inversion tasks."""
    return write_retry_manifest(manifest_path, retry_only=retry_only)


def should_auto_retry_viirs_snpp_inversion_array(manifest_path: str):
    """Return True when any failed VIIRS SNPP task is still auto-retry eligible."""
    return should_auto_retry(manifest_path)


def plan_viirs_snpp_auto_retry_submission(
    manifest_path: str,
    *,
    retry_only: bool = True,
):
    """Return the next retry manifest and compact array payload when auto-retry is allowed."""
    if not should_auto_retry(manifest_path):
        return None
    retry_manifest_path = write_retry_manifest(manifest_path, retry_only=retry_only)
    return {
        "retry_manifest_path": str(retry_manifest_path),
        "retry_payload": render_array_submission_payload_from_manifest(retry_manifest_path),
    }
