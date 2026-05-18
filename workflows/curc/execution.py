"""Executable helpers for CURC workflow steps outside the Slurm inversion array."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import shlex
import subprocess

from spires.sensors.viirs.r0 import build_r0_from_sources

from workflows.curc.config import CurcWorkflowConfig
from workflows.curc.paths import job_log_dir, output_tile_root, r0_dataset_path, r0_dir
from workflows.curc.runtime import default_viirs_lut_file
from workflows.curc.steps import WorkflowStepPlan


def _shell_join(command: tuple[str, ...]) -> str:
    return shlex.join(command)


def _rsync_destination(destination: Path) -> str:
    return f"{destination.expanduser().resolve()}/"


def _group_paths_by_parent(paths: tuple[str, ...]) -> list[tuple[Path, tuple[Path, ...]]]:
    grouped: dict[Path, list[Path]] = {}
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        grouped.setdefault(path.parent, []).append(path)
    return [(parent, tuple(paths)) for parent, paths in sorted(grouped.items())]


def _reflectance_stage_commands(
    step_plan: WorkflowStepPlan,
    *,
    rsync_executable: str,
) -> tuple[tuple[str, ...], ...]:
    destination = Path(step_plan.destination_path).expanduser().resolve()
    commands: list[tuple[str, ...]] = []
    for _, paths in _group_paths_by_parent(step_plan.source_paths):
        commands.append(
            (
                rsync_executable,
                "-av",
                "--ignore-existing",
                *tuple(str(path) for path in paths),
                _rsync_destination(destination),
            )
        )
    return tuple(commands)


def _ancillary_stage_directories(
    config: CurcWorkflowConfig,
    step_plan: WorkflowStepPlan,
) -> tuple[Path, ...]:
    canonical = config.canonicalized()
    directories = [
        Path(step_plan.destination_path).expanduser().resolve(),
        r0_dir(canonical, step_plan.platform).expanduser().resolve() / step_plan.tile,
        job_log_dir(canonical, step_plan.platform, step_plan.tile, step_plan.water_year).expanduser().resolve(),
        output_tile_root(canonical, step_plan.platform, step_plan.tile).expanduser().resolve(),
    ]
    if step_plan.r0_year is not None:
        directories.append(
            r0_dir(canonical, step_plan.platform).expanduser().resolve() / step_plan.tile / str(step_plan.r0_year)
        )
    seen: set[Path] = set()
    unique_directories: list[Path] = []
    for path in directories:
        if path not in seen:
            seen.add(path)
            unique_directories.append(path)
    return tuple(unique_directories)


def _build_r0_output_dataset_path(config: CurcWorkflowConfig, step_plan: WorkflowStepPlan) -> Path:
    canonical = config.canonicalized()
    if step_plan.r0_year is None:
        raise ValueError("build_r0 step requires step_plan.r0_year")
    return r0_dataset_path(canonical, step_plan.platform, step_plan.tile, step_plan.r0_year).expanduser().resolve()


def preview_viirs_snpp_workflow_step_execution(
    config: CurcWorkflowConfig,
    step_plan: WorkflowStepPlan,
    *,
    rsync_executable: str = "rsync",
    lut_file: str | Path | None = None,
    overwrite: bool = False,
    show_progress: bool = False,
) -> dict[str, object]:
    """Return a JSON-serializable preview of how one workflow step would execute."""
    result: dict[str, object] = {
        "step": step_plan.step,
        "sensor": step_plan.sensor,
        "platform": step_plan.platform,
        "tile": step_plan.tile,
        "water_year": step_plan.water_year,
        "date_count": step_plan.date_count,
        "dates": list(step_plan.dates),
        "source_path_count": len(step_plan.source_paths),
        "source_paths": list(step_plan.source_paths),
        "destination_path": str(Path(step_plan.destination_path).expanduser().resolve()),
        "notes": list(step_plan.notes),
        "r0_year": step_plan.r0_year,
        "overwrite": overwrite,
    }

    if step_plan.step == "stage_reflectance":
        commands = _reflectance_stage_commands(step_plan, rsync_executable=rsync_executable)
        result["mode"] = "direct_rsync"
        result["commands"] = [list(command) for command in commands]
        result["shell_commands"] = [_shell_join(command) for command in commands]
        return result

    if step_plan.step == "stage_ancillary":
        directories = _ancillary_stage_directories(config, step_plan)
        commands = tuple(("mkdir", "-p", str(path)) for path in directories)
        result["mode"] = "direct_mkdir"
        result["commands"] = [list(command) for command in commands]
        result["shell_commands"] = [_shell_join(command) for command in commands]
        result["expected_static_files"] = [
            str(Path(step_plan.destination_path).expanduser().resolve() / "canopy_fraction.zarr"),
            str(Path(step_plan.destination_path).expanduser().resolve() / "glacier_ice_fraction.zarr"),
        ]
        return result

    if step_plan.step == "build_r0":
        resolved_lut_file = default_viirs_lut_file(step_plan.platform) if lut_file is None else Path(lut_file)
        output_dataset_path = _build_r0_output_dataset_path(config, step_plan)
        result["mode"] = "python_r0_builder"
        result["output_dataset_path"] = str(output_dataset_path)
        result["lut_file"] = str(resolved_lut_file.expanduser().resolve())
        result["show_progress"] = show_progress
        return result

    if step_plan.step == "run_inversion":
        result["mode"] = "slurm_array"
        result["message"] = "Use the existing inversion-array planning and submission helpers for run_inversion."
        return result

    raise ValueError(f"Unsupported step: {step_plan.step!r}")


def execute_viirs_snpp_workflow_step(
    config: CurcWorkflowConfig,
    step_plan: WorkflowStepPlan,
    *,
    execute: bool = False,
    rsync_executable: str = "rsync",
    lut_file: str | Path | None = None,
    overwrite: bool = False,
    show_progress: bool = False,
) -> dict[str, object]:
    """Preview or execute one planned VIIRS SNPP workflow step."""
    preview = preview_viirs_snpp_workflow_step_execution(
        config,
        step_plan,
        rsync_executable=rsync_executable,
        lut_file=lut_file,
        overwrite=overwrite,
        show_progress=show_progress,
    )
    preview["executed"] = False

    if not execute:
        return preview

    if step_plan.step == "stage_reflectance":
        destination = Path(step_plan.destination_path).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        command_results = []
        for command in _reflectance_stage_commands(step_plan, rsync_executable=rsync_executable):
            completed = subprocess.run(command, check=True, text=True, capture_output=True)
            command_results.append(
                {
                    "command": list(command),
                    "stdout": completed.stdout.strip(),
                    "stderr": completed.stderr.strip(),
                }
            )
        preview["executed"] = True
        preview["command_results"] = command_results
        return preview

    if step_plan.step == "stage_ancillary":
        created_paths = []
        for path in _ancillary_stage_directories(config, step_plan):
            path.mkdir(parents=True, exist_ok=True)
            created_paths.append(str(path))
        preview["executed"] = True
        preview["created_paths"] = created_paths
        return preview

    if step_plan.step == "build_r0":
        output_dataset_path = _build_r0_output_dataset_path(config, step_plan)
        output_dataset_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_lut_file = default_viirs_lut_file(step_plan.platform) if lut_file is None else Path(lut_file)
        dataset = build_r0_from_sources(
            list(step_plan.source_paths),
            r0_path=output_dataset_path,
            overwrite=overwrite,
            lut_file=resolved_lut_file,
            show_progress=show_progress,
        )
        preview["executed"] = True
        preview["output_dataset_path"] = str(output_dataset_path)
        preview["dataset_attrs"] = dataset.attrs.copy()
        preview["dataset_sizes"] = dict(dataset.sizes)
        return preview

    if step_plan.step == "run_inversion":
        raise ValueError("run_inversion execution should use the Slurm-array submission helpers, not direct step execution")

    raise ValueError(f"Unsupported step: {step_plan.step!r}")


def resolve_viirs_snpp_workflow_step(
    config: CurcWorkflowConfig,
    *,
    tile: str,
    water_year: int,
    step: str,
    target_dates: tuple[str, ...] | list[str] = (),
    r0_year: int | None = None,
) -> WorkflowStepPlan:
    """Resolve one step name into its planned VIIRS SNPP workflow step."""
    from workflows.curc.planner import plan_viirs_snpp_workflow_steps

    steps = plan_viirs_snpp_workflow_steps(
        config,
        tile=tile,
        water_year=water_year,
        target_dates=target_dates,
        r0_year=r0_year,
    )
    for candidate in steps:
        if candidate.step == step:
            return candidate
    raise KeyError(f"No workflow step named {step!r}")


def serialize_step_plan(step_plan: WorkflowStepPlan) -> dict[str, object]:
    """Return a stable JSON-serializable view of one step plan."""
    return asdict(step_plan)
