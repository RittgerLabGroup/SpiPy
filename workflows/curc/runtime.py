"""Runtime helpers for CURC Slurm array tasks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import traceback
from typing import Any

from spires.logging_utils import configure_spires_file_logger, log_event
from spires.sensors.io import load_output_dataset_if_valid, write_output_dataset
from spires.sensors.viirs.workflow import run_viirs_inversion
import xarray as xr

from workflows.curc.config import CurcWorkflowConfig
from workflows.curc.paths import r0_dataset_path
from workflows.curc.steps import InversionTaskPlan
from workflows.curc.task_manifest import resolve_inversion_task_from_manifest


@dataclass(frozen=True)
class InversionRuntimeContext:
    """Resolved runtime context for one logical inversion task."""

    task: InversionTaskPlan
    manifest_path: str
    scratch_root: str
    staged_reflectance_paths: tuple[str, ...]
    ancillary_root: str
    r0_root: str
    r0_path: str
    canopy_fraction_path: str | None
    ice_fraction_path: str | None
    lut_file: str
    output_path: str
    output_dataset_path: str
    log_path: str


def default_viirs_lut_file(platform: str) -> Path:
    """Return the repository LUT path for a canonical VIIRS platform."""
    repo_root = Path(__file__).resolve().parents[2]
    lut_by_platform = {
        "snpp": repo_root / "data" / "viirs" / "lut" / "lut_viirs_snpp_i1_i2_i3_m2_m4_m8_m11_3um_dust_bandpass.mat",
        "noaa20": repo_root / "data" / "viirs" / "lut" / "lut_viirs_noaa20_i1_i2_i3_m2_m4_m8_m11_3um_dust_bandpass.mat",
        "noaa21": repo_root / "data" / "viirs" / "lut" / "lut_viirs_noaa21_i1_i2_i3_m2_m4_m8_m11_3um_dust_bandpass.mat",
    }
    try:
        return lut_by_platform[platform].resolve()
    except KeyError as exc:
        raise ValueError(f"Unsupported VIIRS platform for LUT resolution: {platform!r}") from exc


def _first_existing_path(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if path.exists():
            return path.resolve()
    return None


def _infer_static_fraction_path(ancillary_root: Path, stem: str) -> Path | None:
    return _first_existing_path(
        [
            ancillary_root / f"{stem}.zarr",
            ancillary_root / f"{stem}.tif",
            ancillary_root / f"{stem}.tiff",
            ancillary_root / f"{stem}.nc",
        ]
    )


def slurm_metadata_from_env() -> dict[str, object]:
    """Return compact Slurm metadata from the process environment."""
    keys = (
        "SLURM_JOB_ID",
        "SLURM_ARRAY_JOB_ID",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_JOB_NAME",
        "SLURM_CLUSTER_NAME",
        "SLURM_SUBMIT_DIR",
        "SLURM_CPUS_PER_TASK",
    )
    metadata: dict[str, object] = {}
    for key in keys:
        value = os.environ.get(key)
        if value is not None:
            metadata[key.lower()] = value
    return metadata


def _has_slurm_context(slurm_fields: dict[str, object]) -> bool:
    return any(key in slurm_fields for key in ("slurm_job_id", "slurm_array_job_id", "slurm_array_task_id"))


def resolve_array_task_index(task_index: int | None = None) -> int:
    """Return the explicit task index or read it from `SLURM_ARRAY_TASK_ID`."""
    if task_index is not None:
        return int(task_index)
    if "SLURM_ARRAY_TASK_ID" not in os.environ:
        raise ValueError("task_index was not provided and SLURM_ARRAY_TASK_ID is not set")
    return int(os.environ["SLURM_ARRAY_TASK_ID"])


def infer_scratch_root_from_output_path(output_path: str | Path) -> Path:
    """Infer the scratch root from an output path under `<scratch_root>/output/...`."""
    path = Path(output_path).expanduser().resolve()
    parts = path.parts
    if "output" not in parts:
        raise ValueError(f"Could not infer scratch root from output path without 'output' segment: {path}")
    output_index = parts.index("output")
    return Path(*parts[:output_index])


def _runtime_r0_dataset_path(scratch_root: Path, task: InversionTaskPlan) -> Path:
    config = CurcWorkflowConfig(
        scratch_root=scratch_root,
        input_source_root=scratch_root,
        sensor=task.sensor,
        platforms=(task.platform,),
        tiles=(task.tile,),
        years=(),
        water_years=(task.water_year,),
    )
    return r0_dataset_path(config, task.platform, task.tile, task.r0_year)


def build_viirs_snpp_inversion_runtime_context(
    manifest_path: str | Path,
    *,
    task_index: int | None = None,
    lut_file: str | Path | None = None,
) -> InversionRuntimeContext:
    """Resolve one VIIRS SNPP array task into concrete runtime paths."""
    resolved_task_index = resolve_array_task_index(task_index)
    task = resolve_inversion_task_from_manifest(manifest_path, resolved_task_index)
    scratch_root = infer_scratch_root_from_output_path(task.output_path)
    reflectance_root = scratch_root / "input" / task.sensor / task.platform / "reflectance" / task.tile / str(task.water_year)
    staged_reflectance_paths = tuple(str(reflectance_root / Path(path).name) for path in task.source_paths)
    ancillary_root = scratch_root / "input" / task.sensor / task.platform / "ancillary" / task.tile
    r0_root = scratch_root / "input" / task.sensor / task.platform / "ancillary" / "r0" / task.tile / str(task.r0_year)
    resolved_lut_file = default_viirs_lut_file(task.platform) if lut_file is None else Path(lut_file).expanduser().resolve()
    r0_path = _runtime_r0_dataset_path(scratch_root, task)
    canopy_fraction_path = _infer_static_fraction_path(ancillary_root, "canopy_fraction")
    ice_fraction_path = _infer_static_fraction_path(ancillary_root, "glacier_ice_fraction")

    return InversionRuntimeContext(
        task=task,
        manifest_path=str(Path(manifest_path).expanduser().resolve()),
        scratch_root=str(scratch_root),
        staged_reflectance_paths=staged_reflectance_paths,
        ancillary_root=str(ancillary_root),
        r0_root=str(r0_root),
        r0_path=str(r0_path),
        canopy_fraction_path=str(canopy_fraction_path) if canopy_fraction_path is not None else None,
        ice_fraction_path=str(ice_fraction_path) if ice_fraction_path is not None else None,
        lut_file=str(resolved_lut_file),
        output_path=task.output_path,
        output_dataset_path=str(Path(task.output_path).expanduser().resolve() / "inversion.nc"),
        log_path=task.log_path,
    )


def summarize_viirs_snpp_runtime_requirements(context: InversionRuntimeContext) -> dict[str, list[str]]:
    """Return missing required inputs for one runtime context."""
    missing: dict[str, list[str]] = {
        "staged_reflectance_paths": [],
        "r0_path": [],
        "lut_file": [],
    }
    if len(context.staged_reflectance_paths) != 1:
        raise ValueError(
            "Current VIIRS SNPP task executor expects exactly one staged reflectance file per date; "
            f"got {len(context.staged_reflectance_paths)}"
        )
    for path in context.staged_reflectance_paths:
        if not Path(path).exists():
            missing["staged_reflectance_paths"].append(path)
    if not Path(context.r0_path).exists():
        missing["r0_path"].append(context.r0_path)
    if not Path(context.lut_file).exists():
        missing["lut_file"].append(context.lut_file)
    return missing


def validate_viirs_snpp_runtime_context(context: InversionRuntimeContext) -> None:
    """Validate that the resolved runtime context has the required inputs."""
    missing = summarize_viirs_snpp_runtime_requirements(context)
    if any(missing.values()):
        raise FileNotFoundError(f"Missing runtime inputs: {missing}")


def _task_logger_name(context: InversionRuntimeContext) -> str:
    return (
        f"spires.curc.{context.task.sensor}.{context.task.platform}."
        f"{context.task.tile}.{context.task.date}"
    )


def _failure_fields_from_missing_inputs(missing: dict[str, list[str]]) -> dict[str, object]:
    if missing["staged_reflectance_paths"]:
        return {
            "failure_code": "missing_staged_reflectance",
            "retry_recommended": False,
        }
    if missing["r0_path"]:
        return {
            "failure_code": "missing_r0",
            "retry_recommended": False,
        }
    if missing["lut_file"]:
        return {
            "failure_code": "missing_lut",
            "retry_recommended": False,
        }
    return {
        "failure_code": "ready",
        "retry_recommended": False,
    }


def _classify_runtime_exception(exc: Exception, *, slurm_fields: dict[str, object]) -> dict[str, object]:
    failure_code = "python_exception"
    retry_recommended = False
    if isinstance(exc, FileNotFoundError):
        failure_code = "missing_runtime_input"
    elif isinstance(exc, ValueError):
        failure_code = "invalid_runtime_input"
    elif isinstance(exc, OSError):
        failure_code = "filesystem_error"
        retry_recommended = True
    if _has_slurm_context(slurm_fields) and failure_code == "filesystem_error":
        retry_recommended = True
    return {
        "failure_code": failure_code,
        "retry_recommended": retry_recommended,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback_tail": traceback.format_exc(limit=3).strip().splitlines()[-1],
    }


def execute_viirs_snpp_inversion_task(
    manifest_path: str | Path,
    *,
    task_index: int | None = None,
    lut_file: str | Path | None = None,
    execution_profile: str = "cluster",
    overwrite: bool = False,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Execute or dry-run one manifest-backed VIIRS SNPP inversion task."""
    context = build_viirs_snpp_inversion_runtime_context(
        manifest_path,
        task_index=task_index,
        lut_file=lut_file,
    )
    logger = configure_spires_file_logger(
        context.log_path,
        logger_name=_task_logger_name(context),
        mode="a",
    )
    slurm_fields = slurm_metadata_from_env()
    common_fields = {
        "sensor": context.task.sensor,
        "platform": context.task.platform,
        "tile": context.task.tile,
        "water_year": context.task.water_year,
        "date": context.task.date,
        "r0_year": context.task.r0_year,
        "manifest_path": context.manifest_path,
        "task_index": context.task.task_index,
        "retry_count": context.task.retry_count,
        "output_dataset_path": context.output_dataset_path,
        "log_path": context.log_path,
        "dry_run": dry_run,
        **slurm_fields,
    }
    log_event(
        logger,
        "curc_run_viirs_snpp_inversion_task",
        stage="curc_runtime",
        event_type="start",
        status="started",
        **common_fields,
    )

    existing = None if overwrite else load_output_dataset_if_valid(context.output_dataset_path)
    if existing is not None:
        log_event(
            logger,
            "curc_run_viirs_snpp_inversion_task",
            stage="curc_runtime",
            event_type="summary",
            status="loaded_existing",
            failure_code="none",
            retry_recommended=False,
            output_shape=list(existing["raw_viewable_snow_fraction"].shape),
            **common_fields,
        )
        existing.close()
        return {
            "status": "loaded_existing",
            "context": asdict(context),
        }

    missing = summarize_viirs_snpp_runtime_requirements(context)
    if dry_run:
        ready = not any(missing.values())
        status = "dry_run_ready" if ready else "dry_run_missing_inputs"
        failure_fields = {"failure_code": "none", "retry_recommended": False} if ready else _failure_fields_from_missing_inputs(missing)
        log_event(
            logger,
            "curc_run_viirs_snpp_inversion_task",
            stage="curc_runtime",
            event_type="summary",
            status=status,
            **failure_fields,
            staged_reflectance_paths=list(context.staged_reflectance_paths),
            r0_path=context.r0_path,
            lut_file=context.lut_file,
            canopy_fraction_path=context.canopy_fraction_path,
            ice_fraction_path=context.ice_fraction_path,
            missing_inputs=missing,
            **common_fields,
        )
        return {
            "status": status,
            "context": asdict(context),
            "missing_inputs": missing,
            **failure_fields,
        }

    validate_viirs_snpp_runtime_context(context)

    try:
        run_kwargs: dict[str, Any] = {
            "lut_file": context.lut_file,
            "execution_profile": execution_profile,
            "logger": logger,
        }
        if context.canopy_fraction_path is not None:
            run_kwargs["canopy_fraction"] = context.canopy_fraction_path
        if context.ice_fraction_path is not None:
            run_kwargs["ice_fraction"] = context.ice_fraction_path

        results = run_viirs_inversion(
            context.staged_reflectance_paths[0],
            context.r0_path,
            **run_kwargs,
        )
        written_path = write_output_dataset(results, context.output_dataset_path)
        log_event(
            logger,
            "curc_run_viirs_snpp_inversion_task",
            stage="curc_runtime",
            event_type="summary",
            status="completed",
            failure_code="none",
            retry_recommended=False,
            output_shape=list(results["raw_viewable_snow_fraction"].shape),
            staged_reflectance_paths=list(context.staged_reflectance_paths),
            r0_path=context.r0_path,
            lut_file=context.lut_file,
            output_path=str(written_path),
            **common_fields,
        )
        return {
            "status": "completed",
            "context": asdict(context),
            "written_path": str(written_path),
            "failure_code": "none",
            "retry_recommended": False,
        }
    except Exception as exc:
        failure_fields = _classify_runtime_exception(exc, slurm_fields=slurm_fields)
        log_event(
            logger,
            "curc_run_viirs_snpp_inversion_task",
            stage="curc_runtime",
            event_type="summary",
            status="failed",
            **failure_fields,
            **common_fields,
        )
        raise
