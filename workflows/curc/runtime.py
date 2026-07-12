"""Runtime helpers for CURC Slurm array tasks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import traceback
from typing import Any

from spires.albedo import DEFAULT_ALBEDO_LUT_PATH, DEFAULT_RADIATIVE_FORCING_LUT_PATH
from spires.logging_utils import configure_spires_file_logger, log_event, remove_empty_log_file
from spires.sensors.io import load_output_dataset_if_valid, write_output_dataset
from spires.sensors.viirs.workflow import run_viirs_inversion
import xarray as xr

from workflows.curc.config import CurcWorkflowConfig, SlurmProfile
from workflows.curc.paths import r0_dataset_path
from workflows.curc.steps import InversionTaskPlan
from workflows.curc.task_manifest import load_inversion_array_manifest, resolve_inversion_task_from_manifest


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
    cloud_mask_path: str | None
    water_mask_path: str | None
    playa_mask_path: str | None
    lut_file: str
    include_albedo: bool
    include_radiative_forcing: bool
    include_delta_vis: bool
    albedo_lut_path: str | None
    radiative_forcing_lut_path: str | None
    terrain_ancillary_root: str | None
    slope_path: str | None
    aspect_path: str | None
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


def _infer_tile_ancillary_path(ancillary_root: Path, tile: str, stems: tuple[str, ...]) -> Path | None:
    candidates: list[Path] = []
    for stem in stems:
        candidates.extend(
            [
                ancillary_root / f"{stem}.zarr",
                ancillary_root / f"{stem}.tif",
                ancillary_root / f"{stem}.tiff",
                ancillary_root / f"{stem}.nc",
                ancillary_root / f"{tile}_{stem}.zarr",
                ancillary_root / f"{tile}_{stem}.tif",
                ancillary_root / f"{tile}_{stem}.tiff",
                ancillary_root / f"{tile}_{stem}.nc",
            ]
        )
    return _first_existing_path(candidates)


def _cloud_mask_path(scratch_root: Path, task: InversionTaskPlan) -> Path | None:
    date_token = task.date.replace("-", "")
    return _first_existing_path(
        [
            scratch_root
            / "input"
            / task.sensor
            / task.platform
            / "ancillary"
            / "cloud"
            / task.tile
            / str(task.water_year)
            / f"{task.platform}_{task.tile}_{date_token}_cloud_mask.tif",
            scratch_root
            / "input"
            / task.sensor
            / task.platform
            / "ancillary"
            / task.tile
            / "cloud"
            / str(task.water_year)
            / f"{task.platform}_{task.tile}_{date_token}_cloud_mask.tif",
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


def _inversion_output_filename(task: InversionTaskPlan) -> str:
    date_token = task.date.replace("-", "")
    return f"{task.platform}_raw_output_{task.tile}_{date_token}.nc"


def _inversion_output_dataset_path(task: InversionTaskPlan) -> Path:
    return Path(task.output_path).expanduser().resolve() / _inversion_output_filename(task)


def resolve_runtime_task_log_path(task: InversionTaskPlan, *, slurm_job_id: str | None = None) -> Path:
    """Resolve runtime log path and add a per-job suffix when available."""
    base = Path(task.log_path).expanduser().resolve()
    if slurm_job_id is None:
        return base
    return base.with_name(f"{base.stem}_job{slurm_job_id}{base.suffix}")


def resolve_slurm_stdout_path(
    manifest_path: str | Path,
    *,
    slurm_job_name: str | None = None,
    slurm_array_job_id: str | None = None,
    slurm_array_task_id: str | None = None,
) -> Path | None:
    """Resolve the Slurm stdout path for the current array task when identifiable."""
    if slurm_array_job_id is None or slurm_array_task_id is None:
        return None
    payload = load_inversion_array_manifest(manifest_path)
    job_name = slurm_job_name if slurm_job_name is not None else str(payload["job_name"])
    slurm_profile = SlurmProfile.from_payload(payload.get("slurm_profile"))
    stdout_dir = Path(manifest_path).expanduser().resolve().parent if slurm_profile.output_dir is None else slurm_profile.output_dir
    return stdout_dir.expanduser().resolve() / f"{job_name}_{slurm_array_job_id}_{slurm_array_task_id}.out"


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
    manifest_payload = load_inversion_array_manifest(manifest_path)
    scratch_root = infer_scratch_root_from_output_path(task.output_path)
    reflectance_root = scratch_root / "input" / task.sensor / task.platform / "reflectance" / task.tile / str(task.water_year)
    staged_reflectance_paths = tuple(str(reflectance_root / Path(path).name) for path in task.source_paths)
    ancillary_root = scratch_root / "input" / task.sensor / task.platform / "ancillary" / task.tile
    r0_root = scratch_root / "input" / task.sensor / task.platform / "ancillary" / "r0" / task.tile / str(task.r0_year)
    resolved_lut_file = default_viirs_lut_file(task.platform) if lut_file is None else Path(lut_file).expanduser().resolve()
    include_albedo = bool(manifest_payload.get("include_albedo", True))
    include_radiative_forcing = bool(manifest_payload.get("include_radiative_forcing", True))
    include_delta_vis = bool(manifest_payload.get("include_delta_vis", True))
    albedo_lut_path = manifest_payload.get("albedo_lut_path")
    radiative_forcing_lut_path = manifest_payload.get("radiative_forcing_lut_path")
    terrain_ancillary_root = manifest_payload.get("terrain_ancillary_root")
    if albedo_lut_path is None and include_albedo:
        albedo_lut_path = str(DEFAULT_ALBEDO_LUT_PATH)
    if radiative_forcing_lut_path is None and (include_radiative_forcing or include_delta_vis):
        radiative_forcing_lut_path = str(DEFAULT_RADIATIVE_FORCING_LUT_PATH)
    if terrain_ancillary_root is None:
        terrain_ancillary_root = str(scratch_root / "input" / task.sensor / task.platform / "ancillary")
    terrain_root = Path(str(terrain_ancillary_root)).expanduser()
    slope_path = terrain_root / task.tile / f"{task.tile}_slope_gmted_med075.tif"
    aspect_path = terrain_root / task.tile / f"{task.tile}_aspect_gmted_med075_ccw_from_south.tif"
    r0_path = _runtime_r0_dataset_path(scratch_root, task)
    canopy_fraction_path = _infer_static_fraction_path(ancillary_root, "canopy_fraction")
    ice_fraction_path = (
        _infer_static_fraction_path(ancillary_root, "glacier_ice_fraction")
        or _infer_tile_ancillary_path(ancillary_root, task.tile, ("ice_rgi60_202309", "ice"))
    )
    cloud_mask_path = _cloud_mask_path(scratch_root, task)
    water_mask_path = _infer_tile_ancillary_path(ancillary_root, task.tile, ("water_mod44_50", "water_mask", "water"))
    playa_mask_path = _infer_tile_ancillary_path(
        ancillary_root,
        task.tile,
        ("stc_false_positive_manual_20230920_mod44_50", "stc_false_positive", "playa"),
    )

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
        cloud_mask_path=str(cloud_mask_path) if cloud_mask_path is not None else None,
        water_mask_path=str(water_mask_path) if water_mask_path is not None else None,
        playa_mask_path=str(playa_mask_path) if playa_mask_path is not None else None,
        lut_file=str(resolved_lut_file),
        include_albedo=include_albedo,
        include_radiative_forcing=include_radiative_forcing,
        include_delta_vis=include_delta_vis,
        albedo_lut_path=str(albedo_lut_path) if albedo_lut_path is not None else None,
        radiative_forcing_lut_path=str(radiative_forcing_lut_path) if radiative_forcing_lut_path is not None else None,
        terrain_ancillary_root=str(terrain_ancillary_root) if terrain_ancillary_root is not None else None,
        slope_path=str(slope_path),
        aspect_path=str(aspect_path),
        output_path=task.output_path,
        output_dataset_path=str(_inversion_output_dataset_path(task)),
        log_path=task.log_path,
    )


def summarize_viirs_snpp_runtime_requirements(context: InversionRuntimeContext) -> dict[str, list[str]]:
    """Return missing required inputs for one runtime context."""
    missing: dict[str, list[str]] = {
        "staged_reflectance_paths": [],
        "r0_path": [],
        "lut_file": [],
        "cloud_mask_path": [],
        "water_mask_path": [],
        "ice_fraction_path": [],
        "playa_mask_path": [],
        "albedo_lut_path": [],
        "radiative_forcing_lut_path": [],
        "slope_path": [],
        "aspect_path": [],
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
    if context.include_albedo:
        if context.albedo_lut_path is None or not Path(context.albedo_lut_path).exists():
            missing["albedo_lut_path"].append(str(DEFAULT_ALBEDO_LUT_PATH))
        if context.slope_path is None or not Path(context.slope_path).exists():
            missing["slope_path"].append(
                str(Path(context.terrain_ancillary_root or context.ancillary_root) / context.task.tile / f"{context.task.tile}_slope_gmted_med075.tif")
            )
        if context.aspect_path is None or not Path(context.aspect_path).exists():
            missing["aspect_path"].append(
                str(
                    Path(context.terrain_ancillary_root or context.ancillary_root)
                    / context.task.tile
                    / f"{context.task.tile}_aspect_gmted_med075_ccw_from_south.tif"
                )
            )
    if context.include_radiative_forcing or context.include_delta_vis:
        if context.radiative_forcing_lut_path is None or not Path(context.radiative_forcing_lut_path).exists():
            missing["radiative_forcing_lut_path"].append(str(DEFAULT_RADIATIVE_FORCING_LUT_PATH))
    if context.cloud_mask_path is None or not Path(context.cloud_mask_path).exists():
        missing["cloud_mask_path"].append(
            str(
                Path(context.scratch_root)
                / "input"
                / context.task.sensor
                / context.task.platform
                / "ancillary"
                / "cloud"
                / context.task.tile
                / str(context.task.water_year)
                / f"{context.task.platform}_{context.task.tile}_{context.task.date.replace('-', '')}_cloud_mask.tif"
            )
        )
    if context.water_mask_path is None or not Path(context.water_mask_path).exists():
        missing["water_mask_path"].append(str(Path(context.ancillary_root) / f"{context.task.tile}_water_mod44_50.tif"))
    if context.ice_fraction_path is None or not Path(context.ice_fraction_path).exists():
        missing["ice_fraction_path"].append(str(Path(context.ancillary_root) / f"{context.task.tile}_ice_rgi60_202309.tif"))
    if context.playa_mask_path is None or not Path(context.playa_mask_path).exists():
        missing["playa_mask_path"].append(
            str(Path(context.ancillary_root) / f"{context.task.tile}_stc_false_positive_manual_20230920_mod44_50.tif")
        )
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
    for key in (
        "albedo_lut_path",
        "radiative_forcing_lut_path",
        "slope_path",
        "aspect_path",
        "cloud_mask_path",
        "water_mask_path",
        "ice_fraction_path",
        "playa_mask_path",
    ):
        if missing.get(key):
            return {
                "failure_code": f"missing_{key}",
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
    apply_valid_inversion_mask: bool | None = None,
    mask_low_reflectance_for_inversion: bool | None = None,
    low_reflectance_threshold: float | None = None,
    include_grouped_reflectance_rmse: bool | None = None,
    use_grouping: bool | None = None,
    grouping_method: str | None = None,
    include_albedo: bool | None = None,
    include_radiative_forcing: bool | None = None,
    include_delta_vis: bool | None = None,
    albedo_lut_path: str | Path | None = None,
    radiative_forcing_lut_path: str | Path | None = None,
    terrain_ancillary_root: str | Path | None = None,
) -> dict[str, Any]:
    """Execute or dry-run one manifest-backed VIIRS SNPP inversion task."""
    context = build_viirs_snpp_inversion_runtime_context(
        manifest_path,
        task_index=task_index,
        lut_file=lut_file,
    )
    slurm_fields = slurm_metadata_from_env()
    slurm_stdout_path = resolve_slurm_stdout_path(
        context.manifest_path,
        slurm_job_name=None if slurm_fields.get("slurm_job_name") is None else str(slurm_fields["slurm_job_name"]),
        slurm_array_job_id=None if slurm_fields.get("slurm_array_job_id") is None else str(slurm_fields["slurm_array_job_id"]),
        slurm_array_task_id=None if slurm_fields.get("slurm_array_task_id") is None else str(slurm_fields["slurm_array_task_id"]),
    )
    runtime_log_path = resolve_runtime_task_log_path(
        context.task,
        slurm_job_id=None if slurm_fields.get("slurm_job_id") is None else str(slurm_fields["slurm_job_id"]),
    )
    logger = configure_spires_file_logger(
        runtime_log_path,
        logger_name=_task_logger_name(context),
        mode="a",
        log_to_stdout=False,
    )
    preamble_fields = {
        "platform": context.task.platform,
        "tile": context.task.tile,
        "date": context.task.date,
        "log_path": str(runtime_log_path),
        "manifest_path": context.manifest_path,
        "output_dataset_path": context.output_dataset_path,
        "r0_year": context.task.r0_year,
        "retry_count": context.task.retry_count,
        "sensor": context.task.sensor,
        **slurm_fields,
        "task_index": context.task.task_index,
        "water_year": context.task.water_year,
    }
    common_fields = {
        "date": context.task.date,
        "dry_run": dry_run,
        **slurm_fields,
    }
    log_event(
        logger,
        "curc_runtime_context",
        stage="curc_runtime",
        event_type="context",
        status="ready",
        title="SUBMISSION PARAMETERS",
        scope=True,
        **preamble_fields,
    )
    log_event(
        logger,
        "curc_run_viirs_snpp_inversion_task",
        stage="curc_runtime",
        event_type="start",
        status="started",
        scope=True,
        task_index=context.task.task_index,
        retry_count=context.task.retry_count,
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
            scope=True,
            failure_code="none",
            retry_recommended=False,
            output_shape=list(existing["raw_viewable_snow_fraction"].shape),
            task_index=context.task.task_index,
            retry_count=context.task.retry_count,
            **common_fields,
        )
        existing.close()
        if slurm_stdout_path is not None:
            remove_empty_log_file(slurm_stdout_path)
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
            scope=True,
            **failure_fields,
            staged_reflectance_paths=list(context.staged_reflectance_paths),
            r0_path=context.r0_path,
            lut_file=context.lut_file,
            missing_inputs=missing,
            task_index=context.task.task_index,
            retry_count=context.task.retry_count,
            **common_fields,
        )
        return {
            "status": status,
            "context": asdict(context),
            "missing_inputs": missing,
            **failure_fields,
        }

    validate_viirs_snpp_runtime_context(context)
    manifest_payload = load_inversion_array_manifest(context.manifest_path)
    manifest_apply_valid_mask = bool(manifest_payload.get("apply_valid_inversion_mask", False))
    manifest_mask_low_reflectance = bool(manifest_payload.get("mask_low_reflectance_for_inversion", False))
    manifest_low_reflectance_threshold = float(manifest_payload.get("low_reflectance_threshold", 0.1))
    manifest_include_grouped_reflectance_rmse = bool(manifest_payload.get("include_grouped_reflectance_rmse", False))
    manifest_use_grouping = bool(manifest_payload.get("use_grouping", True))
    manifest_grouping_method = str(manifest_payload.get("grouping_method", "chunk_bin_mean"))
    manifest_include_albedo = bool(manifest_payload.get("include_albedo", True))
    manifest_include_radiative_forcing = bool(manifest_payload.get("include_radiative_forcing", True))
    manifest_include_delta_vis = bool(manifest_payload.get("include_delta_vis", True))
    resolved_include_albedo = manifest_include_albedo if include_albedo is None else include_albedo
    resolved_include_radiative_forcing = (
        manifest_include_radiative_forcing if include_radiative_forcing is None else include_radiative_forcing
    )
    resolved_include_delta_vis = manifest_include_delta_vis if include_delta_vis is None else include_delta_vis
    resolved_albedo_lut_path = context.albedo_lut_path if albedo_lut_path is None else str(Path(albedo_lut_path).expanduser())
    resolved_radiative_forcing_lut_path = (
        context.radiative_forcing_lut_path
        if radiative_forcing_lut_path is None
        else str(Path(radiative_forcing_lut_path).expanduser())
    )
    resolved_terrain_ancillary_root = (
        context.terrain_ancillary_root
        if terrain_ancillary_root is None
        else str(Path(terrain_ancillary_root).expanduser())
    )

    try:
        run_kwargs: dict[str, Any] = {
            "lut_file": context.lut_file,
            "execution_profile": execution_profile,
            "logger": logger,
            "apply_valid_inversion_mask": manifest_apply_valid_mask if apply_valid_inversion_mask is None else apply_valid_inversion_mask,
            "mask_low_reflectance_for_inversion": (
                manifest_mask_low_reflectance
                if mask_low_reflectance_for_inversion is None
                else mask_low_reflectance_for_inversion
            ),
            "low_reflectance_threshold": (
                manifest_low_reflectance_threshold
                if low_reflectance_threshold is None
                else low_reflectance_threshold
            ),
            "include_grouped_reflectance_rmse": (
                manifest_include_grouped_reflectance_rmse
                if include_grouped_reflectance_rmse is None
                else include_grouped_reflectance_rmse
            ),
            "use_grouping": manifest_use_grouping if use_grouping is None else use_grouping,
            "grouping_method": manifest_grouping_method if grouping_method is None else grouping_method,
            "include_albedo": resolved_include_albedo,
            "include_radiative_forcing": resolved_include_radiative_forcing,
            "include_delta_vis": resolved_include_delta_vis,
            "albedo_lut_path": resolved_albedo_lut_path,
            "radiative_forcing_lut_path": resolved_radiative_forcing_lut_path,
            "terrain_ancillary_root": resolved_terrain_ancillary_root,
        }
        if resolved_include_albedo:
            run_kwargs["slope_path"] = context.slope_path
            run_kwargs["aspect_path"] = context.aspect_path
        if context.canopy_fraction_path is not None:
            run_kwargs["canopy_fraction"] = context.canopy_fraction_path
        if context.ice_fraction_path is not None:
            run_kwargs["ice_fraction"] = context.ice_fraction_path
        external_inversion_mask_sources = {}
        if context.water_mask_path is not None:
            external_inversion_mask_sources["water_external"] = context.water_mask_path
        if context.ice_fraction_path is not None:
            external_inversion_mask_sources["ice_external"] = context.ice_fraction_path
        if context.playa_mask_path is not None:
            external_inversion_mask_sources["stc_false_positive"] = context.playa_mask_path
        if external_inversion_mask_sources:
            run_kwargs["external_inversion_mask_sources"] = external_inversion_mask_sources
        if context.cloud_mask_path is not None:
            run_kwargs["cloud_mask_source"] = context.cloud_mask_path

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
            scope=True,
            failure_code="none",
            retry_recommended=False,
            output_shape=list(results["raw_viewable_snow_fraction"].shape),
            output_path=str(written_path),
            task_index=context.task.task_index,
            retry_count=context.task.retry_count,
            **common_fields,
        )
        if slurm_stdout_path is not None:
            remove_empty_log_file(slurm_stdout_path)
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
            scope=True,
            **failure_fields,
            task_index=context.task.task_index,
            retry_count=context.task.retry_count,
            **common_fields,
        )
        raise
