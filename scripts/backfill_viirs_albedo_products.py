#!/usr/bin/env python3
"""Backfill LUT-derived albedo, radiative forcing, and delta VIS into VIIRS outputs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import re
import shlex
import sys
from time import perf_counter

from netCDF4 import Dataset as NetCDFDataset
import numpy as np
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spires.albedo import (
    ALBEDO_PRODUCT_NAMES,
    DEFAULT_ALBEDO_LUT_PATH,
    DEFAULT_RADIATIVE_FORCING_LUT_PATH,
    DELTA_VIS_PRODUCT_NAME,
    RADIATIVE_FORCING_PRODUCT_NAME,
    AlbedoLuts,
    AlbedoProductFlags,
    RadiativeForcingLuts,
    generate_albedo_products,
    load_albedo_luts,
    load_radiative_forcing_luts,
    validate_requested_lut_paths,
)
from spires.sensors.full_workflow import as_yx_dataarray, open_dataarray_like


DEFAULT_OUTPUT_ROOTS = (
    Path("/scratch/alpine/ropa5718/spipy/output/viirs"),
    Path("/scratch/alpine/ropa5718/spipy/output/all_masks_pre_inversion"),
    Path("/scratch/alpine/ropa5718/spipy/output/no_masks_pre_inversion"),
)
DEFAULT_TERRAIN_ROOT = Path("/scratch/alpine/ropa5718/spipy/input/viirs")
DEFAULT_LOG_DIR = Path("/scratch/alpine/ropa5718/spipy/logs/albedo_backfill")

OUTPUT_NAME_RE = re.compile(
    r"(?P<platform>snpp|noaa20|noaa21)_raw_output_"
    r"(?P<tile>h\d{2}v\d{2})_(?P<date>\d{8})\.nc$"
)
WATER_YEAR_PART_RE = re.compile(r"^wy(?P<year>\d{4})$")


@dataclass(frozen=True)
class BackfillTask:
    output_path: Path
    platform: str
    tile: str
    date: str


@dataclass(frozen=True)
class LoadedLuts:
    albedo: AlbedoLuts | None
    radiative_forcing: RadiativeForcingLuts | None


def _split_csv(value: str | None) -> set[str] | None:
    if value is None or value.strip() == "":
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def _product_names(flags: AlbedoProductFlags) -> tuple[str, ...]:
    return flags.requested_variables


def _has_complete_products(output_path: Path, flags: AlbedoProductFlags) -> bool:
    try:
        with NetCDFDataset(output_path, "r") as output:
            return all(name in output.variables for name in _product_names(flags))
    except Exception:
        return False


def _metadata_from_output_name(output_path: Path) -> tuple[str, str, str] | None:
    match = OUTPUT_NAME_RE.match(output_path.name)
    if match is None:
        return None
    return match.group("platform"), match.group("tile"), match.group("date")


def _output_in_water_year(output_path: Path, water_year_parts: set[str] | None) -> bool:
    if water_year_parts is None:
        return True
    return any(part in water_year_parts for part in output_path.parts)


def discover_tasks(
    *,
    output_roots: tuple[Path, ...],
    flags: AlbedoProductFlags,
    platforms: set[str] | None = None,
    tiles: set[str] | None = None,
    water_years: set[str] | None = None,
    limit: int | None = None,
    only_missing: bool = False,
) -> tuple[list[BackfillTask], list[tuple[Path, str]]]:
    tasks: list[BackfillTask] = []
    skipped: list[tuple[Path, str]] = []
    water_year_parts = None if water_years is None else {f"wy{year}" for year in water_years}

    for root in output_roots:
        if not root.exists():
            skipped.append((root, "output_root_missing"))
            continue
        for output_path in sorted(root.rglob("*.nc")):
            parsed = _metadata_from_output_name(output_path)
            if parsed is None:
                skipped.append((output_path, "unrecognized_output_filename"))
                continue
            platform, tile, ymd = parsed
            if platforms is not None and platform not in platforms:
                continue
            if tiles is not None and tile not in tiles:
                continue
            if not _output_in_water_year(output_path, water_year_parts):
                continue
            if only_missing and _has_complete_products(output_path, flags):
                skipped.append((output_path, "existing_albedo_products"))
                continue

            tasks.append(BackfillTask(output_path=output_path, platform=platform, tile=tile, date=ymd))
            if limit is not None and len(tasks) >= limit:
                return tasks, skipped

    return tasks, skipped


def select_task_chunk(tasks: list[BackfillTask], *, chunk_index: int | None, chunk_count: int | None) -> list[BackfillTask]:
    if chunk_index is None and chunk_count is None:
        return tasks
    if chunk_index is None or chunk_count is None:
        raise ValueError("chunk_index and chunk_count must be provided together")
    if chunk_count <= 0:
        raise ValueError("chunk_count must be positive")
    if chunk_index < 0 or chunk_index >= chunk_count:
        raise ValueError("chunk_index must satisfy 0 <= chunk_index < chunk_count")
    return [task for index, task in enumerate(tasks) if index % chunk_count == chunk_index]


def _terrain_paths(task: BackfillTask, terrain_root: Path) -> tuple[Path, Path]:
    tile_root = terrain_root / task.platform / "ancillary" / task.tile
    return (
        tile_root / f"{task.tile}_slope_gmted_med075.tif",
        tile_root / f"{task.tile}_aspect_gmted_med075_ccw_from_south.tif",
    )


def _load_terrain(task: BackfillTask, terrain_root: Path, template: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    slope_path, aspect_path = _terrain_paths(task, terrain_root)
    if not slope_path.exists():
        raise FileNotFoundError(f"Missing slope GeoTIFF: {slope_path}")
    if not aspect_path.exists():
        raise FileNotFoundError(f"Missing aspect GeoTIFF: {aspect_path}")

    slope = as_yx_dataarray(
        open_dataarray_like(slope_path, data_var_name="slope"),
        template,
        name="slope",
        sensor_display_name="VIIRS",
    )
    aspect = as_yx_dataarray(
        open_dataarray_like(aspect_path, data_var_name="aspect"),
        template,
        name="aspect",
        sensor_display_name="VIIRS",
    )
    return slope.load(), aspect.load()


def _required_dataset_variables(flags: AlbedoProductFlags) -> tuple[str, ...]:
    required = ["grain_size", "dust_concentration", "solar_zenith"]
    if flags.include_albedo:
        required.append("solar_azimuth")
    return tuple(required)


def _load_required_dataset(output_path: Path, flags: AlbedoProductFlags) -> xr.Dataset:
    required = _required_dataset_variables(flags)
    with xr.open_dataset(output_path) as dataset:
        missing = [name for name in required if name not in dataset]
        if missing:
            raise ValueError(f"Output is missing variables required for albedo products: {missing}")
        loaded = dataset[list(required)].load()
        loaded.attrs.update(dataset.attrs)
    return loaded


def _load_luts(
    flags: AlbedoProductFlags,
    *,
    albedo_lut_path: Path,
    radiative_forcing_lut_path: Path,
) -> LoadedLuts:
    validate_requested_lut_paths(
        flags,
        albedo_lut_path=albedo_lut_path,
        radiative_forcing_lut_path=radiative_forcing_lut_path,
    )
    return LoadedLuts(
        albedo=load_albedo_luts(albedo_lut_path) if flags.requires_albedo_lut else None,
        radiative_forcing=(
            load_radiative_forcing_luts(radiative_forcing_lut_path)
            if flags.requires_radiative_forcing_lut
            else None
        ),
    )


def _create_or_update_product_variable(
    output,
    name: str,
    values: xr.DataArray,
    *,
    overwrite: bool,
    compression_level: int | None,
) -> bool:
    if name in output.variables:
        if not overwrite:
            return False
        variable = output.variables[name]
        if variable.shape != values.shape:
            raise ValueError(f"existing {name} shape {variable.shape} != {values.shape}")
    else:
        create_kwargs: dict[str, object] = {}
        if compression_level is not None and compression_level >= 0:
            create_kwargs.update(zlib=True, complevel=int(compression_level), shuffle=True)
        try:
            variable = output.createVariable(name, "f4", ("y", "x"), **create_kwargs)
        except TypeError:
            variable = output.createVariable(name, "f4", ("y", "x"))

    variable[:, :] = np.asarray(values, dtype=np.float32)
    for attr_name, attr_value in values.attrs.items():
        if attr_value is not None:
            variable.setncattr(attr_name, attr_value)
    return True


def _write_products(
    output_path: Path,
    products: xr.Dataset,
    *,
    flags: AlbedoProductFlags,
    overwrite: bool,
    compression_level: int | None,
    albedo_lut_path: Path,
    radiative_forcing_lut_path: Path,
    slope_path: Path | None,
    aspect_path: Path | None,
) -> list[str]:
    written: list[str] = []
    with NetCDFDataset(output_path, "a") as output:
        if "y" not in output.dimensions or "x" not in output.dimensions:
            raise ValueError("output dataset must have y and x dimensions")
        expected_shape = (len(output.dimensions["y"]), len(output.dimensions["x"]))
        for name in _product_names(flags):
            if name not in products:
                continue
            if products[name].shape != expected_shape:
                raise ValueError(f"{name} shape {products[name].shape} != output grid {expected_shape}")
            if _create_or_update_product_variable(
                output,
                name,
                products[name],
                overwrite=overwrite,
                compression_level=compression_level,
            ):
                written.append(name)

        now = datetime.now(UTC).isoformat()
        output.setncattr("albedo_backfill_timestamp", now)
        output.setncattr("include_albedo", int(flags.include_albedo))
        output.setncattr("include_radiative_forcing", int(flags.include_radiative_forcing))
        output.setncattr("include_delta_vis", int(flags.include_delta_vis))
        if flags.requires_albedo_lut:
            output.setncattr("albedo_lut_file", str(albedo_lut_path))
            output.setncattr("albedo_slope_source", "" if slope_path is None else str(slope_path))
            output.setncattr("albedo_aspect_source", "" if aspect_path is None else str(aspect_path))
        if flags.requires_radiative_forcing_lut:
            output.setncattr("radiative_forcing_lut_file", str(radiative_forcing_lut_path))
    return written


def attach_albedo_products_to_output(
    task: BackfillTask,
    *,
    flags: AlbedoProductFlags,
    terrain_root: Path = DEFAULT_TERRAIN_ROOT,
    albedo_lut_path: Path = DEFAULT_ALBEDO_LUT_PATH,
    radiative_forcing_lut_path: Path = DEFAULT_RADIATIVE_FORCING_LUT_PATH,
    luts: LoadedLuts | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
    compression_level: int | None = 1,
) -> dict[str, object]:
    output_path = task.output_path.expanduser().resolve()
    if _has_complete_products(output_path, flags) and not overwrite:
        return {
            "output_path": str(output_path),
            "status": "skipped_existing",
            "existing_variables": list(_product_names(flags)),
        }

    if dry_run:
        return {
            "output_path": str(output_path),
            "status": "dry_run",
            "variables": list(_product_names(flags)),
        }

    dataset = _load_required_dataset(output_path, flags)
    slope = None
    aspect = None
    slope_path = None
    aspect_path = None
    if flags.include_albedo:
        slope_path, aspect_path = _terrain_paths(task, terrain_root)
        slope, aspect = _load_terrain(task, terrain_root, dataset["grain_size"])

    loaded_luts = luts or _load_luts(
        flags,
        albedo_lut_path=albedo_lut_path,
        radiative_forcing_lut_path=radiative_forcing_lut_path,
    )
    products = generate_albedo_products(
        dataset,
        flags=flags,
        albedo_luts=loaded_luts.albedo,
        radiative_forcing_luts=loaded_luts.radiative_forcing,
        slope=slope,
        aspect=aspect,
    ).load()

    written = _write_products(
        output_path,
        products,
        flags=flags,
        overwrite=overwrite,
        compression_level=compression_level,
        albedo_lut_path=albedo_lut_path,
        radiative_forcing_lut_path=radiative_forcing_lut_path,
        slope_path=slope_path,
        aspect_path=aspect_path,
    )
    return {
        "output_path": str(output_path),
        "status": "updated" if written else "skipped_existing",
        "variables": written,
    }


def _run_one(
    task: BackfillTask,
    *,
    flags: AlbedoProductFlags,
    terrain_root: Path,
    albedo_lut_path: Path,
    radiative_forcing_lut_path: Path,
    luts: LoadedLuts | None,
    dry_run: bool,
    overwrite: bool,
    compression_level: int | None,
) -> dict[str, object]:
    started = perf_counter()
    try:
        result = attach_albedo_products_to_output(
            task,
            flags=flags,
            terrain_root=terrain_root,
            albedo_lut_path=albedo_lut_path,
            radiative_forcing_lut_path=radiative_forcing_lut_path,
            luts=luts,
            dry_run=dry_run,
            overwrite=overwrite,
            compression_level=compression_level,
        )
        return {
            "status": str(result.get("status", "updated")),
            "platform": task.platform,
            "tile": task.tile,
            "date": task.date,
            "output_path": str(task.output_path),
            "elapsed_seconds": round(perf_counter() - started, 3),
            "result": result,
        }
    except Exception as exc:
        return {
            "status": "error",
            "platform": task.platform,
            "tile": task.tile,
            "date": task.date,
            "output_path": str(task.output_path),
            "elapsed_seconds": round(perf_counter() - started, 3),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _write_skipped(path: Path, skipped: list[tuple[Path, str]]) -> None:
    with path.open("w", encoding="ascii") as handle:
        for skipped_path, reason in skipped:
            handle.write(json.dumps({"path": str(skipped_path), "reason": reason}) + "\n")


def _compact_values_label(values: set[str], *, singular_empty: str, mixed_label: str) -> str:
    if not values:
        return singular_empty
    if len(values) == 1:
        return next(iter(values))
    if len(values) <= 3:
        return "-".join(sorted(values))
    return mixed_label


def _log_scope_label(
    tasks: list[BackfillTask],
    *,
    chunk_index: int | None,
    chunk_count: int | None,
) -> str:
    platforms = {task.platform for task in tasks}
    tiles = {task.tile for task in tasks}
    dates = sorted({task.date for task in tasks})

    parts = [
        _compact_values_label(platforms, singular_empty="no-platform", mixed_label="multi-platform"),
        _compact_values_label(tiles, singular_empty="no-tile", mixed_label="multi-tile"),
    ]
    if not dates:
        parts.append("no-dates")
    elif len(dates) == 1:
        parts.append(dates[0])
    else:
        parts.append(f"{dates[0]}-to-{dates[-1]}")

    if chunk_index is not None and chunk_count is not None:
        width = max(2, len(str(chunk_count - 1)))
        parts.append(f"chunk{chunk_index:0{width}d}-of-{chunk_count}")

    return "_".join(parts)


def _write_progress(handle, event: str, task: BackfillTask, extra: dict[str, object] | None = None) -> None:
    payload = {
        "event": event,
        "platform": task.platform,
        "tile": task.tile,
        "date": task.date,
        "output_path": str(task.output_path),
        "timestamp": datetime.now(UTC).isoformat(),
    }
    if extra:
        payload.update(extra)
    handle.write(json.dumps(payload) + "\n")
    handle.flush()


def _run_tasks(
    tasks: list[BackfillTask],
    *,
    flags: AlbedoProductFlags,
    terrain_root: Path,
    albedo_lut_path: Path,
    radiative_forcing_lut_path: Path,
    dry_run: bool,
    overwrite: bool,
    compression_level: int | None,
    results_path: Path,
    progress_path: Path,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    luts = None if dry_run else _load_luts(
        flags,
        albedo_lut_path=albedo_lut_path,
        radiative_forcing_lut_path=radiative_forcing_lut_path,
    )
    with results_path.open("w", encoding="ascii") as handle, progress_path.open("w", encoding="ascii") as progress:
        for task in tasks:
            _write_progress(progress, "task_start", task)
            result = _run_one(
                task,
                flags=flags,
                terrain_root=terrain_root,
                albedo_lut_path=albedo_lut_path,
                radiative_forcing_lut_path=radiative_forcing_lut_path,
                luts=luts,
                dry_run=dry_run,
                overwrite=overwrite,
                compression_level=compression_level,
            )
            counts[result["status"]] = counts.get(result["status"], 0) + 1
            handle.write(json.dumps(result) + "\n")
            handle.flush()
            _write_progress(progress, "task_done", task, {"status": result["status"]})
    return counts


def _bool_arg(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def _render_sbatch_command(args: argparse.Namespace, *, repo_root: Path) -> list[str]:
    array_count = int(args.sbatch_array_count)
    if array_count <= 0:
        raise ValueError("--sbatch-array-count must be positive")
    array_spec = f"0-{array_count - 1}"
    if args.sbatch_max_concurrent is not None:
        array_spec += f"%{args.sbatch_max_concurrent}"

    script = repo_root / "scripts" / "backfill_viirs_albedo_products.py"
    script_parts = [
        "mamba",
        "run",
        "-n",
        "spipy14",
        "python",
        str(script),
    ]
    for root in args.output_roots or []:
        script_parts.extend(["--output-root", str(root)])
    slurm_task_id_placeholder = "__SLURM_ARRAY_TASK_ID__"
    script_parts.extend(
        [
            "--terrain-root",
            str(args.terrain_root),
            "--log-dir",
            str(args.log_dir),
            "--albedo-lut-path",
            str(args.albedo_lut_path),
            "--radiative-forcing-lut-path",
            str(args.radiative_forcing_lut_path),
            "--chunk-index",
            slurm_task_id_placeholder,
            "--chunk-count",
            str(array_count),
            "--compression-level",
            str(args.compression_level),
            "--include-albedo",
            str(args.include_albedo).lower(),
            "--include-radiative-forcing",
            str(args.include_radiative_forcing).lower(),
            "--include-delta-vis",
            str(args.include_delta_vis).lower(),
        ]
    )
    if args.platforms:
        script_parts.extend(["--platforms", args.platforms])
    if args.tiles:
        script_parts.extend(["--tiles", args.tiles])
    if args.water_years:
        script_parts.extend(["--water-years", args.water_years])
    if args.only_missing:
        script_parts.append("--only-missing")
    if args.overwrite:
        script_parts.append("--overwrite")
    if args.dry_run:
        script_parts.append("--dry-run")
    def _quote_wrapped_part(part: str) -> str:
        if part == slurm_task_id_placeholder:
            return "${SLURM_ARRAY_TASK_ID}"
        return shlex.quote(part)

    wrapped_command = "module load miniforge && " + " ".join(_quote_wrapped_part(part) for part in script_parts)

    command = [
        "sbatch",
        "--parsable",
        "--job-name",
        args.sbatch_job_name,
        "--partition",
        args.sbatch_partition,
        "--account",
        args.sbatch_account,
        "--time",
        args.sbatch_time,
        "--mem",
        args.sbatch_mem,
        "--cpus-per-task",
        str(args.sbatch_cpus_per_task),
        "--array",
        array_spec,
        "--output",
        str(args.log_dir / f"{args.sbatch_job_name}_%A_%a.out"),
        "--wrap",
        wrapped_command,
    ]
    return command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", action="append", type=Path, dest="output_roots")
    parser.add_argument("--terrain-root", type=Path, default=DEFAULT_TERRAIN_ROOT)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--albedo-lut-path", type=Path, default=DEFAULT_ALBEDO_LUT_PATH)
    parser.add_argument("--radiative-forcing-lut-path", type=Path, default=DEFAULT_RADIATIVE_FORCING_LUT_PATH)
    parser.add_argument("--platforms", default=None, help="Comma-separated subset: snpp,noaa20,noaa21")
    parser.add_argument("--tiles", default=None, help="Comma-separated tile subset, e.g. h08v05,h09v05")
    parser.add_argument("--water-years", default=None, help="Comma-separated water years, e.g. 2024")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only-missing", action="store_true")
    parser.add_argument("--chunk-index", type=int, default=None)
    parser.add_argument("--chunk-count", type=int, default=None)
    parser.add_argument("--compression-level", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--include-albedo", type=_bool_arg, default=True)
    parser.add_argument("--include-radiative-forcing", type=_bool_arg, default=True)
    parser.add_argument("--include-delta-vis", type=_bool_arg, default=True)
    parser.add_argument("--print-sbatch", action="store_true")
    parser.add_argument("--sbatch-array-count", type=int, default=60)
    parser.add_argument("--sbatch-max-concurrent", type=int, default=20)
    parser.add_argument("--sbatch-job-name", default="spipy-viirs-albedo-backfill")
    parser.add_argument("--sbatch-partition", default="blanca-rittger,blanca-snow")
    parser.add_argument("--sbatch-account", default="blanca-rittger")
    parser.add_argument("--sbatch-time", default="1-00:00:00")
    parser.add_argument("--sbatch-mem", default="16G")
    parser.add_argument("--sbatch-cpus-per-task", type=int, default=1)
    args = parser.parse_args(argv)

    args.log_dir.mkdir(parents=True, exist_ok=True)
    if args.print_sbatch:
        command = _render_sbatch_command(args, repo_root=REPO_ROOT)
        print(" ".join(shlex.quote(part) for part in command))
        return 0

    flags = AlbedoProductFlags(
        include_albedo=bool(args.include_albedo),
        include_radiative_forcing=bool(args.include_radiative_forcing),
        include_delta_vis=bool(args.include_delta_vis),
    )
    if not flags.requested_variables:
        raise ValueError("At least one product flag must be enabled")

    output_roots = tuple(args.output_roots) if args.output_roots else DEFAULT_OUTPUT_ROOTS
    tasks, skipped = discover_tasks(
        output_roots=output_roots,
        flags=flags,
        platforms=_split_csv(args.platforms),
        tiles=_split_csv(args.tiles),
        water_years=_split_csv(args.water_years),
        limit=args.limit,
        only_missing=args.only_missing,
    )
    discovered_tasks = len(tasks)
    tasks = select_task_chunk(tasks, chunk_index=args.chunk_index, chunk_count=args.chunk_count)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    log_scope = _log_scope_label(tasks, chunk_index=args.chunk_index, chunk_count=args.chunk_count)
    results_path = args.log_dir / f"viirs_albedo_backfill_results_{log_scope}_{timestamp}.jsonl"
    skipped_path = args.log_dir / f"viirs_albedo_backfill_skipped_{log_scope}_{timestamp}.jsonl"
    progress_path = args.log_dir / f"viirs_albedo_backfill_progress_{log_scope}_{timestamp}.jsonl"
    _write_skipped(skipped_path, skipped)

    summary = {
        "output_roots": [str(path) for path in output_roots],
        "terrain_root": str(args.terrain_root),
        "discovered_tasks": discovered_tasks,
        "tasks": len(tasks),
        "skipped": len(skipped),
        "requested_variables": list(flags.requested_variables),
        "chunk_index": args.chunk_index,
        "chunk_count": args.chunk_count,
        "dry_run": args.dry_run,
        "overwrite": args.overwrite,
        "only_missing": args.only_missing,
        "log_scope": log_scope,
        "results_path": str(results_path),
        "skipped_path": str(skipped_path),
        "progress_path": str(progress_path),
    }
    print(json.dumps({"event": "start", **summary}, indent=2), flush=True)

    started = perf_counter()
    counts = _run_tasks(
        tasks,
        flags=flags,
        terrain_root=args.terrain_root,
        albedo_lut_path=args.albedo_lut_path,
        radiative_forcing_lut_path=args.radiative_forcing_lut_path,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        compression_level=None if args.compression_level < 0 else args.compression_level,
        results_path=results_path,
        progress_path=progress_path,
    )
    final = {
        "event": "summary",
        **summary,
        "status_counts": counts,
        "elapsed_seconds": round(perf_counter() - started, 3),
    }
    print(json.dumps(final, indent=2), flush=True)
    return 1 if counts.get("error", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
