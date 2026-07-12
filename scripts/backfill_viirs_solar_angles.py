#!/usr/bin/env python3
"""Attach VIIRS solar geometry to existing inversion outputs.

This intentionally does not run VIIRS scene preparation. It reads only the
1 km solar zenith/azimuth fields from the matching VNP09GA/VJ109GA/VJ209GA
HDF file, expands them to the existing output grid, and appends the two
variables to the output NetCDF.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import re
import sys
from time import perf_counter

import h5py
from netCDF4 import Dataset as NetCDFDataset
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spires.sensors.base import collect_attrs, read_scaled_array


DEFAULT_OUTPUT_ROOTS = (
    Path("/scratch/alpine/ropa5718/spipy/output/all_masks_pre_inversion"),
    Path("/scratch/alpine/ropa5718/spipy/output/no_masks_pre_inversion"),
)
DEFAULT_INPUT_ROOT = Path("/scratch/alpine/ropa5718/spipy/input/viirs")
DEFAULT_LOG_DIR = Path("/scratch/alpine/ropa5718/spipy/logs/solar_angle_backfill")

VIIRS_1KM_GRID = "HDFEOS/GRIDS/VIIRS_Grid_1km_2D"
VIIRS_SOLAR_FIELDS = {
    "solar_zenith": "SolarZenith_1",
    "solar_azimuth": "SolarAzimuth_1",
}

PLATFORM_PRODUCT = {
    "snpp": "VNP09GA",
    "noaa20": "VJ109GA",
    "noaa21": "VJ209GA",
}

OUTPUT_NAME_RE = re.compile(
    r"(?P<platform>snpp|noaa20|noaa21)_raw_output_"
    r"(?P<tile>h\d{2}v\d{2})_(?P<date>\d{8})\.nc$"
)
WATER_YEAR_PART_RE = re.compile(r"^wy(?P<year>\d{4})$")


@dataclass(frozen=True)
class BackfillTask:
    output_path: Path
    scene_path: Path
    platform: str
    tile: str
    date: str


@dataclass(frozen=True)
class SolarGeometry:
    solar_zenith: np.ndarray
    solar_azimuth: np.ndarray
    attrs: dict[str, dict[str, object]]


def _split_csv(value: str | None) -> set[str] | None:
    if value is None or value.strip() == "":
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def _data_field_path(dataset_name: str) -> str:
    return f"{VIIRS_1KM_GRID}/Data Fields/{dataset_name}"


def _read_solar_field(hdf: h5py.File, dataset_name: str) -> tuple[np.ndarray, dict[str, object]]:
    dataset = hdf[_data_field_path(dataset_name)]
    array = read_scaled_array(
        dataset,
        apply_scale=True,
        mask_fill=True,
        mask_valid_range=True,
        dtype=np.float32,
    )
    return np.asarray(array, dtype=np.float32), collect_attrs(dataset)


def _grid_expansion_factors(array: np.ndarray, *, y_size: int, x_size: int, name: str) -> tuple[int, int]:
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array, got shape {array.shape}")
    if array.shape == (y_size, x_size):
        return 1, 1
    if y_size % array.shape[0] != 0 or x_size % array.shape[1] != 0:
        raise ValueError(
            f"{name} shape {array.shape} cannot be expanded to output grid {(y_size, x_size)}"
        )
    return y_size // array.shape[0], x_size // array.shape[1]


def _expanded_output_rows(
    array: np.ndarray,
    *,
    y_start: int,
    y_stop: int,
    y_factor: int,
    x_factor: int,
) -> np.ndarray:
    src_start = y_start // y_factor
    src_stop = (y_stop + y_factor - 1) // y_factor
    expanded = np.repeat(np.repeat(array[src_start:src_stop], y_factor, axis=0), x_factor, axis=1)
    offset = y_start - src_start * y_factor
    return np.asarray(expanded[offset : offset + (y_stop - y_start)], dtype=np.float32)


def read_viirs_solar_geometry(scene_path: Path) -> SolarGeometry:
    arrays: dict[str, np.ndarray] = {}
    attrs: dict[str, dict[str, object]] = {}
    with h5py.File(scene_path, "r") as hdf:
        for variable_name, dataset_name in VIIRS_SOLAR_FIELDS.items():
            array, field_attrs = _read_solar_field(hdf, dataset_name)
            arrays[variable_name] = array
            attrs[variable_name] = field_attrs
    return SolarGeometry(
        solar_zenith=arrays["solar_zenith"],
        solar_azimuth=arrays["solar_azimuth"],
        attrs=attrs,
    )


def _output_grid_and_existing(output_path: Path) -> tuple[int, int, list[str]]:
    with NetCDFDataset(output_path, "r") as output:
        if "y" not in output.dimensions or "x" not in output.dimensions:
            raise ValueError("output dataset must have y and x dimensions")
        y_size = len(output.dimensions["y"])
        x_size = len(output.dimensions["x"])
        existing = [name for name in VIIRS_SOLAR_FIELDS if name in output.variables]
    return y_size, x_size, existing


def _has_complete_solar_angles(output_path: Path) -> bool:
    try:
        _, _, existing = _output_grid_and_existing(output_path)
    except Exception:
        return False
    return len(existing) == len(VIIRS_SOLAR_FIELDS)


def _set_attrs(variable, attrs: dict[str, object], *, fallback_long_name: str) -> None:
    variable.long_name = str(attrs.get("long_name", fallback_long_name))
    variable.units = str(attrs.get("units", "degrees"))


def _create_or_update_variable(
    output,
    name: str,
    source_values: np.ndarray,
    attrs: dict[str, object],
    *,
    y_size: int,
    x_size: int,
    overwrite: bool,
    block_rows: int,
    compression_level: int | None,
) -> bool:
    y_factor, x_factor = _grid_expansion_factors(source_values, y_size=y_size, x_size=x_size, name=name)
    if name in output.variables:
        if not overwrite:
            return False
        variable = output.variables[name]
        if variable.shape != (y_size, x_size):
            raise ValueError(f"existing {name} shape {variable.shape} != {(y_size, x_size)}")
    else:
        create_kwargs: dict[str, object] = {}
        if compression_level is not None and compression_level >= 0:
            create_kwargs.update(
                zlib=True,
                complevel=int(compression_level),
                shuffle=True,
                chunksizes=(min(max(1, block_rows), y_size), x_size),
            )
        try:
            variable = output.createVariable(name, "f4", ("y", "x"), **create_kwargs)
        except TypeError:
            variable = output.createVariable(name, "f4", ("y", "x"))

    for y_start in range(0, y_size, block_rows):
        y_stop = min(y_size, y_start + block_rows)
        variable[y_start:y_stop, :] = _expanded_output_rows(
            source_values,
            y_start=y_start,
            y_stop=y_stop,
            y_factor=y_factor,
            x_factor=x_factor,
        )
    _set_attrs(variable, attrs, fallback_long_name=name.replace("_", " ").title())
    return True


def attach_solar_angles_to_output(
    output_path: Path,
    scene_path: Path,
    *,
    overwrite: bool = False,
    dry_run: bool = False,
    block_rows: int = 256,
    compression_level: int | None = 1,
) -> dict[str, object]:
    output_path = output_path.expanduser().resolve()
    scene_path = scene_path.expanduser().resolve()

    y_size, x_size, existing = _output_grid_and_existing(output_path)
    if len(existing) == 2 and not overwrite:
        return {
            "output_path": str(output_path),
            "scene_path": str(scene_path),
            "status": "skipped_existing",
            "existing_variables": existing,
        }

    variables_to_write = [name for name in VIIRS_SOLAR_FIELDS if overwrite or name not in existing]
    if dry_run:
        return {
            "output_path": str(output_path),
            "scene_path": str(scene_path),
            "status": "dry_run",
            "existing_variables": existing,
            "variables": variables_to_write,
        }

    if block_rows <= 0:
        raise ValueError("block_rows must be positive")

    geometry = read_viirs_solar_geometry(scene_path)

    written: list[str] = []
    with NetCDFDataset(output_path, "a") as output:
        current_existing = [name for name in VIIRS_SOLAR_FIELDS if name in output.variables]
        if len(current_existing) == 2 and not overwrite:
            return {
                "output_path": str(output_path),
                "scene_path": str(scene_path),
                "status": "skipped_existing",
                "existing_variables": current_existing,
            }
        values_by_name = {
            "solar_zenith": geometry.solar_zenith,
            "solar_azimuth": geometry.solar_azimuth,
        }
        for name in VIIRS_SOLAR_FIELDS:
            if _create_or_update_variable(
                output,
                name,
                values_by_name[name],
                geometry.attrs.get(name, {}),
                y_size=y_size,
                x_size=x_size,
                overwrite=overwrite,
                block_rows=block_rows,
                compression_level=compression_level,
            ):
                written.append(name)

    return {
        "output_path": str(output_path),
        "scene_path": str(scene_path),
        "status": "updated" if written else "skipped_existing",
        "overwrote_existing": bool(overwrite and existing),
        "variables": written,
    }


def _scene_for_output(output_path: Path, *, input_root: Path) -> BackfillTask | tuple[Path, str]:
    match = OUTPUT_NAME_RE.match(output_path.name)
    if match is None:
        return output_path, "unrecognized_output_filename"

    platform = match.group("platform")
    tile = match.group("tile")
    ymd = match.group("date")
    date = datetime.strptime(ymd, "%Y%m%d")
    doy = date.timetuple().tm_yday
    product = PLATFORM_PRODUCT[platform]

    candidate_years = [date.year]
    for part in output_path.parts:
        water_year_match = WATER_YEAR_PART_RE.match(part)
        if water_year_match is None:
            continue
        water_year = int(water_year_match.group("year"))
        if water_year not in candidate_years:
            candidate_years.append(water_year)
        break
    if date.month >= 10 and date.year + 1 not in candidate_years:
        candidate_years.append(date.year + 1)

    pattern = f"{product}.A{date.year}{doy:03d}.{tile}.*.h5"
    attempts: list[tuple[Path, list[Path]]] = []
    for year in candidate_years:
        scene_dir = input_root / platform / "reflectance" / tile / str(year)
        matches = sorted(scene_dir.glob(pattern))
        attempts.append((scene_dir / pattern, matches))
        if len(matches) == 1:
            return BackfillTask(
                output_path=output_path,
                scene_path=matches[0],
                platform=platform,
                tile=tile,
                date=ymd,
            )
        if len(matches) > 1:
            return output_path, f"scene_match_count={len(matches)} pattern={scene_dir / pattern}"

    attempted_patterns = ";".join(str(pattern_path) for pattern_path, _ in attempts)
    return output_path, f"scene_match_count=0 patterns={attempted_patterns}"


def discover_tasks(
    *,
    output_roots: tuple[Path, ...],
    input_root: Path,
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
            match = OUTPUT_NAME_RE.match(output_path.name)
            if match is None:
                skipped.append((output_path, "unrecognized_output_filename"))
                continue
            if platforms is not None and match.group("platform") not in platforms:
                continue
            if tiles is not None and match.group("tile") not in tiles:
                continue
            if water_year_parts is not None and not any(part in water_year_parts for part in output_path.parts):
                continue
            if only_missing and _has_complete_solar_angles(output_path):
                skipped.append((output_path, "existing_solar_angles"))
                continue

            task_or_skip = _scene_for_output(output_path, input_root=input_root)
            if isinstance(task_or_skip, BackfillTask):
                tasks.append(task_or_skip)
                if limit is not None and len(tasks) >= limit:
                    return tasks, skipped
            else:
                skipped.append(task_or_skip)

    return tasks, skipped


def _run_one(
    task: BackfillTask,
    *,
    dry_run: bool,
    overwrite: bool,
    block_rows: int,
    compression_level: int | None,
) -> dict[str, object]:
    started = perf_counter()
    try:
        result = attach_solar_angles_to_output(
            task.output_path,
            task.scene_path,
            overwrite=overwrite,
            dry_run=dry_run,
            block_rows=block_rows,
            compression_level=compression_level,
        )
        status = str(result.get("status", "updated"))
        return {
            "status": status,
            "platform": task.platform,
            "tile": task.tile,
            "date": task.date,
            "output_path": str(task.output_path),
            "scene_path": str(task.scene_path),
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
            "scene_path": str(task.scene_path),
            "elapsed_seconds": round(perf_counter() - started, 3),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _worker(payload: tuple[BackfillTask, bool, bool, int, int | None]) -> dict[str, object]:
    task, dry_run, overwrite, block_rows, compression_level = payload
    return _run_one(
        task,
        dry_run=dry_run,
        overwrite=overwrite,
        block_rows=block_rows,
        compression_level=compression_level,
    )


def _write_skipped(path: Path, skipped: list[tuple[Path, str]]) -> None:
    with path.open("w") as handle:
        for skipped_path, reason in skipped:
            handle.write(json.dumps({"path": str(skipped_path), "reason": reason}) + "\n")


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
        "scene_path": str(task.scene_path),
        "timestamp": datetime.now(UTC).isoformat(),
    }
    if extra:
        payload.update(extra)
    handle.write(json.dumps(payload) + "\n")
    handle.flush()


def _run_tasks(
    tasks: list[BackfillTask],
    *,
    workers: int,
    max_tasks_per_child: int | None,
    dry_run: bool,
    overwrite: bool,
    block_rows: int,
    compression_level: int | None,
    results_path: Path,
    progress_path: Path,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    payloads = [(task, dry_run, overwrite, block_rows, compression_level) for task in tasks]
    with results_path.open("w") as handle, progress_path.open("w") as progress:
        if workers == 1:
            for payload in payloads:
                _write_progress(progress, "task_start", payload[0])
                result = _worker(payload)
                counts[result["status"]] = counts.get(result["status"], 0) + 1
                handle.write(json.dumps(result) + "\n")
                handle.flush()
                _write_progress(progress, "task_done", payload[0], {"status": result["status"]})
            return counts

        executor_kwargs: dict[str, object] = {"max_workers": workers}
        if max_tasks_per_child is not None and max_tasks_per_child > 0:
            executor_kwargs["max_tasks_per_child"] = max_tasks_per_child
        with ProcessPoolExecutor(**executor_kwargs) as executor:
            futures = [executor.submit(_worker, payload) for payload in payloads]
            for future in as_completed(futures):
                result = future.result()
                counts[result["status"]] = counts.get(result["status"], 0) + 1
                handle.write(json.dumps(result) + "\n")
                handle.flush()
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        action="append",
        type=Path,
        dest="output_roots",
        help="Output root to scan. May be repeated. Defaults to both mask experiment roots.",
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of local worker processes. Defaults to 1; use Slurm arrays for large runs.",
    )
    parser.add_argument(
        "--max-tasks-per-child",
        type=int,
        default=0,
        help="Recycle worker processes after this many files. Defaults to 0 to disable.",
    )
    parser.add_argument(
        "--block-rows",
        type=int,
        default=256,
        help="Number of output rows to expand/write at a time.",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=1,
        help="NetCDF zlib compression level for newly created variables. Use -1 to disable compression.",
    )
    parser.add_argument("--platforms", default=None, help="Comma-separated subset: snpp,noaa20,noaa21")
    parser.add_argument("--tiles", default=None, help="Comma-separated tile subset, e.g. h08v05,h09v05")
    parser.add_argument("--water-years", default=None, help="Comma-separated water years, e.g. 2024")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N matched files")
    parser.add_argument("--only-missing", action="store_true", help="Schedule only outputs missing one or both solar variables.")
    parser.add_argument("--chunk-index", type=int, default=None, help="Modulo chunk index for Slurm array execution.")
    parser.add_argument("--chunk-count", type=int, default=None, help="Modulo chunk count for Slurm array execution.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    output_roots = tuple(args.output_roots) if args.output_roots else DEFAULT_OUTPUT_ROOTS
    platforms = _split_csv(args.platforms)
    tiles = _split_csv(args.tiles)
    water_years = _split_csv(args.water_years)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    tasks, skipped = discover_tasks(
        output_roots=output_roots,
        input_root=args.input_root,
        platforms=platforms,
        tiles=tiles,
        water_years=water_years,
        limit=args.limit,
        only_missing=args.only_missing,
    )
    discovered_tasks = len(tasks)
    tasks = select_task_chunk(tasks, chunk_index=args.chunk_index, chunk_count=args.chunk_count)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    log_scope = _log_scope_label(tasks, chunk_index=args.chunk_index, chunk_count=args.chunk_count)
    results_path = args.log_dir / f"viirs_solar_backfill_results_{log_scope}_{timestamp}.jsonl"
    skipped_path = args.log_dir / f"viirs_solar_backfill_skipped_{log_scope}_{timestamp}.jsonl"
    progress_path = args.log_dir / f"viirs_solar_backfill_progress_{log_scope}_{timestamp}.jsonl"
    _write_skipped(skipped_path, skipped)

    summary = {
        "output_roots": [str(path) for path in output_roots],
        "input_root": str(args.input_root),
        "discovered_tasks": discovered_tasks,
        "tasks": len(tasks),
        "skipped": len(skipped),
        "workers": args.workers,
        "max_tasks_per_child": args.max_tasks_per_child,
        "block_rows": args.block_rows,
        "compression_level": args.compression_level,
        "only_missing": args.only_missing,
        "chunk_index": args.chunk_index,
        "chunk_count": args.chunk_count,
        "dry_run": args.dry_run,
        "overwrite": args.overwrite,
        "log_scope": log_scope,
        "results_path": str(results_path),
        "skipped_path": str(skipped_path),
        "progress_path": str(progress_path),
    }
    print(json.dumps({"event": "start", **summary}, indent=2), flush=True)

    started = perf_counter()
    counts = _run_tasks(
        tasks,
        workers=args.workers,
        max_tasks_per_child=args.max_tasks_per_child,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        block_rows=args.block_rows,
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
