import csv
import json
from pathlib import Path

import numpy as np
import xarray as xr

from workflows.curc.status import scan_inversion_array_status, write_run_group_summary_artifacts, write_status_summary_artifacts


def _write_valid_output(path: Path) -> None:
    y = [0, 1]
    x = [0, 1]
    ds = xr.Dataset(
        data_vars={
            "raw_viewable_snow_fraction": xr.DataArray(np.full((2, 2), 0.5, dtype=np.float32), dims=("y", "x"), coords={"y": y, "x": x}),
            "raw_shade_fraction": xr.DataArray(np.full((2, 2), 0.1, dtype=np.float32), dims=("y", "x"), coords={"y": y, "x": x}),
            "dust_concentration": xr.DataArray(np.full((2, 2), 1.0, dtype=np.float32), dims=("y", "x"), coords={"y": y, "x": x}),
            "grain_size": xr.DataArray(np.full((2, 2), 100.0, dtype=np.float32), dims=("y", "x"), coords={"y": y, "x": x}),
            "valid_inversion_mask": xr.DataArray(np.ones((2, 2), dtype=bool), dims=("y", "x"), coords={"y": y, "x": x}),
        }
    )
    ds.to_netcdf(path)


def _write_manifest(path: Path, *, tasks: list[dict[str, object]], tile: str = "h08v05", run_group_id: str = "20260519_143044_viirs_snpp_wy2023_full") -> None:
    run_group_dir = path.parents[2]
    tile_run_dir = path.parents[1]
    payload = {
        "job_name": f"spipy-viirs-snpp-{tile}-wy2023",
        "step": "run_inversion",
        "sensor": "viirs",
        "platform": "snpp",
        "tile": tile,
        "water_year": 2023,
        "task_count": len(tasks),
        "array_indices": list(range(len(tasks))),
        "max_concurrent_tasks": None,
        "max_auto_retry_count": 3,
        "apply_valid_inversion_mask": False,
        "use_grouping": True,
        "grouping_method": "chunk_bin_mean",
        "r0_year": 2022,
        "run_group_id": run_group_id,
        "run_group_dir": str(run_group_dir),
        "tile_run_dir": str(tile_run_dir),
        "tile_detailed_log_dir": str(path.parent),
        "scope_kind": "full",
        "slurm_profile": {},
        "tasks": tasks,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="ascii")


def _task(*, task_index: int, date: str, retry_count: int, log_dir: Path, output_dir: Path, tile: str = "h08v05") -> dict[str, object]:
    return {
        "task_index": task_index,
        "sensor": "viirs",
        "platform": "snpp",
        "tile": tile,
        "water_year": 2023,
        "date": date,
        "source_paths": [f"/tmp/{date}.h5"],
        "output_path": str(output_dir),
        "log_path": str(log_dir / f"run_inversion_{date}.log"),
        "r0_year": 2022,
        "retry_count": retry_count,
    }


def _write_task_log(path: Path, *, date: str, retry_count: int, status: str, failure_code: str, job_id: str) -> None:
    lines = [
        (
            f'2026-05-19 14:31:00,000 INFO spires.curc.viirs.snpp.h08v05.{date} '
            f'event="curc_run_viirs_snpp_inversion_task" stage="curc_runtime" event_type="start" status="started" '
            f'date="{date}" retry_count={retry_count} slurm_array_job_id="26186675" slurm_array_task_id="0" '
            f'slurm_cluster_name="blanca" slurm_job_id="{job_id}" slurm_job_name="spipy-viirs-snpp-h08v05-wy2023"'
        ),
        (
            f'2026-05-19 14:32:00,000 INFO spires.curc.viirs.snpp.h08v05.{date} '
            f'====== SUMMARY ====== event="curc_run_viirs_snpp_inversion_task" stage="curc_runtime" '
            f'event_type="summary" status="{status}" date="{date}" retry_count={retry_count} '
            f'slurm_array_job_id="26186675" slurm_array_task_id="0" slurm_cluster_name="blanca" '
            f'slurm_job_id="{job_id}" slurm_job_name="spipy-viirs-snpp-h08v05-wy2023" '
            f'failure_code="{failure_code}" retry_recommended={"true" if failure_code != "none" else "false"}'
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_curc_status_summary_artifacts_capture_attempt_history(tmp_path: Path) -> None:
    top_log_dir = tmp_path / "logs" / "20260519_143044_viirs_snpp_wy2023_full"
    tile_dir = top_log_dir / "h08v05"
    log_dir = tile_dir / "detailed_logs"
    output_dir = tmp_path / "output"
    log_dir.mkdir(parents=True)
    output_dir.mkdir()

    initial_manifest = log_dir / "spipy-viirs-snpp-h08v05-wy2023_array_manifest.json"
    retry_manifest = log_dir / "spipy-viirs-snpp-h08v05-wy2023_array_manifest_retry.json"

    _write_manifest(
        initial_manifest,
        tasks=[
            _task(task_index=0, date="2023-03-16", retry_count=0, log_dir=log_dir, output_dir=output_dir),
            _task(task_index=1, date="2023-03-17", retry_count=0, log_dir=log_dir, output_dir=output_dir),
        ],
    )
    _write_manifest(
        retry_manifest,
        tasks=[
            _task(task_index=0, date="2023-03-17", retry_count=1, log_dir=log_dir, output_dir=output_dir),
        ],
    )

    _write_valid_output(output_dir / "snpp_raw_output_h08v05_20230316.nc")
    _write_valid_output(output_dir / "snpp_raw_output_h08v05_20230317.nc")

    _write_task_log(
        log_dir / "run_inversion_2023-03-16_job26186917.log",
        date="2023-03-16",
        retry_count=0,
        status="loaded_existing",
        failure_code="none",
        job_id="26186917",
    )
    _write_task_log(
        log_dir / "run_inversion_2023-03-17_job26186918.log",
        date="2023-03-17",
        retry_count=0,
        status="failed",
        failure_code="missing_lut",
        job_id="26186918",
    )
    _write_task_log(
        log_dir / "run_inversion_2023-03-17_job26186919.log",
        date="2023-03-17",
        retry_count=1,
        status="completed",
        failure_code="none",
        job_id="26186919",
    )

    report = scan_inversion_array_status(initial_manifest)
    assert report.tasks[0].status == "loaded_existing"
    assert report.tasks[1].status == "completed"

    csv_path, txt_path = write_status_summary_artifacts(initial_manifest, report=report)
    assert csv_path.parent == tile_dir
    assert txt_path.parent == tile_dir
    assert csv_path.name == "run_inversion_h08v05_wy2023_summary.csv"
    assert txt_path.name == "run_inversion_h08v05_wy2023_summary.txt"

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert all(row["run_group_id"] == "20260519_143044_viirs_snpp_wy2023_full" for row in rows)
    assert [row["scene_date"] for row in rows] == ["2023-03-16", "2023-03-17", "2023-03-17"]
    assert rows[0]["status"] == "loaded_existing"
    assert rows[0]["last_attempt_for_date"] == "True"
    assert rows[1]["status"] == "failed"
    assert rows[1]["last_attempt_for_date"] == "False"
    assert rows[2]["status"] == "completed"
    assert rows[2]["retry_count"] == "1"
    assert rows[2]["last_attempt_for_date"] == "True"

    summary = txt_path.read_text(encoding="utf-8")
    assert "TILE h08v05 WATER YEAR 2023" in summary
    assert "run_group_id=20260519_143044_viirs_snpp_wy2023_full" in summary
    assert "manifest=" in summary
    assert "wall_time_start_utc=2026-05-19T14:31:00Z" in summary
    assert "wall_time_end_utc=2026-05-19T14:32:00Z" in summary
    assert "total_wall_time_seconds=60.000 (00:01:00)" in summary
    assert "submission_start_utc=2026-05-19T14:30:44Z" in summary
    assert "submission_to_completion_wall_time_seconds=76.000 (00:01:16)" in summary
    assert "\nTOTALS\n" in summary
    assert "loaded_existing=1" in summary
    assert "completed=1" in summary
    assert "auto_retried_dates=1" in summary
    assert "total_attempt_rows=3" in summary
    assert "2023-03-16   loaded_existing" in summary
    assert "2023-03-17   completed" in summary


def test_run_group_summary_artifacts_merge_tiles(tmp_path: Path) -> None:
    run_group_dir = tmp_path / "logs" / "20260519_143044_viirs_snpp_wy2023_full"
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    for tile, date, job_id in (
        ("h08v04", "2023-03-16", "26186917"),
        ("h08v05", "2023-03-17", "26186918"),
    ):
        tile_log_dir = run_group_dir / tile / "detailed_logs"
        tile_log_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = tile_log_dir / f"spipy-viirs-snpp-{tile}-wy2023_array_manifest.json"
        _write_manifest(
            manifest_path,
            tile=tile,
            tasks=[_task(task_index=0, date=date, retry_count=0, log_dir=tile_log_dir, output_dir=output_dir, tile=tile)],
        )
        _write_valid_output(output_dir / f"snpp_raw_output_{tile}_{date.replace('-', '')}.nc")
        _write_task_log(
            tile_log_dir / f"run_inversion_{date}_job{job_id}.log",
            date=date,
            retry_count=0,
            status="completed",
            failure_code="none",
            job_id=job_id,
        )

    csv_path, txt_path = write_run_group_summary_artifacts(run_group_dir)
    assert csv_path.parent == run_group_dir
    assert txt_path.parent == run_group_dir

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["tile"] for row in rows] == ["h08v04", "h08v05"]

    summary = txt_path.read_text(encoding="utf-8")
    assert "RUN GROUP 20260519_143044_viirs_snpp_wy2023_full" in summary
    assert "submission_start_utc=2026-05-19T14:30:44Z" in summary
    assert "submission_to_completion_wall_time_seconds=76.000 (00:01:16)" in summary
    assert "\nPER-TILE TOTALS\n" in summary
    assert "h08v04" in summary
    assert "h08v05" in summary
