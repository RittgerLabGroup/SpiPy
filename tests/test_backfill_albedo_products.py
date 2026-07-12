from pathlib import Path
from argparse import Namespace

import numpy as np
import pytest
import xarray as xr

from scripts.backfill_viirs_albedo_products import (
    BackfillTask,
    LoadedLuts,
    _log_scope_label,
    _render_sbatch_command,
    attach_albedo_products_to_output,
    discover_tasks,
    select_task_chunk,
)
from spires.albedo import (
    ALBEDO_PRODUCT_NAMES,
    DELTA_VIS_PRODUCT_NAME,
    RADIATIVE_FORCING_PRODUCT_NAME,
    AlbedoLutFunction,
    AlbedoLuts,
    AlbedoProductFlags,
    RadiativeForcingLuts,
)


def _lut_function(name, dimensions, points, value):
    shape = tuple(len(point) for point in points)
    return AlbedoLutFunction(
        name=name,
        points=tuple(np.asarray(point, dtype=np.float32) for point in points),
        values=np.full(shape, value, dtype=np.float32),
        dimensions=dimensions,
    )


def _synthetic_luts() -> LoadedLuts:
    mu0 = np.array([0.01, 1.0], dtype=np.float32)
    muz = np.array([0.0, 1.0], dtype=np.float32)
    gs = np.array([5.0, 50.0], dtype=np.float32)
    dust = np.array([0.0, 1.0], dtype=np.float32)
    soot = np.array([0.0, 1.0e-5], dtype=np.float32)
    return LoadedLuts(
        albedo=AlbedoLuts(
            clean=_lut_function("Fclean", ("mu0", "muZ", "gs"), (mu0, muz, gs), 0.8),
            dirty=_lut_function("Fdirty", ("mu0", "muZ", "gs", "dust", "soot"), (mu0, muz, gs, dust, soot), 0.6),
        ),
        radiative_forcing=RadiativeForcingLuts(
            darken=_lut_function("Fdarken", ("mu0", "muZ", "gs", "dust", "soot"), (mu0, muz, gs, dust, soot), 0.1),
            force=_lut_function("Fforce", ("mu0", "muZ", "gs", "dust", "soot"), (mu0, muz, gs, dust, soot), 25.0),
        ),
    )


def _output_dataset(shape=(2, 2)) -> xr.Dataset:
    coords = {"y": np.arange(shape[0]), "x": np.arange(shape[1])}
    return xr.Dataset(
        data_vars={
            "grain_size": xr.DataArray(np.full(shape, 100.0, dtype=np.float32), dims=("y", "x"), coords=coords),
            "dust_concentration": xr.DataArray(np.full(shape, 10.0, dtype=np.float32), dims=("y", "x"), coords=coords),
            "solar_zenith": xr.DataArray(np.full(shape, 40.0, dtype=np.float32), dims=("y", "x"), coords=coords),
            "solar_azimuth": xr.DataArray(np.full(shape, 180.0, dtype=np.float32), dims=("y", "x"), coords=coords),
        },
        attrs={"platform": "noaa20", "tile": "h08v04", "acquisition_date": "2024-01-01"},
    )


def test_attach_albedo_products_appends_all_requested_variables(monkeypatch, tmp_path: Path) -> None:
    output_path = tmp_path / "noaa20_raw_output_h08v04_20240101.nc"
    _output_dataset().to_netcdf(output_path)
    task = BackfillTask(output_path=output_path, platform="noaa20", tile="h08v04", date="20240101")

    def fake_load_terrain(task, terrain_root, template):
        return xr.zeros_like(template).rename("slope"), xr.zeros_like(template).rename("aspect")

    monkeypatch.setattr("scripts.backfill_viirs_albedo_products._load_terrain", fake_load_terrain)

    result = attach_albedo_products_to_output(
        task,
        flags=AlbedoProductFlags(),
        luts=_synthetic_luts(),
        albedo_lut_path=tmp_path / "albedo.mat",
        radiative_forcing_lut_path=tmp_path / "rf.mat",
    )

    assert result["status"] == "updated"
    with xr.open_dataset(output_path) as updated:
        assert set((*ALBEDO_PRODUCT_NAMES, RADIATIVE_FORCING_PRODUCT_NAME, DELTA_VIS_PRODUCT_NAME)).issubset(
            updated.data_vars
        )
        np.testing.assert_allclose(updated["albedo_clean_flat"], np.full((2, 2), 0.8, dtype=np.float32))
        np.testing.assert_allclose(updated["radiative_forcing"], np.full((2, 2), 25.0, dtype=np.float32))
        assert updated.attrs["include_albedo"] == 1


def test_attach_albedo_products_skips_complete_existing_output(tmp_path: Path) -> None:
    output_path = tmp_path / "snpp_raw_output_h08v04_20240101.nc"
    output = _output_dataset()
    for name in (*ALBEDO_PRODUCT_NAMES, RADIATIVE_FORCING_PRODUCT_NAME, DELTA_VIS_PRODUCT_NAME):
        output[name] = xr.zeros_like(output["grain_size"])
    output.to_netcdf(output_path)
    task = BackfillTask(output_path=output_path, platform="snpp", tile="h08v04", date="20240101")

    result = attach_albedo_products_to_output(task, flags=AlbedoProductFlags(), luts=_synthetic_luts())

    assert result["status"] == "skipped_existing"


def test_discover_tasks_only_missing_filters_existing_products(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    complete_path = output_root / "viirs" / "snpp" / "h08v04" / "raw" / "wy2024" / "snpp_raw_output_h08v04_20240101.nc"
    missing_path = output_root / "viirs" / "snpp" / "h08v04" / "raw" / "wy2024" / "snpp_raw_output_h08v04_20240102.nc"
    complete_path.parent.mkdir(parents=True)

    complete = _output_dataset()
    for name in (*ALBEDO_PRODUCT_NAMES, RADIATIVE_FORCING_PRODUCT_NAME, DELTA_VIS_PRODUCT_NAME):
        complete[name] = xr.zeros_like(complete["grain_size"])
    complete.to_netcdf(complete_path)
    _output_dataset().to_netcdf(missing_path)

    tasks, skipped = discover_tasks(
        output_roots=(output_root,),
        flags=AlbedoProductFlags(),
        platforms={"snpp"},
        tiles={"h08v04"},
        water_years={"2024"},
        only_missing=True,
    )

    assert [task.output_path for task in tasks] == [missing_path]
    assert skipped == [(complete_path, "existing_albedo_products")]


def test_select_task_chunk_uses_stable_modulo_selection(tmp_path: Path) -> None:
    tasks = [
        BackfillTask(tmp_path / f"out_{index}.nc", "snpp", "h08v04", f"202401{index:02d}")
        for index in range(6)
    ]

    selected = select_task_chunk(tasks, chunk_index=2, chunk_count=3)

    assert [task.date for task in selected] == ["20240102", "20240105"]


def test_log_scope_label_includes_platform_tile_date_range_and_chunk(tmp_path: Path) -> None:
    tasks = [
        BackfillTask(tmp_path / "a.nc", "snpp", "h08v04", "20240101"),
        BackfillTask(tmp_path / "b.nc", "snpp", "h08v04", "20240105"),
    ]

    label = _log_scope_label(tasks, chunk_index=2, chunk_count=60)

    assert label == "snpp_h08v04_20240101-to-20240105_chunk02-of-60"


def test_log_scope_label_collapses_broad_runs(tmp_path: Path) -> None:
    tasks = [
        BackfillTask(tmp_path / "a.nc", "snpp", "h08v04", "20240101"),
        BackfillTask(tmp_path / "b.nc", "noaa20", "h09v04", "20240102"),
        BackfillTask(tmp_path / "c.nc", "noaa21", "h10v04", "20240103"),
        BackfillTask(tmp_path / "d.nc", "snpp", "h11v04", "20240104"),
    ]

    label = _log_scope_label(tasks, chunk_index=None, chunk_count=None)

    assert label == "noaa20-noaa21-snpp_multi-tile_20240101-to-20240104"


def test_render_sbatch_command_targets_rittger_and_snow_partitions(tmp_path: Path) -> None:
    args = Namespace(
        sbatch_array_count=10,
        sbatch_max_concurrent=4,
        sbatch_job_name="albedo-test",
        sbatch_partition="blanca-rittger,blanca-snow",
        sbatch_account="blanca-rittger",
        sbatch_time="01:00:00",
        sbatch_mem="8G",
        sbatch_cpus_per_task=1,
        output_roots=[tmp_path / "output"],
        terrain_root=tmp_path / "input" / "viirs",
        log_dir=tmp_path / "logs",
        albedo_lut_path=tmp_path / "albedo.mat",
        radiative_forcing_lut_path=tmp_path / "rf.mat",
        compression_level=1,
        platforms="snpp",
        tiles="h08v04",
        water_years="2024",
        include_albedo=True,
        include_radiative_forcing=True,
        include_delta_vis=True,
        only_missing=True,
        overwrite=False,
        dry_run=False,
    )

    command = _render_sbatch_command(args, repo_root=tmp_path / "repo")

    assert command[:2] == ["sbatch", "--parsable"]
    assert "blanca-rittger,blanca-snow" in command
    assert "0-9%4" in command
    assert "backfill_viirs_albedo_products.py" in command[-1]
