from pathlib import Path

import h5py
import numpy as np
import xarray as xr

from scripts.backfill_viirs_solar_angles import (
    BackfillTask,
    _log_scope_label,
    attach_solar_angles_to_output,
    discover_tasks,
    select_task_chunk,
)


def _minimal_inversion_output(shape: tuple[int, int] = (4, 4)) -> xr.Dataset:
    y = np.arange(shape[0])
    x = np.arange(shape[1]) + 10
    coords = {"y": y, "x": x}
    valid = xr.DataArray(
        np.ones(shape, dtype=bool),
        dims=("y", "x"),
        coords=coords,
    )
    return xr.Dataset(
        data_vars={
            "raw_viewable_snow_fraction": xr.DataArray(np.full(shape, 0.5, dtype=np.float32), dims=("y", "x"), coords=coords),
            "raw_shade_fraction": xr.DataArray(np.zeros(shape, dtype=np.float32), dims=("y", "x"), coords=coords),
            "dust_concentration": xr.DataArray(np.full(shape, 10.0, dtype=np.float32), dims=("y", "x"), coords=coords),
            "grain_size": xr.DataArray(np.full(shape, 250.0, dtype=np.float32), dims=("y", "x"), coords=coords),
            "valid_inversion_mask": valid,
        }
    )


def _write_viirs_solar_hdf(path: Path) -> None:
    with h5py.File(path, "w") as hdf:
        fields = hdf.create_group("HDFEOS/GRIDS/VIIRS_Grid_1km_2D/Data Fields")
        zenith = fields.create_dataset(
            "SolarZenith_1",
            data=np.array([[4000, 4100], [4200, 4300]], dtype=np.int16),
        )
        azimuth = fields.create_dataset(
            "SolarAzimuth_1",
            data=np.array([[15000, 15100], [15200, 15300]], dtype=np.int16),
        )
        for dataset in (zenith, azimuth):
            dataset.attrs["scale_factor"] = 0.01
            dataset.attrs["add_offset"] = 0.0
            dataset.attrs["_FillValue"] = np.int16(-28672)
            dataset.attrs["valid_range"] = np.array([-18000, 18000], dtype=np.int16)
            dataset.attrs["units"] = "degrees"


def test_attach_solar_angles_reads_only_hdf_geometry_and_expands_to_output_grid(tmp_path: Path) -> None:
    output_path = tmp_path / "raw_output.nc"
    scene_path = tmp_path / "VJ109GA.A2024001.h08v04.002.2024002000000.h5"
    _minimal_inversion_output().to_netcdf(output_path)
    _write_viirs_solar_hdf(scene_path)

    result = attach_solar_angles_to_output(output_path, scene_path)

    assert result["status"] == "updated"
    assert result["variables"] == ["solar_zenith", "solar_azimuth"]
    with xr.open_dataset(output_path) as updated:
        assert "solar_zenith" in updated
        assert "solar_azimuth" in updated
        np.testing.assert_allclose(
            updated["solar_zenith"],
            [
                [40.0, 40.0, 41.0, 41.0],
                [40.0, 40.0, 41.0, 41.0],
                [42.0, 42.0, 43.0, 43.0],
                [42.0, 42.0, 43.0, 43.0],
            ],
        )
        np.testing.assert_allclose(
            updated["solar_azimuth"],
            [
                [150.0, 150.0, 151.0, 151.0],
                [150.0, 150.0, 151.0, 151.0],
                [152.0, 152.0, 153.0, 153.0],
                [152.0, 152.0, 153.0, 153.0],
            ],
        )


def test_attach_solar_angles_can_write_in_single_row_blocks_without_compression(tmp_path: Path) -> None:
    output_path = tmp_path / "raw_output.nc"
    scene_path = tmp_path / "VJ109GA.A2024001.h08v04.002.2024002000000.h5"
    _minimal_inversion_output().to_netcdf(output_path)
    _write_viirs_solar_hdf(scene_path)

    result = attach_solar_angles_to_output(
        output_path,
        scene_path,
        block_rows=1,
        compression_level=None,
    )

    assert result["status"] == "updated"
    with xr.open_dataset(output_path) as updated:
        assert updated["solar_zenith"].shape == (4, 4)
        np.testing.assert_allclose(updated["solar_zenith"].isel(y=3), [42.0, 42.0, 43.0, 43.0])


def test_attach_solar_angles_skips_existing_before_opening_source_hdf(tmp_path: Path) -> None:
    output_path = tmp_path / "raw_output.nc"
    output = _minimal_inversion_output()
    output["solar_zenith"] = xr.zeros_like(output["grain_size"])
    output["solar_azimuth"] = xr.zeros_like(output["grain_size"])
    output.to_netcdf(output_path)

    result = attach_solar_angles_to_output(output_path, tmp_path / "does_not_exist.h5")

    assert result["status"] == "skipped_existing"
    assert result["existing_variables"] == ["solar_zenith", "solar_azimuth"]


def test_attach_solar_angles_dry_run_does_not_open_source_hdf(tmp_path: Path) -> None:
    output_path = tmp_path / "raw_output.nc"
    _minimal_inversion_output().to_netcdf(output_path)

    result = attach_solar_angles_to_output(
        output_path,
        tmp_path / "does_not_exist.h5",
        dry_run=True,
    )

    assert result["status"] == "dry_run"
    assert result["existing_variables"] == []
    assert result["variables"] == ["solar_zenith", "solar_azimuth"]


def test_discover_tasks_falls_back_to_water_year_reflectance_directory(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    input_root = tmp_path / "input"
    output_path = (
        output_root
        / "viirs"
        / "snpp"
        / "h08v04"
        / "raw"
        / "wy2024"
        / "snpp_raw_output_h08v04_20231024.nc"
    )
    output_path.parent.mkdir(parents=True)
    output_path.touch()

    scene_path = (
        input_root
        / "snpp"
        / "reflectance"
        / "h08v04"
        / "2024"
        / "VNP09GA.A2023297.h08v04.002.2023298000000.h5"
    )
    scene_path.parent.mkdir(parents=True)
    scene_path.touch()

    tasks, skipped = discover_tasks(
        output_roots=(output_root,),
        input_root=input_root,
        platforms={"snpp"},
        tiles={"h08v04"},
        water_years={"2024"},
    )

    assert skipped == []
    assert len(tasks) == 1
    assert tasks[0].output_path == output_path
    assert tasks[0].scene_path == scene_path


def test_discover_tasks_only_missing_skips_existing_without_scene_lookup(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    input_root = tmp_path / "input"
    output_path = (
        output_root
        / "viirs"
        / "snpp"
        / "h08v04"
        / "raw"
        / "wy2024"
        / "snpp_raw_output_h08v04_20231024.nc"
    )
    output_path.parent.mkdir(parents=True)
    output = _minimal_inversion_output()
    output["solar_zenith"] = xr.zeros_like(output["grain_size"])
    output["solar_azimuth"] = xr.zeros_like(output["grain_size"])
    output.to_netcdf(output_path)

    tasks, skipped = discover_tasks(
        output_roots=(output_root,),
        input_root=input_root,
        platforms={"snpp"},
        tiles={"h08v04"},
        water_years={"2024"},
        only_missing=True,
    )

    assert tasks == []
    assert skipped == [(output_path, "existing_solar_angles")]


def test_select_task_chunk_uses_stable_modulo_selection(tmp_path: Path) -> None:
    tasks = [
        BackfillTask(tmp_path / f"out_{index}.nc", tmp_path / f"in_{index}.h5", "snpp", "h08v04", f"202401{index:02d}")
        for index in range(6)
    ]

    selected = select_task_chunk(tasks, chunk_index=1, chunk_count=3)

    assert [task.date for task in selected] == ["20240101", "20240104"]


def test_log_scope_label_includes_platform_tile_date_range_and_chunk(tmp_path: Path) -> None:
    tasks = [
        BackfillTask(tmp_path / "a.nc", tmp_path / "a.h5", "noaa20", "h10v04", "20231107"),
        BackfillTask(tmp_path / "b.nc", tmp_path / "b.h5", "noaa20", "h10v04", "20240528"),
    ]

    label = _log_scope_label(tasks, chunk_index=27, chunk_count=60)

    assert label == "noaa20_h10v04_20231107-to-20240528_chunk27-of-60"


def test_log_scope_label_handles_empty_task_slice() -> None:
    label = _log_scope_label([], chunk_index=3, chunk_count=60)

    assert label == "no-platform_no-tile_no-dates_chunk03-of-60"
