from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from spires.sensors.modis.workflow import (
    get_modis_execution_profile,
    run_modis_inversion,
)


TEST_LUT_FILE = "SpiPy/tests/data/lut_modis_1_2_3_4_5_6_7_3um_dust_bandpass.mat"


def build_mock_prepared_scene() -> xr.Dataset:
    band_names = ["1", "2", "3", "4", "5", "6", "7"]
    reflectance = xr.DataArray(
        np.full((2, 2, len(band_names)), 0.2, dtype=np.float32),
        dims=("y", "x", "band"),
        coords={"y": [0, 1], "x": [0, 1], "band": band_names},
    )
    solar_zenith = xr.DataArray(
        np.full((2, 2), 40.0, dtype=np.float32),
        dims=("y", "x"),
        coords={"y": [0, 1], "x": [0, 1]},
    )
    valid_inversion_mask = xr.DataArray(
        np.array([[True, False], [True, True]]),
        dims=("y", "x"),
        coords={"y": [0, 1], "x": [0, 1]},
    )
    sensor_zenith = xr.DataArray(
        np.zeros((2, 2), dtype=np.float32),
        dims=("y", "x"),
        coords={"y": [0, 1], "x": [0, 1]},
    )
    return xr.Dataset(
        data_vars={
            "reflectance": reflectance,
            "solar_zenith": solar_zenith,
            "sensor_zenith": sensor_zenith,
            "valid_inversion_mask": valid_inversion_mask,
        },
        coords={"y": [0, 1], "x": [0, 1], "band": band_names},
        attrs={"platform": "terra", "acquisition_date": "2026-04-24", "tile": "h08v05"},
    )


def build_mock_r0() -> xr.Dataset:
    band_names = ["1", "2", "3", "4", "5", "6", "7"]
    return xr.Dataset(
        data_vars={
            "r0_reflectance": xr.DataArray(
                np.full((2, 2, len(band_names)), 0.1, dtype=np.float32),
                dims=("y", "x", "band"),
                coords={"y": [0, 1], "x": [0, 1], "band": band_names},
            )
        },
        coords={"y": [0, 1], "x": [0, 1], "band": band_names},
    )


class DummyInterpolator:
    def __init__(self, lut_file):
        self.lut_file = Path(lut_file)
        self.bands = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.float32)


def test_get_modis_execution_profile_returns_expected_defaults():
    local = get_modis_execution_profile("local")
    cluster = get_modis_execution_profile("cluster")

    assert local.chunks == {"time": 1, "y": 256, "x": 256, "band": -1}
    assert not local.scatter_lut
    assert cluster.scatter_lut
    assert cluster.persist_inputs


def test_run_modis_inversion_calls_core_inverter_and_masks_results(monkeypatch):
    scene = build_mock_prepared_scene()
    r0 = build_mock_r0()
    captured = {}

    def fake_speedy_invert_dask(
        *,
        spectra_targets,
        spectra_backgrounds,
        obs_solar_angles,
        interpolator,
        max_eval,
        x0,
        algorithm,
        client,
        scatter_lut,
        valid_mask,
        use_grouping,
        grouping_method,
        grouping_tolerance,
        grouping_reflectance_tol,
        grouping_background_tol,
        grouping_solar_zenith_tol,
    ):
        captured["target_chunks"] = spectra_targets.chunks
        captured["background_chunks"] = spectra_backgrounds.chunks
        captured["angle_chunks"] = obs_solar_angles.chunks
        captured["scatter_lut"] = scatter_lut
        captured["algorithm"] = algorithm
        captured["x0"] = x0
        captured["valid_mask_chunks"] = valid_mask.chunks
        dims = tuple(dim for dim in spectra_targets.dims if dim != "band")
        coords = {dim: spectra_targets.coords[dim] for dim in dims}
        fsca = xr.DataArray(np.full((2, 2), 0.75, dtype=np.float32), dims=dims, coords=coords)
        return xr.Dataset(
            data_vars={
                "fsca": fsca,
                "fshade": xr.ones_like(fsca) * 0.05,
                "dust_concentration": xr.ones_like(fsca) * 10.0,
                "grain_size": xr.ones_like(fsca) * 250.0,
            }
        )

    monkeypatch.setattr("spires.sensors.modis.workflow.LutInterpolator", DummyInterpolator)
    monkeypatch.setattr("spires.sensors.modis.workflow.speedy_invert_dask", fake_speedy_invert_dask)

    result = run_modis_inversion(scene, r0, lut_file=TEST_LUT_FILE, execution_profile="local", algorithm=5)

    assert captured["target_chunks"] is not None
    assert captured["background_chunks"] is not None
    assert captured["angle_chunks"] is not None
    assert captured["valid_mask_chunks"] is not None
    assert captured["scatter_lut"] is False
    assert captured["algorithm"] == 5
    np.testing.assert_allclose(captured["x0"], np.array([0.5, 0.05, 10, 250], dtype=np.float64))
    assert np.isnan(result["fsca"].isel(y=0, x=1))
    assert result.attrs["lut_file"].endswith(".mat")
    assert result.attrs["execution_profile"] == "local"
    assert list(result.attrs["selected_bands"]) == ["1", "2", "3", "4", "5", "6", "7"]
    assert "valid_inversion_mask" in result
    assert "raw_viewable_snow_fraction" in result
    assert "raw_shade_fraction" in result
    assert "raw_canopy_fraction" in result
    assert "raw_snow_fraction" in result


def test_run_modis_inversion_can_keep_outputs_unmasked(monkeypatch):
    scene = build_mock_prepared_scene()
    r0 = build_mock_r0()
    captured = {}

    def fake_speedy_invert_dask(
        *,
        spectra_targets,
        spectra_backgrounds,
        obs_solar_angles,
        interpolator,
        max_eval,
        x0,
        algorithm,
        client,
        scatter_lut,
        valid_mask,
        use_grouping,
        grouping_method,
        grouping_tolerance,
        grouping_reflectance_tol,
        grouping_background_tol,
        grouping_solar_zenith_tol,
    ):
        captured["valid_mask"] = valid_mask
        dims = tuple(dim for dim in spectra_targets.dims if dim != "band")
        coords = {dim: spectra_targets.coords[dim] for dim in dims}
        fsca = xr.DataArray(np.full((2, 2), 0.75, dtype=np.float32), dims=dims, coords=coords)
        return xr.Dataset(
            data_vars={
                "fsca": fsca,
                "fshade": xr.ones_like(fsca) * 0.05,
                "dust_concentration": xr.ones_like(fsca) * 10.0,
                "grain_size": xr.ones_like(fsca) * 250.0,
            }
        )

    monkeypatch.setattr("spires.sensors.modis.workflow.LutInterpolator", DummyInterpolator)
    monkeypatch.setattr("spires.sensors.modis.workflow.speedy_invert_dask", fake_speedy_invert_dask)

    result = run_modis_inversion(
        scene,
        r0,
        lut_file=TEST_LUT_FILE,
        execution_profile="local",
        apply_valid_inversion_mask=False,
    )

    assert captured["valid_mask"] is None
    assert result["fsca"].isel(y=0, x=1).item() == pytest.approx(0.75)
    assert result["raw_viewable_snow_fraction"].isel(y=0, x=1).item() == pytest.approx(0.75)
    assert not bool(result["valid_inversion_mask"].isel(y=0, x=1))
    assert result.attrs["valid_inversion_mask_applied"] == 0
    assert result.attrs["valid_inversion_mask_mode"] == "output_only"


def test_run_modis_inversion_rejects_scene_r0_band_mismatch(monkeypatch):
    scene = build_mock_prepared_scene()
    r0 = build_mock_r0().assign_coords(band=["1", "2", "3", "4", "5", "7", "6"])

    monkeypatch.setattr("spires.sensors.modis.workflow.LutInterpolator", DummyInterpolator)

    with pytest.raises(ValueError, match="Scene and R0 band order do not match"):
        run_modis_inversion(scene, r0, lut_file=TEST_LUT_FILE)


def test_run_modis_inversion_preserves_reflectance_in_output(monkeypatch, tmp_path):
    scene = build_mock_prepared_scene()
    r0 = build_mock_r0()

    def fake_speedy_invert_dask(
        *,
        spectra_targets,
        spectra_backgrounds,
        obs_solar_angles,
        interpolator,
        max_eval,
        x0,
        algorithm,
        client,
        scatter_lut,
        valid_mask,
        use_grouping,
        grouping_method,
        grouping_tolerance,
        grouping_reflectance_tol,
        grouping_background_tol,
        grouping_solar_zenith_tol,
    ):
        dims = tuple(dim for dim in spectra_targets.dims if dim != "band")
        coords = {dim: spectra_targets.coords[dim] for dim in dims}
        fsca = xr.DataArray(np.full((2, 2), 0.75, dtype=np.float32), dims=dims, coords=coords)
        return xr.Dataset(
            data_vars={
                "fsca": fsca,
                "fshade": xr.ones_like(fsca) * 0.05,
                "dust_concentration": xr.ones_like(fsca) * 10.0,
                "grain_size": xr.ones_like(fsca) * 250.0,
            }
        )

    monkeypatch.setattr("spires.sensors.full_workflow.LutInterpolator", DummyInterpolator)
    monkeypatch.setattr("spires.sensors.full_workflow.speedy_invert_dask", fake_speedy_invert_dask)

    result = run_modis_inversion(scene, r0, lut_file=TEST_LUT_FILE, execution_profile="local")

    assert "reflectance" in result
    assert result["reflectance"].dims == ("y", "x", "band")
    np.testing.assert_allclose(result["reflectance"].values, scene["reflectance"].values)

    output_path = tmp_path / "modis_inversion_with_reflectance.nc"
    result.to_netcdf(output_path)
    with xr.open_dataset(output_path) as written:
        np.testing.assert_allclose(written["reflectance"].values, scene["reflectance"].values)
