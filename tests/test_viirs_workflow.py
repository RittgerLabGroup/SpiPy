from pathlib import Path
import json

import numpy as np
import pytest
import xarray as xr

from spires.sensors.full_workflow import find_default_canopy_fraction
from spires.sensors.viirs.workflow import (
    get_viirs_execution_profile,
    run_viirs_inversion,
)


TEST_LUT_FILE = "SpiPy/tests/data/lut_viirs_noaa20_i1_i2_i3_m2_m4_m8_m11_3um_dust_bandpass.mat"


def build_mock_prepared_scene() -> xr.Dataset:
    band_names = ["I1", "I2", "I3", "M2", "M4", "M8", "M11"]
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
        attrs={"platform": "noaa20", "acquisition_date": "2026-04-24"},
    )


def build_mock_r0() -> xr.Dataset:
    band_names = ["I1", "I2", "I3", "M2", "M4", "M8", "M11"]
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


def test_get_viirs_execution_profile_returns_expected_defaults():
    local = get_viirs_execution_profile("local")
    cluster = get_viirs_execution_profile("cluster")

    assert local.chunks == {"time": 1, "y": 256, "x": 256, "band": -1}
    assert not local.scatter_lut
    assert cluster.scatter_lut
    assert cluster.persist_inputs


def test_run_viirs_inversion_calls_core_inverter_and_masks_results(monkeypatch):
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
        captured["use_grouping"] = use_grouping
        captured["grouping_method"] = grouping_method
        captured["grouping_tolerance"] = grouping_tolerance
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

    result = run_viirs_inversion(
        scene,
        r0,
        lut_file=TEST_LUT_FILE,
        execution_profile="local",
        algorithm=5,
        use_grouping=True,
        grouping_method="first",
        grouping_tolerance=0.02,
    )

    assert captured["target_chunks"] is not None
    assert captured["background_chunks"] is not None
    assert captured["angle_chunks"] is not None
    assert captured["valid_mask_chunks"] is not None
    assert captured["scatter_lut"] is False
    assert captured["algorithm"] == 5
    assert captured["use_grouping"] is True
    assert captured["grouping_method"] == "first"
    assert captured["grouping_tolerance"] == pytest.approx(0.02)
    np.testing.assert_allclose(captured["x0"], np.array([0.5, 0.05, 10, 250], dtype=np.float64))
    assert np.isnan(result["raw_viewable_snow_fraction"].isel(y=0, x=1))
    assert result.attrs["lut_file"].endswith(".mat")
    assert result.attrs["execution_profile"] == "local"
    assert list(result.attrs["selected_bands"]) == ["I1", "I2", "I3", "M2", "M4", "M8", "M11"]
    assert "valid_inversion_mask" in result
    assert "raw_viewable_snow_fraction" in result
    assert "raw_shade_fraction" in result
    assert "raw_canopy_fraction" in result
    assert "raw_snow_fraction" in result
    assert result["raw_canopy_fraction"].isel(y=0, x=0).compute().item() == pytest.approx(0.0)
    assert result["raw_snow_fraction"].isel(y=0, x=0).compute().item() == pytest.approx(0.75 / 0.95)


def test_run_viirs_inversion_can_keep_outputs_unmasked(monkeypatch):
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

    monkeypatch.setattr("spires.sensors.full_workflow.LutInterpolator", DummyInterpolator)
    monkeypatch.setattr("spires.sensors.full_workflow.speedy_invert_dask", fake_speedy_invert_dask)

    result = run_viirs_inversion(
        scene,
        r0,
        lut_file=TEST_LUT_FILE,
        execution_profile="local",
        apply_valid_inversion_mask=False,
    )

    assert captured["valid_mask"] is None
    assert result["raw_viewable_snow_fraction"].isel(y=0, x=1).item() == pytest.approx(0.75)
    assert result["raw_viewable_snow_fraction"].isel(y=0, x=1).item() == pytest.approx(0.75)
    assert not bool(result["valid_inversion_mask"].isel(y=0, x=1))
    assert result.attrs["valid_inversion_mask_applied"] == 0
    assert result.attrs["valid_inversion_mask_mode"] == "output_only"


def test_run_viirs_inversion_applies_canopy_and_ice_snow_fraction_adjustment(monkeypatch, tmp_path):
    scene = build_mock_prepared_scene()
    r0 = build_mock_r0()
    canopy_fraction = xr.DataArray(
        np.full((2, 2), 0.20, dtype=np.float32),
        dims=("y", "x"),
        coords={"y": [0, 1], "x": [0, 1]},
        attrs={
            "TIFFTAG_SOFTWARE": "MATLAB",
            "_FillValue": -32768,
            "add_offset": 0.0,
            "DIMENSION_LIST": '[["<HDF5 object reference>"]]',
            "valid_range": [0, 18000],
        },
    )

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
        fsca = xr.DataArray(np.full((2, 2), 0.50, dtype=np.float32), dims=dims, coords=coords)
        return xr.Dataset(
            data_vars={
                "fsca": fsca,
                "fshade": xr.ones_like(fsca) * 0.10,
                "dust_concentration": xr.ones_like(fsca) * 10.0,
                "grain_size": xr.ones_like(fsca) * 250.0,
            }
        )

    monkeypatch.setattr("spires.sensors.full_workflow.LutInterpolator", DummyInterpolator)
    monkeypatch.setattr("spires.sensors.full_workflow.speedy_invert_dask", fake_speedy_invert_dask)

    result = run_viirs_inversion(
        scene,
        r0,
        lut_file=TEST_LUT_FILE,
        execution_profile="local",
        apply_valid_inversion_mask=False,
        canopy_fraction=canopy_fraction,
        ice_fraction=0.05,
    )

    assert result["raw_viewable_snow_fraction"].isel(y=0, x=0).item() == pytest.approx(0.50)
    assert result["raw_shade_fraction"].isel(y=0, x=0).item() == pytest.approx(0.10)
    assert result["raw_canopy_fraction"].isel(y=0, x=0).item() == pytest.approx(0.20)
    assert result["raw_snow_fraction"].isel(y=0, x=0).item() == pytest.approx(0.50 / (1.0 - 0.35))
    assert "TIFFTAG_SOFTWARE" not in result["raw_canopy_fraction"].attrs
    assert "_FillValue" not in result["raw_canopy_fraction"].attrs
    assert "DIMENSION_LIST" not in result["raw_canopy_fraction"].attrs
    assert result.attrs["canopy_correction_applied"] == 1
    assert result.attrs["ice_fraction_applied"] == 1
    result.to_netcdf(tmp_path / "viirs_canopy_inversion.nc")


def test_run_viirs_inversion_rejects_conflicting_mask_keywords(monkeypatch):
    scene = build_mock_prepared_scene()
    r0 = build_mock_r0()

    monkeypatch.setattr("spires.sensors.full_workflow.LutInterpolator", DummyInterpolator)

    with pytest.raises(ValueError, match="different values"):
        run_viirs_inversion(
            scene,
            r0,
            lut_file=TEST_LUT_FILE,
            apply_valid_inversion_mask=False,
            mask_with_valid_inversion_mask=True,
        )


def test_run_viirs_inversion_returns_netcdf_serializable_attrs(monkeypatch, tmp_path):
    scene = build_mock_prepared_scene()
    scene.attrs["units_by_band"] = {
        "I1": "percent reflectance",
        "I2": "percent reflectance",
        "I3": "percent reflectance",
    }
    scene["x"].attrs["_FillValue"] = "N/A"
    scene["x"].attrs["two_dimensional_attr"] = np.array([[1, 2], [3, 4]], dtype=np.int16)
    scene["valid_inversion_mask"].attrs["valid_range"] = np.array([0, 255], dtype=np.int16)
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

    result = run_viirs_inversion(scene, r0, lut_file=TEST_LUT_FILE, execution_profile="local")

    assert json.loads(result.attrs["units_by_band"])["I1"] == "percent reflectance"
    assert json.loads(result["x"].attrs["two_dimensional_attr"]) == [[1, 2], [3, 4]]
    assert result["valid_inversion_mask"].attrs["flag_values"] == [0, 1]
    assert "valid_range" not in result["valid_inversion_mask"].attrs
    result.to_netcdf(tmp_path / "viirs_inversion.nc")


def test_run_viirs_inversion_rejects_scene_r0_band_mismatch(monkeypatch):
    scene = build_mock_prepared_scene()
    r0 = build_mock_r0().sel(band=["I1", "I2", "I3", "M2", "M4", "M11", "M8"])

    monkeypatch.setattr("spires.sensors.full_workflow.LutInterpolator", DummyInterpolator)

    with pytest.raises(ValueError, match="Scene and R0 band order do not match"):
        run_viirs_inversion(scene, r0, lut_file=TEST_LUT_FILE)


def test_run_viirs_inversion_rejects_scene_lut_platform_mismatch(monkeypatch):
    scene = build_mock_prepared_scene()
    scene.attrs["platform"] = "snpp"
    r0 = build_mock_r0()

    monkeypatch.setattr("spires.sensors.full_workflow.LutInterpolator", DummyInterpolator)

    with pytest.raises(ValueError, match="does not match LUT platform"):
        run_viirs_inversion(scene, r0, lut_file=TEST_LUT_FILE)


def test_run_viirs_inversion_accepts_noaa21_lut_platform(monkeypatch, tmp_path):
    scene = build_mock_prepared_scene()
    scene.attrs["platform"] = "noaa21"
    r0 = build_mock_r0()
    lut_file = tmp_path / "lut_viirs_noaa21_i1_i2_i3_m2_m4_m8_m11_3um_dust_bandpass.mat"
    lut_file.touch()

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

    result = run_viirs_inversion(scene, r0, lut_file=lut_file)

    assert result.attrs["platform"] == "noaa21"


def test_default_viirs_canopy_lookup_is_platform_agnostic(monkeypatch, tmp_path):
    data_root = tmp_path / "data" / "viirs" / "ancillary" / "tiles" / "h08v05" / "static"
    data_root.mkdir(parents=True)
    canopy_path = data_root / "canopy_fraction.zarr"
    canopy_path.mkdir()
    scene = xr.Dataset(attrs={"platform": "noaa21", "tile": "h08v05"})

    monkeypatch.chdir(tmp_path)

    assert find_default_canopy_fraction(scene, sensor_name="viirs") == canopy_path
