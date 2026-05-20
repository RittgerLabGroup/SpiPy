import numpy as np
import pytest
import xarray as xr

from spires.logging_utils import configure_spires_file_logger
from spires.sensors import build_r0_from_sources as build_sensor_r0_from_sources
from spires.sensors.viirs.r0 import (
    build_viirs_r0,
    build_viirs_r0_candidate_metrics,
    build_viirs_timeseries,
    compute_viirs_r0_indices,
    reduce_viirs_prepared_scene_for_r0,
)


def build_mock_prepared_scene(
    acquisition_date: str,
    spectra: np.ndarray,
    *,
    sensor_zenith: np.ndarray | None = None,
    sensor_azimuth: np.ndarray | None = None,
) -> xr.Dataset:
    band_names = ["I1", "I2", "I3", "M2", "M4", "M8", "M11"]
    reflectance = xr.DataArray(
        spectra.reshape(1, 2, len(band_names)),
        dims=("y", "x", "band"),
        coords={"y": [0], "x": [0, 1], "band": band_names},
    )
    zenith_values = np.array([[10.0, 10.0]], dtype=np.float32) if sensor_zenith is None else sensor_zenith.astype(np.float32)
    azimuth_values = np.array([[0.0, 90.0]], dtype=np.float32) if sensor_azimuth is None else sensor_azimuth.astype(np.float32)
    zenith = xr.DataArray(zenith_values, dims=("y", "x"), coords={"y": [0], "x": [0, 1]})
    azimuth = xr.DataArray(azimuth_values, dims=("y", "x"), coords={"y": [0], "x": [0, 1]})
    valid_mask = xr.DataArray(np.array([[True, True]]), dims=("y", "x"), coords={"y": [0], "x": [0, 1]})
    return xr.Dataset(
        data_vars={
            "reflectance": reflectance,
            "sensor_zenith": zenith,
            "sensor_azimuth": azimuth,
            "valid_r0_mask": valid_mask,
        },
        coords={"y": [0], "x": [0, 1], "band": band_names},
        attrs={"acquisition_date": acquisition_date},
    )


def test_compute_viirs_r0_indices_uses_expected_bands():
    scene = build_mock_prepared_scene(
        "2026-06-01",
        np.array(
            [
                [0.2, 0.6, 0.1, 0.3, 0.5, 0.4, 0.4],
                [0.3, 0.4, 0.2, 0.25, 0.45, 0.35, 0.35],
            ],
            dtype=np.float32,
        ),
    )

    indices = compute_viirs_r0_indices(scene)

    expected_ndvi = (0.6 - 0.2) / (0.6 + 0.2)
    expected_ndsi = (0.5 - 0.1) / (0.5 + 0.1)

    assert np.isclose(indices["ndvi"].isel(y=0, x=0).item(), expected_ndvi)
    assert np.isclose(indices["ndsi"].isel(y=0, x=0).item(), expected_ndsi)
    assert np.isclose(indices["blue_metric"].isel(y=0, x=0).item(), 0.3)


def test_build_viirs_r0_matches_modis_style_selection_rule():
    # Pixel 0: no negative-NDSI candidate exists, so use the min-blue (M2) day => time index 1
    # Pixel 1: a negative-NDSI candidate exists, so use the max-NDVI day among those candidates => time index 0
    scene_1 = build_mock_prepared_scene(
        "2026-06-01",
        np.array(
            [
                [0.2, 0.3, 0.1, 0.30, 0.5, 0.4, 0.4],
                [0.2, 0.6, 0.4, 0.40, 0.2, 0.3, 0.3],
            ],
            dtype=np.float32,
        ),
    )
    scene_2 = build_mock_prepared_scene(
        "2026-07-01",
        np.array(
            [
                [0.2, 0.4, 0.1, 0.20, 0.3, 0.4, 0.4],
                [0.3, 0.5, 0.1, 0.20, 0.2, 0.3, 0.3],
            ],
            dtype=np.float32,
        ),
    )

    timeseries = build_viirs_timeseries([scene_1, scene_2])
    r0 = build_viirs_r0(timeseries)

    assert r0["r0_source_index"].isel(y=0, x=0).compute().item() == 1
    assert r0["r0_source_index"].isel(y=0, x=1).compute().item() == 0

    np.testing.assert_allclose(
        r0["r0_reflectance"].isel(y=0, x=0).values,
        scene_2["reflectance"].isel(y=0, x=0).values,
    )
    np.testing.assert_allclose(
        r0["r0_reflectance"].isel(y=0, x=1).values,
        scene_1["reflectance"].isel(y=0, x=1).values,
    )
    assert not bool(r0["has_negative_ndsi"].isel(y=0, x=0))
    assert bool(r0["has_negative_ndsi"].isel(y=0, x=1))


def test_build_viirs_r0_candidate_metrics_masks_ndvi_to_negative_ndsi_candidates():
    scene_1 = build_mock_prepared_scene(
        "2026-06-01",
        np.array(
            [
                [0.2, 0.5, 0.1, 0.30, 0.4, 0.4, 0.4],  # positive NDSI
                [0.2, 0.6, 0.4, 0.40, 0.2, 0.3, 0.3],  # negative NDSI
            ],
            dtype=np.float32,
        ),
    )
    scene_2 = build_mock_prepared_scene(
        "2026-07-01",
        np.array(
            [
                [0.2, 0.4, 0.1, 0.20, 0.3, 0.4, 0.4],  # positive NDSI
                [0.3, 0.5, 0.1, 0.20, 0.2, 0.3, 0.3],  # positive NDSI
            ],
            dtype=np.float32,
        ),
    )

    timeseries = build_viirs_timeseries([scene_1, scene_2])
    candidates = build_viirs_r0_candidate_metrics(timeseries)

    assert not bool(candidates["has_negative_ndsi"].isel(y=0, x=0))
    assert bool(candidates["has_negative_ndsi"].isel(y=0, x=1))
    assert np.isnan(candidates["candidate_ndvi"].isel(time=0, y=0, x=0))
    assert np.isnan(candidates["candidate_ndvi"].isel(time=1, y=0, x=0))
    assert not np.isnan(candidates["candidate_ndvi"].isel(time=0, y=0, x=1))


def test_build_viirs_r0_prefers_near_nadir_candidate_when_ndvi_is_within_epsilon():
    scene_1 = build_mock_prepared_scene(
        "2026-06-01",
        np.array(
            [
                [0.20, 0.596, 0.4, 0.30, 0.2, 0.4, 0.4],
                [0.2, 0.6, 0.4, 0.40, 0.2, 0.3, 0.3],
            ],
            dtype=np.float32,
        ),
        sensor_zenith=np.array([[28.0, 10.0]], dtype=np.float32),
        sensor_azimuth=np.array([[70.0, 10.0]], dtype=np.float32),
    )
    scene_2 = build_mock_prepared_scene(
        "2026-07-01",
        np.array(
            [
                [0.20, 0.58, 0.4, 0.20, 0.2, 0.4, 0.4],
                [0.3, 0.5, 0.1, 0.20, 0.2, 0.3, 0.3],
            ],
            dtype=np.float32,
        ),
        sensor_zenith=np.array([[10.0, 10.0]], dtype=np.float32),
        sensor_azimuth=np.array([[5.0, 20.0]], dtype=np.float32),
    )

    r0 = build_viirs_r0(build_viirs_timeseries([scene_1, scene_2]), ndvi_tie_epsilon=0.02)

    assert r0["r0_source_index"].isel(y=0, x=0).item() == 1
    assert np.isclose(r0["r0_sensor_zenith"].isel(y=0, x=0).item(), 10.0)
    assert np.isclose(r0["r0_sensor_azimuth"].isel(y=0, x=0).item(), 5.0)


def test_build_viirs_timeseries_concatenates_prepared_scenes_by_time():
    scene_2 = build_mock_prepared_scene("2026-07-01", np.full((2, 7), 0.2, dtype=np.float32))
    scene_1 = build_mock_prepared_scene("2026-06-01", np.full((2, 7), 0.1, dtype=np.float32))

    timeseries = build_viirs_timeseries([scene_2, scene_1])

    assert timeseries.sizes["time"] == 2
    assert str(timeseries["time"].values[0])[:10] == "2026-06-01"
    assert str(timeseries["time"].values[1])[:10] == "2026-07-01"


def test_build_viirs_timeseries_accepts_show_progress_false_by_default():
    scene = build_mock_prepared_scene("2026-06-01", np.full((2, 7), 0.1, dtype=np.float32))
    timeseries = build_viirs_timeseries([scene], show_progress=False)
    assert timeseries.sizes["time"] == 1


def test_reduce_viirs_prepared_scene_for_r0_keeps_only_required_variables():
    scene = build_mock_prepared_scene("2026-06-01", np.full((2, 7), 0.1, dtype=np.float32))
    scene["extra"] = xr.DataArray(np.ones((1, 2), dtype=np.float32), dims=("y", "x"), coords={"y": [0], "x": [0, 1]})

    reduced = reduce_viirs_prepared_scene_for_r0(scene)

    assert {"reflectance", "sensor_zenith", "sensor_azimuth", "valid_r0_mask"}.issubset(reduced.data_vars)


def test_build_viirs_timeseries_can_keep_only_r0_variables():
    scene_1 = build_mock_prepared_scene("2026-06-01", np.full((2, 7), 0.1, dtype=np.float32))
    scene_2 = build_mock_prepared_scene("2026-07-01", np.full((2, 7), 0.2, dtype=np.float32))
    scene_1["extra"] = xr.DataArray(np.ones((1, 2), dtype=np.float32), dims=("y", "x"), coords={"y": [0], "x": [0, 1]})
    scene_2["extra"] = xr.DataArray(np.ones((1, 2), dtype=np.float32), dims=("y", "x"), coords={"y": [0], "x": [0, 1]})

    timeseries = build_viirs_timeseries([scene_1, scene_2], keep_variables="r0")

    assert {"reflectance", "sensor_zenith", "sensor_azimuth", "valid_r0_mask"}.issubset(timeseries.data_vars)


def test_build_viirs_timeseries_can_write_reduced_stack_to_zarr(tmp_path):
    scene_1 = build_mock_prepared_scene("2026-06-01", np.full((2, 7), 0.1, dtype=np.float32))
    scene_2 = build_mock_prepared_scene("2026-07-01", np.full((2, 7), 0.2, dtype=np.float32))
    zarr_path = tmp_path / "viirs_r0_stack.zarr"

    timeseries = build_viirs_timeseries(
        [scene_1, scene_2],
        keep_variables="r0",
        zarr_path=zarr_path,
        chunks={"time": 1, "y": 1, "x": 1, "band": -1},
    )

    assert zarr_path.exists()
    assert {"reflectance", "sensor_zenith", "sensor_azimuth", "valid_r0_mask"}.issubset(timeseries.data_vars)
    assert timeseries.sizes["time"] == 2


def test_unified_viirs_r0_from_sources_matches_timeseries_builder():
    scene_1 = build_mock_prepared_scene(
        "2026-06-01",
        np.array(
            [
                [0.2, 0.3, 0.1, 0.30, 0.5, 0.4, 0.4],
                [0.2, 0.6, 0.4, 0.40, 0.2, 0.3, 0.3],
            ],
            dtype=np.float32,
        ),
    )
    scene_2 = build_mock_prepared_scene(
        "2026-07-01",
        np.array(
            [
                [0.2, 0.4, 0.1, 0.20, 0.3, 0.4, 0.4],
                [0.3, 0.5, 0.1, 0.20, 0.2, 0.3, 0.3],
            ],
            dtype=np.float32,
        ),
    )

    from_timeseries = build_viirs_r0(build_viirs_timeseries([scene_1, scene_2]))
    from_sources = build_sensor_r0_from_sources([scene_1, scene_2], sensor="viirs", platform="snpp")

    np.testing.assert_allclose(from_timeseries["r0_reflectance"].values, from_sources["r0_reflectance"].values)
    np.testing.assert_array_equal(from_timeseries["r0_source_index"].values, from_sources["r0_source_index"].values)
    np.testing.assert_array_equal(
        from_timeseries["r0_used_min_blue_rule"].values,
        from_sources["r0_used_min_blue_rule"].values,
    )


def test_build_viirs_r0_writes_selected_sensor_angles():
    scene_1 = build_mock_prepared_scene(
        "2026-06-01",
        np.array(
            [
                [0.2, 0.3, 0.1, 0.30, 0.5, 0.4, 0.4],
                [0.2, 0.6, 0.4, 0.40, 0.2, 0.3, 0.3],
            ],
            dtype=np.float32,
        ),
        sensor_zenith=np.array([[12.0, 14.0]], dtype=np.float32),
        sensor_azimuth=np.array([[25.0, 35.0]], dtype=np.float32),
    )
    scene_2 = build_mock_prepared_scene(
        "2026-07-01",
        np.array(
            [
                [0.2, 0.4, 0.1, 0.20, 0.3, 0.4, 0.4],
                [0.3, 0.5, 0.1, 0.20, 0.2, 0.3, 0.3],
            ],
            dtype=np.float32,
        ),
        sensor_zenith=np.array([[18.0, 20.0]], dtype=np.float32),
        sensor_azimuth=np.array([[55.0, 65.0]], dtype=np.float32),
    )

    r0 = build_viirs_r0(build_viirs_timeseries([scene_1, scene_2]))

    assert np.isclose(r0["r0_sensor_zenith"].isel(y=0, x=0).item(), 18.0)
    assert np.isclose(r0["r0_sensor_azimuth"].isel(y=0, x=0).item(), 55.0)
    assert np.isclose(r0["r0_sensor_zenith"].isel(y=0, x=1).item(), 14.0)
    assert np.isclose(r0["r0_sensor_azimuth"].isel(y=0, x=1).item(), 35.0)


def test_unified_viirs_r0_from_sources_loads_existing_file_and_logs_path(tmp_path):
    log_path = tmp_path / "viirs_r0.log"
    logger = configure_spires_file_logger(log_path, logger_name="spires.test.viirs.r0.reuse", log_to_stdout=False)
    r0_path = tmp_path / "existing_r0.nc"

    scene = build_mock_prepared_scene("2026-06-01", np.full((2, 7), 0.2, dtype=np.float32))
    original = build_sensor_r0_from_sources([scene], sensor="viirs", platform="snpp", r0_path=r0_path, logger=logger)

    loaded = build_sensor_r0_from_sources([], sensor="viirs", platform="snpp", r0_path=r0_path, logger=logger)

    assert r0_path.exists()
    np.testing.assert_allclose(original["r0_reflectance"].values, loaded["r0_reflectance"].values)

    for handler in logger.handlers:
        handler.flush()

    contents = log_path.read_text()
    assert 'status="loaded_existing"' in contents
    assert f'r0_path="{r0_path.resolve()}"' in contents


def test_unified_viirs_r0_from_sources_drops_scene_specific_attrs(tmp_path):
    r0_path = tmp_path / "snpp_r0_h08v05_2026.nc"
    scene = build_mock_prepared_scene("2026-06-01", np.full((2, 7), 0.2, dtype=np.float32))
    scene.attrs["processing_timestamp"] = "2026160123456"
    scene.attrs["lut_file"] = "/tmp/lut_viirs.mat"

    result = build_sensor_r0_from_sources([scene], sensor="viirs", platform="snpp", r0_path=r0_path)

    assert "acquisition_date" not in result.attrs
    assert "processing_timestamp" not in result.attrs
    assert "lut_file" not in result.attrs

    written = xr.open_dataset(r0_path)
    try:
        assert "acquisition_date" not in written.attrs
        assert "processing_timestamp" not in written.attrs
        assert "lut_file" not in written.attrs
    finally:
        written.close()


def test_build_viirs_r0_accepts_chunked_timeseries():
    da = pytest.importorskip("dask.array")

    scene_1 = build_mock_prepared_scene(
        "2026-06-01",
        np.array(
            [
                [0.2, 0.3, 0.1, 0.30, 0.5, 0.4, 0.4],
                [0.2, 0.6, 0.4, 0.40, 0.2, 0.3, 0.3],
            ],
            dtype=np.float32,
        ),
    )
    scene_2 = build_mock_prepared_scene(
        "2026-07-01",
        np.array(
            [
                [0.2, 0.4, 0.1, 0.20, 0.3, 0.4, 0.4],
                [0.3, 0.5, 0.1, 0.20, 0.2, 0.3, 0.3],
            ],
            dtype=np.float32,
        ),
    )

    timeseries = build_viirs_timeseries([scene_1, scene_2]).chunk({"time": -1, "y": 1, "x": 1, "band": -1})
    chunked = timeseries.copy()
    chunked["reflectance"] = xr.DataArray(
        da.from_array(timeseries["reflectance"].values, chunks=(2, 1, 1, 7)),
        dims=timeseries["reflectance"].dims,
        coords=timeseries["reflectance"].coords,
    )
    chunked["sensor_zenith"] = xr.DataArray(
        da.from_array(timeseries["sensor_zenith"].values, chunks=(2, 1, 1)),
        dims=timeseries["sensor_zenith"].dims,
        coords=timeseries["sensor_zenith"].coords,
    )
    chunked["valid_r0_mask"] = xr.DataArray(
        da.from_array(timeseries["valid_r0_mask"].values, chunks=(2, 1, 1)),
        dims=timeseries["valid_r0_mask"].dims,
        coords=timeseries["valid_r0_mask"].coords,
    )

    r0 = build_viirs_r0(chunked)

    assert r0["r0_source_index"].isel(y=0, x=0).compute().item() == 1
    assert r0["r0_source_index"].isel(y=0, x=1).compute().item() == 0


def test_build_viirs_timeseries_logs_batch_start_before_scene_details(tmp_path):
    log_path = tmp_path / "viirs_timeseries.log"
    logger = configure_spires_file_logger(log_path, logger_name="spires.test.viirs", log_to_stdout=False)

    scene_2 = build_mock_prepared_scene("2026-07-01", np.full((2, 7), 0.2, dtype=np.float32))
    scene_1 = build_mock_prepared_scene("2026-06-01", np.full((2, 7), 0.1, dtype=np.float32))

    build_viirs_timeseries([scene_2, scene_1], logger=logger)

    for handler in logger.handlers:
        handler.flush()

    contents = log_path.read_text()
    assert 'event="build_viirs_timeseries" stage="timeseries" event_type="start" status="started"' in contents
    assert 'requested_time_coverage_start="2026-06-01"' in contents
    assert 'requested_time_coverage_end="2026-07-01"' in contents
    assert contents.index('event="build_viirs_timeseries" stage="timeseries" event_type="start" status="started"') < contents.index(
        'event="build_viirs_timeseries_scene" stage="timeseries" event_type="detail"'
    )
