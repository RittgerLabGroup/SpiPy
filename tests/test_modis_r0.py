import numpy as np

from spires.sensors.modis import (
    build_modis_r0,
    build_modis_r0_candidate_metrics,
    build_modis_timeseries,
)
from tests.test_modis_hdf import build_mock_modis_raw_dataset


def _scene_with_date(acquisition_date: str, values: list[float]):
    scene = build_mock_modis_raw_dataset()
    reflectance = scene["reflectance_500m"].copy()
    for band_index, value in enumerate(values):
        reflectance[..., band_index] = value
    scene["reflectance_500m"] = reflectance
    scene.attrs["acquisition_date"] = acquisition_date
    return scene


def test_build_modis_r0_candidate_metrics_marks_negative_ndsi_dates():
    scene = _scene_with_date("2026-06-15", [0.2, 0.7, 0.15, 0.1, 0.3, 0.4, 0.2])
    timeseries = build_modis_timeseries([scene], keep_variables="r0")

    candidates = build_modis_r0_candidate_metrics(timeseries)

    assert bool(candidates["has_negative_ndsi"].all())
    assert bool(candidates["candidate_negative_ndsi_mask"].all())


def test_build_modis_r0_uses_max_ndvi_when_any_negative_ndsi_exists():
    scene_1 = _scene_with_date("2026-06-15", [0.2, 0.6, 0.15, 0.1, 0.3, 0.4, 0.2])
    scene_2 = _scene_with_date("2026-06-16", [0.2, 0.8, 0.25, 0.1, 0.3, 0.4, 0.2])
    timeseries = build_modis_timeseries([scene_1, scene_2], keep_variables="r0")

    r0 = build_modis_r0(timeseries)

    assert bool((r0["r0_source_index"] == 1).all())
    assert bool((r0["r0_used_min_blue_rule"] == 0).all())


def test_build_modis_r0_uses_min_blue_when_ndsi_stays_positive():
    scene_1 = _scene_with_date("2026-06-15", [0.2, 0.6, 0.3, 0.5, 0.3, 0.2, 0.2])
    scene_2 = _scene_with_date("2026-06-16", [0.2, 0.8, 0.2, 0.5, 0.3, 0.2, 0.2])
    timeseries = build_modis_timeseries([scene_1, scene_2], keep_variables="r0")

    r0 = build_modis_r0(timeseries)

    assert bool((r0["r0_source_index"] == 1).all())
    assert bool(r0["r0_used_min_blue_rule"].all())


def test_build_modis_timeseries_can_write_reduced_stack_to_zarr(tmp_path):
    scene_1 = _scene_with_date("2026-06-15", [0.2, 0.6, 0.3, 0.5, 0.3, 0.2, 0.2])
    scene_2 = _scene_with_date("2026-06-16", [0.2, 0.8, 0.2, 0.5, 0.3, 0.2, 0.2])
    zarr_path = tmp_path / "modis_r0_stack.zarr"

    timeseries = build_modis_timeseries(
        [scene_1, scene_2],
        keep_variables="r0",
        zarr_path=zarr_path,
        chunks={"time": 1, "y": 1, "x": 1, "band": -1},
    )

    assert zarr_path.exists()
    assert {"reflectance", "sensor_zenith", "valid_r0_mask"}.issubset(timeseries.data_vars)
    assert timeseries.sizes["time"] == 2
