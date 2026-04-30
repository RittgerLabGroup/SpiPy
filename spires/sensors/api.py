"""Generic sensor-dispatch API built on top of the sensor registry."""

from __future__ import annotations

from pathlib import Path

import xarray as xr

from spires.sensors.io import (
    load_output_dataset_if_valid,
    validate_inversion_output_dataset,
    write_output_dataset,
)
from spires.sensors.registry import (
    describe_sensor,
    get_sensor_adapter,
    list_supported_sensor_platforms,
    list_supported_sensors,
    normalize_platform_name,
    normalize_sensor_name,
)


def open_surface_reflectance(
    source: str | Path,
    *,
    sensor: str,
    platform: str | None = None,
    **kwargs,
) -> xr.Dataset:
    """Open a surface-reflectance source using the registered sensor adapter."""
    adapter = get_sensor_adapter(sensor, platform)
    return adapter.open_surface_reflectance(source, **kwargs)


def prepare_scene_for_inversion(
    source,
    *,
    sensor: str,
    platform: str | None = None,
    **kwargs,
) -> xr.Dataset:
    """Prepare a scene for inversion using the registered sensor adapter."""
    adapter = get_sensor_adapter(sensor, platform)
    return adapter.prepare_scene_for_inversion(source, **kwargs)


def build_timeseries(
    sources,
    *,
    sensor: str,
    platform: str | None = None,
    **kwargs,
) -> xr.Dataset:
    """Build a prepared time series using the registered sensor adapter."""
    adapter = get_sensor_adapter(sensor, platform)
    return adapter.build_timeseries(sources, **kwargs)


def build_r0(
    prepared_timeseries: xr.Dataset,
    *,
    sensor: str,
    platform: str | None = None,
    **kwargs,
) -> xr.Dataset:
    """Build an R0 composite from a prepared time series."""
    adapter = get_sensor_adapter(sensor, platform)
    return adapter.build_r0(prepared_timeseries, **kwargs)


def build_r0_from_sources(
    sources,
    *,
    sensor: str,
    platform: str | None = None,
    **kwargs,
) -> xr.Dataset:
    """Build an R0 composite from raw or prepared input sources."""
    adapter = get_sensor_adapter(sensor, platform)
    return adapter.build_r0_from_sources(sources, **kwargs)


def run_inversion(
    scene,
    r0,
    *,
    sensor: str,
    platform: str | None = None,
    **kwargs,
) -> xr.Dataset:
    """Run SPIReS inversion using the registered sensor adapter."""
    adapter = get_sensor_adapter(sensor, platform)
    return adapter.run_inversion(scene, r0, **kwargs)


def get_execution_profile(
    name: str,
    *,
    sensor: str,
    platform: str | None = None,
):
    """Return a named execution profile for a registered sensor adapter."""
    adapter = get_sensor_adapter(sensor, platform)
    return adapter.get_execution_profile(name)


__all__ = [
    "build_r0",
    "build_r0_from_sources",
    "build_timeseries",
    "describe_sensor",
    "get_execution_profile",
    "load_output_dataset_if_valid",
    "list_supported_sensor_platforms",
    "list_supported_sensors",
    "normalize_platform_name",
    "normalize_sensor_name",
    "open_surface_reflectance",
    "prepare_scene_for_inversion",
    "run_inversion",
    "validate_inversion_output_dataset",
    "write_output_dataset",
]
