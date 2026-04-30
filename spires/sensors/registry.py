"""Sensor registry and adapter definitions for generic sensor workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import xarray as xr

import spires.sensors.modis as modis
import spires.sensors.viirs as viirs


OpenFn = Callable[..., xr.Dataset]
PrepareFn = Callable[..., xr.Dataset]
BuildTimeseriesFn = Callable[..., xr.Dataset]
BuildR0Fn = Callable[..., xr.Dataset]
BuildR0FromSourcesFn = Callable[..., xr.Dataset]
RunInversionFn = Callable[..., xr.Dataset]
GetExecutionProfileFn = Callable[[str], object]


@dataclass(frozen=True)
class SensorAdapter:
    """Callable adapter surface for a supported sensor family."""

    sensor: str
    supported_platforms: tuple[str, ...]
    open_surface_reflectance: OpenFn
    prepare_scene_for_inversion: PrepareFn
    build_timeseries: BuildTimeseriesFn
    build_r0: BuildR0Fn
    build_r0_from_sources: BuildR0FromSourcesFn
    run_inversion: RunInversionFn
    get_execution_profile: GetExecutionProfileFn
    notes: str = ""


SENSOR_ALIASES = {
    "modis": "modis",
    "terra": "modis",
    "aqua": "modis",
    "viirs": "viirs",
    "snpp": "viirs",
    "npp": "viirs",
    "noaa20": "viirs",
    "j01": "viirs",
    "noaa21": "viirs",
    "j02": "viirs",
}

PLATFORM_ALIASES = {
    "modis": {
        "terra": "terra",
        "mod09ga": "terra",
        "aqua": "aqua",
        "myd09ga": "aqua",
    },
    "viirs": {
        "snpp": "snpp",
        "npp": "snpp",
        "vnp09ga": "snpp",
        "noaa20": "noaa20",
        "j01": "noaa20",
        "vj109ga": "noaa20",
        "noaa21": "noaa21",
        "j02": "noaa21",
        "vj209ga": "noaa21",
    },
}


def _normalize_token(value: str | Path | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower().replace("_", "").replace("-", "").replace(" ", "")
    return text or None


def normalize_sensor_name(sensor: str) -> str:
    """Return the canonical sensor family name."""
    key = _normalize_token(sensor)
    if key is None:
        raise ValueError("sensor must be provided")
    try:
        return SENSOR_ALIASES[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported sensor {sensor!r}") from exc


def normalize_platform_name(sensor: str, platform: str | None) -> str | None:
    """Return the canonical platform name for a sensor, if provided."""
    if platform is None:
        return None
    canonical_sensor = normalize_sensor_name(sensor)
    key = _normalize_token(platform)
    if key is None:
        return None
    aliases = PLATFORM_ALIASES.get(canonical_sensor, {})
    try:
        return aliases[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported platform {platform!r} for sensor {canonical_sensor!r}") from exc


MODIS_ADAPTER = SensorAdapter(
    sensor="modis",
    supported_platforms=("terra", "aqua"),
    open_surface_reflectance=modis.open_modis_surface_reflectance,
    prepare_scene_for_inversion=modis.prepare_modis_scene_for_inversion,
    build_timeseries=modis.build_modis_timeseries,
    build_r0=modis.build_modis_r0,
    build_r0_from_sources=modis.build_modis_r0_from_sources,
    run_inversion=modis.run_modis_inversion,
    get_execution_profile=modis.get_modis_execution_profile,
    notes="Shared MODIS LUT workflow for Terra and Aqua.",
)

VIIRS_ADAPTER = SensorAdapter(
    sensor="viirs",
    supported_platforms=("snpp", "noaa20", "noaa21"),
    open_surface_reflectance=viirs.open_viirs_surface_reflectance,
    prepare_scene_for_inversion=viirs.prepare_viirs_scene_for_inversion,
    build_timeseries=viirs.build_viirs_timeseries,
    build_r0=viirs.build_viirs_r0,
    build_r0_from_sources=viirs.build_r0_from_sources,
    run_inversion=viirs.run_viirs_inversion,
    get_execution_profile=viirs.get_viirs_execution_profile,
    notes="Platform-specific LUT workflow for SNPP, NOAA-20, and NOAA-21.",
)

SENSOR_REGISTRY: dict[str, SensorAdapter] = {
    "modis": MODIS_ADAPTER,
    "viirs": VIIRS_ADAPTER,
}


def get_sensor_adapter(sensor: str, platform: str | None = None) -> SensorAdapter:
    """Return the adapter for a sensor/platform pair."""
    canonical_sensor = normalize_sensor_name(sensor)
    canonical_platform = normalize_platform_name(canonical_sensor, platform)
    adapter = SENSOR_REGISTRY[canonical_sensor]
    if canonical_platform is not None and canonical_platform not in adapter.supported_platforms:
        raise ValueError(
            f"Platform {canonical_platform!r} is not supported for sensor {canonical_sensor!r}. "
            f"Supported platforms: {adapter.supported_platforms}"
        )
    return adapter


def list_supported_sensors() -> tuple[str, ...]:
    """Return the canonical sensor families currently registered."""
    return tuple(SENSOR_REGISTRY)


def list_supported_sensor_platforms() -> dict[str, tuple[str, ...]]:
    """Return canonical platform names for each registered sensor family."""
    return {
        sensor: adapter.supported_platforms
        for sensor, adapter in SENSOR_REGISTRY.items()
    }


def describe_sensor(sensor: str, platform: str | None = None) -> dict[str, Any]:
    """Return a compact metadata summary for a registered adapter."""
    adapter = get_sensor_adapter(sensor, platform)
    return {
        "sensor": adapter.sensor,
        "supported_platforms": adapter.supported_platforms,
        "notes": adapter.notes,
    }
