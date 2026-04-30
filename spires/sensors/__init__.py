"""Sensor-specific readers, helpers, and generic dispatch APIs."""

from spires.sensors.api import (
    build_r0,
    build_r0_from_sources,
    build_timeseries,
    describe_sensor,
    get_execution_profile,
    load_output_dataset_if_valid,
    open_surface_reflectance,
    prepare_scene_for_inversion,
    run_inversion,
    validate_inversion_output_dataset,
    write_output_dataset,
)
from spires.sensors.registry import (
    SensorAdapter,
    get_sensor_adapter,
    list_supported_sensor_platforms,
    list_supported_sensors,
    normalize_platform_name,
    normalize_sensor_name,
)

__all__ = [
    "build_r0",
    "build_r0_from_sources",
    "build_timeseries",
    "describe_sensor",
    "get_execution_profile",
    "get_sensor_adapter",
    "load_output_dataset_if_valid",
    "list_supported_sensor_platforms",
    "list_supported_sensors",
    "normalize_platform_name",
    "normalize_sensor_name",
    "open_surface_reflectance",
    "prepare_scene_for_inversion",
    "run_inversion",
    "SensorAdapter",
    "validate_inversion_output_dataset",
    "write_output_dataset",
]
