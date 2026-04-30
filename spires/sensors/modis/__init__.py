"""MODIS surface reflectance readers and helpers."""

from spires.sensors.modis.ancillary import (
    create_modis_ancillary_layout,
    modis_annual_ancillary_path,
    modis_ancillary_path,
    modis_ancillary_root,
    modis_sensor_root,
    modis_static_ancillary_path,
    modis_tile_ancillary_root,
)
from spires.sensors.modis.bands import (
    MODIS_DEFAULT_BAND_NAMES,
    MODIS_PRODUCT_TO_PLATFORM,
    normalize_modis_band_names,
    resolve_modis_inversion_bands,
)
from spires.sensors.modis.geospatial import MODIS_SINUSOIDAL_CRS
from spires.sensors.modis.hdf import (
    open_modis_surface_reflectance,
    parse_modis_surface_reflectance_filename,
    prepare_modis_scene_for_inversion,
)
from spires.sensors.modis.qa import decode_modis_qa_masks, load_external_cloud_masks
from spires.sensors.modis.r0 import (
    build_modis_r0,
    build_modis_r0_candidate_metrics,
    build_modis_r0_from_sources,
    build_modis_timeseries,
    compute_modis_r0_indices,
    reduce_modis_prepared_scene_for_r0,
)
from spires.sensors.modis.workflow import (
    AUTO_CANOPY_FRACTION,
    ModisExecutionProfile,
    get_modis_execution_profile,
    run_modis_inversion,
)

__all__ = [
    "create_modis_ancillary_layout",
    "MODIS_DEFAULT_BAND_NAMES",
    "MODIS_PRODUCT_TO_PLATFORM",
    "MODIS_SINUSOIDAL_CRS",
    "AUTO_CANOPY_FRACTION",
    "build_modis_r0",
    "build_modis_r0_candidate_metrics",
    "build_modis_r0_from_sources",
    "build_modis_timeseries",
    "compute_modis_r0_indices",
    "decode_modis_qa_masks",
    "get_modis_execution_profile",
    "load_external_cloud_masks",
    "ModisExecutionProfile",
    "modis_annual_ancillary_path",
    "modis_ancillary_path",
    "modis_ancillary_root",
    "modis_sensor_root",
    "modis_static_ancillary_path",
    "modis_tile_ancillary_root",
    "normalize_modis_band_names",
    "open_modis_surface_reflectance",
    "parse_modis_surface_reflectance_filename",
    "prepare_modis_scene_for_inversion",
    "reduce_modis_prepared_scene_for_r0",
    "resolve_modis_inversion_bands",
    "run_modis_inversion",
]
