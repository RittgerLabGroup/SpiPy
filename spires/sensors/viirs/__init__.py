"""VIIRS surface reflectance readers."""

from spires.sensors.viirs.ancillary import (
    create_viirs_ancillary_layout,
    viirs_annual_ancillary_path,
    viirs_ancillary_path,
    viirs_ancillary_root,
    viirs_sensor_root,
    viirs_static_ancillary_path,
    viirs_tile_ancillary_root,
)
from spires.sensors.viirs.hdf import (
    open_viirs_surface_reflectance,
    parse_viirs_surface_reflectance_filename,
    prepare_viirs_scene_for_inversion,
)
from spires.sensors.viirs.qa import decode_viirs_qa_masks, load_external_cloud_masks
from spires.sensors.viirs.r0 import (
    build_viirs_r0,
    build_viirs_r0_from_sources,
    build_viirs_r0_candidate_metrics,
    build_viirs_timeseries,
    compute_viirs_r0_indices,
    reduce_viirs_prepared_scene_for_r0,
)
from spires.sensors.viirs.workflow import (
    ViirsExecutionProfile,
    get_viirs_execution_profile,
    run_viirs_inversion,
)

__all__ = [
    "build_viirs_r0",
    "build_viirs_r0_from_sources",
    "build_viirs_r0_candidate_metrics",
    "build_viirs_timeseries",
    "compute_viirs_r0_indices",
    "create_viirs_ancillary_layout",
    "decode_viirs_qa_masks",
    "get_viirs_execution_profile",
    "load_external_cloud_masks",
    "open_viirs_surface_reflectance",
    "parse_viirs_surface_reflectance_filename",
    "prepare_viirs_scene_for_inversion",
    "reduce_viirs_prepared_scene_for_r0",
    "run_viirs_inversion",
    "viirs_annual_ancillary_path",
    "viirs_ancillary_path",
    "viirs_ancillary_root",
    "viirs_sensor_root",
    "viirs_static_ancillary_path",
    "viirs_tile_ancillary_root",
    "ViirsExecutionProfile",
]
