"""High-level MODIS inversion workflow helpers."""

from __future__ import annotations

import logging
from pathlib import Path

import xarray as xr

from spires.sensors.full_workflow import (
    AUTO_CANOPY_FRACTION,
    SensorExecutionProfile,
    get_default_execution_profile,
    run_sensor_inversion,
)
from spires.sensors.io import sanitize_netcdf_dataset
from spires.sensors.modis.bands import (
    infer_modis_lut_band_names_from_metadata,
    infer_modis_lut_band_names_from_path,
    normalize_modis_band_names,
    resolve_modis_inversion_bands,
)
from spires.sensors.modis.geospatial import copy_spatial_metadata
from spires.sensors.modis.hdf import prepare_modis_scene_for_inversion


LOGGER = logging.getLogger(__name__)
ModisExecutionProfile = SensorExecutionProfile
_sanitize_netcdf_attrs = sanitize_netcdf_dataset


def get_modis_execution_profile(name: str) -> ModisExecutionProfile:
    """Return a named MODIS execution profile."""
    try:
        return get_default_execution_profile(name)
    except ValueError as exc:
        raise ValueError(f"Unknown MODIS execution profile: {name!r}") from exc


def _resolve_lut_band_names(lut_file: str | Path) -> list[str]:
    lut_bands = infer_modis_lut_band_names_from_metadata(lut_file)
    if lut_bands is None:
        lut_bands = infer_modis_lut_band_names_from_path(lut_file)
    if lut_bands is None:
        lut_bands = resolve_modis_inversion_bands(lut_file=lut_file)
    return normalize_modis_band_names(lut_bands)


def run_modis_inversion(
    scene,
    r0,
    *,
    lut_file: str | Path,
    client=None,
    bands: list[str] | tuple[str, ...] | None = None,
    apply_valid_inversion_mask: bool | None = None,
    mask_with_valid_inversion_mask: bool | None = None,
    chunk_config: dict[str, int] | None = None,
    scatter_lut: bool | None = None,
    max_eval: int = 100,
    x0=None,
    algorithm: int = 2,
    use_grouping: bool = False,
    grouping_method: str = "chunk_bin_mean",
    grouping_tolerance=0.02,
    grouping_reflectance_tol=None,
    grouping_background_tol=None,
    grouping_solar_zenith_tol=None,
    canopy_fraction=AUTO_CANOPY_FRACTION,
    ice_fraction=None,
    canopy_vertical_to_horizontal_crown_radius: float = 2.7,
    execution_profile: str | ModisExecutionProfile | None = None,
    logger: logging.Logger | None = None,
    **prepare_kwargs,
) -> xr.Dataset:
    """
    Run SPIRES inversion for a MODIS scene using a prepared or on-disk R0 background.

    Set ``apply_valid_inversion_mask=False`` to keep the generated
    ``valid_inversion_mask`` in the output while leaving inversion result
    variables unmasked. The legacy ``mask_with_valid_inversion_mask`` keyword is
    still accepted as an alias.
    """
    return run_sensor_inversion(
        scene,
        r0,
        sensor_name="modis",
        sensor_display_name="MODIS",
        event_name="run_modis_inversion",
        lut_file=lut_file,
        prepare_scene_fn=prepare_modis_scene_for_inversion,
        normalize_band_names_fn=normalize_modis_band_names,
        resolve_lut_band_names_fn=_resolve_lut_band_names,
        copy_spatial_metadata_fn=copy_spatial_metadata,
        client=client,
        bands=bands,
        apply_valid_inversion_mask=apply_valid_inversion_mask,
        mask_with_valid_inversion_mask=mask_with_valid_inversion_mask,
        chunk_config=chunk_config,
        scatter_lut=scatter_lut,
        max_eval=max_eval,
        x0=x0,
        algorithm=algorithm,
        use_grouping=use_grouping,
        grouping_method=grouping_method,
        grouping_tolerance=grouping_tolerance,
        grouping_reflectance_tol=grouping_reflectance_tol,
        grouping_background_tol=grouping_background_tol,
        grouping_solar_zenith_tol=grouping_solar_zenith_tol,
        canopy_fraction=canopy_fraction,
        ice_fraction=ice_fraction,
        canopy_vertical_to_horizontal_crown_radius=canopy_vertical_to_horizontal_crown_radius,
        execution_profile=execution_profile,
        logger=logger or LOGGER,
        **prepare_kwargs,
    )
