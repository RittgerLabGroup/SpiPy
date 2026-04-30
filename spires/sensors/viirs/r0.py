"""Background reflectance (R0) helpers for VIIRS prepared scenes."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import xarray as xr

from spires.logging_utils import log_event
from spires.sensors.r0_core import (
    build_r0 as build_shared_r0,
    build_r0_candidate_metrics as build_shared_r0_candidate_metrics,
    build_timeseries as build_shared_timeseries,
    compute_r0_indices as compute_shared_r0_indices,
    gather_spectra_by_index,
    infer_source_date_bounds,
    load_existing_r0_if_valid,
    reduce_prepared_scene_for_r0 as reduce_shared_prepared_scene_for_r0,
    scalar_dataarray_to_float,
    select_time_indices,
    write_r0_dataset_atomically,
)
from spires.sensors.viirs.geospatial import copy_spatial_metadata
from spires.sensors.viirs.hdf import parse_viirs_surface_reflectance_filename, prepare_viirs_scene_for_inversion


VIIRS_R0_NDVI_RED_BAND = "I1"
VIIRS_R0_NDVI_NIR_BAND = "I2"
VIIRS_R0_NDSI_VISIBLE_BAND = "M4"
VIIRS_R0_NDSI_SWIR_BAND = "I3"
VIIRS_R0_BLUE_BAND = "M2"
VIIRS_R0_STAGING_VARIABLES = (
    "reflectance",
    "sensor_zenith",
    "valid_r0_mask",
)
LOGGER = logging.getLogger(__name__)


def reduce_viirs_prepared_scene_for_r0(prepared_ds: xr.Dataset) -> xr.Dataset:
    """Keep only the variables required for VIIRS R0 screening and compositing."""
    return reduce_shared_prepared_scene_for_r0(
        prepared_ds,
        staging_variables=VIIRS_R0_STAGING_VARIABLES,
        sensor_display_name="VIIRS",
        copy_spatial_metadata_fn=copy_spatial_metadata,
    )


def compute_viirs_r0_indices(
    prepared_ds: xr.Dataset,
    *,
    max_sensor_zenith: float = 30.0,
    ndvi_red_band: str = VIIRS_R0_NDVI_RED_BAND,
    ndvi_nir_band: str = VIIRS_R0_NDVI_NIR_BAND,
    ndsi_visible_band: str = VIIRS_R0_NDSI_VISIBLE_BAND,
    ndsi_swir_band: str = VIIRS_R0_NDSI_SWIR_BAND,
    blue_band: str = VIIRS_R0_BLUE_BAND,
    min_blue_reflectance: float = 0.10,
) -> xr.Dataset:
    """Compute the VIIRS screening indices used by the shared R0 compositing logic."""
    return compute_shared_r0_indices(
        prepared_ds,
        max_sensor_zenith=max_sensor_zenith,
        ndvi_red_band=ndvi_red_band,
        ndvi_nir_band=ndvi_nir_band,
        ndsi_visible_band=ndsi_visible_band,
        ndsi_swir_band=ndsi_swir_band,
        blue_band=blue_band,
        min_blue_reflectance=min_blue_reflectance,
    )


def build_viirs_r0_candidate_metrics(
    prepared_timeseries: xr.Dataset,
    *,
    max_sensor_zenith: float = 30.0,
    ndvi_red_band: str = VIIRS_R0_NDVI_RED_BAND,
    ndvi_nir_band: str = VIIRS_R0_NDVI_NIR_BAND,
    ndsi_visible_band: str = VIIRS_R0_NDSI_VISIBLE_BAND,
    ndsi_swir_band: str = VIIRS_R0_NDSI_SWIR_BAND,
    blue_band: str = VIIRS_R0_BLUE_BAND,
    min_blue_reflectance: float = 0.10,
) -> xr.Dataset:
    """Build screened VIIRS candidate metrics for downstream R0 selection."""
    return build_shared_r0_candidate_metrics(
        prepared_timeseries,
        max_sensor_zenith=max_sensor_zenith,
        ndvi_red_band=ndvi_red_band,
        ndvi_nir_band=ndvi_nir_band,
        ndsi_visible_band=ndsi_visible_band,
        ndsi_swir_band=ndsi_swir_band,
        blue_band=blue_band,
        min_blue_reflectance=min_blue_reflectance,
    )


def build_viirs_r0(
    prepared_timeseries: xr.Dataset,
    *,
    logger: logging.Logger | None = None,
    max_sensor_zenith: float = 30.0,
    ndvi_red_band: str = VIIRS_R0_NDVI_RED_BAND,
    ndvi_nir_band: str = VIIRS_R0_NDVI_NIR_BAND,
    ndsi_visible_band: str = VIIRS_R0_NDSI_VISIBLE_BAND,
    ndsi_swir_band: str = VIIRS_R0_NDSI_SWIR_BAND,
    blue_band: str = VIIRS_R0_BLUE_BAND,
    min_blue_reflectance: float = 0.10,
) -> xr.Dataset:
    """Build a VIIRS R0 composite from a prepared time series."""
    return build_shared_r0(
        prepared_timeseries,
        logger=logger or LOGGER,
        event_name="build_viirs_r0",
        max_sensor_zenith=max_sensor_zenith,
        ndvi_red_band=ndvi_red_band,
        ndvi_nir_band=ndvi_nir_band,
        ndsi_visible_band=ndsi_visible_band,
        ndsi_swir_band=ndsi_swir_band,
        blue_band=blue_band,
        min_blue_reflectance=min_blue_reflectance,
        copy_spatial_metadata_fn=copy_spatial_metadata,
    )


def build_viirs_timeseries(
    sources: list[str | Path | xr.Dataset],
    *,
    lut_file: str | Path | None = None,
    logger: logging.Logger | None = None,
    show_progress: bool = False,
    progress_desc: str = "Preparing VIIRS scenes",
    keep_variables: tuple[str, ...] | list[str] | str | None = None,
    zarr_path: str | Path | None = None,
    zarr_mode: str = "w",
    chunks: dict[str, int] | None = None,
    **prepare_kwargs,
) -> xr.Dataset:
    """Prepare and concatenate VIIRS scenes into a time stack for R0 workflows."""
    return build_shared_timeseries(
        sources,
        lut_file=lut_file,
        logger=logger or LOGGER,
        show_progress=show_progress,
        progress_desc=progress_desc,
        keep_variables=keep_variables,
        zarr_path=zarr_path,
        zarr_mode=zarr_mode,
        chunks=chunks,
        parse_filename_fn=parse_viirs_surface_reflectance_filename,
        prepare_scene_fn=prepare_viirs_scene_for_inversion,
        reduce_scene_for_r0_fn=reduce_viirs_prepared_scene_for_r0,
        start_event_name="build_viirs_timeseries",
        scene_event_name="build_viirs_timeseries_scene",
        summary_event_name="build_viirs_timeseries",
        **prepare_kwargs,
    )


def build_viirs_r0_from_sources(
    sources: list[str | Path | xr.Dataset],
    *,
    r0_path: str | Path | None = None,
    overwrite: bool = False,
    lut_file: str | Path | None = None,
    logger: logging.Logger | None = None,
    show_progress: bool = False,
    progress_desc: str = "Building VIIRS R0",
    max_sensor_zenith: float = 30.0,
    ndvi_red_band: str = VIIRS_R0_NDVI_RED_BAND,
    ndvi_nir_band: str = VIIRS_R0_NDVI_NIR_BAND,
    ndsi_visible_band: str = VIIRS_R0_NDSI_VISIBLE_BAND,
    ndsi_swir_band: str = VIIRS_R0_NDSI_SWIR_BAND,
    blue_band: str = VIIRS_R0_BLUE_BAND,
    min_blue_reflectance: float = 0.10,
    **prepare_kwargs,
) -> xr.Dataset:
    """Build a VIIRS R0 composite incrementally from input scenes."""
    logger = logger or LOGGER
    resolved_r0_path = Path(r0_path).expanduser().resolve() if r0_path is not None else None
    max_scene_index = len(sources) - 1 if sources else None

    if resolved_r0_path is not None and resolved_r0_path.exists() and not overwrite:
        result = load_existing_r0_if_valid(
            resolved_r0_path,
            expected_band_count=None,
            max_scene_index=max_scene_index,
        )
        if result is not None:
            log_event(
                logger,
                "build_viirs_r0_from_sources",
                stage="r0",
                event_type="summary",
                status="loaded_existing",
                r0_path=str(resolved_r0_path),
                time_coverage_start=result.attrs.get("time_coverage_start"),
                time_coverage_end=result.attrs.get("time_coverage_end"),
                selected_bands=result["band"].values.tolist() if "band" in result.coords else None,
                output_shape=list(result["r0_reflectance"].shape) if "r0_reflectance" in result else None,
            )
            return result
        log_event(
            logger,
            "build_viirs_r0_from_sources",
            stage="r0",
            event_type="detail",
            status="invalid_existing_rebuild",
            r0_path=str(resolved_r0_path),
        )

    iterator = sources
    if show_progress:
        try:
            from tqdm.auto import tqdm
        except ImportError as exc:
            raise ImportError("show_progress=True requires tqdm to be installed") from exc
        iterator = tqdm(sources, desc=progress_desc)

    requested_start_date, requested_end_date = infer_source_date_bounds(
        sources,
        parse_filename_fn=parse_viirs_surface_reflectance_filename,
    )
    total_sources = len(sources)
    prepared_count = 0

    log_event(
        logger,
        "build_viirs_r0_from_sources",
        stage="r0",
        event_type="start",
        status="started",
        scenes_requested=total_sources,
        requested_time_coverage_start=requested_start_date,
        requested_time_coverage_end=requested_end_date,
        r0_path=str(resolved_r0_path) if resolved_r0_path is not None else None,
    )

    first_prepared: xr.Dataset | None = None
    best_ndvi: np.ndarray | None = None
    best_ndvi_spectrum: np.ndarray | None = None
    best_ndvi_index: np.ndarray | None = None
    best_ndvi_time = None
    min_blue_metric_arr: np.ndarray | None = None
    min_blue_spectrum: np.ndarray | None = None
    min_blue_index: np.ndarray | None = None
    min_blue_time = None
    min_ndsi_arr: np.ndarray | None = None
    has_negative_ndsi_arr: np.ndarray | None = None
    valid_count_arr: np.ndarray | None = None
    band_values: list[str] | None = None
    observed_times: list[np.datetime64] = []

    for idx, source in enumerate(iterator):
        if isinstance(source, xr.Dataset):
            prepared = source
        else:
            prepared = prepare_viirs_scene_for_inversion(source, lut_file=lut_file, logger=logger, **prepare_kwargs)

        prepared = reduce_viirs_prepared_scene_for_r0(prepared)
        acquisition_date = prepared.attrs.get("acquisition_date")
        if acquisition_date is None:
            raise ValueError("Prepared VIIRS scene is missing 'acquisition_date' in attrs")

        time_value = np.datetime64(acquisition_date)
        observed_times.append(time_value)

        if first_prepared is None:
            first_prepared = prepared
            band_values = prepared["band"].values.tolist()
            spatial_shape = (prepared.sizes["y"], prepared.sizes["x"])
            spectral_shape = spatial_shape + (prepared.sizes["band"],)
            best_ndvi = np.full(spatial_shape, np.nan, dtype=np.float32)
            best_ndvi_spectrum = np.full(spectral_shape, np.nan, dtype=np.float32)
            best_ndvi_index = np.full(spatial_shape, -1, dtype=np.int32)
            best_ndvi_time = np.full(spatial_shape, np.datetime64("NaT"), dtype="datetime64[ns]")
            min_blue_metric_arr = np.full(spatial_shape, np.nan, dtype=np.float32)
            min_blue_spectrum = np.full(spectral_shape, np.nan, dtype=np.float32)
            min_blue_index = np.full(spatial_shape, -1, dtype=np.int32)
            min_blue_time = np.full(spatial_shape, np.datetime64("NaT"), dtype="datetime64[ns]")
            min_ndsi_arr = np.full(spatial_shape, np.nan, dtype=np.float32)
            has_negative_ndsi_arr = np.zeros(spatial_shape, dtype=bool)
            valid_count_arr = np.zeros(spatial_shape, dtype=np.int32)
        else:
            if prepared.sizes["y"] != first_prepared.sizes["y"] or prepared.sizes["x"] != first_prepared.sizes["x"]:
                raise ValueError("All VIIRS sources must share the same spatial shape for incremental R0 building")
            if prepared["band"].values.tolist() != band_values:
                raise ValueError("All VIIRS sources must share the same band order for incremental R0 building")

        indices_ds = compute_viirs_r0_indices(
            prepared,
            max_sensor_zenith=max_sensor_zenith,
            ndvi_red_band=ndvi_red_band,
            ndvi_nir_band=ndvi_nir_band,
            ndsi_visible_band=ndsi_visible_band,
            ndsi_swir_band=ndsi_swir_band,
            blue_band=blue_band,
            min_blue_reflectance=min_blue_reflectance,
        )

        reflectance_values = prepared["reflectance"].values.astype(np.float32, copy=False)
        valid_r0_mask = prepared["valid_r0_mask"].values.astype(bool, copy=False)
        ndsi_values = indices_ds["ndsi"].values.astype(np.float32, copy=False)
        ndvi_values = indices_ds["ndvi"].values.astype(np.float32, copy=False)
        blue_values = indices_ds["blue_metric"].values.astype(np.float32, copy=False)

        valid_count_arr += valid_r0_mask.astype(np.int32)
        has_negative_current = np.isfinite(ndsi_values) & (ndsi_values < 0)
        has_negative_ndsi_arr |= has_negative_current

        replace_min_ndsi = np.isfinite(ndsi_values) & (~np.isfinite(min_ndsi_arr) | (ndsi_values < min_ndsi_arr))
        min_ndsi_arr[replace_min_ndsi] = ndsi_values[replace_min_ndsi]

        candidate_ndvi = np.where(has_negative_current, ndvi_values, np.nan)
        replace_best_ndvi = np.isfinite(candidate_ndvi) & (~np.isfinite(best_ndvi) | (candidate_ndvi > best_ndvi))
        best_ndvi[replace_best_ndvi] = candidate_ndvi[replace_best_ndvi]
        best_ndvi_spectrum[replace_best_ndvi, :] = reflectance_values[replace_best_ndvi, :]
        best_ndvi_index[replace_best_ndvi] = idx
        best_ndvi_time[replace_best_ndvi] = time_value

        candidate_blue = np.where(valid_r0_mask, blue_values, np.nan)
        replace_min_blue = np.isfinite(candidate_blue) & (
            ~np.isfinite(min_blue_metric_arr) | (candidate_blue < min_blue_metric_arr)
        )
        min_blue_metric_arr[replace_min_blue] = candidate_blue[replace_min_blue]
        min_blue_spectrum[replace_min_blue, :] = reflectance_values[replace_min_blue, :]
        min_blue_index[replace_min_blue] = idx
        min_blue_time[replace_min_blue] = time_value

        prepared_count += 1

    if first_prepared is None or band_values is None:
        raise ValueError("At least one VIIRS scene is required to build incremental R0")

    use_min_blue = ~has_negative_ndsi_arr
    invalid_final = np.where(use_min_blue, ~np.isfinite(min_blue_metric_arr), ~np.isfinite(best_ndvi))
    r0_values = np.where(use_min_blue[..., None], min_blue_spectrum, best_ndvi_spectrum).astype(np.float32, copy=True)
    r0_values[invalid_final, :] = np.nan

    source_index = np.where(use_min_blue, min_blue_index, best_ndvi_index).astype(np.int32)
    source_index[invalid_final] = -1

    source_time = np.where(use_min_blue, min_blue_time, best_ndvi_time).astype("datetime64[ns]")
    source_time[invalid_final] = np.datetime64("NaT")
    valid_source = source_index >= 0

    result = xr.Dataset(
        data_vars={
            "r0_reflectance": xr.DataArray(
                r0_values,
                dims=("y", "x", "band"),
                coords={"y": first_prepared["y"].values, "x": first_prepared["x"].values, "band": first_prepared["band"].values},
            ),
            "r0_source_index": xr.DataArray(
                source_index,
                dims=("y", "x"),
                coords={"y": first_prepared["y"].values, "x": first_prepared["x"].values},
            ),
            "r0_source_time": xr.DataArray(
                source_time,
                dims=("y", "x"),
                coords={"y": first_prepared["y"].values, "x": first_prepared["x"].values},
            ),
            "r0_used_min_blue_rule": xr.DataArray(
                use_min_blue.astype(bool),
                dims=("y", "x"),
                coords={"y": first_prepared["y"].values, "x": first_prepared["x"].values},
            ),
            "r0_count": xr.DataArray(
                valid_count_arr.astype(np.int32),
                dims=("y", "x"),
                coords={"y": first_prepared["y"].values, "x": first_prepared["x"].values},
            ),
            "max_ndvi": xr.DataArray(
                best_ndvi,
                dims=("y", "x"),
                coords={"y": first_prepared["y"].values, "x": first_prepared["x"].values},
            ),
            "min_ndsi": xr.DataArray(
                min_ndsi_arr,
                dims=("y", "x"),
                coords={"y": first_prepared["y"].values, "x": first_prepared["x"].values},
            ),
            "min_blue_metric": xr.DataArray(
                min_blue_metric_arr,
                dims=("y", "x"),
                coords={"y": first_prepared["y"].values, "x": first_prepared["x"].values},
            ),
            "has_negative_ndsi": xr.DataArray(
                has_negative_ndsi_arr.astype(bool),
                dims=("y", "x"),
                coords={"y": first_prepared["y"].values, "x": first_prepared["x"].values},
            ),
        },
        attrs=first_prepared.attrs.copy(),
    )
    result = copy_spatial_metadata(first_prepared, result)

    result.attrs["time_coverage_start"] = str(min(observed_times))[:10]
    result.attrs["time_coverage_end"] = str(max(observed_times))[:10]
    result.attrs["build_status"] = "complete"

    used_min_blue_fraction = float(np.mean(use_min_blue)) if use_min_blue.size else float("nan")
    valid_source_fraction = float(np.mean(valid_source)) if valid_source.size else float("nan")

    if resolved_r0_path is not None:
        resolved_r0_path.parent.mkdir(parents=True, exist_ok=True)
        write_r0_dataset_atomically(
            result,
            resolved_r0_path,
            expected_band_count=len(band_values),
            max_scene_index=max_scene_index,
        )

    log_event(
        logger,
        "build_viirs_r0_from_sources",
        stage="r0",
        event_type="summary",
        status="completed",
        time_count=len(observed_times),
        time_coverage_start=result.attrs["time_coverage_start"],
        time_coverage_end=result.attrs["time_coverage_end"],
        selected_bands=band_values,
        output_shape=list(result["r0_reflectance"].shape),
        scenes_requested=total_sources,
        scenes_prepared=prepared_count,
        r0_path=str(resolved_r0_path) if resolved_r0_path is not None else None,
        used_min_blue_fraction=round(used_min_blue_fraction, 6),
        valid_source_fraction=round(valid_source_fraction, 6),
        mean_r0_count=round(float(valid_count_arr.mean()), 6),
    )

    return result
