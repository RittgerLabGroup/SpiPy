"""Background reflectance (R0) helpers for VIIRS prepared scenes."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import xarray as xr

from spires.logging_utils import log_event
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


def _infer_source_date_bounds(
    sources: list[str | Path | xr.Dataset],
) -> tuple[str | None, str | None]:
    """Infer requested date bounds from source paths or dataset metadata when possible."""
    dates: list[np.datetime64] = []

    for source in sources:
        acquisition_date: str | None = None

        if isinstance(source, xr.Dataset):
            if "time" in source.coords and source["time"].size:
                time_values = source["time"].values
                dates.append(np.datetime64(time_values.min()))
                dates.append(np.datetime64(time_values.max()))
                continue

            acquisition_date = source.attrs.get("acquisition_date")
            if acquisition_date is None:
                source_path = source.attrs.get("source_path")
                if source_path:
                    try:
                        acquisition_date = parse_viirs_surface_reflectance_filename(source_path).acquisition_date
                    except ValueError:
                        acquisition_date = None
        else:
            try:
                acquisition_date = parse_viirs_surface_reflectance_filename(source).acquisition_date
            except ValueError:
                acquisition_date = None

        if acquisition_date is not None:
            dates.append(np.datetime64(acquisition_date))

    if not dates:
        return None, None

    return str(min(dates))[:10], str(max(dates))[:10]


def _safe_normalized_difference(numerator: xr.DataArray, denominator: xr.DataArray) -> xr.DataArray:
    """Compute a normalized difference while masking zero denominators."""
    total = numerator + denominator
    return xr.where(total != 0, (numerator - denominator) / total, np.nan)


def _scalar_dataarray_to_float(value: xr.DataArray) -> float:
    """Convert a scalar DataArray to float, computing it first if needed."""
    if hasattr(value.data, "compute"):
        value = value.compute()
    return float(value.item())


def reduce_viirs_prepared_scene_for_r0(prepared_ds: xr.Dataset) -> xr.Dataset:
    """Keep only the variables required for VIIRS R0 screening and compositing."""
    missing = [name for name in VIIRS_R0_STAGING_VARIABLES if name not in prepared_ds]
    if missing:
        raise ValueError(f"Prepared VIIRS scene is missing required R0 staging variables: {missing}")

    reduced = prepared_ds[list(VIIRS_R0_STAGING_VARIABLES)].copy()
    reduced.attrs = prepared_ds.attrs.copy()
    return copy_spatial_metadata(prepared_ds, reduced)


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
    """
    Compute the VIIRS indices used by the MODIS-inspired R0 compositing logic.

    Policy follows the existing MATLAB `createR0.m` design:
    - NDVI is masked by `valid_r0_mask`
    - NDSI is not masked by cloud/snow, but is masked for high view angles
    - the blue-band minimum uses `M2` as the MODIS band-3 substitute and
      excludes values below `min_blue_reflectance`
    """
    reflectance = prepared_ds["reflectance"]
    sensor_zenith = prepared_ds["sensor_zenith"]
    valid_r0_mask = prepared_ds["valid_r0_mask"]

    low_zenith = sensor_zenith <= max_sensor_zenith

    red = reflectance.sel(band=ndvi_red_band)
    nir = reflectance.sel(band=ndvi_nir_band)
    visible = reflectance.sel(band=ndsi_visible_band)
    swir = reflectance.sel(band=ndsi_swir_band)
    blue = reflectance.sel(band=blue_band)

    ndvi = _safe_normalized_difference(nir, red)
    ndvi = ndvi.where(valid_r0_mask & low_zenith)
    ndvi.name = "ndvi"

    ndsi = _safe_normalized_difference(visible, swir)
    ndsi = ndsi.where(low_zenith)
    ndsi.name = "ndsi"

    blue_metric = blue.where(blue >= min_blue_reflectance)
    blue_metric = blue_metric.where(low_zenith)
    blue_metric.name = "blue_metric"

    return xr.Dataset(
        data_vars={
            "ndvi": ndvi,
            "ndsi": ndsi,
            "blue_metric": blue_metric,
            "r0_low_sensor_zenith_mask": low_zenith.astype(bool),
        }
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
    """
    Build notebook-style screened candidate metrics for VIIRS R0 selection.

    Screening policy:
    - `candidate_ndvi`: observations that are valid for R0 and also have `ndsi < 0`
    - `candidate_blue_metric`: observations that are valid for R0 and have
      blue reflectance above the minimum threshold
    """
    indices_ds = compute_viirs_r0_indices(
        prepared_timeseries,
        max_sensor_zenith=max_sensor_zenith,
        ndvi_red_band=ndvi_red_band,
        ndvi_nir_band=ndvi_nir_band,
        ndsi_visible_band=ndsi_visible_band,
        ndsi_swir_band=ndsi_swir_band,
        blue_band=blue_band,
        min_blue_reflectance=min_blue_reflectance,
    )

    valid_r0_mask = prepared_timeseries["valid_r0_mask"]
    ndsi = indices_ds["ndsi"]
    ndvi = indices_ds["ndvi"]
    blue_metric = indices_ds["blue_metric"]

    has_negative_ndsi = (ndsi < 0).any(dim="time")
    candidate_negative_ndsi_mask = valid_r0_mask & (ndsi < 0)
    candidate_blue_mask = valid_r0_mask & blue_metric.notnull()

    candidate_ndvi = ndvi.where(candidate_negative_ndsi_mask)
    candidate_blue_metric = blue_metric.where(candidate_blue_mask)

    return xr.Dataset(
        data_vars={
            **indices_ds.data_vars,
            "candidate_ndvi": candidate_ndvi,
            "candidate_blue_metric": candidate_blue_metric,
            "candidate_negative_ndsi_mask": candidate_negative_ndsi_mask.astype(bool),
            "candidate_blue_mask": candidate_blue_mask.astype(bool),
            "has_negative_ndsi": has_negative_ndsi.astype(bool),
        }
    )


def _gather_spectra_by_index(
    reflectance: xr.DataArray,
    source_index: xr.DataArray,
    invalid_mask: xr.DataArray,
) -> xr.DataArray:
    """Gather spectra lazily from a `(time, y, x, band)` cube using per-pixel time indices."""
    source_index = xr.DataArray(
        source_index.data,
        dims=source_index.dims,
        coords={dim: source_index.coords[dim] for dim in source_index.dims},
    )
    invalid_mask = xr.DataArray(
        invalid_mask.data,
        dims=invalid_mask.dims,
        coords={dim: invalid_mask.coords[dim] for dim in invalid_mask.dims},
    )

    # xarray vectorized indexing does not currently support chunked indexers.
    # Materializing the 2-D per-pixel index is far cheaper than gathering the
    # full 4-D reflectance cube eagerly.
    if hasattr(source_index.data, "compute"):
        source_index = source_index.compute()
    if hasattr(invalid_mask.data, "compute"):
        invalid_mask = invalid_mask.compute()

    safe_index = source_index.where(~invalid_mask, other=0)
    gathered = reflectance.isel(time=safe_index)
    return gathered.where(~invalid_mask).astype(np.float32)


def _select_time_indices(metric: xr.DataArray, *, mode: str) -> tuple[xr.DataArray, xr.DataArray]:
    """Select per-pixel time indices from a metric with NaNs while preserving lazy execution."""
    metric = xr.DataArray(
        metric.data,
        dims=metric.dims,
        coords={dim: metric.coords[dim] for dim in metric.dims},
        name=metric.name,
    )
    invalid = metric.isnull().all(dim="time")

    if mode == "max":
        filled = metric.fillna(-np.inf)
        index = filled.argmax(dim="time")
    elif mode == "min":
        filled = metric.fillna(np.inf)
        index = filled.argmin(dim="time")
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    index = index.where(~invalid, other=-1).astype(np.int32)
    return index, invalid


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
    """
    Build a VIIRS R0 composite from a prepared time series.

    The selection rule mirrors the existing MODIS MATLAB implementation:
    - use the spectrum from the min-blue day if min-NDSI stays positive
    - otherwise use the spectrum from the max-NDVI day
    """
    if "time" not in prepared_timeseries.dims:
        raise ValueError("prepared_timeseries must have a 'time' dimension")
    logger = logger or LOGGER

    candidate_ds = build_viirs_r0_candidate_metrics(
        prepared_timeseries,
        max_sensor_zenith=max_sensor_zenith,
        ndvi_red_band=ndvi_red_band,
        ndvi_nir_band=ndvi_nir_band,
        ndsi_visible_band=ndsi_visible_band,
        ndsi_swir_band=ndsi_swir_band,
        blue_band=blue_band,
        min_blue_reflectance=min_blue_reflectance,
    )

    ndvi = candidate_ds["ndvi"]
    ndsi = candidate_ds["ndsi"]
    blue_metric = candidate_ds["blue_metric"]
    candidate_ndvi = candidate_ds["candidate_ndvi"]
    candidate_blue_metric = candidate_ds["candidate_blue_metric"]
    has_negative_ndsi = candidate_ds["has_negative_ndsi"]

    idx_max_ndvi, invalid_ndvi = _select_time_indices(candidate_ndvi, mode="max")
    idx_min_blue, invalid_blue = _select_time_indices(candidate_blue_metric, mode="min")

    reflectance = prepared_timeseries["reflectance"]
    spectra_from_max_ndvi = _gather_spectra_by_index(reflectance, idx_max_ndvi, invalid_ndvi)
    spectra_from_min_blue = _gather_spectra_by_index(reflectance, idx_min_blue, invalid_blue)

    min_ndsi = ndsi.min(dim="time", skipna=True)
    max_ndvi = candidate_ndvi.max(dim="time", skipna=True)
    min_blue = candidate_blue_metric.min(dim="time", skipna=True)

    use_min_blue = (~has_negative_ndsi).fillna(False).astype(bool)
    invalid_final = xr.where(use_min_blue, invalid_blue, invalid_ndvi)
    r0_values = xr.where(use_min_blue, spectra_from_min_blue, spectra_from_max_ndvi)
    r0_values = r0_values.where(~invalid_final).astype(np.float32)

    source_index = xr.where(use_min_blue, idx_min_blue, idx_max_ndvi)
    source_index = source_index.where(~invalid_final, other=-1).astype(np.int32)

    safe_source_index = xr.DataArray(
        source_index.data,
        dims=source_index.dims,
        coords={dim: source_index.coords[dim] for dim in source_index.dims},
    )
    if hasattr(safe_source_index.data, "compute"):
        safe_source_index = safe_source_index.compute()
    safe_source_index = safe_source_index.where(safe_source_index >= 0, other=0)
    source_time = prepared_timeseries["time"].isel(time=safe_source_index)
    source_time = source_time.where(source_index >= 0)

    valid_count = prepared_timeseries["valid_r0_mask"].sum(dim="time").astype(np.int32)
    band_values = prepared_timeseries["band"].values.tolist()
    time_values = prepared_timeseries["time"].values
    used_min_blue_fraction = _scalar_dataarray_to_float(use_min_blue.mean()) if use_min_blue.size else float("nan")
    valid_source = source_index >= 0
    valid_source_fraction = _scalar_dataarray_to_float(valid_source.mean()) if valid_source.size else float("nan")

    result = xr.Dataset(
        data_vars={
            "r0_reflectance": xr.DataArray(
                r0_values.data,
                dims=("y", "x", "band"),
                coords={
                    "y": prepared_timeseries["y"].values,
                    "x": prepared_timeseries["x"].values,
                    "band": prepared_timeseries["band"].values,
                },
            ),
            "r0_source_index": xr.DataArray(
                source_index.data,
                dims=("y", "x"),
                coords={"y": prepared_timeseries["y"].values, "x": prepared_timeseries["x"].values},
            ),
            "r0_source_time": xr.DataArray(
                source_time.data,
                dims=("y", "x"),
                coords={"y": prepared_timeseries["y"].values, "x": prepared_timeseries["x"].values},
            ),
            "r0_used_min_blue_rule": xr.DataArray(
                use_min_blue.data,
                dims=("y", "x"),
                coords={"y": prepared_timeseries["y"].values, "x": prepared_timeseries["x"].values},
            ),
            "r0_count": valid_count.astype(np.int32),
            "max_ndvi": max_ndvi,
            "min_ndsi": min_ndsi,
            "min_blue_metric": min_blue,
            "has_negative_ndsi": has_negative_ndsi,
        },
        attrs=prepared_timeseries.attrs.copy(),
    )
    result = copy_spatial_metadata(prepared_timeseries, result)

    log_event(
        logger,
        "build_viirs_r0",
        stage="r0",
        event_type="summary",
        time_count=int(prepared_timeseries.sizes["time"]),
        time_coverage_start=str(time_values.min())[:10],
        time_coverage_end=str(time_values.max())[:10],
        selected_bands=band_values,
        output_shape=list(result["r0_reflectance"].shape),
        used_min_blue_fraction=round(used_min_blue_fraction, 6),
        valid_source_fraction=round(valid_source_fraction, 6),
        mean_r0_count=round(_scalar_dataarray_to_float(valid_count.mean()), 6),
    )

    return result


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
    """Build VIIRS R0 incrementally from individual sources without forming a full time cube."""
    logger = logger or LOGGER
    resolved_r0_path = Path(r0_path).expanduser().resolve() if r0_path is not None else None

    if resolved_r0_path is not None and resolved_r0_path.exists() and not overwrite:
        result = xr.open_dataset(resolved_r0_path)
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

    iterator = sources

    if show_progress:
        try:
            from tqdm.auto import tqdm
        except ImportError as exc:
            raise ImportError("show_progress=True requires tqdm to be installed") from exc
        iterator = tqdm(sources, desc=progress_desc)

    requested_start_date, requested_end_date = _infer_source_date_bounds(sources)
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
            y_values = prepared["y"].values
            x_values = prepared["x"].values
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

    used_min_blue_fraction = float(np.mean(use_min_blue)) if use_min_blue.size else float("nan")
    valid_source_fraction = float(np.mean(valid_source)) if valid_source.size else float("nan")

    if resolved_r0_path is not None:
        resolved_r0_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_netcdf(resolved_r0_path)

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
    """Prepare and concatenate multiple VIIRS scenes into a single time stack."""
    prepared_scenes = []
    times = []
    iterator = sources
    logger = logger or LOGGER

    if show_progress:
        try:
            from tqdm.auto import tqdm
        except ImportError as exc:
            raise ImportError("show_progress=True requires tqdm to be installed") from exc
        iterator = tqdm(sources, desc=progress_desc)

    total_sources = len(sources)
    prepared_count = 0
    requested_start_date, requested_end_date = _infer_source_date_bounds(sources)
    resolved_zarr_path = Path(zarr_path).expanduser().resolve() if zarr_path is not None else None
    wrote_any_scene = False

    log_event(
        logger,
        "build_viirs_timeseries",
        stage="timeseries",
        event_type="start",
        status="started",
        scenes_requested=total_sources,
        requested_time_coverage_start=requested_start_date,
        requested_time_coverage_end=requested_end_date,
    )

    for idx, source in enumerate(iterator, start=1):
        if isinstance(source, xr.Dataset) and "time" in source.dims:
            prepared_scenes.append(source)
            times.extend(source["time"].values.tolist())
            prepared_count += int(source.sizes["time"])
            log_event(
                logger,
                "build_viirs_timeseries_scene",
                stage="timeseries",
                event_type="detail",
                status="reused_dataset",
                source_type="dataset_with_time",
                processed_scenes=f"{idx}/{total_sources}",
            )
            continue

        if isinstance(source, xr.Dataset) and "reflectance" in source.data_vars:
            prepared = source
            scene_name = prepared.attrs.get("source_path") or f"dataset_{idx}"
            log_event(
                logger,
                "build_viirs_timeseries_scene",
                stage="timeseries",
                event_type="detail",
                status="reused_dataset",
                source_type="prepared_dataset",
                scene_name=Path(scene_name).name,
                processed_scenes=f"{idx}/{total_sources}",
            )
        else:
            prepared = prepare_viirs_scene_for_inversion(source, lut_file=lut_file, logger=logger, **prepare_kwargs)
            log_event(
                logger,
                "build_viirs_timeseries_scene",
                stage="timeseries",
                event_type="detail",
                status="prepared",
                source_type="path",
                scene_name=Path(source).name,
                input_path=str(source),
                processed_scenes=f"{idx}/{total_sources}",
            )

        acquisition_date = prepared.attrs.get("acquisition_date")
        if acquisition_date is None:
            raise ValueError("Prepared VIIRS scene is missing 'acquisition_date' in attrs")

        if keep_variables == "r0":
            prepared = reduce_viirs_prepared_scene_for_r0(prepared)
        elif keep_variables is not None:
            variable_names = list(keep_variables)
            missing = [name for name in variable_names if name not in prepared]
            if missing:
                raise ValueError(f"Prepared VIIRS scene is missing requested variables: {missing}")
            prepared = prepared[variable_names].copy()
            prepared.attrs = prepared.attrs.copy()

        prepared = prepared.expand_dims(time=[np.datetime64(acquisition_date)])
        times.append(np.datetime64(acquisition_date))
        prepared_count += 1

        if chunks is not None:
            normalized_chunks = {
                dim: prepared.sizes[dim] if size == -1 else size
                for dim, size in chunks.items()
                if dim in prepared.dims
            }
            if normalized_chunks:
                prepared = prepared.chunk(normalized_chunks)

        if resolved_zarr_path is not None:
            if not wrote_any_scene:
                resolved_zarr_path.parent.mkdir(parents=True, exist_ok=True)
                prepared.to_zarr(resolved_zarr_path, mode=zarr_mode)
                wrote_any_scene = True
            else:
                prepared.to_zarr(resolved_zarr_path, mode="a", append_dim="time")
        else:
            prepared_scenes.append(prepared)

    if not prepared_scenes and resolved_zarr_path is None:
        raise ValueError("At least one VIIRS scene is required to build a timeseries")

    if resolved_zarr_path is not None:
        if not wrote_any_scene:
            raise ValueError("At least one VIIRS scene is required to build a timeseries")
        timeseries = xr.open_zarr(resolved_zarr_path)
    else:
        timeseries = xr.concat(prepared_scenes, dim="time")

    if "time" in timeseries.coords:
        timeseries = timeseries.sortby("time")
    timeseries.attrs["time_coverage_start"] = str(timeseries["time"].min().values)
    timeseries.attrs["time_coverage_end"] = str(timeseries["time"].max().values)
    log_event(
        logger,
        "build_viirs_timeseries",
        stage="timeseries",
        event_type="summary",
        status="completed",
        time_count=int(timeseries.sizes["time"]),
        time_coverage_start=timeseries.attrs["time_coverage_start"][:10],
        time_coverage_end=timeseries.attrs["time_coverage_end"][:10],
        selected_bands=timeseries["band"].values.tolist() if "band" in timeseries.coords else None,
        output_shape=list(timeseries["reflectance"].shape) if "reflectance" in timeseries else None,
        scenes_requested=total_sources,
        scenes_prepared=prepared_count,
        zarr_path=str(resolved_zarr_path) if resolved_zarr_path is not None else None,
    )
    return timeseries
