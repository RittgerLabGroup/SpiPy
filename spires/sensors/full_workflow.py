"""Shared full inversion workflow helpers for multisensor SPIReS adapters."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from numbers import Number
from pathlib import Path
from typing import Callable

import numpy as np
import xarray as xr

from spires.interpolator import LutInterpolator
from spires.invert import speedy_invert_dask
from spires.logging_utils import log_event
from spires.sensors.io import sanitize_netcdf_dataset


LOGGER = logging.getLogger(__name__)
AUTO_CANOPY_FRACTION = "auto"

PrepareSceneFn = Callable[..., xr.Dataset]
NormalizeBandNamesFn = Callable[[tuple[str, ...] | list[str]], list[str]]
ResolveLutBandNamesFn = Callable[[str | Path], list[str]]
CopySpatialMetadataFn = Callable[[xr.Dataset, xr.Dataset], xr.Dataset]
InferLutPlatformFn = Callable[[str | Path], str | None]


@dataclass(frozen=True)
class SensorExecutionProfile:
    """Execution settings tuned for local or distributed inversion workflows."""

    name: str
    chunks: dict[str, int]
    scatter_lut: bool
    persist_inputs: bool = False
    write_intermediate_zarr: bool = False


def get_default_execution_profile(name: str) -> SensorExecutionProfile:
    profiles = {
        "local": SensorExecutionProfile(
            name="local",
            chunks={"time": 1, "y": 256, "x": 256, "band": -1},
            scatter_lut=False,
        ),
        "cluster": SensorExecutionProfile(
            name="cluster",
            chunks={"time": 1, "y": 512, "x": 512, "band": -1},
            scatter_lut=True,
            persist_inputs=True,
            write_intermediate_zarr=True,
        ),
    }
    try:
        return profiles[name]
    except KeyError as exc:
        raise ValueError(f"Unknown execution profile: {name!r}") from exc


def open_dataset_like(source: str | Path | xr.Dataset | xr.DataArray, *, data_var_name: str | None = None) -> xr.Dataset:
    if isinstance(source, xr.Dataset):
        return source
    if isinstance(source, xr.DataArray):
        if data_var_name is None:
            data_var_name = source.name or "data"
        return source.to_dataset(name=data_var_name)

    path = Path(source).expanduser().resolve()
    if path.suffix == ".zarr":
        return xr.open_zarr(path)
    return xr.open_dataset(path)


def ensure_r0_dataset(r0: str | Path | xr.Dataset | xr.DataArray) -> xr.Dataset:
    return open_dataset_like(r0, data_var_name="r0_reflectance")


def ensure_prepared_scene(
    scene: str | Path | xr.Dataset | xr.DataArray,
    *,
    sensor_display_name: str,
    lut_file: str | Path,
    bands: tuple[str, ...] | list[str] | None,
    logger: logging.Logger,
    prepare_scene_fn: PrepareSceneFn,
    prepare_kwargs: dict[str, object],
) -> xr.Dataset:
    if isinstance(scene, xr.DataArray):
        raise TypeError(
            f"scene must be a prepared xr.Dataset or a {sensor_display_name} surface-reflectance path, not a DataArray"
        )
    if isinstance(scene, xr.Dataset):
        return scene
    return prepare_scene_fn(scene, bands=bands, lut_file=lut_file, logger=logger, **prepare_kwargs)


def find_default_canopy_fraction(scene_ds: xr.Dataset, *, sensor_name: str):
    tile = scene_ds.attrs.get("tile")
    if not tile:
        return None

    repo_root = Path(__file__).resolve().parents[2]
    candidate_roots = [Path.cwd(), repo_root]
    candidate_names = [
        f"{tile}_canopycover_LC100_global_v301_2019.tif",
        "canopy_fraction.tif",
        "canopy_fraction.tiff",
        "canopy_fraction.zarr",
    ]
    for root in candidate_roots:
        static_root = root / "data" / sensor_name / "ancillary" / "tiles" / tile / "static"
        for name in candidate_names:
            candidate = static_root / name
            if candidate.exists():
                return candidate
    return None


def resolve_canopy_fraction_source(canopy_fraction, scene_ds: xr.Dataset, *, sensor_name: str):
    if isinstance(canopy_fraction, str) and canopy_fraction == AUTO_CANOPY_FRACTION:
        return find_default_canopy_fraction(scene_ds, sensor_name=sensor_name)
    return canopy_fraction


def open_dataarray_like(source, *, data_var_name: str) -> xr.DataArray:
    if source is None:
        raise TypeError("source cannot be None")
    if isinstance(source, Number):
        return xr.DataArray(source)
    if isinstance(source, xr.DataArray):
        return source
    if isinstance(source, xr.Dataset):
        if data_var_name in source:
            return source[data_var_name]
        data_vars = list(source.data_vars)
        if len(data_vars) == 1:
            return source[data_vars[0]]
        raise ValueError(
            f"Dataset ancillary input must contain {data_var_name!r} or exactly one data variable"
        )

    path = Path(source).expanduser().resolve()
    if path.suffix.lower() in {".tif", ".tiff"}:
        try:
            import rioxarray
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Reading GeoTIFF ancillary inputs requires rioxarray") from exc
        data = rioxarray.open_rasterio(path, masked=True).squeeze(drop=True)
        return data.rename(data_var_name)

    return open_dataset_like(path, data_var_name=data_var_name)[data_var_name]


def as_yx_dataarray(data: xr.DataArray, template: xr.DataArray, *, name: str, sensor_display_name: str) -> xr.DataArray:
    if data.ndim == 0:
        return xr.zeros_like(template, dtype=np.float32) + data.astype(np.float32)

    rename_dims = {}
    if "y" not in data.dims:
        rename_dims[data.dims[-2]] = "y"
    if "x" not in data.dims:
        rename_dims[data.dims[-1]] = "x"
    if rename_dims:
        data = data.rename(rename_dims)

    if set(data.dims) != {"y", "x"}:
        data = data.squeeze(drop=True)
    if set(data.dims) != {"y", "x"}:
        raise ValueError(f"{name} must be a 2-D y/x raster after squeezing; got dims {data.dims}")

    data = data.transpose("y", "x")
    template_yx = template.transpose("y", "x")
    if data.sizes == template_yx.sizes:
        return data.assign_coords(y=template_yx["y"], x=template_yx["x"]).astype(np.float32).rename(name)

    try:
        data = data.interp(y=template_yx["y"], x=template_yx["x"], method="nearest")
    except Exception as exc:
        raise ValueError(
            f"{name} could not be aligned to the {sensor_display_name} scene grid. "
            f"For this temporary GeoTIFF path, provide data already on the prepared {sensor_display_name} y/x grid."
        ) from exc
    return data.astype(np.float32).rename(name)


def normalize_fraction(data: xr.DataArray) -> xr.DataArray:
    try:
        max_value = float(data.max(skipna=True))
    except (TypeError, ValueError):
        max_value = np.nan
    if np.isfinite(max_value) and max_value > 1.5:
        return data / 100.0
    return data


def resolve_fraction_layer(source, template: xr.DataArray, *, name: str, sensor_display_name: str) -> xr.DataArray:
    data = open_dataarray_like(source, data_var_name=name)
    data = as_yx_dataarray(data, template, name=name, sensor_display_name=sensor_display_name)
    data = normalize_fraction(data)
    return data.clip(min=0.0, max=1.0)


def view_adjust_canopy_fraction(
    canopy_fraction: xr.DataArray,
    sensor_zenith: xr.DataArray,
    *,
    vertical_to_horizontal_crown_radius: float,
) -> xr.DataArray:
    sensor_zenith = sensor_zenith.astype(np.float32)
    view_angle = np.arctan(vertical_to_horizontal_crown_radius * np.tan(np.deg2rad(sensor_zenith)))
    exponent = 1.0 / np.cos(view_angle)
    adjusted = 1.0 - (1.0 - canopy_fraction.astype(np.float32)) ** exponent
    adjusted = adjusted.clip(min=0.0, max=1.0).rename("raw_canopy_fraction")
    adjusted.attrs = {}
    adjusted.encoding.clear()
    return adjusted


def add_snow_fraction_layers(
    results: xr.Dataset,
    scene_ds: xr.Dataset,
    *,
    sensor_name: str,
    sensor_display_name: str,
    canopy_fraction,
    ice_fraction,
    canopy_vertical_to_horizontal_crown_radius: float,
) -> xr.Dataset:
    if "fsca" not in results or "fshade" not in results:
        return results

    canopy_fraction = resolve_canopy_fraction_source(canopy_fraction, scene_ds, sensor_name=sensor_name)
    results = results.rename({"fsca": "raw_viewable_snow_fraction", "fshade": "raw_shade_fraction"})
    raw_viewable = results["raw_viewable_snow_fraction"].clip(min=0.0, max=1.0)
    raw_shade = results["raw_shade_fraction"].clip(min=0.0, max=1.0)

    template = raw_viewable
    if canopy_fraction is None:
        raw_canopy = xr.zeros_like(template, dtype=np.float32).rename("raw_canopy_fraction")
        canopy_source = "none"
    else:
        if "sensor_zenith" not in scene_ds:
            raise ValueError("scene dataset must contain 'sensor_zenith' when canopy_fraction is provided")
        canopy = resolve_fraction_layer(
            canopy_fraction,
            template,
            name="canopy_fraction",
            sensor_display_name=sensor_display_name,
        )
        sensor_zenith = as_yx_dataarray(
            scene_ds["sensor_zenith"],
            template,
            name="sensor_zenith",
            sensor_display_name=sensor_display_name,
        )
        raw_canopy = view_adjust_canopy_fraction(
            canopy,
            sensor_zenith,
            vertical_to_horizontal_crown_radius=canopy_vertical_to_horizontal_crown_radius,
        )
        canopy_source = str(canopy_fraction) if not isinstance(canopy_fraction, (xr.DataArray, xr.Dataset)) else "xarray"

    if ice_fraction is None:
        ice = xr.zeros_like(template, dtype=np.float32).rename("ice_fraction")
        ice_source = "none"
    else:
        ice = resolve_fraction_layer(
            ice_fraction,
            template,
            name="ice_fraction",
            sensor_display_name=sensor_display_name,
        ).rename("ice_fraction")
        ice_source = str(ice_fraction) if not isinstance(ice_fraction, (xr.DataArray, xr.Dataset)) else "xarray"

    obscured_fraction = (raw_shade + raw_canopy + ice).clip(min=0.0, max=0.99)
    raw_snow = (raw_viewable / (1.0 - obscured_fraction)).clip(min=0.0, max=1.0)
    raw_snow = xr.where(raw_snow < ice, ice, raw_snow).rename("raw_snow_fraction")

    results["raw_viewable_snow_fraction"] = raw_viewable
    results["raw_shade_fraction"] = raw_shade
    results["raw_canopy_fraction"] = raw_canopy
    results["raw_snow_fraction"] = raw_snow

    results["raw_viewable_snow_fraction"].attrs.update(
        long_name=f"Raw viewable {sensor_display_name} SPIReS snow fraction",
        units="1",
    )
    results["raw_shade_fraction"].attrs.update(
        long_name=f"Raw {sensor_display_name} SPIReS shade fraction",
        units="1",
    )
    results["raw_canopy_fraction"].attrs.update(
        long_name="View-angle-adjusted canopy obstruction fraction",
        units="1",
        source=canopy_source,
        vertical_to_horizontal_crown_radius=canopy_vertical_to_horizontal_crown_radius,
    )
    results["raw_snow_fraction"].attrs.update(
        long_name=f"Raw {sensor_display_name} snow fraction adjusted for shade, canopy, and ice",
        units="1",
        ice_fraction_source=ice_source,
    )
    return results


def normalize_chunks(chunks: dict[str, int] | None, scene_ds: xr.Dataset, *, default_chunks: dict[str, int]) -> dict[str, int]:
    if chunks is None:
        return {dim: size for dim, size in default_chunks.items() if dim in scene_ds["reflectance"].dims}

    normalized: dict[str, int] = {}
    for dim, size in chunks.items():
        if dim not in scene_ds["reflectance"].dims:
            continue
        normalized[dim] = scene_ds.sizes[dim] if size == -1 else size
    return normalized


def chunk_if_possible(data: xr.DataArray, chunks: dict[str, int]) -> xr.DataArray:
    if not chunks:
        return data
    try:
        return data.chunk(chunks)
    except ValueError:
        filtered_chunks = {dim: size for dim, size in chunks.items() if dim in data.dims}
        if filtered_chunks:
            return data.chunk(filtered_chunks)
        return data


def resolve_valid_mask_application(
    *,
    apply_valid_inversion_mask: bool | None,
    mask_with_valid_inversion_mask: bool | None,
) -> bool:
    if apply_valid_inversion_mask is None and mask_with_valid_inversion_mask is None:
        return True
    if apply_valid_inversion_mask is None:
        return bool(mask_with_valid_inversion_mask)
    if mask_with_valid_inversion_mask is None:
        return bool(apply_valid_inversion_mask)
    if apply_valid_inversion_mask != mask_with_valid_inversion_mask:
        raise ValueError(
            "apply_valid_inversion_mask and mask_with_valid_inversion_mask were both provided with different values"
        )
    return bool(apply_valid_inversion_mask)


def validate_inversion_inputs(
    scene_ds: xr.Dataset,
    r0_ds: xr.Dataset,
    lut_file: str | Path,
    interpolator: LutInterpolator,
    *,
    normalize_band_names_fn: NormalizeBandNamesFn,
    resolve_lut_band_names_fn: ResolveLutBandNamesFn,
    infer_lut_platform_fn: InferLutPlatformFn | None = None,
) -> tuple[list[str], list[str], list[str]]:
    if "reflectance" not in scene_ds:
        raise ValueError("scene dataset must contain 'reflectance'")
    if "solar_zenith" not in scene_ds:
        raise ValueError("scene dataset must contain 'solar_zenith'")
    if "valid_inversion_mask" not in scene_ds:
        raise ValueError("scene dataset must contain 'valid_inversion_mask'")
    if "r0_reflectance" not in r0_ds:
        raise ValueError("R0 dataset must contain 'r0_reflectance'")

    scene_bands = normalize_band_names_fn(scene_ds["band"].values.tolist())
    r0_bands = normalize_band_names_fn(r0_ds["band"].values.tolist())
    lut_bands = resolve_lut_band_names_fn(lut_file)

    if scene_bands != r0_bands:
        raise ValueError(f"Scene and R0 band order do not match: {scene_bands} != {r0_bands}")
    if scene_bands != lut_bands:
        raise ValueError(f"Scene and LUT band order do not match: {scene_bands} != {lut_bands}")
    if len(scene_bands) != len(interpolator.bands):
        raise ValueError(
            "Scene band count and LUT interpolator band count do not match: "
            f"{len(scene_bands)} != {len(interpolator.bands)}"
        )

    if infer_lut_platform_fn is not None:
        scene_platform = scene_ds.attrs.get("platform")
        lut_platform = infer_lut_platform_fn(lut_file)
        if scene_platform is not None and lut_platform is not None and scene_platform != lut_platform:
            raise ValueError(f"Scene platform {scene_platform!r} does not match LUT platform {lut_platform!r}")

    return scene_bands, r0_bands, lut_bands


def run_sensor_inversion(
    scene: str | Path | xr.Dataset,
    r0: str | Path | xr.Dataset | xr.DataArray,
    *,
    sensor_name: str,
    sensor_display_name: str,
    event_name: str,
    lut_file: str | Path,
    prepare_scene_fn: PrepareSceneFn,
    normalize_band_names_fn: NormalizeBandNamesFn,
    resolve_lut_band_names_fn: ResolveLutBandNamesFn,
    copy_spatial_metadata_fn: CopySpatialMetadataFn,
    infer_lut_platform_fn: InferLutPlatformFn | None = None,
    client=None,
    bands: list[str] | tuple[str, ...] | None = None,
    apply_valid_inversion_mask: bool | None = None,
    mask_with_valid_inversion_mask: bool | None = None,
    chunk_config: dict[str, int] | None = None,
    scatter_lut: bool | None = None,
    max_eval: int = 100,
    x0: np.ndarray | None = None,
    algorithm: int = 2,
    use_grouping: bool = False,
    grouping_method: str = "chunk_bin_mean",
    grouping_tolerance: float | np.ndarray = 0.02,
    grouping_reflectance_tol: float | np.ndarray | None = None,
    grouping_background_tol: float | np.ndarray | None = None,
    grouping_solar_zenith_tol: float | np.ndarray | None = None,
    canopy_fraction=AUTO_CANOPY_FRACTION,
    ice_fraction=None,
    canopy_vertical_to_horizontal_crown_radius: float = 2.7,
    execution_profile: str | SensorExecutionProfile | None = None,
    logger: logging.Logger | None = None,
    **prepare_kwargs,
) -> xr.Dataset:
    logger = logger or LOGGER
    lut_file = Path(lut_file).expanduser().resolve()
    apply_valid_mask = resolve_valid_mask_application(
        apply_valid_inversion_mask=apply_valid_inversion_mask,
        mask_with_valid_inversion_mask=mask_with_valid_inversion_mask,
    )

    profile: SensorExecutionProfile | None = None
    if isinstance(execution_profile, str):
        profile = get_default_execution_profile(execution_profile)
    elif execution_profile is not None:
        profile = execution_profile

    scene_ds = ensure_prepared_scene(
        scene,
        sensor_display_name=sensor_display_name,
        lut_file=lut_file,
        bands=bands,
        logger=logger,
        prepare_scene_fn=prepare_scene_fn,
        prepare_kwargs=prepare_kwargs,
    )
    r0_ds = ensure_r0_dataset(r0)
    interpolator = LutInterpolator(lut_file=str(lut_file))

    if chunk_config is None and profile is not None:
        chunk_config = profile.chunks
    chunks = normalize_chunks(
        chunk_config,
        scene_ds,
        default_chunks=get_default_execution_profile("local").chunks,
    )
    if scatter_lut is None:
        scatter_lut = profile.scatter_lut if profile is not None else True

    scene_bands, _, _ = validate_inversion_inputs(
        scene_ds,
        r0_ds,
        lut_file,
        interpolator,
        normalize_band_names_fn=normalize_band_names_fn,
        resolve_lut_band_names_fn=resolve_lut_band_names_fn,
        infer_lut_platform_fn=infer_lut_platform_fn,
    )

    scene_reflectance = chunk_if_possible(scene_ds["reflectance"], chunks)
    scene_solar_zenith = chunk_if_possible(scene_ds["solar_zenith"], {k: v for k, v in chunks.items() if k != "band"})
    scene_valid_mask = chunk_if_possible(scene_ds["valid_inversion_mask"], {k: v for k, v in chunks.items() if k != "band"})
    inversion_valid_mask = scene_valid_mask if apply_valid_mask else None
    r0_reflectance = chunk_if_possible(r0_ds["r0_reflectance"], {k: v for k, v in chunks.items() if k != "time"})

    log_event(
        logger,
        event_name,
        stage="inversion",
        event_type="start",
        status="started",
        selected_bands=scene_bands,
        lut_file=str(lut_file),
        scene_platform=scene_ds.attrs.get("platform"),
        chunk_config=chunks,
        apply_valid_inversion_mask=apply_valid_mask,
        mask_with_valid_inversion_mask=apply_valid_mask,
        execution_profile=profile.name if profile is not None else None,
        use_grouping=use_grouping,
        grouping_method=grouping_method if use_grouping else None,
        grouping_tolerance=grouping_tolerance if use_grouping else None,
        grouping_reflectance_tol=grouping_reflectance_tol if use_grouping else None,
        grouping_background_tol=grouping_background_tol if use_grouping else None,
        grouping_solar_zenith_tol=grouping_solar_zenith_tol if use_grouping else None,
    )

    results = speedy_invert_dask(
        spectra_targets=scene_reflectance,
        spectra_backgrounds=r0_reflectance,
        obs_solar_angles=scene_solar_zenith,
        interpolator=interpolator,
        max_eval=max_eval,
        x0=np.array([0.5, 0.05, 10, 250], dtype=np.float64) if x0 is None else x0,
        algorithm=algorithm,
        client=client,
        scatter_lut=scatter_lut,
        valid_mask=inversion_valid_mask,
        use_grouping=use_grouping,
        grouping_method=grouping_method,
        grouping_tolerance=grouping_tolerance,
        grouping_reflectance_tol=grouping_reflectance_tol,
        grouping_background_tol=grouping_background_tol,
        grouping_solar_zenith_tol=grouping_solar_zenith_tol,
    )

    if apply_valid_mask:
        results = results.where(scene_valid_mask)

    resolved_canopy_fraction = resolve_canopy_fraction_source(canopy_fraction, scene_ds, sensor_name=sensor_name)
    results = add_snow_fraction_layers(
        results,
        scene_ds,
        sensor_name=sensor_name,
        sensor_display_name=sensor_display_name,
        canopy_fraction=resolved_canopy_fraction,
        ice_fraction=ice_fraction,
        canopy_vertical_to_horizontal_crown_radius=canopy_vertical_to_horizontal_crown_radius,
    )

    results = results.assign_coords(scene_ds["reflectance"].coords)
    results["valid_inversion_mask"] = scene_valid_mask.astype(bool)
    results["valid_inversion_mask"].attrs = {
        "long_name": f"Valid {sensor_display_name} SPIReS inversion mask",
        "flag_values": [0, 1],
        "flag_meanings": "invalid valid",
    }
    if "time" in scene_ds.coords:
        results = results.assign_coords(time=scene_ds["time"])

    results.attrs.update(scene_ds.attrs)
    results.attrs["lut_file"] = str(lut_file)
    results.attrs["selected_bands"] = scene_bands
    results.attrs["valid_inversion_mask_applied"] = apply_valid_mask
    results.attrs["valid_inversion_mask_mode"] = "applied_to_outputs" if apply_valid_mask else "output_only"
    results.attrs["canopy_correction_applied"] = resolved_canopy_fraction is not None
    results.attrs["canopy_vertical_to_horizontal_crown_radius"] = canopy_vertical_to_horizontal_crown_radius
    results.attrs["canopy_fraction_source"] = str(resolved_canopy_fraction) if resolved_canopy_fraction is not None else "none"
    results.attrs["ice_fraction_applied"] = ice_fraction is not None
    results.attrs["grouping_enabled"] = bool(use_grouping)
    results.attrs["grouping_method"] = grouping_method if use_grouping else "none"
    results.attrs["grouping_tolerance"] = grouping_tolerance if use_grouping else None
    results.attrs["grouping_reflectance_tol"] = grouping_reflectance_tol if use_grouping else None
    results.attrs["grouping_background_tol"] = grouping_background_tol if use_grouping else None
    results.attrs["grouping_solar_zenith_tol"] = grouping_solar_zenith_tol if use_grouping else None
    if profile is not None:
        results.attrs["execution_profile"] = profile.name
    results = copy_spatial_metadata_fn(scene_ds, results)
    results = sanitize_netcdf_dataset(results)

    log_event(
        logger,
        event_name,
        stage="inversion",
        event_type="summary",
        status="completed",
        selected_bands=scene_bands,
        lut_file=str(lut_file),
        output_shape=list(results["raw_viewable_snow_fraction"].shape),
        chunk_config=chunks,
        apply_valid_inversion_mask=apply_valid_mask,
        mask_with_valid_inversion_mask=apply_valid_mask,
        execution_profile=profile.name if profile is not None else None,
        use_grouping=use_grouping,
        grouping_method=grouping_method if use_grouping else None,
        grouping_tolerance=grouping_tolerance if use_grouping else None,
        grouping_reflectance_tol=grouping_reflectance_tol if use_grouping else None,
        grouping_background_tol=grouping_background_tol if use_grouping else None,
        grouping_solar_zenith_tol=grouping_solar_zenith_tol if use_grouping else None,
    )

    return results
