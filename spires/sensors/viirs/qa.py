"""QA decoding and external mask helpers for VIIRS surface reflectance scenes."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import xarray as xr


def _extract_bits(values: xr.DataArray, start_bit: int, width: int = 1) -> xr.DataArray:
    """Extract a bitfield from an unsigned integer QA byte array."""
    mask = (1 << width) - 1
    data = (values.astype(np.uint16) >> start_bit) & mask
    coords = {dim: values.coords[dim].values for dim in values.dims}
    return xr.DataArray(data.astype(np.uint8), dims=values.dims, coords=coords)


def decode_viirs_qa_masks(
    qa_qf1: xr.DataArray,
    qa_qf2: xr.DataArray,
    qa_qf7: xr.DataArray,
) -> xr.Dataset:
    """
    Decode core VIIRS QA masks used for inversion and R0 workflows.

    Current policy:
    - cloud: probably cloudy, confidently cloudy, thin cirrus, or adjacent-to-cloud
    - cloud shadow: native shadow bit
    - snow: native snow/ice or snow-present flags
    """
    cloud_confidence = _extract_bits(qa_qf1, start_bit=2, width=2)
    cloud_mask_quality = _extract_bits(qa_qf1, start_bit=0, width=2)

    qf2_shadow = _extract_bits(qa_qf2, start_bit=3).astype(bool)
    qf2_snow_ice = _extract_bits(qa_qf2, start_bit=5).astype(bool)
    qf2_thin_cirrus_reflective = _extract_bits(qa_qf2, start_bit=6).astype(bool)
    qf2_thin_cirrus_emissive = _extract_bits(qa_qf2, start_bit=7).astype(bool)

    qf7_thin_cirrus = _extract_bits(qa_qf7, start_bit=4).astype(bool)
    qf7_adjacent_to_cloud = _extract_bits(qa_qf7, start_bit=1).astype(bool)
    qf7_snow_present = _extract_bits(qa_qf7, start_bit=0).astype(bool)

    mask_cloud = (
        (cloud_confidence >= 2)
        | qf2_thin_cirrus_reflective
        | qf2_thin_cirrus_emissive
        | qf7_thin_cirrus
        | qf7_adjacent_to_cloud
    ).astype(bool)
    mask_cloud_shadow = qf2_shadow.astype(bool)
    mask_snow = (qf2_snow_ice | qf7_snow_present).astype(bool)

    return xr.Dataset(
        data_vars={
            "qa_cloud_confidence": cloud_confidence,
            "qa_cloud_mask_quality": cloud_mask_quality,
            "qa_thin_cirrus_reflective": qf2_thin_cirrus_reflective,
            "qa_thin_cirrus_emissive": qf2_thin_cirrus_emissive,
            "qa_thin_cirrus_flag": qf7_thin_cirrus,
            "qa_adjacent_to_cloud": qf7_adjacent_to_cloud,
            "qa_shadow_flag": qf2_shadow,
            "qa_snow_ice_flag": qf2_snow_ice,
            "qa_snow_present_flag": qf7_snow_present,
            "mask_cloud_qa": mask_cloud,
            "mask_cloud_shadow_qa": mask_cloud_shadow,
            "mask_snow_qa": mask_snow,
        }
    )


def _false_mask_like(target_x: xr.DataArray, target_y: xr.DataArray) -> xr.DataArray:
    return xr.DataArray(
        np.zeros((target_y.size, target_x.size), dtype=bool),
        dims=("y", "x"),
        coords={"y": target_y.values, "x": target_x.values},
    )


def _normalize_external_mask_dataarray(
    data_array: xr.DataArray,
    *,
    target_x: xr.DataArray,
    target_y: xr.DataArray,
) -> xr.DataArray:
    spatial_dims = {"y", "x", "y_500m", "x_500m"}
    squeeze_dims = [dim for dim in data_array.dims if dim not in spatial_dims and data_array.sizes[dim] == 1]
    if squeeze_dims:
        data_array = data_array.squeeze(dim=squeeze_dims, drop=True)
    rename_map = {}
    if "y_500m" in data_array.dims:
        rename_map["y_500m"] = "y"
    if "x_500m" in data_array.dims:
        rename_map["x_500m"] = "x"
    normalized = data_array.rename(rename_map)

    if normalized.dims != ("y", "x"):
        raise ValueError(f"External mask must have dims ('y', 'x') or ('y_500m', 'x_500m'); got {normalized.dims}")

    target_sizes = {"y": target_y.size, "x": target_x.size}
    if normalized.sizes == target_sizes:
        normalized = normalized.assign_coords(y=target_y.values, x=target_x.values)
    elif "y" in normalized.coords and "x" in normalized.coords:
        normalized = normalized.interp(
            y=target_y.values,
            x=target_x.values,
            method="nearest",
            kwargs={"fill_value": "extrapolate"},
        )
    else:
        raise ValueError(
            "External mask shape differs from the target grid and lacks y/x coordinates for nearest-neighbor "
            f"resampling: mask sizes={dict(normalized.sizes)}, target sizes={target_sizes}"
        )
    return (normalized.notnull() & (normalized != 0)).astype(bool).load()


def _open_mask_source(source: str | Path | xr.Dataset | xr.DataArray, *, mask_var: str) -> xr.Dataset:
    if isinstance(source, xr.DataArray):
        return xr.Dataset({mask_var: source})
    if isinstance(source, xr.Dataset):
        return source

    path = Path(source)
    try:
        return xr.open_dataset(path)
    except ValueError:
        data_array = xr.open_dataarray(path)
        return xr.Dataset({mask_var: data_array})


def load_external_cloud_masks(
    source: str | Path | xr.Dataset | xr.DataArray,
    *,
    target_x: xr.DataArray,
    target_y: xr.DataArray,
    cloud_mask_var: str = "mask_cloud",
    cloud_shadow_mask_var: str = "mask_cloud_shadow",
) -> xr.Dataset:
    """
    Load external cloud and cloud-shadow masks on the prepared 500 m grid.

    If a DataArray is provided, it is treated as the cloud mask and the cloud
    shadow mask defaults to all-False.
    """
    close_dataset = None if isinstance(source, (xr.DataArray, xr.Dataset)) else _open_mask_source(source, mask_var=cloud_mask_var)
    dataset = source if isinstance(source, xr.Dataset) else xr.Dataset({cloud_mask_var: source}) if isinstance(source, xr.DataArray) else close_dataset

    try:
        if cloud_mask_var not in dataset:
            data_vars = list(dataset.data_vars)
            if len(data_vars) != 1:
                raise ValueError(f"External cloud mask source does not contain variable {cloud_mask_var!r}")
            cloud_data = dataset[data_vars[0]]
        else:
            cloud_data = dataset[cloud_mask_var]

        mask_cloud = _normalize_external_mask_dataarray(cloud_data, target_x=target_x, target_y=target_y)
        if cloud_shadow_mask_var in dataset:
            mask_cloud_shadow = _normalize_external_mask_dataarray(
                dataset[cloud_shadow_mask_var],
                target_x=target_x,
                target_y=target_y,
            )
        else:
            mask_cloud_shadow = _false_mask_like(target_x, target_y)

        return xr.Dataset(
            data_vars={
                "mask_cloud_external": mask_cloud,
                "mask_cloud_shadow_external": mask_cloud_shadow,
            }
        )
    finally:
        if close_dataset is not None:
            close_dataset.close()


def load_external_inversion_masks(
    sources: Mapping[str, str | Path | xr.Dataset | xr.DataArray] | None,
    *,
    target_x: xr.DataArray,
    target_y: xr.DataArray,
) -> xr.Dataset:
    """Load named external invalid-pixel masks on the prepared 500 m grid."""
    if not sources:
        return xr.Dataset()

    data_vars: dict[str, xr.DataArray] = {}
    close_datasets = []
    try:
        for raw_name, source in sources.items():
            name = str(raw_name)
            mask_var = f"mask_{name}"
            if isinstance(source, xr.Dataset):
                dataset = source
            elif isinstance(source, xr.DataArray):
                dataset = xr.Dataset({mask_var: source})
            else:
                dataset = _open_mask_source(source, mask_var=mask_var)
                close_datasets.append(dataset)

            if mask_var in dataset:
                data_array = dataset[mask_var]
            elif name in dataset:
                data_array = dataset[name]
            else:
                variables = list(dataset.data_vars)
                if len(variables) != 1:
                    raise ValueError(
                        f"External inversion mask {name!r} must contain {mask_var!r}, "
                        f"{name!r}, or exactly one data variable"
                    )
                data_array = dataset[variables[0]]

            data_vars[mask_var] = _normalize_external_mask_dataarray(
                data_array,
                target_x=target_x,
                target_y=target_y,
            )
        return xr.Dataset(data_vars=data_vars)
    finally:
        for dataset in close_datasets:
            dataset.close()
