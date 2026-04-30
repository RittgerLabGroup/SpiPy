"""Sensor-agnostic helpers for safe output serialization and validation."""

from __future__ import annotations

import json
import logging
import os
from numbers import Number
from pathlib import Path
import uuid

import numpy as np
import xarray as xr


LOGGER = logging.getLogger(__name__)
NETCDF_ATTR_TYPES = (str, Number, np.number, bytes)
NETCDF_RESERVED_ATTRS = {"_FillValue", "missing_value", "add_offset", "scale_factor", "DIMENSION_LIST"}
INVERSION_REQUIRED_VARS = (
    "raw_viewable_snow_fraction",
    "raw_shade_fraction",
    "dust_concentration",
    "grain_size",
    "valid_inversion_mask",
)


def _netcdf_safe_attr_value(value):
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        if value.ndim == 1:
            return value.tolist()
        return json.dumps(value.tolist())
    if isinstance(value, (list, tuple)):
        array_value = np.asarray(value, dtype=object)
        if array_value.ndim <= 1:
            return list(value)
        return json.dumps(value)
    if isinstance(value, NETCDF_ATTR_TYPES):
        return value
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _is_fill_value_compatible(variable: xr.DataArray, value) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.upper() in {"N/A", "NA"}:
        return False
    try:
        np.asarray(value, dtype=variable.dtype)
    except (TypeError, ValueError):
        return False
    return True


def _is_valid_range_compatible(variable: xr.DataArray, value) -> bool:
    try:
        values = np.asarray(value, dtype=variable.dtype)
    except (TypeError, ValueError):
        return False
    return values.ndim <= 1 and values.size == 2


def _sanitize_variable_attrs(variable: xr.DataArray) -> dict[str, object]:
    sanitized_attrs = {}
    for key, value in variable.attrs.items():
        if key in NETCDF_RESERVED_ATTRS:
            continue
        if key in {"_FillValue", "missing_value"} and not _is_fill_value_compatible(variable, value):
            continue
        if key == "valid_range" and not _is_valid_range_compatible(variable, value):
            continue
        sanitized_attrs[key] = _netcdf_safe_attr_value(value)
    return sanitized_attrs


def sanitize_netcdf_dataset(dataset: xr.Dataset) -> xr.Dataset:
    """Return a copy of a dataset with NetCDF-safe attrs."""
    sanitized = dataset.copy()
    sanitized.attrs = {key: _netcdf_safe_attr_value(value) for key, value in dataset.attrs.items()}
    for variable_name in sanitized.variables:
        sanitized[variable_name].attrs = _sanitize_variable_attrs(sanitized[variable_name])
    return sanitized


def validate_inversion_output_dataset(
    dataset: xr.Dataset,
    *,
    expected_attrs: dict[str, object] | None = None,
) -> xr.Dataset:
    """Validate a saved inversion dataset before reuse or promotion."""
    missing = [name for name in INVERSION_REQUIRED_VARS if name not in dataset]
    if missing:
        raise ValueError(f"Inversion dataset is missing required variables: {missing}")

    for name in INVERSION_REQUIRED_VARS:
        if dataset[name].dims != ("y", "x"):
            raise ValueError(f"Unexpected dims for {name}: {dataset[name].dims}")

    if "raw_canopy_fraction" in dataset and dataset["raw_canopy_fraction"].dims != ("y", "x"):
        raise ValueError(f"Unexpected dims for raw_canopy_fraction: {dataset['raw_canopy_fraction'].dims}")
    if "raw_snow_fraction" in dataset and dataset["raw_snow_fraction"].dims != ("y", "x"):
        raise ValueError(f"Unexpected dims for raw_snow_fraction: {dataset['raw_snow_fraction'].dims}")

    finite_fraction_vars = (
        "raw_viewable_snow_fraction",
        "raw_shade_fraction",
        "dust_concentration",
        "grain_size",
    )
    if not any(bool(np.isfinite(dataset[name]).any().item()) for name in finite_fraction_vars):
        raise ValueError("Inversion dataset contains no finite retrieval values")

    mask_dtype = dataset["valid_inversion_mask"].dtype
    if mask_dtype.kind not in {"b", "i", "u"}:
        raise ValueError(f"valid_inversion_mask has unsupported dtype: {mask_dtype}")

    for name in ("raw_viewable_snow_fraction", "raw_shade_fraction"):
        values = dataset[name]
        finite = np.isfinite(values)
        if bool(finite.any().item()):
            min_value = float(values.where(finite).min().item())
            max_value = float(values.where(finite).max().item())
            if min_value < -1e-6 or max_value > 1.0 + 1e-6:
                raise ValueError(f"{name} has values outside [0, 1]: min={min_value}, max={max_value}")

    build_status = dataset.attrs.get("build_status")
    if build_status is not None and build_status != "complete":
        raise ValueError(f"Inversion dataset has non-complete build_status={build_status!r}")

    if expected_attrs is not None:
        for key, expected_value in expected_attrs.items():
            if expected_value is None:
                continue
            actual_value = dataset.attrs.get(key)
            if actual_value != expected_value:
                raise ValueError(f"Inversion dataset attr mismatch for {key!r}: {actual_value!r} != {expected_value!r}")

    return dataset


def load_output_dataset_if_valid(
    path: str | Path,
    *,
    validator=None,
    expected_attrs: dict[str, object] | None = None,
) -> xr.Dataset | None:
    """Open an output dataset if it exists and passes validation."""
    resolved_path = Path(path).expanduser().resolve()
    if not resolved_path.exists():
        return None
    try:
        dataset = xr.open_dataset(resolved_path)
    except Exception as exc:
        LOGGER.warning("Failed to open inversion output %s; ignoring. Error: %s", resolved_path, exc)
        return None

    validate = validate_inversion_output_dataset if validator is None else validator
    try:
        return validate(dataset, expected_attrs=expected_attrs)
    except Exception as exc:
        dataset.close()
        LOGGER.warning("Inversion output %s failed validation; ignoring. Error: %s", resolved_path, exc)
        return None


def write_output_dataset(
    dataset: xr.Dataset,
    path: str | Path,
    *,
    validator=None,
    expected_attrs: dict[str, object] | None = None,
) -> Path:
    """Write an output dataset atomically after a lightweight validation pass."""
    resolved_path = Path(path).expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    prepared = sanitize_netcdf_dataset(dataset)
    prepared.attrs = prepared.attrs.copy()
    prepared.attrs["build_status"] = "complete"

    temp_path = resolved_path.parent / f".{resolved_path.name}.{uuid.uuid4().hex}.tmp"
    validate = validate_inversion_output_dataset if validator is None else validator
    try:
        prepared.to_netcdf(temp_path)
        with xr.open_dataset(temp_path) as written:
            validate(written, expected_attrs=expected_attrs)
        os.replace(temp_path, resolved_path)
    except Exception:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        raise

    return resolved_path
