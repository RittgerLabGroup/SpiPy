"""Sensor-agnostic helpers for albedo, radiative forcing, and delta VIS products."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import xarray as xr


ALBEDO_PRODUCT_NAMES = (
    "albedo_dirty_flat",
    "albedo_dirty_terrain_corrected",
    "albedo_clean_flat",
    "albedo_clean_terrain_corrected",
)
RADIATIVE_FORCING_PRODUCT_NAME = "radiative_forcing"
DELTA_VIS_PRODUCT_NAME = "delta_vis"
DEFAULT_ALBEDO_LUT_PATH = Path(
    "/scratch/alpine/ropa5718/modis/input_spires_from_Jeff_202404/SnowAlbedo_SanJuanDust_BlackCarbon.mat"
)
DEFAULT_RADIATIVE_FORCING_LUT_PATH = Path(
    "/scratch/alpine/ropa5718/modis/input_spires_from_Jeff_202404/Darkening_RadiativeForcing_SanJuanDust_BlackCarbon.mat"
)

PRODUCT_ATTRS = {
    "albedo_dirty_flat": {
        "long_name": "Dirty snow albedo on flat terrain",
        "units": "1",
    },
    "albedo_dirty_terrain_corrected": {
        "long_name": "Dirty snow albedo with terrain correction",
        "units": "1",
    },
    "albedo_clean_flat": {
        "long_name": "Clean snow albedo on flat terrain",
        "units": "1",
    },
    "albedo_clean_terrain_corrected": {
        "long_name": "Clean snow albedo with terrain correction",
        "units": "1",
    },
    "radiative_forcing": {
        "long_name": "Light absorbing particle radiative forcing",
        "units": "W m-2",
    },
    "delta_vis": {
        "long_name": "Visible albedo reduction from light absorbing particles",
        "units": "1",
    },
}

ALBEDO_LUT_FUNCTIONS = {
    "Fclean": {
        "grid_ref": "/#refs#/e",
        "values": "/#refs#/i",
        "dimensions": ("mu0", "muZ", "gs"),
    },
    "Fdirty": {
        "grid_ref": "/#refs#/l",
        "values": "/#refs#/r",
        "dimensions": ("mu0", "muZ", "gs", "dust", "soot"),
    },
}

RADIATIVE_FORCING_LUT_FUNCTIONS = {
    "Fdarken": {
        "grid_ref": "/#refs#/e",
        "values": "/#refs#/k",
        "dimensions": ("mu0", "muZ", "gs", "dust", "soot"),
    },
    "Fforce": {
        "grid_ref": "/#refs#/n",
        "values": "/#refs#/t",
        "dimensions": ("mu0", "muZ", "gs", "dust", "soot"),
    },
}


@dataclass(frozen=True)
class AlbedoProductFlags:
    """Product switches for LUT-derived albedo products."""

    include_albedo: bool = True
    include_radiative_forcing: bool = True
    include_delta_vis: bool = True

    @property
    def requested_variables(self) -> tuple[str, ...]:
        names: list[str] = []
        if self.include_albedo:
            names.extend(ALBEDO_PRODUCT_NAMES)
        if self.include_radiative_forcing:
            names.append(RADIATIVE_FORCING_PRODUCT_NAME)
        if self.include_delta_vis:
            names.append(DELTA_VIS_PRODUCT_NAME)
        return tuple(names)

    @property
    def requires_albedo_lut(self) -> bool:
        return self.include_albedo

    @property
    def requires_radiative_forcing_lut(self) -> bool:
        return self.include_radiative_forcing or self.include_delta_vis

    @property
    def requires_terrain(self) -> bool:
        return self.include_albedo


@dataclass(frozen=True)
class AlbedoLutFunction:
    """One no-extrapolation LUT function loaded in MATLAB input order."""

    name: str
    points: tuple[np.ndarray, ...]
    values: np.ndarray
    dimensions: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.points) != len(self.dimensions):
            raise ValueError(f"{self.name} has {len(self.points)} grid vectors for {len(self.dimensions)} dimensions")
        expected_shape = tuple(point.size for point in self.points)
        if self.values.shape != expected_shape:
            raise ValueError(f"{self.name} values shape {self.values.shape} does not match grid shape {expected_shape}")

    @property
    def ranges(self) -> dict[str, tuple[float, float]]:
        return {
            name: (float(points[0]), float(points[-1]))
            for name, points in zip(self.dimensions, self.points, strict=True)
        }

    def make_interpolator(self) -> RegularGridInterpolator:
        return RegularGridInterpolator(
            self.points,
            self.values,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )

    def evaluate(self, *coordinates: Any) -> xr.DataArray:
        if len(coordinates) != len(self.points):
            raise ValueError(f"{self.name} requires {len(self.points)} coordinates, got {len(coordinates)}")

        interpolator = self.make_interpolator()

        def _evaluate_blocks(*blocks):
            broadcast = np.broadcast_arrays(*[np.asarray(block) for block in blocks])
            points = np.stack([block.ravel() for block in broadcast], axis=-1)
            output = np.full(points.shape[0], np.nan, dtype=np.float32)
            finite = np.all(np.isfinite(points), axis=1)
            if np.any(finite):
                output[finite] = interpolator(points[finite]).astype(np.float32)
            return output.reshape(broadcast[0].shape)

        return xr.apply_ufunc(
            _evaluate_blocks,
            *coordinates,
            dask="parallelized",
            output_dtypes=[np.float32],
            keep_attrs=False,
        )


@dataclass(frozen=True)
class AlbedoLuts:
    clean: AlbedoLutFunction
    dirty: AlbedoLutFunction


@dataclass(frozen=True)
class RadiativeForcingLuts:
    darken: AlbedoLutFunction
    force: AlbedoLutFunction


def _any_xarray(*values: Any) -> bool:
    return any(isinstance(value, (xr.DataArray, xr.Dataset)) for value in values)


def _where(condition: Any, value_if_true: Any, value_if_false: Any) -> Any:
    if _any_xarray(condition, value_if_true, value_if_false):
        return xr.where(condition, value_if_true, value_if_false)
    return np.where(condition, value_if_true, value_if_false)


def cosd(angle_degrees: Any) -> Any:
    """Return cosine for degree-valued input."""
    return np.cos(np.deg2rad(angle_degrees))


def sind(angle_degrees: Any) -> Any:
    """Return sine for degree-valued input."""
    return np.sin(np.deg2rad(angle_degrees))


def solar_azimuth_to_phi0(solar_azimuth: Any) -> Any:
    """Convert source solar azimuth to ParBal/sunslope convention.

    The output convention is degrees, with 0 = south and positive values
    counter-clockwise from south.
    """
    phi0 = 180.0 - solar_azimuth
    return _where(phi0 > 180.0, phi0 - 360.0, phi0)


def dust_ppm_to_lut_fraction(dust_concentration: Any) -> Any:
    """Convert SpiPy physical dust concentration in ppm to LUT fraction."""
    return dust_concentration / 1000.0


def grain_size_micrometers_to_lut_gs(grain_size: Any) -> Any:
    """Convert grain size in micrometers to the LUT square-root coordinate."""
    return np.sqrt(grain_size)


def sunslope(mu0: Any, phi0: Any, slope: Any, aspect_ccw_from_south: Any) -> Any:
    """Compute terrain-corrected illumination cosine.

    Parameters use ParBal/sunslope azimuth convention for `phi0` and `aspect`:
    0 = south, positive counter-clockwise from south, degrees. Negative
    illumination is clipped to 0 and values above 1 are clipped to 1.
    """
    solar_zenith_sine = np.sqrt(np.clip(1.0 - np.square(mu0), 0.0, None))
    mu_z = (
        mu0 * cosd(slope)
        + solar_zenith_sine * sind(slope) * cosd(phi0 - aspect_ccw_from_south)
    )
    return _where(mu_z < 0.0, 0.0, _where(mu_z > 1.0, 1.0, mu_z))


def derived_product_attrs(variable_name: str) -> dict[str, str]:
    """Return NetCDF attrs for a derived albedo product variable."""
    try:
        return dict(PRODUCT_ATTRS[variable_name])
    except KeyError as exc:
        raise ValueError(f"Unknown albedo product variable: {variable_name}") from exc


def _read_hdf5_vector(h5: h5py.File, path: str) -> np.ndarray:
    vector = np.asarray(h5[path][()], dtype=np.float32).ravel(order="F")
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"Expected non-empty LUT coordinate vector at {path}")
    return vector


def _read_grid_vectors(h5: h5py.File, grid_ref_path: str) -> tuple[np.ndarray, ...]:
    refs = np.asarray(h5[grid_ref_path][()]).ravel(order="F")
    vectors: list[np.ndarray] = []
    for ref in refs:
        vectors.append(_read_hdf5_vector(h5, h5[ref].name))
    return tuple(vectors)


def _load_lut_function(h5: h5py.File, name: str, mapping: dict[str, object]) -> AlbedoLutFunction:
    dimensions = tuple(mapping["dimensions"])
    points = _read_grid_vectors(h5, str(mapping["grid_ref"]))
    values_hdf = np.asarray(h5[str(mapping["values"])][()], dtype=np.float32)
    # MATLAB v7.3/HDF5 stores numeric arrays with dimensions reversed relative
    # to MATLAB indexing. Transpose back to the griddedInterpolant input order.
    values = values_hdf.transpose(tuple(reversed(range(values_hdf.ndim))))
    return AlbedoLutFunction(name=name, points=points, values=values, dimensions=dimensions)


def load_albedo_luts(path: str | Path) -> AlbedoLuts:
    """Load clean and dirty albedo LUT functions from the production MAT file."""
    with h5py.File(Path(path).expanduser(), "r") as h5:
        return AlbedoLuts(
            clean=_load_lut_function(h5, "Fclean", ALBEDO_LUT_FUNCTIONS["Fclean"]),
            dirty=_load_lut_function(h5, "Fdirty", ALBEDO_LUT_FUNCTIONS["Fdirty"]),
        )


def load_radiative_forcing_luts(path: str | Path) -> RadiativeForcingLuts:
    """Load delta VIS and radiative forcing LUT functions from the production MAT file."""
    with h5py.File(Path(path).expanduser(), "r") as h5:
        return RadiativeForcingLuts(
            darken=_load_lut_function(h5, "Fdarken", RADIATIVE_FORCING_LUT_FUNCTIONS["Fdarken"]),
            force=_load_lut_function(h5, "Fforce", RADIATIVE_FORCING_LUT_FUNCTIONS["Fforce"]),
        )


def _ensure_albedo_luts(luts: AlbedoLuts | str | Path | None) -> AlbedoLuts:
    if isinstance(luts, AlbedoLuts):
        return luts
    if luts is None:
        raise ValueError("Requested albedo products require albedo_luts or albedo_lut_path")
    return load_albedo_luts(luts)


def _ensure_radiative_forcing_luts(luts: RadiativeForcingLuts | str | Path | None) -> RadiativeForcingLuts:
    if isinstance(luts, RadiativeForcingLuts):
        return luts
    if luts is None:
        raise ValueError("Requested radiative forcing or delta VIS products require radiative_forcing_luts")
    return load_radiative_forcing_luts(luts)


def _require_variables(dataset: xr.Dataset, variable_names: tuple[str, ...]) -> None:
    missing = [name for name in variable_names if name not in dataset]
    if missing:
        raise ValueError(f"Dataset is missing variables required for albedo products: {missing}")


def _empty_like(template: xr.DataArray) -> xr.DataArray:
    return xr.zeros_like(template, dtype=np.float32) * np.float32(np.nan)


def generate_albedo_products(
    dataset: xr.Dataset,
    *,
    flags: AlbedoProductFlags | None = None,
    albedo_luts: AlbedoLuts | str | Path | None = None,
    radiative_forcing_luts: RadiativeForcingLuts | str | Path | None = None,
    slope: xr.DataArray | None = None,
    aspect: xr.DataArray | None = None,
) -> xr.Dataset:
    """Generate requested LUT-derived products from one inversion output dataset."""
    flags = AlbedoProductFlags() if flags is None else flags
    if not flags.requested_variables:
        return xr.Dataset(coords={name: coord for name, coord in dataset.coords.items() if name in dataset.dims})

    _require_variables(dataset, ("grain_size", "dust_concentration", "solar_zenith"))
    grain_size = dataset["grain_size"].astype(np.float32)
    dust_concentration = dataset["dust_concentration"].astype(np.float32)
    solar_zenith = dataset["solar_zenith"].astype(np.float32)
    mu0 = cosd(solar_zenith)
    gs = grain_size_micrometers_to_lut_gs(grain_size)
    dust = dust_ppm_to_lut_fraction(dust_concentration)
    soot = xr.zeros_like(dust, dtype=np.float32)

    products = xr.Dataset(coords={dim: grain_size.coords[dim] for dim in grain_size.dims if dim in grain_size.coords})

    if flags.include_albedo:
        if slope is None or aspect is None:
            raise ValueError("Terrain-corrected albedo products require slope and aspect")
        _require_variables(dataset, ("solar_azimuth",))
        albedo = _ensure_albedo_luts(albedo_luts)
        solar_azimuth = dataset["solar_azimuth"].astype(np.float32)
        phi0 = solar_azimuth_to_phi0(solar_azimuth)
        mu_z = sunslope(mu0, phi0, slope.astype(np.float32), aspect.astype(np.float32))

        product_values = {
            "albedo_clean_flat": albedo.clean.evaluate(mu0, mu0, gs),
            "albedo_dirty_flat": albedo.dirty.evaluate(mu0, mu0, gs, dust, soot),
            "albedo_clean_terrain_corrected": albedo.clean.evaluate(mu0, mu_z, gs),
            "albedo_dirty_terrain_corrected": albedo.dirty.evaluate(mu0, mu_z, gs, dust, soot),
        }
        for name in ALBEDO_PRODUCT_NAMES:
            products[name] = product_values[name].astype(np.float32).rename(name)
            products[name].attrs.update(derived_product_attrs(name))

    if flags.include_radiative_forcing or flags.include_delta_vis:
        rf_luts = _ensure_radiative_forcing_luts(radiative_forcing_luts)
        if flags.include_delta_vis:
            products[DELTA_VIS_PRODUCT_NAME] = rf_luts.darken.evaluate(mu0, mu0, gs, dust, soot).astype(np.float32)
            products[DELTA_VIS_PRODUCT_NAME] = products[DELTA_VIS_PRODUCT_NAME].rename(DELTA_VIS_PRODUCT_NAME)
            products[DELTA_VIS_PRODUCT_NAME].attrs.update(derived_product_attrs(DELTA_VIS_PRODUCT_NAME))
        if flags.include_radiative_forcing:
            products[RADIATIVE_FORCING_PRODUCT_NAME] = rf_luts.force.evaluate(mu0, mu0, gs, dust, soot).astype(np.float32)
            products[RADIATIVE_FORCING_PRODUCT_NAME] = products[RADIATIVE_FORCING_PRODUCT_NAME].rename(
                RADIATIVE_FORCING_PRODUCT_NAME
            )
            products[RADIATIVE_FORCING_PRODUCT_NAME].attrs.update(derived_product_attrs(RADIATIVE_FORCING_PRODUCT_NAME))

    for name in flags.requested_variables:
        if name not in products:
            products[name] = _empty_like(grain_size).rename(name)
            products[name].attrs.update(derived_product_attrs(name))
    return products


def validate_requested_lut_paths(
    flags: AlbedoProductFlags,
    *,
    albedo_lut_path: str | Path | None = None,
    radiative_forcing_lut_path: str | Path | None = None,
) -> None:
    """Raise if requested products do not have loadable LUT paths."""
    required_paths: list[tuple[str, str | Path | None]] = []
    if flags.requires_albedo_lut:
        required_paths.append(("albedo", albedo_lut_path))
    if flags.requires_radiative_forcing_lut:
        required_paths.append(("radiative forcing/delta VIS", radiative_forcing_lut_path))

    for label, path in required_paths:
        if path is None:
            raise ValueError(f"Requested {label} products require a LUT path")
        resolved = Path(path).expanduser()
        if not resolved.exists():
            raise FileNotFoundError(f"Requested {label} LUT does not exist: {resolved}")
