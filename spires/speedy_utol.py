from __future__ import annotations

from dataclasses import dataclass

import numpy as np


RepresentativeMethod = str


@dataclass(frozen=True)
class GroupedSpectra:
    """Chunk-local grouped spectra ready for representative inversion."""

    representative_targets: np.ndarray
    representative_backgrounds: np.ndarray
    representative_solar_zenith: np.ndarray | None
    inverse_indices: np.ndarray
    counts: np.ndarray
    valid_flat_indices: np.ndarray
    representative_indices: np.ndarray
    representative_method: RepresentativeMethod
    reflectance_tol: np.ndarray
    background_tol: np.ndarray
    solar_zenith_tol: np.ndarray | None
    original_shape: tuple[int, ...]

    @property
    def n_groups(self) -> int:
        return int(self.counts.size)

    @property
    def n_valid(self) -> int:
        return int(self.valid_flat_indices.size)


def _as_float64_2d(array: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array with shape (n_samples, n_bands); got {arr.shape}")
    return np.ascontiguousarray(arr)


def _as_float64_1d(array: np.ndarray | None, name: str) -> np.ndarray | None:
    if array is None:
        return None
    arr = np.asarray(array, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array with shape (n_samples,); got {arr.shape}")
    return np.ascontiguousarray(arr)


def _normalize_tolerance(
    value: float | np.ndarray | None,
    fallback: float | np.ndarray,
    size: int,
    name: str,
) -> np.ndarray:
    base = fallback if value is None else value
    arr = np.asarray(base, dtype=np.float64)
    if arr.ndim == 0:
        out = np.full(size, float(arr), dtype=np.float64)
    elif arr.ndim == 1 and arr.size == size:
        out = arr.astype(np.float64, copy=False)
    else:
        raise ValueError(
            f"{name} must be a scalar or a 1D array of length {size}; got shape {arr.shape}"
        )
    if np.any(out <= 0):
        raise ValueError(f"{name} must be strictly positive")
    return np.ascontiguousarray(out)


def _scale_tolerance(
    value: float | np.ndarray,
    factor: float,
) -> float | np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        return float(arr) * factor
    return arr * factor


def _build_valid_mask(
    spectra_targets: np.ndarray,
    spectra_backgrounds: np.ndarray,
    obs_solar_angles: np.ndarray | None,
    valid_mask: np.ndarray | None,
) -> np.ndarray:
    finite_mask = np.all(np.isfinite(spectra_targets), axis=1) & np.all(np.isfinite(spectra_backgrounds), axis=1)
    if obs_solar_angles is not None:
        finite_mask &= np.isfinite(obs_solar_angles)

    if valid_mask is None:
        return finite_mask

    provided = np.asarray(valid_mask, dtype=bool).reshape(-1)
    if provided.shape[0] != spectra_targets.shape[0]:
        raise ValueError(
            "valid_mask must have the same number of samples as the input spectra; "
            f"got {provided.shape[0]} and {spectra_targets.shape[0]}"
        )
    return finite_mask & provided


def _quantize(values: np.ndarray, tolerance: np.ndarray) -> np.ndarray:
    return np.rint(values / tolerance).astype(np.int64, copy=False)


def _row_unique_inverse(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if keys.size == 0:
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )

    packed = np.ascontiguousarray(keys).view(
        np.dtype((np.void, keys.dtype.itemsize * keys.shape[1]))
    ).reshape(-1)
    _, representative_indices, inverse_indices, counts = np.unique(
        packed,
        return_index=True,
        return_inverse=True,
        return_counts=True,
    )
    return representative_indices, inverse_indices, counts


def _group_means(values: np.ndarray, inverse_indices: np.ndarray, n_groups: int, counts: np.ndarray) -> np.ndarray:
    means = np.zeros((n_groups, values.shape[1]), dtype=np.float64)
    np.add.at(means, inverse_indices, values)
    means /= counts[:, None]
    return means


def group_spectra_rows(
    spectra_targets: np.ndarray,
    spectra_backgrounds: np.ndarray,
    obs_solar_angles: np.ndarray | None = None,
    *,
    valid_mask: np.ndarray | None = None,
    representative_method: RepresentativeMethod = "chunk_bin_mean",
    tolerance: float | np.ndarray = 0.02,
    reflectance_tol: float | np.ndarray | None = None,
    background_tol: float | np.ndarray | None = None,
    solar_zenith_tol: float | np.ndarray | None = None,
) -> GroupedSpectra:
    """
    Group chunk-local spectra into approximate unique sets for fast inversion.

    Parameters
    ----------
    spectra_targets
        2D array with shape ``(n_samples, n_bands)`` containing observed reflectance.
    spectra_backgrounds
        2D array with shape ``(n_samples, n_bands)`` containing background reflectance.
    obs_solar_angles
        Optional 1D array with shape ``(n_samples,)`` containing solar zenith angles.
    valid_mask
        Optional boolean mask with shape ``(n_samples,)``. Invalid rows are excluded
        from grouping and inversion.
    representative_method
        One of:
        - ``"first"``: representative is the first original row encountered in
          each quantized group. This matches the MATLAB-style member-row
          representative behavior most closely.
        - ``"chunk_bin_mean"``: representative is the mean of all rows in the bin
          within the current chunk.
    tolerance
        Default tolerance for all variables. Scalar or per-band 1D array.
    reflectance_tol
        Optional override for observed-reflectance tolerance.
    background_tol
        Optional override for background-reflectance tolerance.
    solar_zenith_tol
        Optional override for solar-zenith tolerance in degrees. If not provided,
        the fallback is ``tolerance * 100`` so a single scalar tolerance can still
        be interpreted sensibly across reflectance-like values and solar zenith.

    Returns
    -------
    GroupedSpectra
        Grouped representatives plus inverse indices for broadcasting inversion
        results back to the valid original rows.
    """
    targets = _as_float64_2d(spectra_targets, "spectra_targets")
    backgrounds = _as_float64_2d(spectra_backgrounds, "spectra_backgrounds")

    if targets.shape != backgrounds.shape:
        raise ValueError(
            "spectra_targets and spectra_backgrounds must have the same shape; "
            f"got {targets.shape} and {backgrounds.shape}"
        )

    solar = _as_float64_1d(obs_solar_angles, "obs_solar_angles")
    if solar is not None and solar.shape[0] != targets.shape[0]:
        raise ValueError(
            "obs_solar_angles must have the same number of samples as the input spectra; "
            f"got {solar.shape[0]} and {targets.shape[0]}"
        )

    valid = _build_valid_mask(targets, backgrounds, solar, valid_mask)
    valid_flat_indices = np.flatnonzero(valid)

    reflectance_tol_arr = _normalize_tolerance(
        reflectance_tol, tolerance, targets.shape[1], "reflectance_tol"
    )
    background_tol_arr = _normalize_tolerance(
        background_tol, tolerance, backgrounds.shape[1], "background_tol"
    )
    solar_tol_arr = None
    if solar is not None:
        solar_fallback = _scale_tolerance(tolerance, 100.0)
        solar_tol_arr = _normalize_tolerance(
            solar_zenith_tol, solar_fallback, 1, "solar_zenith_tol"
        )

    representative_method = representative_method.lower()
    if representative_method not in {"first", "chunk_bin_mean"}:
        raise ValueError(
            "representative_method must be one of {'first', 'chunk_bin_mean'}; "
            f"got {representative_method!r}"
        )

    if valid_flat_indices.size == 0:
        empty_groups = np.empty((0, targets.shape[1]), dtype=np.float64)
        empty_inverse = np.empty((0,), dtype=np.int64)
        empty_counts = np.empty((0,), dtype=np.int64)
        empty_indices = np.empty((0,), dtype=np.int64)
        empty_solar = None if solar is None else np.empty((0,), dtype=np.float64)
        return GroupedSpectra(
            representative_targets=empty_groups,
            representative_backgrounds=empty_groups.copy(),
            representative_solar_zenith=empty_solar,
            inverse_indices=empty_inverse,
            counts=empty_counts,
            valid_flat_indices=valid_flat_indices,
            representative_indices=empty_indices,
            representative_method=representative_method,
            reflectance_tol=reflectance_tol_arr,
            background_tol=background_tol_arr,
            solar_zenith_tol=solar_tol_arr,
            original_shape=targets.shape,
        )

    valid_targets = targets[valid]
    valid_backgrounds = backgrounds[valid]
    key_parts = [
        _quantize(valid_targets, reflectance_tol_arr),
        _quantize(valid_backgrounds, background_tol_arr),
    ]
    valid_solar = None
    if solar is not None:
        valid_solar = solar[valid]
        key_parts.append(_quantize(valid_solar[:, None], solar_tol_arr))

    key_matrix = np.concatenate(key_parts, axis=1)
    representative_indices, inverse_indices, counts = _row_unique_inverse(key_matrix)
    n_groups = int(counts.size)

    if representative_method == "first":
        representative_targets = valid_targets[representative_indices]
        representative_backgrounds = valid_backgrounds[representative_indices]
        representative_solar = None if valid_solar is None else valid_solar[representative_indices]
    else:
        representative_targets = _group_means(valid_targets, inverse_indices, n_groups, counts)
        representative_backgrounds = _group_means(valid_backgrounds, inverse_indices, n_groups, counts)
        representative_solar = None
        if valid_solar is not None:
            solar_sums = np.zeros(n_groups, dtype=np.float64)
            np.add.at(solar_sums, inverse_indices, valid_solar)
            representative_solar = solar_sums / counts

    return GroupedSpectra(
        representative_targets=np.ascontiguousarray(representative_targets),
        representative_backgrounds=np.ascontiguousarray(representative_backgrounds),
        representative_solar_zenith=None if representative_solar is None else np.ascontiguousarray(representative_solar),
        inverse_indices=np.ascontiguousarray(inverse_indices),
        counts=np.ascontiguousarray(counts),
        valid_flat_indices=np.ascontiguousarray(valid_flat_indices),
        representative_indices=np.ascontiguousarray(valid_flat_indices[representative_indices]),
        representative_method=representative_method,
        reflectance_tol=reflectance_tol_arr,
        background_tol=background_tol_arr,
        solar_zenith_tol=solar_tol_arr,
        original_shape=targets.shape,
    )


def group_spectra_block(
    spectra_targets: np.ndarray,
    spectra_backgrounds: np.ndarray,
    obs_solar_angles: np.ndarray | None = None,
    *,
    valid_mask: np.ndarray | None = None,
    representative_method: RepresentativeMethod = "chunk_bin_mean",
    tolerance: float | np.ndarray = 0.02,
    reflectance_tol: float | np.ndarray | None = None,
    background_tol: float | np.ndarray | None = None,
    solar_zenith_tol: float | np.ndarray | None = None,
) -> GroupedSpectra:
    """
    Group an image-like block without xarray stack/materialize patterns.

    Parameters
    ----------
    spectra_targets
        Array with shape ``(..., band)``. All leading dimensions are flattened into a
        local sample axis.
    spectra_backgrounds
        Array with same shape as ``spectra_targets``.
    obs_solar_angles
        Optional array with shape matching the leading spatial dimensions of
        ``spectra_targets``.
    valid_mask
        Optional boolean array with the same leading spatial dimensions.
    """
    targets = np.asarray(spectra_targets, dtype=np.float64)
    backgrounds = np.asarray(spectra_backgrounds, dtype=np.float64)

    if targets.shape != backgrounds.shape:
        raise ValueError(
            "spectra_targets and spectra_backgrounds must have the same shape; "
            f"got {targets.shape} and {backgrounds.shape}"
        )
    if targets.ndim < 2:
        raise ValueError(
            "spectra_targets must have at least one sample dimension plus a trailing band dimension; "
            f"got {targets.shape}"
        )

    spatial_shape = targets.shape[:-1]
    n_bands = targets.shape[-1]

    flat_targets = np.ascontiguousarray(targets.reshape(-1, n_bands))
    flat_backgrounds = np.ascontiguousarray(backgrounds.reshape(-1, n_bands))

    flat_solar = None
    if obs_solar_angles is not None:
        solar = np.asarray(obs_solar_angles, dtype=np.float64)
        if solar.shape != spatial_shape:
            raise ValueError(
                "obs_solar_angles must match the leading sample dimensions of the spectra; "
                f"got {solar.shape} and {spatial_shape}"
            )
        flat_solar = np.ascontiguousarray(solar.reshape(-1))

    flat_valid_mask = None
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != spatial_shape:
            raise ValueError(
                "valid_mask must match the leading sample dimensions of the spectra; "
                f"got {mask.shape} and {spatial_shape}"
            )
        flat_valid_mask = np.ascontiguousarray(mask.reshape(-1))

    grouped = group_spectra_rows(
        flat_targets,
        flat_backgrounds,
        flat_solar,
        valid_mask=flat_valid_mask,
        representative_method=representative_method,
        tolerance=tolerance,
        reflectance_tol=reflectance_tol,
        background_tol=background_tol,
        solar_zenith_tol=solar_zenith_tol,
    )

    return GroupedSpectra(
        representative_targets=grouped.representative_targets,
        representative_backgrounds=grouped.representative_backgrounds,
        representative_solar_zenith=grouped.representative_solar_zenith,
        inverse_indices=grouped.inverse_indices,
        counts=grouped.counts,
        valid_flat_indices=grouped.valid_flat_indices,
        representative_indices=grouped.representative_indices,
        representative_method=grouped.representative_method,
        reflectance_tol=grouped.reflectance_tol,
        background_tol=grouped.background_tol,
        solar_zenith_tol=grouped.solar_zenith_tol,
        original_shape=spatial_shape + (n_bands,),
    )


def scatter_group_results(
    grouped: GroupedSpectra,
    group_results: np.ndarray,
    *,
    fill_value: float = np.nan,
    n_properties: int | None = None,
) -> np.ndarray:
    """
    Broadcast group-level inversion results back to the original flattened samples.
    """
    results = np.asarray(group_results, dtype=np.float64)
    if results.ndim == 1:
        results = results[:, None]
    if results.ndim != 2:
        raise ValueError(f"group_results must be a 1D or 2D array; got shape {results.shape}")
    if results.shape[0] != grouped.n_groups:
        raise ValueError(
            "group_results first dimension must match the number of groups; "
            f"got {results.shape[0]} and {grouped.n_groups}"
        )

    n_samples = int(np.prod(grouped.original_shape[:-1], dtype=np.int64))
    n_properties = results.shape[1] if n_properties is None else int(n_properties)
    if n_properties != results.shape[1]:
        raise ValueError(
            "n_properties must match the second dimension of group_results; "
            f"got {n_properties} and {results.shape[1]}"
        )

    full = np.full((n_samples, n_properties), fill_value, dtype=np.float64)
    if grouped.n_valid > 0:
        full[grouped.valid_flat_indices] = results[grouped.inverse_indices]
    return full


def scatter_group_results_block(
    grouped: GroupedSpectra,
    group_results: np.ndarray,
    *,
    fill_value: float = np.nan,
) -> np.ndarray:
    """
    Broadcast group-level inversion results back to the original block shape.
    """
    flat = scatter_group_results(grouped, group_results, fill_value=fill_value)
    spatial_shape = grouped.original_shape[:-1]
    return flat.reshape(spatial_shape + (flat.shape[-1],))
