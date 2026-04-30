#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from netCDF4 import Dataset as NetCDFDataset
import numpy as np
import xarray as xr


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spires.sensors.modis.hdf import prepare_modis_scene_for_inversion


def _percentiles(values: np.ndarray) -> str:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return "no finite values"
    p1, p50, p99 = np.percentile(finite, [1, 50, 99])
    return (
        f"min={finite.min():.4f} "
        f"p1={p1:.4f} "
        f"p50={p50:.4f} "
        f"p99={p99:.4f} "
        f"max={finite.max():.4f}"
    )


def _print_raw_band_stats(scene_path: Path, band_names: list[str]) -> None:
    print("Raw band stats")
    with NetCDFDataset(scene_path) as dataset:
        dataset.set_auto_mask(False)
        dataset.set_auto_scale(False)
        for band_name in band_names:
            variable_name = f"sur_refl_b{int(band_name):02d}_1"
            variable = dataset.variables[variable_name]
            array = variable[:]
            fill_value = getattr(variable, "_FillValue", None)
            valid_range = getattr(variable, "valid_range", None)

            print(f"  {variable_name}")
            print(f"    dtype={array.dtype} shape={array.shape}")
            print(f"    fill_value={fill_value} scale_factor={getattr(variable, 'scale_factor', None)}")
            if fill_value is not None:
                print(f"    fill_count={(array == fill_value).sum()}")
            if valid_range is not None:
                low, high = valid_range
                outside = ((array < low) | (array > high)).sum()
                print(f"    valid_range=[{low}, {high}] outside_valid_range={outside}")


def _print_prepared_stats(prepared: xr.Dataset) -> None:
    print("Prepared-scene stats")
    reflectance = prepared["reflectance"].values
    band_names = prepared["band"].values.tolist()
    for band_index, band_name in enumerate(band_names):
        band = reflectance[:, :, band_index]
        print(f"  reflectance band {band_name}: finite_frac={np.isfinite(band).mean():.4f} {_percentiles(band)}")

    for variable_name in [
        "valid_inversion_mask",
        "valid_r0_mask",
        "mask_invalid_reflectance",
        "mask_bad_geometry",
        "mask_water",
        "mask_low_observation_support",
        "mask_bad_modland_qa",
        "mask_cloud",
        "mask_cloud_shadow",
        "mask_snow",
    ]:
        data = prepared[variable_name].values
        print(f"  {variable_name}: true_frac={data.mean():.4f}")

    invalid = prepared["mask_invalid_reflectance"].values
    row_invalid = invalid.mean(axis=1)
    col_invalid = invalid.mean(axis=0)
    print(
        "  invalid row coverage:",
        f"min={row_invalid.min():.4f}",
        f"p50={np.percentile(row_invalid, 50):.4f}",
        f"p99={np.percentile(row_invalid, 99):.4f}",
        f"max={row_invalid.max():.4f}",
    )
    print(
        "  invalid col coverage:",
        f"min={col_invalid.min():.4f}",
        f"p50={np.percentile(col_invalid, 50):.4f}",
        f"p99={np.percentile(col_invalid, 99):.4f}",
        f"max={col_invalid.max():.4f}",
    )


def _print_inversion_stats(inversion: xr.Dataset, prepared: xr.Dataset) -> None:
    print("Saved inversion stats")
    for variable_name in ["fsca", "fshade", "raw_snow_fraction"]:
        if variable_name not in inversion:
            continue
        values = inversion[variable_name].values
        print(f"  {variable_name}: finite_frac={np.isfinite(values).mean():.4f} {_percentiles(values)}")

    if "fsca" not in inversion:
        return

    fsca = inversion["fsca"].values
    masks = {
        "all": np.ones(fsca.shape, dtype=bool),
        "valid_mask": prepared["valid_inversion_mask"].values,
        "invalid_mask": ~prepared["valid_inversion_mask"].values,
        "water": prepared["mask_water"].values,
        "land": ~prepared["mask_water"].values,
    }
    for name, mask in masks.items():
        masked = fsca[mask]
        finite = masked[np.isfinite(masked)]
        if finite.size == 0:
            print(f"  fsca over {name}: no finite values")
            continue
        p1, p50, p99 = np.percentile(finite, [1, 50, 99])
        print(
            f"  fsca over {name}:",
            f"n={finite.size}",
            f"mean={finite.mean():.4f}",
            f"p1={p1:.4f}",
            f"p50={p50:.4f}",
            f"p99={p99:.4f}",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose a MODIS scene through raw, prepared, and saved inversion stages.")
    parser.add_argument("scene_path", type=Path, help="Path to a MOD09GA or MYD09GA HDF scene")
    parser.add_argument("--lut-file", type=Path, required=True, help="Path to the MODIS LUT")
    parser.add_argument(
        "--cloud-mask-policy",
        default="ignore_cloud_and_shadow",
        choices=["strict", "snow_wins", "ignore_cloud", "ignore_cloud_and_shadow"],
        help="Cloud policy passed into prepare_modis_scene_for_inversion",
    )
    parser.add_argument(
        "--inversion-path",
        type=Path,
        help="Optional path to a saved inversion NetCDF for the same scene",
    )
    args = parser.parse_args()

    prepared = prepare_modis_scene_for_inversion(
        args.scene_path,
        lut_file=args.lut_file,
        cloud_mask_policy=args.cloud_mask_policy,
    )

    print(f"Scene: {args.scene_path}")
    print(f"Platform: {prepared.attrs.get('platform')}")
    print(f"Acquisition date: {prepared.attrs.get('acquisition_date')}")
    print(f"Bands: {prepared['band'].values.tolist()}")
    print()

    _print_raw_band_stats(args.scene_path, prepared["band"].values.tolist())
    print()
    _print_prepared_stats(prepared)

    if args.inversion_path is not None:
        print()
        inversion = xr.open_dataset(args.inversion_path)
        try:
            _print_inversion_stats(inversion, prepared)
        finally:
            inversion.close()


if __name__ == "__main__":
    main()
