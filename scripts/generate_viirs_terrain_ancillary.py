#!/usr/bin/env python
"""Generate VIIRS slope/aspect ancillary GeoTIFFs from elevation tiles."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import rasterio


DEFAULT_TILES = ("h08v04", "h08v05", "h09v04", "h09v05", "h10v04")
DEFAULT_SOURCE_DIR = Path("/pl/active/rittger_ops/modis_ancillary/v3.2/elevation")
DEFAULT_TARGET_ROOT = Path("/scratch/alpine/ropa5718/spipy/input/viirs/snpp/ancillary")
OUTPUT_NODATA = np.float32(-9999.0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--target-root", type=Path, default=DEFAULT_TARGET_ROOT)
    parser.add_argument("--tiles", nargs="+", default=list(DEFAULT_TILES))
    parser.add_argument("--replace", action="store_true", help="Remove existing aspect/slope TIFFs before writing outputs.")
    return parser.parse_args()


def _tile_elevation_name(tile: str) -> str:
    return f"{tile}_elevation_gmted_med075.tif"


def _tile_slope_name(tile: str) -> str:
    return f"{tile}_slope_gmted_med075.tif"


def _tile_aspect_name(tile: str) -> str:
    return f"{tile}_aspect_gmted_med075_ccw_from_south.tif"


def _remove_existing_terrain_tifs(tile_dir: Path) -> list[Path]:
    removed: list[Path] = []
    for pattern in ("*aspect*.tif", "*slope*.tif"):
        for path in sorted(tile_dir.glob(pattern)):
            path.unlink()
            removed.append(path)
    return removed


def _copy_elevation(source_dir: Path, tile_dir: Path, tile: str) -> Path:
    source = source_dir / _tile_elevation_name(tile)
    if not source.exists():
        raise FileNotFoundError(f"Missing source elevation file: {source}")
    destination = tile_dir / source.name
    shutil.copy2(source, destination)
    return destination


def _terrain_from_elevation(elevation: np.ndarray, transform: rasterio.Affine, nodata: float | int | None) -> tuple[np.ndarray, np.ndarray]:
    elevation = elevation.astype(np.float64, copy=False)
    invalid = ~np.isfinite(elevation)
    if nodata is not None and np.isfinite(nodata):
        invalid |= elevation == nodata

    working = elevation.copy()
    working[invalid] = np.nan

    dx = abs(float(transform.a))
    dy = abs(float(transform.e))
    if dx <= 0 or dy <= 0:
        raise ValueError(f"Invalid pixel spacing from transform: dx={dx}, dy={dy}")

    # Axis 0 increases southward in the raster, so flip that derivative to northing.
    dz_drow, dz_dx = np.gradient(working, dy, dx, edge_order=2)
    dz_dy = -dz_drow

    slope = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy))).astype(np.float32)

    downhill_east = -dz_dx
    downhill_north = -dz_dy
    aspect_north_clockwise = (np.degrees(np.arctan2(downhill_east, downhill_north)) + 360.0) % 360.0

    # ParBal/sunslope convention: 0=south, positive counter-clockwise, degrees.
    aspect_ccw_from_south = 180.0 - aspect_north_clockwise
    aspect_ccw_from_south = np.where(
        aspect_ccw_from_south > 180.0,
        aspect_ccw_from_south - 360.0,
        aspect_ccw_from_south,
    ).astype(np.float32)

    flat = np.isfinite(slope) & (slope == 0.0)
    aspect_ccw_from_south[flat] = 0.0

    output_invalid = invalid | ~np.isfinite(slope) | ~np.isfinite(aspect_ccw_from_south)
    slope[output_invalid] = OUTPUT_NODATA
    aspect_ccw_from_south[output_invalid] = OUTPUT_NODATA
    return slope, aspect_ccw_from_south


def _write_float_geotiff(path: Path, values: np.ndarray, source_profile: dict) -> None:
    profile = source_profile.copy()
    profile.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        nodata=float(OUTPUT_NODATA),
        compress="deflate",
        predictor=3,
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )
    with rasterio.open(path, "w", **profile) as dataset:
        dataset.write(values.astype(np.float32, copy=False), 1)


def _generate_tile(source_dir: Path, target_root: Path, tile: str, replace: bool) -> dict[str, object]:
    tile_dir = target_root / tile
    tile_dir.mkdir(parents=True, exist_ok=True)
    removed = _remove_existing_terrain_tifs(tile_dir) if replace else []
    elevation_path = _copy_elevation(source_dir, tile_dir, tile)

    with rasterio.open(elevation_path) as dataset:
        elevation = dataset.read(1)
        slope, aspect = _terrain_from_elevation(elevation, dataset.transform, dataset.nodata)
        profile = dataset.profile

    slope_path = tile_dir / _tile_slope_name(tile)
    aspect_path = tile_dir / _tile_aspect_name(tile)
    _write_float_geotiff(slope_path, slope, profile)
    _write_float_geotiff(aspect_path, aspect, profile)

    valid_slope = slope[slope != OUTPUT_NODATA]
    valid_aspect = aspect[aspect != OUTPUT_NODATA]
    return {
        "tile": tile,
        "removed": [str(path) for path in removed],
        "elevation": str(elevation_path),
        "slope": str(slope_path),
        "aspect": str(aspect_path),
        "slope_min": float(np.nanmin(valid_slope)),
        "slope_max": float(np.nanmax(valid_slope)),
        "aspect_min": float(np.nanmin(valid_aspect)),
        "aspect_max": float(np.nanmax(valid_aspect)),
    }


def main() -> None:
    args = _parse_args()
    for tile in args.tiles:
        result = _generate_tile(args.source_dir, args.target_root, tile, args.replace)
        print(
            "{tile}: elevation={elevation} slope={slope} aspect={aspect} "
            "slope_range=[{slope_min:.3f}, {slope_max:.3f}] "
            "aspect_range=[{aspect_min:.3f}, {aspect_max:.3f}] removed={removed_count}".format(
                **result,
                removed_count=len(result["removed"]),
            )
        )


if __name__ == "__main__":
    main()
