# Albedo/Radiative Forcing/Delta VIS Implementation Plan

## Goal

Add LUT-derived albedo, radiative forcing, and delta VIS products to SpiPy inversion outputs. This first implementation is for future outputs written by the normal inversion workflow. A separate backfill implementation for existing NetCDF outputs will come later.

Products should be written into the same output NetCDF as the inversion fields, not as a separate derived product.

## Product Flags

Add individual product flags, defaulting to enabled:

- `include_albedo=True`
- `include_radiative_forcing=True`
- `include_delta_vis=True`

If any requested product requires a LUT that cannot be found or loaded, the whole inversion task should fail.

## Output Variables

Write these variables when the relevant product flag is enabled:

- `albedo_dirty_flat`
- `albedo_dirty_terrain_corrected`
- `albedo_clean_flat`
- `albedo_clean_terrain_corrected`
- `radiative_forcing`
- `delta_vis`

Units/storage:

- Albedos: unitless fractions, not percent.
- `delta_vis`: unitless fraction, not percent.
- `radiative_forcing`: `W m-2`.
- Store as floating-point values in SpiPy outputs. Do not copy old SPIRES packed integer percent/tenths storage.

## Inputs

Required inversion variables:

- `grain_size`, in micrometers.
- `dust_concentration`, in ppm.

Required solar variables:

- `solar_zenith`, degrees.
- `solar_azimuth`, source-product convention; convert during terrain correction.

Required terrain variables for terrain-corrected albedos:

- slope GeoTIFF: `<tile>_slope_gmted_med075.tif`
- aspect GeoTIFF: `<tile>_aspect_gmted_med075_ccw_from_south.tif`

The regenerated aspect files are already in ParBal/sunslope convention: `0 = south`, positive counter-clockwise from south, degrees.

## Dust Scaling Decision

Current SpiPy raw inversion output stores `dust_concentration` as physical ppm, float32, with attrs `units = ppm` and no packing scale factor.

Therefore the LUT input should be:

```python
dust_fraction = dust_concentration / 1000.0
```

Do not divide by 10 in the SpiPy future-output path.

The old `SPIRES_2025_0_1` albedo step used:

```matlab
data.dust_concentration_s = single(data.dust_concentration_s) / 10 / 1000;
```

because old intermediate `dust_concentration_s` was packed as `10 * ppm` in `uint16` storage. That `/10` unpacked old storage; it is not part of the albedo LUT physical conversion.

If we later support old packed SPIRES products directly, unpack them in a compatibility loader before calling the LUT function.

## LUT Inputs And Product Mapping

Definitions:

```python
mu0 = cosd(solar_zenith)
phi0 = 180.0 - solar_azimuth
phi0 = where(phi0 > 180.0, phi0 - 360.0, phi0)
muZ = sunslope(mu0, phi0, slope, aspect_ccw_from_south)
muZ = clip(muZ, None, 1.0)
gs = sqrt(grain_size)
dust = dust_concentration / 1000.0
soot = 0.0
```

Albedo products:

```python
albedo_clean_flat = Fclean(mu0, mu0, gs)
albedo_dirty_flat = Fdirty(mu0, mu0, gs, dust, soot)
albedo_clean_terrain_corrected = Fclean(mu0, muZ, gs)
albedo_dirty_terrain_corrected = Fdirty(mu0, muZ, gs, dust, soot)
```

Radiative forcing and delta VIS:

```python
delta_vis = Fdarken(mu0, mu0, gs, dust, soot)
radiative_forcing = Fforce(mu0, mu0, gs, dust, soot)
```

Do not create terrain-corrected RF or delta VIS products unless we deliberately define those later.

## LUT Inspection Notes

The production LUT files are MATLAB v7.3/HDF5 files containing serialized `griddedInterpolant` objects. Load them with `h5py`, not `scipy.io.loadmat`.

Albedo LUT:

- Path: `/scratch/alpine/ropa5718/modis/input_spires_from_Jeff_202404/SnowAlbedo_SanJuanDust_BlackCarbon.mat`
- Top-level objects: `Fclean`, `Fdirty`, `metadata`.
- `Fclean`:
  - grid vector cell: `/#refs#/e`
  - value array: `/#refs#/i`
  - MATLAB input order: `(mu0, muZ, gs)`
  - Python/HDF5 grid vector lengths: `29, 29, 28`
  - Python/HDF5 value shape: `(28, 29, 29)`; transpose to `(29, 29, 28)` for interpolation in MATLAB input order.
  - coordinate ranges:
    - `mu0`: `0.02` to `1.0`
    - `muZ`: `0.0` to `1.0`
    - `gs`: `5.477226` to `44.72136`
  - value range: about `0.265866` to `0.971845`
- `Fdirty`:
  - grid vector cell: `/#refs#/l`
  - value array: `/#refs#/r`
  - MATLAB input order: `(mu0, muZ, gs, dust, soot)`
  - Python/HDF5 grid vector lengths: `29, 29, 28, 15, 15`
  - Python/HDF5 value shape: `(15, 15, 28, 29, 29)`; transpose to `(29, 29, 28, 15, 15)`.
  - coordinate ranges:
    - `mu0`: `0.02` to `1.0`
    - `muZ`: `0.0` to `1.0`
    - `gs`: `5.477226` to `44.72136`
    - `dust`: `0.0` to `1.0e-3`
    - `soot`: `0.0` to `1.0e-6`
  - value range: about `0.215009` to `0.971845`

RF/delta VIS LUT:

- Path: `/scratch/alpine/ropa5718/modis/input_spires_from_Jeff_202404/Darkening_RadiativeForcing_SanJuanDust_BlackCarbon.mat`
- Top-level objects: `Fdarken`, `Fforce`, `metadata`.
- `Fdarken`:
  - grid vector cell: `/#refs#/e`
  - value array: `/#refs#/k`
  - MATLAB input order: `(mu0, muZ, gs, dust, soot)`
  - Python/HDF5 grid vector lengths: `31, 31, 30, 16, 16`
  - Python/HDF5 value shape: `(16, 16, 30, 31, 31)`; transpose to `(31, 31, 30, 16, 16)`.
  - coordinate ranges:
    - `mu0`: `0.02` to `1.0`
    - `muZ`: `0.0` to `1.0`
    - `gs`: `5.477226` to `44.72136`
    - `dust`: `0.0` to `1.0e-3`
    - `soot`: `0.0` to `1.0e-5`
  - value range: about `0.0` to `0.586219`
- `Fforce`:
  - grid vector cell: `/#refs#/n`
  - value array: `/#refs#/t`
  - MATLAB input order: `(mu0, muZ, gs, dust, soot)`
  - Python/HDF5 grid vector lengths: `31, 31, 30, 16, 16`
  - Python/HDF5 value shape: `(16, 16, 30, 31, 31)`; transpose to `(31, 31, 30, 16, 16)`.
  - coordinate ranges match `Fdarken`.
  - value range: about `0.0` to `455.955`.

The serialized MATLAB objects use linear interpolation and nearest extrapolation. SpiPy should intentionally use linear interpolation with no extrapolation, returning `NaN` outside coordinate ranges.

## Masking Policy

Do not hard-wire `valid_inversion_mask` into the LUT calculation.

Instead, compute each product where its required inputs are finite and in range:

- Flat albedo/RF/delta VIS require finite `grain_size`, `dust_concentration`, and `solar_zenith`.
- Terrain-corrected albedos additionally require finite `solar_azimuth`, `slope`, and `aspect`.

This keeps the same function usable for future temporally interpolated products, where grain size and dust may be filled outside the original valid inversion mask.

## LUT Bounds Policy

Do not extrapolate LUTs.

Pixels outside LUT coordinate ranges should become NaN. Record useful summary counts/ranges in attrs or logs if practical. Clip `muZ` to at most `1.0`; leave lower clipping behavior consistent with `sunslope`, which clips negative illumination to `0`.

## Proposed Code Structure

Add a sensor-agnostic module, for example:

```text
spires/albedo.py
```

Responsibilities:

- Load/validate albedo LUT and RF/delta VIS LUT.
- Expose small calculation functions that accept an `xarray.Dataset` plus optional terrain inputs.
- Implement `sunslope`.
- Apply unit conversions and product mapping.
- Return an `xarray.Dataset` containing only requested derived products.

Keep sensor/workflow-specific path discovery outside the core LUT math where possible.

## Workflow Integration

For future outputs, call the albedo step after inversion results are assembled and after `solar_zenith`/`solar_azimuth` have been attached, before NetCDF serialization.

Relevant current workflow location:

- `spires/sensors/full_workflow.py`
- After `add_solar_geometry_layers(results, scene_ds)`
- Before `sanitize_netcdf_dataset(results)`

The workflow should pass:

- requested product flags
- LUT paths
- ancillary directory or explicit terrain paths
- tile identifier if terrain paths are resolved from tile naming

## Defaults And Path Discovery

Likely defaults:

- albedo LUT: `/scratch/alpine/ropa5718/modis/input_spires_from_Jeff_202404/SnowAlbedo_SanJuanDust_BlackCarbon.mat`
- RF/delta VIS LUT: `/scratch/alpine/ropa5718/modis/input_spires_from_Jeff_202404/Darkening_RadiativeForcing_SanJuanDust_BlackCarbon.mat`
- terrain ancillary root: `/scratch/alpine/ropa5718/spipy/input/viirs/snpp/ancillary`

These should be overrideable by function args and by CURC workflow script arguments.

## Testing Plan

Unit tests:

- `sunslope` returns expected values for flat and simple sloped cases.
- solar azimuth conversion maps source convention into `+ccw from south`.
- dust conversion uses `/1000` only, not `/10/1000`.
- clean albedo calls/use does not require dust or soot.
- missing requested LUT raises an error.
- disabled product flags skip the relevant LUT requirement.
- no extrapolation: out-of-range LUT inputs produce NaN.
- `muZ > 1` is clipped to `1`.
- product variables have expected names, units, and fraction-valued albedo/delta VIS outputs.

Workflow tests:

- Future VIIRS workflow output includes all six products by default when LUTs and terrain are available.
- Individual flags can disable albedo, RF, or delta VIS.
- Terrain-corrected albedos require slope/aspect files; flat albedo/RF/delta VIS do not need terrain.
- Output NetCDF validation still passes with new variables present.

## Implementation Steps

1. Inspect LUT internals one final time and document coordinate order/ranges in code comments or tests.
2. Add the sensor-agnostic albedo module and tests for pure math/conversions.
3. Add LUT loading and no-extrapolation interpolation behavior.
4. Add terrain loading/resampling/alignment for the output grid.
5. Add product generation function returning an `xarray.Dataset`.
6. Wire product flags and LUT/terrain paths into `run_sensor_inversion_workflow`.
7. Add CLI/CURC workflow arguments with defaults.
8. Add workflow-level tests.
9. Run focused tests and one small CURC-style smoke test on an existing scene.

## Later Backfill Plan

The backfill path should reuse the same sensor-agnostic product generation function, but with a separate script that:

- opens existing inversion NetCDF outputs
- verifies required variables
- computes requested products
- writes variables back into the same file
- fails the file/task if requested LUTs or required inputs are missing

Existing raw SpiPy outputs should use `dust_concentration / 1000.0`. Only old packed SPIRES products should be unpacked with `/10` first.
