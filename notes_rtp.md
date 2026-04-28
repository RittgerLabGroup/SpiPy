# VIIRS SPIReS Workflow Notes

## Goal

Keep a compact working spec for running SPIReS spectral unmixing on VIIRS surface-reflectance products.

Assumptions:

- a VIIRS-specific SPIReS LUT will be generated separately and made available to this workflow
- Landsat work is out of scope for this note
- the existing SPIReS core inversion code should remain as unchanged as possible

## Design Principles

- keep SPIReS core inversion logic sensor-agnostic
- isolate VIIRS-specific ingestion, QA handling, geometry, and compositing in a dedicated namespace
- use xarray + dask + zarr for scalable preprocessing and inversion
- preserve a consistent schema for reflectance, geometry, masks, `R0`, and inversion outputs

## Implementation Shape

```text
spires/
  sensors/
    viirs/
      __init__.py
      bands.py
      hdf.py
      qa.py
      geometry.py
      stack.py
      r0.py
      masks.py
      workflow.py
      io.py
examples/
  09_viirs_create_background_reflectance.ipynb
  10_viirs_snow_inversion.ipynb
  11_viirs_postprocess.ipynb
tests/
  test_viirs_hdf.py
  test_viirs_qa.py
  test_viirs_r0.py
  test_viirs_inversion.py
```

## Module Responsibilities

- `bands.py`: canonical VIIRS band definitions and LUT-aligned ordering
- `hdf.py`: raw VIIRS HDF reader for reflectance, geometry, QA, and scene metadata
- `qa.py`: VIIRS QA decoding and mask construction
- `geometry.py`: geometry accessors and grid alignment helpers
- `stack.py`: scaling, cleanup, renaming, and construction of analysis-ready reflectance cubes
- `r0.py`: summer background-reflectance selection and compositing
- `masks.py`: combination of QA masks with ancillary masks
- `workflow.py`: high-level orchestration for stack building, `R0`, and inversion
- `io.py`: zarr-first intermediate I/O plus selected exports

## End-to-End Workflow

### 1. Build an analysis-ready summer stack

- collect June-September VIIRS HDF files for the target tile / year span
- read reflectance, geometry, and QA
- apply scaling and invalid-value filtering
- align all fields to the analysis grid
- write a chunked xarray/zarr time stack

Core variables:

- `reflectance(time, y, x, band)`
- `solar_zenith(time, y, x)`
- `sensor_zenith(time, y, x)`
- `qa_mask(time, y, x)`

### 2. Create background reflectance `R0`

- use June-September observations only
- exclude cloudy, shadowed, invalid, and snow-contaminated observations
- optionally exclude extreme view geometry
- composite valid observations per pixel and band

Core outputs:

- `r0_reflectance(y, x, band)`
- `r0_count(y, x)`
- support / diagnostic layers as needed

### 3. Run inversion

- read or prepare the target scene
- select the LUT-matched band order
- attach `R0` and geometry
- run SPIReS through the dask-enabled inversion path

Core outputs:

- `fsca`
- `fshade`
- `dust_concentration`
- `grain_size`
- `valid_inversion_mask`

## Parallelization Strategy

- preprocess raw HDFs into zarr once when possible
- keep `reflectance` chunked in xarray/zarr
- keep `band` unchunked
- store `R0` as static `y, x, band`
- run inversion with `speedy_invert_dask(...)`

Suggested chunking:

- `time = 1`
- `y = 256-1024`
- `x = 256-1024`
- `band = -1`

This matches the current dask/xarray inversion assumptions and preserves intact spectral vectors.

## Ancillary Data

### Required

- VIIRS SPIReS LUT
- water mask
- forest / canopy mask
- DEM
- VIIRS QA decoding rules
- a snow-free filtering rule for `R0` generation

### Recommended

- land-cover mask
- snow mask or snow prior for `R0`
- permanent snow / ice or glacier mask

### Optional

- terrain-shadow mask derived from DEM + solar geometry
- cloud-mask refinement beyond native VIIRS QA
- region-of-interest boundaries

## Stable Schema Recommendation

Use the following core names consistently:

- `reflectance(time, y, x, band)`
- `solar_zenith(time, y, x)`
- `sensor_zenith(time, y, x)`
- `qa_mask(time, y, x)`
- `r0_reflectance(y, x, band)`
- `r0_count(y, x)`
- `valid_inversion_mask(time, y, x)`
- `fsca(time, y, x)`
- `fshade(time, y, x)`
- `dust_concentration(time, y, x)`
- `grain_size(time, y, x)`

## Candidate Public API

```python
# spires/sensors/viirs/workflow.py

def open_vnp09ga(path) -> xr.Dataset: ...
def open_vnp09ga_many(paths) -> xr.Dataset: ...
def build_viirs_reflectance_stack(paths, band_names, chunks) -> xr.Dataset: ...
def build_viirs_r0(ds, masks, months=(6, 7, 8, 9), method="median") -> xr.Dataset: ...
def prepare_viirs_scene(ds_scene, ds_r0, masks) -> xr.Dataset: ...
def run_viirs_inversion(scene_ds, r0_ds, interpolator, client=None) -> xr.Dataset: ...
```

## Current Status Snapshot (2026-04-27)

The sections above describe the intended architecture. The items below summarize what is now implemented, what decisions have been made, and what still needs work.

### Environment / repo notes

- active repo root is `Codex/SpiPy_RLG/SpiPy`
- in sandboxed terminal sessions, prefer `mamba` over `conda`
- when needed, use a writable cache such as:
  - `XDG_CACHE_HOME=/tmp/mamba-cache mamba run -n spipy14 python -m pytest SpiPy/tests/test_viirs_r0.py`

### Implemented so far

- **LUT groundwork**
  - upstream `SPIRES` was cloned and inspected
  - `build_lt.m` was confirmed compatible with SpiPy when LUTs are saved as MATLAB `-v7.3`
  - `SPIRES/RemoteSensing/multispectral.mat` now includes `VIIRS_SNPP`, `VIIRS_NOAA20`, and `VIIRS_NOAA21`
  - `SPIRES/prepInputs/build_viirs_luts.m` builds platform-specific VIIRS LUTs and writes `SensorTableBandOrder`

- **Reader / scene preparation**
  - Python VIIRS support lives under `spires/sensors/viirs/`
  - implemented APIs include:
    - `parse_viirs_surface_reflectance_filename(path)`
    - `open_viirs_surface_reflectance(path)`
    - `prepare_viirs_scene_for_inversion(source, ...)`
  - supported product families:
    - `VNP09GA` -> `snpp`
    - `VJ109GA` -> `noaa20`
    - `VJ209GA` -> `noaa21`
  - reader behavior:
    - reads native `500 m` and `1 km` layers separately
    - scales reflectance and geometry, handles fill/range filtering, and preserves scene metadata
    - promotes scene preparation onto the canonical `500 m` grid by exact `2x2` nearest-neighbor replication of `1 km` fields
    - merges selected imagery/moderate bands into `reflectance(y, x, band)`

- **QA / masks**
  - first-pass QA decoding produces cloud, cloud-shadow, and snow flags
  - prepared scenes expose transparent component masks rather than one opaque filter:
    - `mask_invalid_reflectance`
    - `mask_bad_geometry`
    - `mask_water`
    - `mask_low_observation_support`
    - `mask_cloud`
    - `mask_cloud_shadow`
    - `mask_snow`
  - both `valid_inversion_mask` and `valid_r0_mask` are produced
  - inversion cloud handling is policy-driven via `cloud_mask_policy`

- **R0 workflow**
  - implemented APIs include:
    - `compute_viirs_r0_indices(...)`
    - `build_viirs_r0_candidate_metrics(...)`
    - `build_viirs_timeseries(...)`
    - `build_viirs_r0(...)`
    - `build_viirs_r0_from_sources(...)`
  - current snow-free selection logic uses:
    - `NDVI = (I2 - I1) / (I2 + I1)`
    - `NDSI = (M4 - I3) / (M4 + I3)`
    - `blue_metric = M2`
  - current rule is:
    - if any valid date has `NDSI < 0`, choose the valid date with maximum `NDVI`
    - otherwise choose the valid date with minimum `M2`
  - incremental `R0` generation exists for laptop-scale runs and can reuse an existing saved `r0_path`

- **Inversion workflow**
  - implemented APIs include:
    - `ViirsExecutionProfile`
    - `get_viirs_execution_profile(name)`
    - `run_viirs_inversion(scene, r0, ..., lut_file=..., ...)`
  - wrapper behavior:
    - accepts a prepared scene or raw HDF input
    - accepts in-memory or on-disk `R0`
    - validates scene/R0/LUT band order and platform consistency
    - runs through `speedy_invert_dask(...)`
    - returns lazy dask-backed outputs until explicitly computed or written
  - current outputs:
    - `fsca`
    - `fshade`
    - `dust_concentration`
    - `grain_size`
    - `valid_inversion_mask`
  - netCDF writing now sanitizes copied attrs so derived outputs serialize cleanly

- **Canopy correction**
  - `run_viirs_inversion(...)` accepts optional `canopy_fraction` and `ice_fraction`
  - current workflow supports temporary GeoTIFF-based canopy input for testing, though tile-level zarr remains the preferred long-term format
  - Ross-style canopy/shade/ice snow adjustment is implemented
  - diagnostic layers now include:
    - `raw_viewable_snow_fraction`
    - `raw_shade_fraction`
    - `raw_canopy_fraction`
    - `raw_snow_fraction`

- **Ancillary layout**
  - ancillary helpers live in `spires/sensors/viirs/ancillary.py`
  - ancillary storage is tile-oriented under `data/viirs/ancillary/`
  - tile-level ancillary paths are sensor-family level rather than platform-specific
  - use `annual/`, not `yearly/`, for year-specific products

- **Notebook / examples**
  - `examples/09_viirs_reader_playground.ipynb` is now a compact batch-style example
  - it can build or reuse `R0`, run per-scene inversion, save netCDF outputs, save quicklook figures, and inspect workflow logs

### Current defaults and decisions

- canonical VIIRS analysis grid: `500 m`
- `1 km` to `500 m` alignment: nearest-neighbor replication only
- raw-reader and prepared-scene layers remain separate
- SNPP, NOAA-20, and NOAA-21 use separate LUTs
- default prepared-scene band order:
  - `I1`, `I2`, `I3`, `M1`, `M2`, `M3`, `M4`, `M5`, `M7`, `M8`, `M10`, `M11`
- snow is excluded from `valid_r0_mask` but not from inversion by default
- inversion cloud masking is policy-driven so snow-preserving experiments can relax cloud QA without changing `R0` filtering
- the current Python masking approach intentionally exposes component masks, even though Ross MATLAB uses a more bundled daily-filter style

### Testing status

- targeted VIIRS tests are passing, including:
  - `tests/test_viirs_hdf.py`
  - `tests/test_viirs_qa.py`
  - `tests/test_viirs_r0.py`
  - `tests/test_viirs_workflow.py`
  - `tests/test_viirs_ancillary.py`
  - `tests/test_invert_dask.py`
  - `tests/test_logging_utils.py`
- workflow regression coverage includes:
  - canopy and ice snow-fraction adjustment
  - GeoTIFF/HDF-derived canopy attrs being stripped before netCDF writes
  - safe serialization of multi-dimensional attrs

### Still open

- grouped-inversion status:
  - added a new NumPy-first grouping module at `spires/speedy_utol.py`
  - grouping is now integrated into the active `speedy_invert_array2d(...)` / `speedy_invert_dask(...)` / `run_viirs_inversion(...)` path
  - current supported representative methods are:
    - `first`: use the first member row in each group (closest to the original MATLAB-style representative choice)
    - `chunk_bin_mean`: use the mean of all rows in the group within the current chunk
  - `bin_center` was tested during benchmarking and later removed from the repo
  - grouping is optional and currently defaults to `chunk_bin_mean`
  - grouping supports:
    - scalar or per-variable tolerances
    - solar-zenith tolerance defaulting to `100 * tolerance` when a single scalar tolerance is used
    - chunk-local grouping without `xarray.stack(...).values`
  - `apply_valid_inversion_mask=False` now also disables grouped prefiltering by `valid_inversion_mask`, so grouped and ungrouped benchmarking use the same solve footprint
- grouped-inversion debugging note:
  - during NOAA-20 notebook benchmarking, grouped runs (`first`, `chunk_bin_mean`) produced coherent missing patches over ocean while ungrouped inversion still returned values there
  - this is unlikely to be explained purely by non-finite per-pixel `R`, `R0`, or solar zenith, because in that case the ungrouped path should also fail on the same pixels
  - current working hypothesis: many ocean pixels collapse into a large group, the chosen group representative fails inversion, and that NaN solution is then broadcast back to the whole group
  - likely next diagnostic: compare a failed group representative against several member pixels and confirm whether the representative returns NaNs while some individual member pixels still invert successfully
  - likely mitigation if confirmed: fallback from failed grouped representatives to finer regrouping or per-pixel inversion for the affected group
- consider early `HDF -> zarr` staging so repeated raw-HDF reopening is not required for large jobs
- optionally allow lighter reader modes that skip QA/support fields when downstream steps only need reflectance and geometry
- replace the temporary GeoTIFF-first canopy path with a tile-level zarr `canopy_fraction` product
- optionally add a real ice-fraction ancillary layer
