# VIIRS VNP09GA SPIReS Workflow Notes

## Goal

Sketch a clean, end-to-end module layout for running SPIReS spectral unmixing on VIIRS `VNP09GA` reflectance data.

Assumptions:

- a VIIRS-specific SPIReS LUT will be generated separately and made available to this workflow
- Landsat work is out of scope for this note
- the existing SPIReS core inversion code should remain as unchanged as possible

## Design Principles

- keep SPIReS core inversion logic sensor-agnostic
- isolate VIIRS-specific ingestion, QA handling, geometry, and compositing in a dedicated namespace
- mirror the existing Sentinel-2 workflow structure where useful
- use xarray + dask + zarr for scalable preprocessing and inversion
- preserve a consistent schema for reflectance, geometry, masks, `R0`, and inversion outputs

## Proposed Module Layout

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

### `bands.py`

- define the canonical VIIRS band set used by the LUT and workflow
- map `VNP09GA` SDS names to SPIReS band names and order
- provide helpers for selecting and ordering reflectance bands to match the LUT

### `hdf.py`

- open raw `VNP09GA` HDF files
- read reflectance SDSs
- read solar zenith / azimuth and sensor zenith / azimuth
- read QA layers and metadata
- return a consistent xarray `Dataset`

### `qa.py`

- decode VIIRS QA bitfields
- produce masks for:
  - invalid / fill
  - cloudy / probably cloudy
  - cloud shadow if available
  - bad aerosol or low-quality retrieval
  - snow flag if useful for `R0` filtering
  - extreme view geometry if desired

### `geometry.py`

- normalize geometry fields onto the reflectance grid
- provide helpers such as:
  - `get_solar_zenith(ds)`
  - `get_sensor_zenith(ds)`
- handle interpolation or nearest-neighbor alignment where geometry resolution differs from reflectance resolution

### `stack.py`

- scale reflectance to physical units
- handle fill values and valid-range filtering
- rename and reorder bands to match the LUT
- build analysis-ready data cubes:
  - `reflectance(y, x, band)`
  - `reflectance(time, y, x, band)`

### `r0.py`

- build background reflectance `R0` from June, July, August, and September VIIRS time series
- apply QA-based filtering
- exclude snow-covered observations when constructing `R0`
- composite valid summer observations using a robust summary statistic such as median
- write quality/support layers such as valid observation counts

### `masks.py`

- combine QA masks with ancillary masks
- create a unified analysis mask for both `R0` creation and inversion
- support masks such as:
  - water
  - forest / canopy
  - permanent snow / ice
  - terrain shadow if derived from DEM + geometry

### `workflow.py`

- provide high-level orchestration wrappers, for example:
  - `build_viirs_timeseries(...)`
  - `build_viirs_r0(...)`
  - `prepare_viirs_scene_for_inversion(...)`
  - `run_viirs_inversion(...)`
  - `run_viirs_inversion_dask(...)`

### `io.py`

- write and read intermediate products
- prefer zarr for chunked reflectance stacks and inversion outputs
- support netCDF or GeoTIFF export for selected derived layers as needed later

## End-to-End Workflow

### 1. Build an analysis-ready summer stack

- collect all June, July, August, and September `VNP09GA` HDF files for a region and year range
- read reflectance, geometry, and QA
- apply scale factors and invalid-value masking
- align geometry and QA to the reflectance grid if needed
- build a chunked xarray/zarr time stack

Suggested variables:

- `reflectance(time, y, x, band)`
- `solar_zenith(time, y, x)`
- `sensor_zenith(time, y, x)`
- `qa_mask(time, y, x)`
- optional raw QA layers for debugging

### 2. Create background reflectance `R0`

Initial version:

- use June through September observations only
- exclude cloudy / shadowed / invalid observations using VIIRS QA
- exclude observations flagged as snow or classified as likely snow
- optionally exclude observations with extreme sensor zenith
- composite by pixel and band using a robust statistic such as median

Suggested outputs:

- `r0_reflectance(y, x, band)`
- `r0_count(y, x, band)` or `r0_count(y, x)`
- `r0_quality(y, x)`

Possible later refinements:

- low-percentile rather than median compositing
- temporal windows by year or multi-year climatologies
- gap-filling and neighborhood interpolation
- land-cover-aware smoothing
- topographic stratification

### 3. Run inversion

For each VIIRS scene or time step:

- read observed reflectance
- select and order the same VIIRS bands used by the LUT
- attach `R0`
- attach solar zenith
- run SPIReS with the existing dask-enabled inversion path

Suggested outputs:

- `fsca`
- `fshade`
- `dust_concentration`
- `grain_size`
- `valid_inversion_mask`
- optional support fields such as observation count or fit residual

## Parallelization Strategy

Recommended approach:

- preprocess raw HDFs into zarr once
- store `reflectance` as chunked xarray/zarr
- keep `band` unchunked
- store `R0` as static `y, x, band`
- run inversion with the existing `speedy_invert_dask(...)` path

Suggested chunking:

- `time = 1`
- `y = 256-1024`
- `x = 256-1024`
- `band = -1`

Rationale:

- this matches the assumptions of the existing dask/xarray inversion wrapper
- it enables parallel processing over time and spatial tiles
- it keeps each spectral vector intact for inversion

## Ancillary Data Needed

### Essential for a first working version

- VIIRS SPIReS LUT
- water mask
- forest / canopy mask
- DEM
- VIIRS QA decoding rules
- a snow-free filtering rule for `R0` generation

### Strongly recommended

#### Water mask

- needed to avoid building invalid `R0` over lakes, reservoirs, and coast-adjacent mixed pixels
- also useful for suppressing inversion over open water

#### Forest / canopy mask

- important where open-snow interpretation is desired
- trees complicate both background reflectance and inversion

#### DEM

Useful for:

- terrain shadow estimation
- slope and aspect
- elevation-based plausibility filters
- future topographic correction or illumination screening

#### Land-cover mask

Useful for:

- excluding or flagging urban areas
- excluding wetlands or problematic bright surfaces
- stratifying background compositing

#### Snow mask or snow prior

Needed to filter snow-contaminated observations out of `R0`.

Possible sources:

- VIIRS QA snow flag
- VIIRS-derived NDSI thresholding
- external snow products if needed later

#### Permanent snow / ice or glacier mask

- helpful to avoid attempting snow-free `R0` in places that never truly become snow-free

### Optional but likely useful later

- terrain shadow mask derived from DEM + solar geometry
- cloud mask refinement beyond native VIIRS QA
- region-of-interest boundaries

## Stable Schema Recommendation

Aim for the following variable naming conventions:

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

## Recommended Next Step

Before implementing code, define the exact xarray schema and confirm:

- which VIIRS bands will be used in the LUT
- which `VNP09GA` SDS names map to those bands
- how QA filtering will be handled for clouds, snow, and invalid pixels
- whether `R0` will be seasonal by year, multi-year, or climatological

## Implementation Status (2026-04-24)

### Repo / session note

- the project was moved from `Codex/SpiPy` to `Codex/SpiPy_RLG/SpiPy`
- a fresh Codex session should be started from the new repo root if possible, otherwise writes may keep requiring escalated filesystem permission because the old writable sandbox root lingers
- in Codex sandboxed terminal sessions, use `mamba` rather than `conda` for the project environment; `spipy14` is discoverable under the local mamba root, while `conda` may look in different env directories
- Codex cannot write to the default mamba cache path at `~/.cache/mamba`, so set a writable temporary cache when running mamba commands:
  - `XDG_CACHE_HOME=/tmp/mamba-cache mamba run -n spipy14 python -m pytest SpiPy/tests/test_viirs_r0.py`

### LUT work completed

- cloned upstream `SPIRES` into `Codex/SpiPy_RLG/SPIRES`
- inspected SPIRES `prepInputs/build_lt.m`
- confirmed it produces a compatible core LUT structure for SpiPy when saved as MATLAB `-v7.3`
- confirmed the existing OLI and Sentinel-2 LUTs share the same non-band grid vectors:
  - grain size: `30:10:1200`
  - dust: `[0 0.1 1:10:991]`
  - solar zenith: `0:90`
- confirmed `VNP09GA` and `VJ109GA` have identical internal HDF dataset names and shapes for the core reflectance / geometry / QA layers
- confirmed SNPP and NOAA-20 VIIRS are not spectrally identical, so sensor-specific LUTs are preferred

### SPIRES MATLAB changes completed

- updated `SPIRES/RemoteSensing/multispectral.mat`
- added new sensor entries:
  - `VIIRS_SNPP`
  - `VIIRS_NOAA20`
- kept the legacy `VIIRS` rows unchanged
- populated `LowerWavelength` / `UpperWavelength` with platform-specific bounds so `build_lt(..., integrate=1)` uses different bandpasses for SNPP and NOAA-20
- added MATLAB LUT build script:
  - `SPIRES/prepInputs/build_viirs_luts.m`
- script behavior:
  - builds both `VIIRS_SNPP` and `VIIRS_NOAA20`
  - saves MATLAB `-v7.3` LUTs compatible with SpiPy
  - writes `SensorTableBandOrder`
  - uses an 8-worker parallel pool
  - restarts the pool between sensors and retries once after pool-related `parfor` failures
- note: `build_lt.m` progress printing is misleading; it prints values approaching about `0.98`, not `100`, because the progress denominator is wrong

### Python VIIRS reader work completed

New files added:

- `spires/sensors/__init__.py`
- `spires/sensors/base.py`
- `spires/sensors/viirs/__init__.py`
- `spires/sensors/viirs/bands.py`
- `spires/sensors/viirs/hdf.py`
- `tests/test_viirs_hdf.py`
- `examples/09_viirs_reader_playground.ipynb`

Package export update:

- `spires/__init__.py` now re-exports the VIIRS reader functions through `spires.sensors.viirs`

Implemented reader API:

- `parse_viirs_surface_reflectance_filename(path)`
- `open_viirs_surface_reflectance(path)`
- `prepare_viirs_scene_for_inversion(source, ...)`

### Current Python reader behavior

`open_viirs_surface_reflectance(path)`:

- supports both `VNP09GA` and `VJ109GA`
- detects platform from filename
- reads native 1 km and 500 m VIIRS grids separately
- can optionally read only the reflectance bands needed by an explicit band list or a provided LUT
- applies scale / fill / valid-range handling to reflectance and geometry layers
- returns an xarray `Dataset` with:
  - `reflectance_1km(y_1km, x_1km, band_1km)`
  - `reflectance_500m(y_500m, x_500m, band_500m)`
  - 1 km geometry fields
  - raw QA/support fields
  - scene attrs like `product`, `platform`, `tile`, `acquisition_date`

`prepare_viirs_scene_for_inversion(source, ...)`:

- uses the 500 m grid as the canonical analysis grid
- expands all 1 km fields onto 500 m using exact nearest-neighbor `2x2` replication
- merges imagery and moderate bands into:
  - `reflectance(y, x, band)`
- carries geometry, QA, and support fields onto the same 500 m grid
- creates `qa_raw_stack(y, x, qa_flag)`
- decodes first-pass VIIRS QA masks into:
  - `mask_cloud_qa`
  - `mask_cloud_shadow_qa`
  - `mask_snow_qa`
- supports overriding cloud and cloud-shadow masking with an external mask source
- creates final analysis masks:
  - `mask_invalid_reflectance`
  - `mask_bad_geometry`
  - `mask_water`
  - `mask_low_observation_support`
  - `mask_cloud`
  - `mask_cloud_shadow`
  - `mask_snow`
- `valid_inversion_mask`
- `valid_r0_mask`
- does not apply `mask_snow` to inversion, but does apply it to `valid_r0_mask`
- supports configurable cloud masking for inversion through `cloud_mask_policy`:
  - `strict`: apply cloud and cloud-shadow masks
  - `snow_wins`: ignore cloud and cloud shadow where snow is also flagged
  - `ignore_cloud`: ignore cloud but keep cloud-shadow masking
  - `ignore_cloud_and_shadow`: ignore both cloud and cloud-shadow masks
- stores effective cloud masks used by inversion as:
  - `mask_cloud_for_inversion`
  - `mask_cloud_shadow_for_inversion`

### Current Python R0 work completed

New file added:

- `spires/sensors/viirs/r0.py`

Implemented R0 API:

- `compute_viirs_r0_indices(prepared_ds, ...)`
- `build_viirs_r0_candidate_metrics(prepared_timeseries, ...)`
- `build_viirs_timeseries(sources, ...)`
- `build_viirs_r0(prepared_timeseries, ...)`

Current VIIRS R0 band choices:

- `NDVI`: `I2`, `I1`
- `NDSI`: `M4`, `I3`
- MODIS `b3` substitute: `M2`

Current VIIRS R0 candidate-filtering logic:

- start from prepared summer scenes on the common `500 m` grid
- compute:
  - `NDVI = (I2 - I1) / (I2 + I1)`
  - `NDSI = (M4 - I3) / (M4 + I3)`
  - `blue_metric = M2`
- define candidate dates per pixel:
  - `candidate_ndvi`: dates where `valid_r0_mask` is true and `NDSI < 0`
  - `candidate_blue_metric`: dates where `valid_r0_mask` is true and `M2 >= 0.10`
- selection rule per pixel:
  - if the pixel has at least one candidate date with `NDSI < 0`, choose the candidate date with maximum `NDVI`
  - otherwise choose the valid date with minimum `M2`

Current R0 outputs:

- `r0_reflectance(y, x, band)`
- `r0_source_index(y, x)`
- `r0_source_time(y, x)`
- `r0_used_min_blue_rule(y, x)`
- `r0_count(y, x)`
- `max_ndvi(y, x)`
- `min_ndsi(y, x)`
- `min_blue_metric(y, x)`
- `has_negative_ndsi(y, x)`

Additional R0 scaling work completed:

- added `reduce_viirs_prepared_scene_for_r0(...)`
- this strips prepared scenes down to the minimum R0 variables:
  - `reflectance`
  - `sensor_zenith`
  - `valid_r0_mask`
- updated `build_viirs_timeseries(...)` to support lighter-weight R0 staging with:
  - `keep_variables="r0"`
  - `zarr_path=...`
  - optional chunked zarr writing
- added `build_viirs_r0_from_sources(...)`
- this provides an incremental R0 path that processes one scene at a time instead of materializing the full summer stack in memory first
- this should be the preferred path for MacBook-scale development when full-scene summer stacks are too large for RAM
- `build_viirs_r0_from_sources(...)` now accepts `r0_path=...`
- if `r0_path` already exists and `overwrite=False`, it loads the saved R0 and logs `status="loaded_existing"` with the R0 path, time coverage, bands, and output shape
- if `r0_path` does not exist, it builds R0 incrementally, writes the file, and logs the output path

### Current Python inversion workflow work completed

New file added:

- `spires/sensors/viirs/workflow.py`

Implemented inversion workflow API:

- `ViirsExecutionProfile`
- `get_viirs_execution_profile(name)`
- `run_viirs_inversion(scene, r0, ..., lut_file=..., ...)`

Current VIIRS inversion wrapper behavior:

- accepts a prepared VIIRS scene dataset or a VIIRS HDF path
- accepts an in-memory R0 dataset or an on-disk R0 file path
- loads the SPIRES LUT through `LutInterpolator`
- validates:
  - scene vs R0 band order
  - scene vs LUT band order
  - scene platform vs LUT filename platform tag
- applies inversion-safe chunking
- supports named execution profiles:
  - `local`
  - `cluster`
- calls the core `speedy_invert_dask(...)` path
- optionally masks outputs with `valid_inversion_mask`
- supports overriding output masking with `apply_valid_inversion_mask=False`
  - inversion outputs are left unmasked
  - `valid_inversion_mask` is still preserved in the returned dataset for diagnostics
  - output attrs record `valid_inversion_mask_mode="output_only"`
- writes start and summary log events
- returns lazy dask-backed outputs unless the caller explicitly computes or writes them
- full inversion work is triggered by calls such as:
  - `inversion_ds.compute()`
  - `inversion_ds.to_netcdf(...)`
  - `inversion_ds.fsca.plot(...)`
- netCDF serialization sanitizes copied metadata:
  - dict attrs such as `units_by_band` are JSON-encoded
  - incompatible `_FillValue`, `missing_value`, and invalid `valid_range` attrs are dropped
  - multi-dimensional array attrs are JSON-encoded because netCDF4 cannot write them directly
  - reserved source-storage attrs such as `_FillValue`, `missing_value`, `add_offset`, `scale_factor`, and `DIMENSION_LIST` are stripped from derived outputs
  - `valid_inversion_mask` is written as a clean boolean mask with flag metadata
- fixed a core dask inversion API mismatch where `speedy_invert_dask(...)` passed `spectrum_shade` into `speedy_invert_array2d(...)`

Current inversion outputs:

- `fsca`
- `fshade`
- `dust_concentration`
- `grain_size`
- `valid_inversion_mask`

Canopy-correction update:

- `run_viirs_inversion(...)` now supports optional ancillary inputs:
  - `canopy_fraction=...`
  - `ice_fraction=...`
- current temporary canopy input can be a GeoTIFF, xarray object, zarr/netCDF dataset, or scalar
- this GeoTIFF support is for immediate testing; preferred longer-term storage remains tile-level zarr under the VIIRS ancillary layout
- default `canopy_fraction="auto"` looks for a local tile-level file such as:
  - `data/viirs/ancillary/tiles/h08v05/static/h08v05_canopycover_LC100_global_v301_2019.tif`
- the first all-zero `raw_canopy_fraction` notebook result was caused by not passing a canopy source; the output attrs showed `canopy_correction_applied=0` and `source="none"`
- the h08v05 test GeoTIFF has real values:
  - native values are `0..100` percent
  - mean canopy cover is about `9%`
  - shape is `2400 x 2400`, matching the VIIRS tile grid
- the canopy adjustment follows the Ross v2025 MODIS logic:
  - source canopy is interpreted as fraction or percent and normalized to `0..1`
  - view-angle-adjusted canopy obstruction is computed from canopy cover and `sensor_zenith`
  - default vertical-to-horizontal crown radius parameter is `2.7`
  - adjusted snow is calculated as `viewable_snow / (1 - min(shade + canopy + ice, 0.99))`, clipped to `0..1`, with ice as a lower bound
- new diagnostic snow layers are:
  - `raw_viewable_snow_fraction`
  - `raw_shade_fraction`
  - `raw_canopy_fraction`
  - `raw_snow_fraction`
- `fsca` and `fshade` are currently retained as backward-compatible aliases for `raw_viewable_snow_fraction` and `raw_shade_fraction`
- for already-computed notebook datasets created before the latest metadata fix, bad GeoTIFF attrs can be removed in-place before writing:

```python
bad_attrs = {"_FillValue", "missing_value", "add_offset", "scale_factor", "DIMENSION_LIST", "valid_range"}
for name in ["raw_canopy_fraction", "raw_snow_fraction", "raw_viewable_snow_fraction", "raw_shade_fraction"]:
    if name in inversion_ds:
        for attr in bad_attrs:
            inversion_ds[name].attrs.pop(attr, None)
        inversion_ds[name].encoding.clear()
```

### Reader optimization notes for later

- consider optional omission of QA and support fields for job types that only need reflectance + geometry or reflectance alone
- consider an early `HDF -> zarr` staging workflow so cluster jobs do not repeatedly reopen raw HDF files for inversion or compositing

### Logging / batch-processing notes

- keep using plain-text structured logs that can be written to `.log` or `.txt` files for Slurm jobs
- keep per-step structured events for reader, `R0`, and inversion stages
- for reader events, record at minimum:
  - full input HDF path
  - LUT file path
  - selected band subset
  - split native `500 m` vs `1 km` band lists
  - band selection source such as explicit list, LUT metadata, LUT filename, or default
  - elapsed time

Default prepared-scene band order is currently:

- `I1`, `I2`, `I3`, `M1`, `M2`, `M3`, `M4`, `M5`, `M7`, `M8`, `M10`, `M11`

### VIIRS ancillary input structure

- added reusable ancillary path helpers in:
  - `spires/sensors/viirs/ancillary.py`
- exported helpers include:
  - `viirs_sensor_root(...)`
  - `viirs_ancillary_root(...)`
  - `viirs_tile_ancillary_root(...)`
  - `viirs_static_ancillary_path(...)`
  - `viirs_annual_ancillary_path(...)`
  - `viirs_ancillary_path(...)`
  - `create_viirs_ancillary_layout(...)`
- local input-oriented scaffold is under:
  - `data/viirs/ancillary/`
- current scaffold:

```text
data/
  viirs/
    ancillary/
      global/
        canopy/
        dem/
        glacier_ice/
        landcover/
        water/
      tiles/
        h08v05/
          static/
          annual/
            2026/
          logs/
```

- static tile-level products should be reused across years and should share the VIIRS 500 m grid:
  - `canopy_fraction.zarr`
  - `water_mask.zarr`
  - `dem.zarr`
  - `glacier_ice_fraction.zarr`
  - `landcover.zarr`
- annual tile-level products are year-specific:
  - `r0_reflectance.zarr` or `r0_reflectance.nc`
  - `r0_support.zarr`
  - `qa_summary.zarr`
  - `snow_prior.zarr`
- example helper usage:

```python
canopy_path = viirs_ancillary_path("data", "h08v05", "canopy_fraction")
r0_path = viirs_ancillary_path("data", "h08v05", "r0_reflectance", year=2026)
```

- large local products under `data/` are ignored by `.gitignore`
- use `annual`, not `yearly`, for year-specific ancillary directories

### Masking comparison note from `SPIRES_2025_0_1_ross`

- Ross MATLAB workflow does not simply invert every pixel by default and mask only after inversion
- it builds daily bitfield filters such as:
  - `daily_nodata_filter_s`
  - `daily_zero_filter_s`
  - `daily_grain_size_dust_s`
- the key pre-inversion switch is `snowInversionFilter` in `SpiresInversor.m`
  - `0`: restrict pixels before inversion using daily filters
  - `10`: comment indicates do not remove anything before inversion
- for VIIRS, the pre-inversion branch excludes pixels based on:
  - `dailyNoDataFilter` bits for no input, background reflectance nodata, STC cloud, and cloud extension
  - `dailyZeroFilter` bits for low NDSI and water
- VIIRS QA bits are read and compacted into `QF_1km`, then used mainly for weights and cloud/saltpan status
- the Ross VIIRS pre-inversion branch does not appear to directly apply the raw QF cloud bit as a simple QA cloud mask
- Ross background reflectance generation uses VIIRS QA more directly:
  - cloud shadow
  - heavy aerosol
  - snow/ice
  - thin cirrus
  - adjacent cloud
  - band-specific bad-quality flags
- current Python VIIRS workflow differs intentionally by exposing transparent component masks:
  - `mask_invalid_reflectance`
  - `mask_bad_geometry`
  - `mask_water`
  - `mask_low_observation_support`
  - `mask_cloud`
  - `mask_cloud_shadow`
  - `mask_snow`
  - `valid_inversion_mask`
- possible follow-up: keep the explicit Python component-mask design, but add a Ross-style policy layer that can choose between:
  - strict QA/support masking
  - snow-preserving relaxed cloud masking
  - Ross-like daily filter behavior
  - run-all-pixels inversion with masks preserved only as output diagnostics

### Testing status

- targeted test files pass:
  - `tests/test_invert_dask.py`
  - `tests/test_viirs_hdf.py`
  - `tests/test_viirs_qa.py`
  - `tests/test_viirs_r0.py`
  - `tests/test_viirs_workflow.py`
  - `tests/test_logging_utils.py`
- latest targeted run completed with:
  - `tests/test_invert_dask.py`
  - `tests/test_viirs_hdf.py`
  - `tests/test_viirs_workflow.py`
  - `tests/test_viirs_r0.py`
- latest canopy/netCDF targeted run completed with:
  - `tests/test_viirs_workflow.py`
- `tests/test_viirs_workflow.py` now includes regression coverage for:
  - Ross-style canopy and ice snow-fraction adjustment
  - GeoTIFF/HDF-style canopy attrs being stripped from `raw_canopy_fraction`
  - multi-dimensional attrs being serialized safely before netCDF writes
- verified against the local example files in `~/Downloads`:
  - `VNP09GA.A2026112.h08v05.002.2026113100255.h5`
  - `VJ109GA.A2026112.h08v05.002.2026113072313.h5`

### Notebook

- batch-style notebook example updated at:
  - `examples/09_viirs_reader_playground.ipynb`
- notebook is now titled `VIIRS Batch-Style Playground`
- notebook now provides a compact batch-ready VIIRS example:
  - configure one VIIRS product family / sensor, LUT, R0-source scene glob, and inversion-scene glob
  - expect HDF inputs under `data/viirs/inputs/<sensor>/r0/` and `data/viirs/inputs/<sensor>/reflectance/`
  - validate that R0-source and inversion scenes match one platform and one tile
  - build or reuse saved `R0` through `build_viirs_r0_from_sources(..., r0_path=...)`
  - write reusable R0 files under `data/viirs/r0/<sensor>/<tile>/<year>/`
  - loop over all scenes matched by `inversion_glob`
  - prepare each inversion scene with the configured cloud-mask policy
  - run each scene with `run_viirs_inversion(...)`
  - override output masking with `apply_valid_inversion_mask=False` so inversion fields remain available while `valid_inversion_mask` is carried as a diagnostic layer
  - explicitly compute each lazy inversion with `ProgressBar(dt=5.0)` and report per-scene timing
  - save a sensor/date-specific netCDF for each inversion scene
  - save a 1x4 quicklook figure for each scene with raw viewable snow, shade, canopy, and snow fraction
  - inspect the workflow log tail
- current default notebook outputs are written under:
  - R0: `data/viirs/r0/<sensor>/<tile>/<year>/`
  - inversion netCDFs: `outputs/viirs/<sensor>/data/`
  - quicklook figures: `outputs/viirs/<sensor>/figs/`
  - logs: `outputs/`
- notebook performance note:
  - `run_viirs_inversion(...)` creates a lazy dask graph quickly; plotting or writing the inversion outputs triggers the expensive full computation
  - use `from dask.diagnostics import ProgressBar` and `with ProgressBar(dt=5.0): inversion_ds = inversion_ds.compute()` for a visible laptop timing run
  - increasing `ProgressBar(dt=...)` helps avoid notebook `IOStream.flush timed out` messages

### Pixel-grouping / uniquetol status

- `spires/utol.py` contains Python helpers for uniquetol-style grouping:
  - `unique_elements_spacetime(...)`
  - `unique_elements_space(...)`
  - `uniquetol_1d(...)`
- these helpers flatten observation reflectance and background reflectance, group approximately repeated spectral vectors, and return labels plus the unique spectra
- current active VIIRS inversion path does not call these helpers
- active VIIRS path is:
  - `run_viirs_inversion(...)`
  - `speedy_invert_dask(...)`
  - `speedy_invert_array2d(...)`
  - C++ `invert_array2d(...)`
- the C++ `invert_array2d(...)` currently loops over every `y, x` pixel and calls `invert(...)` independently, aside from quick NaN handling inside `invert(...)`
- repository search found no Python call sites for:
  - `unique_elements_spacetime`
  - `unique_elements_space`
  - `uniquetol_1d`
  - `spires.utol`
- `utol.py` is referenced by Sphinx docs in `doc/source/reference.rst`
- older Sentinel notebooks mention `implement uniquetol` in cell 1 markdown:
  - `examples/05_sentinel_snow_inversion.ipynb`
  - `examples/speedy_invert_sentinel.ipynb`
- MATLAB-side SPIRES code has analogous uniquetol usage, including:
  - `SPIRES/core/run_spires.m`
  - `SPIRES/core/speedyUniqueTol.m`

## Recommended Next Step

Important next step: implement pixel grouping with the existing `spires/utol.py` uniquetol helpers in the main inversion code path so repeated or near-repeated spectra are inverted once and then mapped back to the full raster. This should be integrated with the dask-backed `speedy_invert_dask(...)` / `run_viirs_inversion(...)` path and covered by tests that confirm grouped and ungrouped inversion outputs match within tolerance.

## Update 2026-04-27: VIIRS three-sensor support

- the VIIRS Python workflow now supports all three current VIIRS surface-reflectance product families:
  - `VNP09GA` -> `snpp`
  - `VJ109GA` -> `noaa20`
  - `VJ209GA` -> `noaa21`
- the filename parser and scene attrs now recognize `VJ209GA` as NOAA-21
- LUT platform validation now recognizes NOAA-21 LUT filenames containing `noaa21`
- `SPIRES/RemoteSensing/multispectral.mat` now includes a `VIIRS_NOAA21` sensor-table entry
- `SPIRES/prepInputs/build_viirs_luts.m` can be reused to build the NOAA-21 LUT from `VIIRS_NOAA21`
- VIIRS ancillary paths remain sensor-family level, not platform-specific:
  - `data/viirs/ancillary/tiles/<tile>/static/...`
  - `data/viirs/ancillary/tiles/<tile>/annual/<year>/...`
- this means VNP/SNPP, VJ1/NOAA-20, and VJ2/NOAA-21 scenes for the same tile all look in the same `data/viirs/ancillary` tree
- current repo contains a saved SNPP R0 example:
  - `examples/outputs/viirs/viirs_r0_snpp_h08v05_2019.nc`
- no NOAA-20/VJ1 R0 file was found in the repo as of this update
- latest focused test run passed:
  - `tests/test_viirs_hdf.py`
  - `tests/test_viirs_workflow.py`
  - `tests/test_viirs_ancillary.py`

Next canopy-adjustment checks:

- later, replace the temporary GeoTIFF path with a zarr-based tile-level `canopy_fraction` ancillary product and keep the GeoTIFF path as a short-term compatibility option
- optionally add a real ice-fraction ancillary layer; the workflow already accepts `ice_fraction`, but the next tests can leave it unset or use a scalar zero

## Important Design Decisions Already Made

- canonical VIIRS analysis grid: `500 m`
- 1 km to 500 m alignment method: nearest neighbor only, no interpolation
- raw-reader layer and prepared-scene layer should remain separate
- SNPP, NOAA-20, and NOAA-21 should use separate LUTs
- snow is not included as an inversion mask by default; snow remains excluded from `valid_r0_mask`
- cloud masking for inversion is policy-driven so snow-preserving tests can relax cloud QA without changing R0 filtering
