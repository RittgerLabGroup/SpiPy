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
- preserves separate intermediate cubes:
  - `reflectance_500m_native`
  - `reflectance_1km_on_500m`
- carries geometry, QA, and support fields onto the same 500 m grid
- creates `qa_raw_stack(y, x, qa_flag)`
- creates initial transparent masks:
  - `mask_invalid_reflectance`
  - `mask_bad_geometry`
  - `mask_water`
  - `mask_low_observation_support`
  - placeholder `mask_cloud`
  - placeholder `mask_cloud_shadow`
  - placeholder `mask_snow`
- `valid_inversion_mask`
- `valid_r0_mask`

### Reader optimization notes for later

- consider optional omission of QA and support fields for job types that only need reflectance + geometry or reflectance alone
- consider an early `HDF -> zarr` staging workflow so cluster jobs do not repeatedly reopen raw HDF files for inversion or compositing

### Logging notes for later

- keep using plain-text structured logs that can be written to `.log` or `.txt` files for Slurm jobs
- log a job-level header for batch runs with items such as:
  - `SLURM_JOB_ID`
  - hostname
  - environment name
  - repo root and, if available, git commit
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

### Testing status

- targeted test file `tests/test_viirs_hdf.py` passes
- verified against the local example files in `~/Downloads`:
  - `VNP09GA.A2026112.h08v05.002.2026113100255.h5`
  - `VJ109GA.A2026112.h08v05.002.2026113072313.h5`

### Notebook

- exploratory notebook created at:
  - `examples/09_viirs_reader_playground.ipynb`
- first cell imports the new reader functions and demonstrates reading a local `VNP09GA` file from `~/Downloads`
- later notebook follow-up:
  - add an example of overriding the QA-derived cloud and cloud-shadow masks with an external mask file once that product exists

## Recommended Next Step

With basic VIIRS QA decoding now implemented, likely next priorities are:

- refine and validate the QA-based cloud, cloud-shadow, and snow rules against real scenes
- support external cloud + cloud-shadow mask products in real workflows once those masks are generated
- add an example notebook workflow that demonstrates the external cloud-mask override path
- decide which decoded QA component layers should remain in default outputs versus debug-only outputs

## Important Design Decisions Already Made

- canonical VIIRS analysis grid: `500 m`
- 1 km to 500 m alignment method: nearest neighbor only, no interpolation
- raw-reader layer and prepared-scene layer should remain separate
- SNPP and NOAA-20 should use separate LUTs
