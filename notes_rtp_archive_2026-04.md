# VIIRS / MODIS SPIReS Workflow Notes (Archive, April 2026)

This archived note preserves the longer implementation log and intermediate diagnostic history that led to the current shared-core sensor architecture.

For the current working note, use:

- `notes_rtp.md`

## Goal

Keep a compact working spec for running SPIReS spectral unmixing on VIIRS and MODIS surface-reflectance products.

Assumptions:

- sensor-specific SPIReS LUTs are generated separately and made available to this workflow
- Landsat work is out of scope for this note
- the existing SPIReS core inversion code should remain as unchanged as possible

## Design Principles

- keep SPIReS core inversion logic sensor-agnostic
- keep sensor-specific ingestion, QA handling, geometry, and compositing isolated in dedicated namespaces
- make higher-level workflows dispatch through a shared sensor registry rather than ad hoc `if sensor == ...` branching
- prefer the richer VIIRS-style API as the reference contract when MODIS / VIIRS behavior diverges
- use xarray + dask + zarr for scalable preprocessing and inversion
- preserve a consistent schema for reflectance, geometry, masks, `R0`, and inversion outputs

## Implementation Shape

```text
spires/
  sensors/
    __init__.py
    api.py
    registry.py
    base.py
    viirs/
      __init__.py
      ancillary.py
      bands.py
      geospatial.py
      hdf.py
      qa.py
      r0.py
      workflow.py
    modis/
      __init__.py
      ancillary.py
      bands.py
      geospatial.py
      hdf.py
      qa.py
      r0.py
      workflow.py
examples/
  09_viirs_reader_playground.ipynb
  10_modis_reader_playground.ipynb
  11_unified_multisensor_playground.ipynb
tests/
  test_viirs_hdf.py
  test_viirs_qa.py
  test_viirs_r0.py
  test_viirs_inversion.py
  test_modis_hdf.py
  test_modis_qa.py
  test_modis_r0.py
```

## Module Responsibilities

- `spires/sensors/registry.py`: sensor/platform normalization, adapter registration, and future extension point for Sentinel-2 / Landsat / HLS
- `spires/sensors/api.py`: generic dispatch layer for `open_surface_reflectance(...)`, `prepare_scene_for_inversion(...)`, `build_timeseries(...)`, `build_r0_from_sources(...)`, and `run_inversion(...)`
- `spires/sensors/<sensor>/bands.py`: canonical band definitions and LUT-aligned ordering
- `spires/sensors/<sensor>/hdf.py`: raw product reader plus prepared-scene construction
- `spires/sensors/<sensor>/qa.py`: QA decoding plus optional external cloud-mask loading
- `spires/sensors/<sensor>/geospatial.py`: grid metadata parsing and spatial-ref propagation
- `spires/sensors/<sensor>/r0.py`: summer background-reflectance selection, timeseries staging, zarr writing, and `R0` compositing
- `spires/sensors/<sensor>/workflow.py`: inversion wrapper, chunk/profile handling, canopy/ice adjustment, and output serialization safeguards

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
# spires/sensors/api.py

def open_surface_reflectance(source, *, sensor, platform=None, **kwargs) -> xr.Dataset: ...
def prepare_scene_for_inversion(source, *, sensor, platform=None, **kwargs) -> xr.Dataset: ...
def build_timeseries(sources, *, sensor, platform=None, **kwargs) -> xr.Dataset: ...
def build_r0(prepared_timeseries, *, sensor, platform=None, **kwargs) -> xr.Dataset: ...
def build_r0_from_sources(sources, *, sensor, platform=None, **kwargs) -> xr.Dataset: ...
def run_inversion(scene, r0, *, sensor, platform=None, **kwargs) -> xr.Dataset: ...
def get_execution_profile(name, *, sensor, platform=None): ...

# spires/sensors/registry.py

class SensorAdapter: ...
def get_sensor_adapter(sensor, platform=None) -> SensorAdapter: ...
def list_supported_sensors() -> tuple[str, ...]: ...
def list_supported_sensor_platforms() -> dict[str, tuple[str, ...]]: ...
```

## Current Status Snapshot (2026-04-29)

The sections above describe the intended architecture. The items below summarize what is now implemented, what decisions have been made, and what still needs work.

### Environment / repo notes

- active repo root is `Codex/SpiPy_RLG/SpiPy`
- in sandboxed terminal sessions, prefer `mamba` over `conda`
- when needed, use a writable cache such as:
  - `XDG_CACHE_HOME=/tmp/mamba-cache mamba run -n spipy14 python -m pytest SpiPy/tests/test_viirs_r0.py`

### Implemented so far

- **Unified sensor wrapper / registry**
  - generic sensor dispatch now lives under:
    - `spires/sensors/registry.py`
    - `spires/sensors/api.py`
  - the current generic public entry points are:
    - `open_surface_reflectance(...)`
    - `prepare_scene_for_inversion(...)`
    - `build_timeseries(...)`
    - `build_r0(...)`
    - `build_r0_from_sources(...)`
    - `run_inversion(...)`
    - `get_execution_profile(...)`
  - the registry currently normalizes and dispatches:
    - `sensor='modis'` with platforms `terra`, `aqua`
    - `sensor='viirs'` with platforms `snpp`, `noaa20`, `noaa21`
  - the adapter structure is intended to scale to future sensors such as Sentinel-2, Landsat 8/9, and HLS without expanding one large branchy wrapper
  - `spires/__init__.py` and `spires/sensors/__init__.py` now expose the generic sensor API

- **LUT groundwork**
  - upstream `SPIRES` was cloned and inspected
  - `build_lt.m` was confirmed compatible with SpiPy when LUTs are saved as MATLAB `-v7.3`
  - `SPIRES/RemoteSensing/multispectral.mat` now includes `VIIRS_SNPP`, `VIIRS_NOAA20`, and `VIIRS_NOAA21`
  - `SPIRES/prepInputs/build_viirs_luts.m` now works as a switchable multispectral LUT builder
  - it can build:
    - platform-specific VIIRS LUTs
    - a shared MODIS bands `1-7` LUT
  - LUT files now write `SensorTableBandOrder`

- **VIIRS reader / scene preparation**
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

- **VIIRS QA / masks**
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

- **VIIRS R0 workflow**
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
  - `build_viirs_timeseries(...)` supports:
    - optional `zarr_path`
    - optional `zarr_mode`
    - optional chunking
    - zarr staging now writes with `zarr_format=2` to avoid noisy zarr-v3 warnings during notebook runs
  - normalized-difference screening now suppresses divide-by-zero runtime noise via `np.errstate(...)` while preserving masked results

- **VIIRS inversion workflow**
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

- **VIIRS canopy correction**
  - `run_viirs_inversion(...)` accepts optional `canopy_fraction` and `ice_fraction`
  - current workflow supports temporary GeoTIFF-based canopy input for testing, though tile-level zarr remains the preferred long-term format
  - Ross-style canopy/shade/ice snow adjustment is implemented
  - diagnostic layers now include:
    - `raw_viewable_snow_fraction`
    - `raw_shade_fraction`
    - `raw_canopy_fraction`
    - `raw_snow_fraction`

- **VIIRS ancillary layout**
  - ancillary helpers live in `spires/sensors/viirs/ancillary.py`
  - ancillary storage is tile-oriented under `data/viirs/ancillary/`
  - tile-level ancillary paths are sensor-family level rather than platform-specific
  - use `annual/`, not `yearly/`, for year-specific products

- **VIIRS notebook / examples**
  - `examples/09_viirs_reader_playground.ipynb` is now a compact batch-style example
  - it can build or reuse `R0`, run per-scene inversion, save netCDF outputs, save quicklook figures, and inspect workflow logs

- **MODIS reader / scene preparation**
  - Python MODIS support now lives under `spires/sensors/modis/`
  - implemented APIs include:
    - `parse_modis_surface_reflectance_filename(path)`
    - `open_modis_surface_reflectance(path)`
    - `prepare_modis_scene_for_inversion(source, ...)`
  - supported product families:
    - `MOD09GA` -> `terra`
    - `MYD09GA` -> `aqua`
  - reader behavior:
    - reads Collection `061` MODIS HDF4 granules through `netCDF4`
    - reads `500 m` reflectance bands `1-7`
    - reads `1 km` geometry and state QA fields plus `500 m` QC/support fields
    - parses HDF-EOS sinusoidal grid metadata and assigns projected `x/y` coordinates
    - promotes `1 km` fields onto the `500 m` grid by exact `2x2` nearest-neighbor replication

- **MODIS QA / masks**
  - first-pass MODIS QA decoding uses `state_1km_1` and `QC_500m_1`
  - current decoded QA layers include:
    - cloud state
    - land/water class
    - aerosol quantity
    - cirrus class
    - internal cloud flag
    - cloud shadow flag
    - MOD35 snow/ice flag
    - adjacent-to-cloud flag
    - internal snow flag
    - MODLAND QA
    - per-band quality classes for bands `1-7`
  - prepared scenes expose transparent component masks including:
    - `mask_invalid_reflectance`
    - `mask_bad_geometry`
    - `mask_water`
    - `mask_low_observation_support`
    - `mask_bad_modland_qa`
    - `mask_cloud`
    - `mask_cloud_shadow`
    - `mask_snow`
    - `mask_cloud_for_inversion`
    - `mask_cloud_shadow_for_inversion`
  - both `valid_inversion_mask` and `valid_r0_mask` are produced
  - MODIS now mirrors the richer VIIRS cloud-mask interface:
    - `cloud_mask_source`
    - `cloud_mask_var`
    - `cloud_shadow_mask_var`
    - `cloud_mask_policy`
  - external cloud / shadow masks are loadable through `spires/sensors/modis/qa.py`

- **MODIS R0 workflow**
  - implemented APIs include:
    - `compute_modis_r0_indices(...)`
    - `build_modis_r0_candidate_metrics(...)`
    - `build_modis_timeseries(...)`
    - `build_modis_r0(...)`
    - `build_modis_r0_from_sources(...)`
  - current MODIS snow-free selection logic uses:
    - `NDVI = (B2 - B1) / (B2 + B1)`
    - `NDSI = (B4 - B6) / (B4 + B6)`
    - `blue_metric = B3`
  - current rule is:
    - if any valid date has `NDSI < 0`, choose the valid date with maximum `NDVI`
    - otherwise choose the valid date with minimum `B3`
  - `build_modis_timeseries(...)` now mirrors the VIIRS timeseries API more closely and supports:
    - optional `zarr_path`
    - optional `zarr_mode`
    - optional chunking
    - per-scene logging similar to the VIIRS batch path
  - zarr staging now strips CF serialization attrs before writing and uses `zarr_format=2`
  - final `R0` netCDF writing now sanitizes attrs before serialization, fixing MODIS equivalents of earlier VIIRS attr-write failures
  - normalized-difference screening now suppresses divide-by-zero runtime noise via `np.errstate(...)` while preserving masked results

- **MODIS inversion workflow**
  - implemented APIs include:
    - `ModisExecutionProfile`
    - `get_modis_execution_profile(name)`
    - `run_modis_inversion(scene, r0, ..., lut_file=..., ...)`
  - wrapper behavior:
    - accepts a prepared scene or raw HDF input
    - accepts in-memory or on-disk `R0`
    - validates scene/R0/LUT band order
    - uses the shared MODIS bands `1-7` LUT for both Terra and Aqua
    - runs through `speedy_invert_dask(...)`
    - supports `apply_valid_inversion_mask=False` identically to the VIIRS wrapper
    - supports optional canopy and ice-fraction adjustment with the same Ross-style postprocessing used for VIIRS
  - generic wrapper usage now works through:
    - `run_inversion(..., sensor='modis', platform='terra' | 'aqua', ...)`

- **MODIS ancillary layout / notebook**
  - local MODIS data layout now mirrors VIIRS under:
    - `data/modis/ancillary/`
    - `data/modis/inputs/terra/`
    - `data/modis/inputs/aqua/`
    - `data/modis/lut/`
    - `data/modis/r0/`
  - `examples/10_modis_reader_playground.ipynb` now exists as a compact batch-style MODIS example
  - it can build or reuse `R0`, run per-scene inversion, save netCDF outputs, save 1x4 quicklook figures, inspect workflow logs, and show notebook-style progress bars for `R0` generation and inversion

- **Unified notebook / examples**
  - `examples/11_unified_multisensor_playground.ipynb` now exists as a generic batch notebook built on the sensor registry / API layer
  - it currently targets all five locally configured sensor/platform combinations:
    - MODIS Terra
    - MODIS Aqua
    - VIIRS SNPP
    - VIIRS NOAA-20
    - VIIRS NOAA-21
  - for each platform it can:
    - build or reuse `R0`
    - stage summer timeseries through zarr when rebuilding `R0`
    - run inversion on the available reflectance scenes
    - save per-scene netCDF outputs
    - save and display 1x4 quicklook figures

### Current defaults and decisions

- the preferred high-level API direction is now:
  - generic wrappers in `spires.sensors.api`
  - sensor-specific details hidden behind registered adapters
- when MODIS / VIIRS behavior diverges, prefer aligning MODIS upward toward the richer VIIRS-style contract where practical
- `sensor` and `platform` normalization now happen through the registry layer rather than being left to notebooks or ad hoc string handling

- canonical VIIRS analysis grid: `500 m`
- `1 km` to `500 m` alignment: nearest-neighbor replication only
- raw-reader and prepared-scene layers remain separate
- SNPP, NOAA-20, and NOAA-21 use separate LUTs
- default prepared-scene band order:
  - `I1`, `I2`, `I3`, `M1`, `M2`, `M3`, `M4`, `M5`, `M7`, `M8`, `M10`, `M11`
- snow is excluded from `valid_r0_mask` but not from inversion by default
- inversion cloud masking is policy-driven so snow-preserving experiments can relax cloud QA without changing `R0` filtering
- the current Python masking approach intentionally exposes component masks, even though Ross MATLAB uses a more bundled daily-filter style
- MODIS analysis grid is also `500 m`
- current MODIS inversion band order is `1`, `2`, `3`, `4`, `5`, `6`, `7`
- current MODIS LUT policy is one shared bands `1-7` LUT for Terra and Aqua
- MODIS uses HDF4 granules but they are readable through `netCDF4` in the current Python environment, so no separate `pyhdf` dependency is currently required
- notebook-local MODIS convention is `r0_input_glob = '**/*.hdf'`
- current local MODIS LUT layout now mirrors VIIRS under `data/modis/lut/`
- for sandboxed `mamba` / notebook runs, `XDG_CACHE_HOME=/tmp/mamba-cache` remains the preferred cache pattern

### Testing status

- generic sensor-wrapper smoke tests are passing in `spipy14`, including:
  - importing `spires`
  - importing the generic `spires.sensors` API
  - registry lookup / alias normalization for MODIS and VIIRS
  - execution-profile lookup through the generic wrapper
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
- targeted MODIS tests are now passing, including:
  - `tests/test_modis_ancillary.py`
  - `tests/test_modis_bands.py`
  - `tests/test_modis_hdf.py`
  - `tests/test_modis_qa.py`
  - `tests/test_modis_r0.py`
  - `tests/test_modis_workflow.py`
- recent regression coverage added during unified-wrapper work includes:
  - MODIS external cloud-mask policy handling
  - MODIS zarr timeseries writing
  - MODIS `R0` attr sanitization before netCDF writes
  - VIIRS / MODIS zarr staging under `zarr_format=2`
  - VIIRS / MODIS normalized-difference warning suppression for dask-backed `R0` workflows

### Still open

- the unified wrapper currently dispatches MODIS and VIIRS only; Sentinel-2, Landsat 8/9, and HLS adapters are still future work
- the generic API exists, but not all sensor-specific kwargs are yet documented centrally in one place
- the unified notebook still needs a full clean end-to-end validation pass across all five real local platform runs after the recent serialization / zarr / warning fixes

- MODIS unified-notebook diagnostic checkpoint (2026-04-29):
  - notebook `examples/11_unified_multisensor_playground.ipynb` was rerun for the local Terra/Aqua scenes with:
    - `cloud_mask_policy='ignore_cloud_and_shadow'`
    - `apply_valid_inversion_mask=False`
    - `use_grouping=False`
  - this means the observed Terra/Aqua issues are not the NOAA-20 grouped-inversion failure mode noted below
  - Aqua striped missing-data artifact:
    - traced primarily to the input scene / prepared reflectance path, not the optimizer
    - for `MYD09GA.A2023075.h08v05.061.2023077025646.hdf`, raw Aqua band 6 contains heavy fill:
      - `sur_refl_b06_1` fill count: `3,423,306 / 5,760,000`
      - prepared-scene band 6 finite fraction: about `0.406`
      - prepared-scene `mask_invalid_reflectance` fraction: about `0.598`
    - because the current MODIS inversion path requires all seven bands, those non-finite band-6 values propagate directly into large striped invalid regions in the Aqua inversion outputs
    - current interpretation: this is more likely a product/data-availability issue for these Aqua scenes than a generic reader bug
  - Terra compressed / flattened output:
    - Terra prepared reflectance itself looks mostly healthy:
      - prepared-scene `mask_invalid_reflectance` fraction: about `0.001`
      - prepared-scene `valid_inversion_mask` fraction: about `0.750`
    - however, the inversion output is suspiciously concentrated near the initial `fsca` guess:
      - for valid non-water land pixels on `2023-03-16`, about `74%` of `fsca` values are essentially `0.5`
    - this points more toward an inversion-side issue, or an `R0` / LUT mismatch, than toward a raw Terra reader problem
    - note that `apply_valid_inversion_mask=False` also allows invalid / water pixels to remain in the plotted outputs, which makes the Terra quicklooks look even flatter over water, but that does not explain the very large fraction of valid land pixels stuck near `0.5`
  - code-level findings from this diagnostic pass:
    - added a reusable MODIS scene diagnostic helper:
      - `scripts/diagnose_modis_scene.py`
    - fixed a real C++ bug in the inversion core:
      - `spires/spires.cpp`
      - `spectrum_has_nan(...)` had an off-by-one loop bound (`<= len_target`) and was reading one element past the end of the spectrum buffer
    - the optimizer path still returns the final parameter vector without logging or exposing the `nlopt` result code, so Terra failures can currently look like ordinary outputs rather than explicit optimization failures
  - suggested immediate next steps:
    - add temporary Terra-focused instrumentation around the MODIS inversion path to capture:
      - optimizer status / exception counts
      - objective value improvement relative to `x0`
      - fraction of pixels returning exactly or nearly the initial parameter vector
    - run a small Terra comparison with:
      - current shared MODIS LUT
      - a few representative valid land pixels
      - saved `R0` spectrum, target spectrum, solar zenith, and final solution
    - verify whether the Terra problem is caused by:
      - optimizer non-convergence silently returning `x0`
      - an `R0` mismatch
      - a LUT spectral-order / sensor-response mismatch
    - for Aqua, decide whether the near-term policy should be:
      - reject scenes with excessive band-6 fill
      - explicitly report a scene-quality warning before inversion
      - or test a reduced-band Aqua path only if a compatible LUT is available
    - rerun the MODIS notebook quicklooks once with `apply_valid_inversion_mask=True` to separate true retrieval behavior from invalid/water plotting artifacts

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
- add a compact MODIS end-to-end real-data validation pass in the notebook and confirm Terra/Aqua outputs against expected behavior
- if Terra/Aqua retrieval differences show up later, revisit whether one shared MODIS LUT remains sufficient or whether separate Terra/Aqua LUTs are warranted
