# VIIRS / MODIS SPIReS Notes

## Purpose

Keep a short current-state note for the shared VIIRS / MODIS SPIReS workflow.

This file should describe:

- the active sensor-agnostic architecture
- current data model and workflow expectations
- known issues that are still open
- the next concrete engineering steps

Historical implementation details from the large April 2026 refactor now live in:

- `notes_rtp_archive_2026-04.md`

## Environment Notes

- in sandboxed terminal sessions, prefer `mamba` over `conda`
- active environment: `spipy14`
- use a writable cache pattern for sandboxed `mamba` or notebook runs:
  - `XDG_CACHE_HOME=/tmp/mamba-cache`
- example command:
  - `XDG_CACHE_HOME=/tmp/mamba-cache mamba run -n spipy14 python -m pytest SpiPy/tests/test_viirs_r0.py`

## Current Architecture

### Shared sensor-agnostic layer

- `spires/sensors/registry.py`
  - sensor / platform normalization and adapter lookup
- `spires/sensors/api.py`
  - generic dispatch for `open_surface_reflectance(...)`
  - `prepare_scene_for_inversion(...)`
  - `build_timeseries(...)`
  - `build_r0(...)`
  - `build_r0_from_sources(...)`
  - `run_inversion(...)`
  - `get_execution_profile(...)`
- `spires/sensors/full_workflow.py`
  - shared inversion workflow core after prepared `R` and `R0` are available
- `spires/sensors/r0_core.py`
  - shared `R0` source-stack, optional Zarr-backed staging, and compositing machinery
- `spires/sensors/io.py`
  - generic safe dataset load / validate / atomic write helpers

### Sensor-specific layer

- `spires/sensors/viirs/`
  - reader, QA, bands, geospatial, ancillary, thin workflow / shared-core `R0` wrappers
- `spires/sensors/modis/`
  - reader, QA, bands, geospatial, ancillary, thin workflow / shared-core `R0` wrappers

### Current design rule

Keep ingestion and QA sensor-specific.
Once a prepared scene has canonical `reflectance`, geometry, masks, and `R0`, the inversion path should stay shared whenever possible.

## Canonical Workflow

1. Read raw sensor files and build a prepared scene on the analysis grid.
2. Build or reuse a validated `R0` background dataset.
   Use the unified `spires.sensors.build_r0_from_sources(...)` API for both MODIS and VIIRS; pass `zarr_path` and time-major chunks for memory-safe local builds.
3. Run the shared inversion workflow with sensor-specific wrappers only where band / LUT logic differs.
4. Write outputs through the shared safe I/O layer.

## Current Output Schema

### Prepared-scene core

- `reflectance(y, x, band)`
- `solar_zenith(y, x)`
- `sensor_zenith(y, x)`
- component masks such as:
  - `mask_invalid_reflectance`
  - `mask_bad_geometry`
  - `mask_water`
  - `mask_low_observation_support`
- `valid_r0_mask`
- `valid_inversion_mask`

### R0 output

- `r0_reflectance(y, x, band)`
- `r0_count(y, x)`
- diagnostic layers such as `r0_source_index`, `r0_source_time`, and rule-selection fields

### Inversion output

- `raw_viewable_snow_fraction`
- `raw_shade_fraction`
- `raw_canopy_fraction`
- `raw_snow_fraction`
- `dust_concentration`
- `grain_size`
- `valid_inversion_mask`

Note:

- legacy `fsca` and `fshade` aliases are intentionally no longer written

## Output-Safety Policy

Current persisted output safeguards:

- `R0` NetCDFs write atomically through a temp file and replace the final path only after validation
- inversion NetCDFs use the same safe-write pattern
- existing outputs are not reused just because the file exists
- reusable outputs must pass lightweight validation
- new files are marked with `build_status="complete"`

This was added after a Terra `R0` file was found to be corrupted after an interrupted notebook run.

## Known Issues Worth Tracking

### 1. MODIS Aqua band-6 data gaps

Some Aqua scenes contain heavy band-6 fill, which propagates into large invalid striped regions after scene preparation and then into inversion outputs because the current MODIS inversion path requires all seven bands.

This is a scene-quality / product-availability issue, not currently a reader bug.

### 2. Grouped inversion still needs targeted debugging

Grouped inversion can still produce coherent missing patches in cases where ungrouped inversion returns values.

Current working hypothesis:

- a representative pixel can fail inversion and that failed solution then gets broadcast to the whole group

### 3. Unified real-data validation pass still needed

The shared wrappers and safe I/O refactor are in place, but the unified notebook still needs a fresh full end-to-end validation pass across the local MODIS and VIIRS platform runs using the current architecture.

## Resolved / No Longer Active

### MODIS Terra flattened retrieval issue

The earlier Terra `fsca ~= 0.5` problem was traced to a corrupted persisted Terra `R0` file rather than a confirmed Terra-specific inversion or LUT problem.

Regenerating the `R0` file fixed the issue.

### Redundant inversion aliases

The older `fsca` / `fshade` alias duplication has been removed from written outputs.

## Near-Term Next Steps

1. Run a clean end-to-end validation pass in `examples/11_unified_multisensor_playground.ipynb` using the shared-core workflow and safe output writing.
2. Decide how Aqua scenes with excessive MODIS band-6 fill should be handled:
   reject, warn, or support only through a future reduced-band path with a compatible LUT.
3. Debug grouped inversion failure modes with representative-vs-member comparisons and fallback behavior for failed groups.
4. Document the generic sensor API kwargs in one central place.
5. Replace the temporary canopy-input path with a durable tile-level ancillary product.

## Longer-Term Opportunities

- future sensor adapters for Sentinel-2, Landsat 8/9, and HLS
- earlier `HDF -> zarr` staging for large batch runs
- lighter reader modes when downstream steps do not need full QA/support fields
- optional ice-fraction ancillary support
- revisit separate Terra / Aqua LUTs only if future evidence suggests one shared MODIS LUT is insufficient
