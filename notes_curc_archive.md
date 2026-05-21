# CURC / VSCode Server Notes

## Purpose

Keep a short current-state note for working in this repository on the CURC VSCode server.

## Confirmed Environment Access

- this server session needs `module load miniforge` before `mamba` is available
- confirmed envs:
  - `base` at `/curc/sw/install/miniforge3/24.11.3-0`
  - `gis2` at `/projects/ropa5718/software/anaconda/envs/gis2`
  - `spipy14` at `/projects/ropa5718/software/anaconda/envs/spipy14`
- confirmed interpreter:
  - `module load miniforge && mamba run -n spipy14 python -V`
  - returned `Python 3.14.4`

## Current Status

- `spipy14` is installed and runnable on this server
- the local checkout now imports cleanly after building the SWIG extension in-place
- current checkout observed during repo reconciliation: `master`
- verified command:
  - `module load miniforge && mamba run -n spipy14 python -c "import spires; print('imports_ok')"`
  - returned `imports_ok`
- verified VIIRS imports:
  - `module load miniforge && mamba run -n spipy14 python -c "import spires; import spires.sensors.api; import spires.sensors.viirs.workflow; print('viirs_imports_ok')"`
  - returned `viirs_imports_ok`
- verified VIIRS pytest subset:
  - `module load miniforge && mamba run -n spipy14 python -m pytest tests/test_viirs_qa.py tests/test_viirs_workflow.py tests/test_viirs_ancillary.py tests/test_viirs_r0.py tests/test_viirs_hdf.py`
  - result: `38 passed, 6 skipped`

## Local Build Step

From the repository `README.md`:

```bash
module load miniforge
mamba run -n spipy14 python setup.py build_ext --inplace
```

After that, re-check:

```bash
module load miniforge
mamba run -n spipy14 python -c "import spires; print('imports_ok')"
```

## Server-Specific Build Notes

- `spipy14` already contains `swig`, `nlopt`, and the needed compiler toolchain
- the initial build failed because `setup.py` mixed Conda headers with hardcoded system include paths like `/usr/include`
- on this server, the working fix was to make `setup.py` prefer `CONDA_PREFIX/include` and `CONDA_PREFIX/lib` during Conda builds, instead of injecting system include paths
- after the extension built, `import spires` still needed `spires.core` loaded earlier in package initialization; importing `spires.core` first in `spires/__init__.py` resolved that

## Dependency Notes

The `README.md` says source builds require conda-forge packages including:

- `swig`
- `gxx`
- `gcc`
- `nlopt`

If the build fails on this server, verify those are installed in `spipy14`.

## Working Command Pattern

For interactive use:

```bash
module load miniforge
mamba activate spipy14
```

For one-off commands:

```bash
module load miniforge
mamba run -n spipy14 <command>
```

Python rule for this server:

- never use `python3` directly in shell commands for this repo
- always run Python via the project environment:
  - `module load miniforge && mamba run -n spipy14 python <...>`

## Git / GitHub Notes

- this server now authenticates to GitHub over SSH using `~/.ssh/curc`
- repo `origin` was switched from HTTPS to SSH:

```bash
git remote set-url origin git@github.com:RittgerLabGroup/SpiPy.git
```

- verify auth with:

```bash
ssh -T git@github.com
```

- current validated push path from this server:

```bash
git push
```

## CURC Workflow Progress

- scratch directory conventions are now established:
  - `/scratch/alpine/ropa5718/spipy/output/<sensor>/<platform>/<tile>/<date>/`
  - `/scratch/alpine/ropa5718/spipy/input/<sensor>/<platform>/reflectance`
  - `/scratch/alpine/ropa5718/spipy/input/<sensor>/<platform>/ancillary`
  - `/scratch/alpine/ropa5718/spipy/input/<sensor>/<platform>/ancillary/r0`
  - `/scratch/alpine/ropa5718/spipy/logs`
- CURC-specific orchestration code now lives outside core `spires/` in:
  - `workflows/curc/`
  - plus user-facing `configs/curc/` and `scripts/`
- the exploratory notebook now uses the CURC workflow modules rather than duplicating logic:
  - `examples/12_curc_sensor_workflow_planning.ipynb`
  - includes a visible user-editable config cell for paths, sensor/platforms, tiles, water years, dates, and dry-run mode
- current concrete example is VIIRS SNPP using:
  - `/pl/active/rittger_ops/vnp09ga.002/input/<tile>/<year>/VNP09GA*.h5`
- water year is defined as:
  - `October 1` of the previous calendar year through `September 30` of the named year
  - example: water year `2024` means `2023-10-01` through `2024-09-30`
- current planning layer can:
  - discover full water-year SNPP reflectance files
  - plan `stage_reflectance`
  - plan `stage_ancillary`
  - plan `build_r0`
  - plan `run_inversion`
- for the real SNPP example, tile `h09v05`, water year `2024` currently discovers `348` reflectance files
- current direct-execution helpers are now in place for:
  - `stage_reflectance` via rendered or executable `rsync` commands
  - `stage_ancillary` via direct directory/layout creation
  - `build_r0` via a Python entrypoint that writes `r0_reflectance.nc`
  - `run_inversion` remains on the manifest-backed Slurm array path
- CURC `build_r0` output naming was updated after initial notebook testing:
  - future outputs now write as `{platform}_r0_{tile}_{year}.nc`
  - example: `snpp_r0_h08v05_2022.nc`
- future persisted `R0` files now omit scene-specific attrs that are not appropriate for a composite product:
  - `acquisition_date`
  - `processing_timestamp`
  - `lut_file`
- notebook testing confirmed that a small direct `stage_reflectance` execution completed successfully and that a direct `build_r0` test run completed for the prior-summer source set

## Immediate Next Steps

- the notebook helper cells for executable step testing are now in active use in:
  - `examples/12_curc_sensor_workflow_planning.ipynb`
- completed notebook validation milestones so far:
  - previewed executable `stage_reflectance`, `stage_ancillary`, and `build_r0`
  - executed a small single-date `stage_reflectance` copy successfully
  - executed `build_r0` for the prior-summer source window successfully
- the next active task is to create an executable path for the actual `run_inversion` step now that `R0` has been built and validated in the notebook workflow
- once executable `run_inversion` passes for the small targeted case, the next stage is to run a full water year
- recommended rationale for that order:
  - keep the first executable inversion test narrow enough to debug runtime path, LUT, ancillary, and output issues quickly
  - only scale to a full water year after the single-case executable inversion path behaves as expected

## Slurm Array / Retry Status

- inversion is currently modeled as:
  - one logical task per acquisition date
  - submitted as one Slurm array rather than many separate `sbatch` calls
- manifest-backed array support is in place:
  - one JSON manifest file stores the per-date tasks
  - array tasks resolve themselves from `SLURM_ARRAY_TASK_ID`
- runtime entrypoint is in place:
  - `scripts/run_curc_inversion_array_task.py`
  - resolves staged reflectance paths, ancillary root, R0 path, output path, and log path
  - dry-run mode currently reports missing staged inputs/R0 cleanly
- runtime R0 lookup now expects the new CURC naming convention:
  - `{platform}_r0_{tile}_{year}.nc`
- current per-task output convention for inversion runtime is:
  - `.../output/<sensor>/<platform>/<tile>/<date>/inversion.nc`
- CURC-side structured logging is now wired into the existing `spires` logging flow
  - runtime task events include Slurm metadata when available
  - submission events use a visually obvious `====== SUBMISSION ======` marker
- initial submissions and auto-retry submissions are explicitly distinguished with:
  - `submission_kind="initial"`
  - `submission_kind="auto_retry"`

## Failure Classification / Auto-Retry Status

- post-run scan tooling is now in place to inspect:
  - manifest tasks
  - per-date outputs
  - per-date logs
- failure classification currently distinguishes at least:
  - `missing_staged_reflectance`
  - `missing_r0`
  - `missing_lut`
  - `missing_runtime_input`
  - `invalid_runtime_input`
  - `filesystem_error`
  - `python_exception`
  - `slurm_or_external_failure`
- important behavioral distinction:
  - `failed_count` means a task did not complete successfully
  - `retryable_count` means a blind retry might make sense
  - example: missing staged reflectance is a failure, but not a retryable one until prerequisites are fixed
- automatic retry control is now configurable through `max_auto_retry_count`
  - default is `3`
  - user can set a different value in config, including `0`
  - `0` disables auto-retry
- retry stopping rule now implemented:
  - auto-submission stops when every failed task is either:
    - `retry_recommended=False`
    - or `retry_recommended=True` with `retry_count >= max_auto_retry_count`
- helper scripts now exist for:
  - initial inversion-array submission
  - manifest scanning
  - auto-retry manifest generation
  - optional auto-retry submission

## Slurm Profile Implementation Status

- user-configurable Slurm resource settings are now implemented in `CurcWorkflowConfig` via a structured `SlurmProfile`
- inversion-array manifests now persist the Slurm profile so retry submissions reuse the same resource settings automatically
- generated `sbatch` commands now render profile-driven flags directly rather than relying only on ad hoc CLI `--sbatch-arg` values
- CLI `--sbatch-arg` values still append after the configured profile so they can be used as explicit one-off overrides
- old logs under `/projects/ropa5718/slurm_out/` were reviewed for baseline settings from the previous codebase
- the current recommended initial inversion-job Slurm profile is:
  - `account = blanca-rittger`
  - `qos = preemptable`
  - `cpus_per_task = 1`
  - `mem = 44G`
  - `time = 15:00:00`
  - `output_dir = /projects/ropa5718/slurm_out`
- `ntasks_per_node` was used in the old codebase, but should not be copied blindly into the new array-task model without deciding whether it is still semantically necessary
- later workflow gap to close:
  - bring in albedo functionality from `ParBal` and `LookupFunctionsSPIReS`
  - current CURC-side VIIRS runtime/output path does not yet include that functionality

## Session Update (2026-05-19)

- CURC logging behavior was overhauled to improve traceability and reduce collisions:
  - runtime task logs now support per-job naming (`run_inversion_<date>_job<SLURM_JOB_ID>.log`) so separate submissions do not append into a single date log
  - near-real-time water-year aggregate log writing is now wired (`run_inversion_wy<year>_aggregate.log`)
  - hierarchical visual indentation was added via context-aware formatting to make parent/child event flow easier to scan
  - submission and auto-retry orchestration logs now include timestamps in filenames to avoid reuse collisions
  - scanner logic now prefers newest job-specific logs and falls back to legacy per-date logs
- log placement was simplified for new runs:
  - CURC job logs now resolve to the top-level `/scratch/alpine/ropa5718/spipy/logs/` directory instead of nested sensor/platform/tile/year folders
- runtime/output behavior updates from this session also include:
  - inversion output naming now uses `{platform}_raw_output_{tile}_{YYYYMMDD}.nc`
  - default inversion masking behavior remains output-only (`valid_inversion_mask` persisted, science layers unmasked)
- notebook-driven Slurm execution is now in place for targeted inversion reruns:
  - the notebook can stage reflectance directly for a selected date subset
  - the notebook can then submit and scan inversion-array jobs using the same CURC scripts used outside the notebook
- five-day targeted validation for tile `h08v05` (`2023-03-16` through `2023-03-20`) was exercised:
  - the first submission failed for four dates because only `2023-03-16` had been staged to scratch
  - after staging the missing reflectance inputs, the five-date run executed successfully under the notebook-driven flow
- log/output organization was revised again during this session:
  - per-run logs now land under timestamped directories directly beneath `/scratch/alpine/ropa5718/spipy/logs/`
  - inversion outputs now land under `/scratch/alpine/ropa5718/spipy/output/<sensor>/<platform>/<tile>/raw/wy<water_year>/`
- aggregate/per-job log formatting was refined toward the edited target example:
  - `curc_submit_inversion_array` now logs only the actual submission event
  - the runtime context is rendered as a dedicated `SUBMISSION PARAMETERS` block
  - scope separators now wrap runtime start/summary pairs and indentation depth is reduced by one level
- full water year notebook execution for VIIRS SNPP tile `h08v05`, water year `2023`, was completed successfully:
  - the latest full-run manifest under `/scratch/alpine/ropa5718/spipy/logs/20260519_143044/` carried `364` logical dates
  - the output directory now contains `364` validated NetCDFs under `/scratch/alpine/ropa5718/spipy/output/viirs/snpp/h08v05/raw/wy2023/`
  - no `auto_retry` submission artifacts or `retry_count > 0` runtime summaries were found for that full run
  - five dates (`2023-03-16` through `2023-03-20`) were reported as `loaded_existing` because they had already been produced during the earlier targeted validation run
- inversion outputs now persist the prepared reflectance cube alongside the science layers:
  - final NetCDFs now include `reflectance(y, x, band)` in addition to the inversion products and `valid_inversion_mask`
  - this was added in the shared sensor inversion path, so both VIIRS and MODIS outputs inherit it
- aggregate reporting was redesigned away from the old human-readability-heavy aggregate log:
  - scans now write `run_inversion_wy<year>_task_attempts.csv` with one row per attempt
  - scans now write `run_inversion_wy<year>_summary.txt` with one row per scene date in water-year order
  - the detailed CSV uses `last_attempt_for_date` to flag the winning/latest attempt for each date
  - the human-readable summary includes the `TOTALS` block immediately after the manifest header and before the per-date table
- timestamped log directory layout was revised again to separate machine summaries from detailed runtime artifacts:
  - each timestamped run directory now keeps only the top-level summary artifacts (`task_attempts.csv` and `summary.txt`)
  - manifests, per-task logs, submission logs, auto-retry logs, Slurm `.out` files, and the aggregate `.log` now live under a `detailed_logs/` subdirectory inside the timestamped run directory

### Next Step

- re-run the notebook scan/reporting cells against the new summary-artifact layout:
  - confirm `run_inversion_wy2023_task_attempts.csv` and `run_inversion_wy2023_summary.txt` are easy to use from the notebook
  - verify the top-level timestamped directory stays uncluttered while `detailed_logs/` retains the full forensic trail
  - decide whether the legacy aggregate `.log` should be retained, reduced further, or removed from the default workflow once the CSV/TXT summaries are proven sufficient

## Session Update (2026-05-20)

- the CURC sensor workflow notebook was retargeted from the old single-tile `WY2023` example to a first-run `WY2024` `VNP09GA` planning/submission flow:
  - notebook path: `examples/12_curc_sensor_workflow_planning.ipynb`
  - current default tile set is:
    - `h08v04`
    - `h08v05`
    - `h09v04`
    - `h09v05`
    - `h10v04`
  - current default source root remains `/pl/active/rittger_ops/vnp09ga.002`
- the notebook now orchestrates the existing single-tile CURC helpers sequentially across that tile set:
  - per-tile discovery
  - per-tile workflow-step planning
  - per-tile manifest creation/submission/scanning
- notebook output volume was reduced after the first multi-tile pass proved too heavy for interactive use:
  - preview and execution helper cells now print compact summaries instead of full `dates` / `source_paths` payloads
  - array planning, submission, and scan cells now also print compact summaries
- the first multi-tile notebook `build_r0` attempt crashed the kernel while processing tile `h08v04`
  - scratch inspection showed `/scratch/alpine/ropa5718/spipy/input/viirs/snpp/ancillary/r0/h08v04/2023/` existed but was empty
  - no `snpp_r0_h08v04_2023.nc` artifact was found anywhere under `/scratch/alpine/ropa5718/spipy`
  - this strongly suggested the crash happened before the final atomic NetCDF rename completed, not after a successful `R0` write
- as an immediate notebook-side safeguard, `build_r0` was restricted to one tile at a time:
  - notebook flag: `R0_BUILD_TILE`
  - `stage_reflectance` and `stage_ancillary` can still iterate across all configured tiles
- more importantly, the lower-memory `R0` path that already existed in the sensor layer was now exposed through the CURC helper layer:
  - `workflows/curc/execution.py` now accepts optional `zarr_path` and `chunks` for the `build_r0` step preview/execute path
  - `workflows/curc/runner.py` now exposes those same options through:
    - `preview_viirs_snpp_step_execution(...)`
    - `run_viirs_snpp_step(...)`
  - the underlying sensor-layer behavior already supported incremental `Zarr` staging and was not invented in this session; this session connected that capability to the CURC workflow entrypoints
- the notebook now uses that new CURC `Zarr` passthrough for `build_r0`:
  - preview flag/control:
    - `R0_USE_ZARR = True`
    - `R0_ZARR_CHUNKS = {"time": 1, "y": 256, "x": 256, "band": -1}`
  - execution flag/control:
    - `R0_BUILD_USE_ZARR = True`
    - `R0_BUILD_CHUNKS = {"time": 1, "y": 256, "x": 256, "band": -1}`
  - per-tile temporary stacks are currently directed under:
    - `/scratch/alpine/ropa5718/spipy/tmp/r0_zarr/`
- focused CURC execution coverage was extended to lock in this new behavior:
  - `tests/test_curc_execution.py` now checks that `build_r0` preview payloads include `zarr_path` / `chunks`
  - the same test module now checks that `execute_viirs_snpp_workflow_step(...)` passes `zarr_path` / `chunks` through to `build_r0_from_sources(...)`
- validated command from this session:
  - `module load miniforge && mamba run -n spipy14 python -m pytest -q tests/test_curc_execution.py`
  - result: `5 passed`

### Immediate Next Step

- rerun the notebook `build_r0` cell for a single tile using the new CURC `Zarr` path before broadening the tile set
- if that succeeds cleanly on real `WY2024` data, revisit whether the notebook still needs a single-tile `R0_BUILD_TILE` guard or can be relaxed to a small tile subset

## Session Update (2026-05-20, later)

- the shared `R0` selector was revised to keep the existing negative-`NDSI` / max-`NDVI` / min-blue structure while adding a geometry-aware tie-break within the negative-`NDSI` branch:
  - for each pixel, if any candidate dates satisfy `NDSI < 0`, the candidate pool is still defined by that rule
  - `NDVI` remains the primary ranking metric
  - candidates within `ndvi_tie_epsilon` of the per-pixel max `NDVI` are now treated as near-ties
  - near-ties are broken by preferring more nadir-like viewing geometry, with `sensor_zenith` as the primary key and directional components derived from `sensor_zenith` / `sensor_azimuth` as secondary keys
  - default `ndvi_tie_epsilon` is currently `0.02`
- the `R0` outputs now retain the selected per-pixel sensor geometry needed for later inversion-side filtering or diagnostics:
  - `r0_sensor_zenith`
  - `r0_sensor_azimuth`
  - no scalar `sensor_view_angle` layer is persisted
- CURC top-level `build_r0` execution now exposes the tie-break control:
  - `workflows/curc/execution.py` preview/execute helpers accept `ndvi_tie_epsilon`
  - `workflows/curc/runner.py` exposes that same option through:
    - `preview_viirs_snpp_step_execution(...)`
    - `run_viirs_snpp_step(...)`
  - `scripts/run_curc_workflow_step.py` now accepts:
    - `--ndvi-tie-epsilon <float>`
- the CURC notebook was updated to surface the new `R0` tie-break control and to relax the prior single-tile `build_r0` restriction:
  - preview cell control:
    - `R0_NDVI_TIE_EPSILON = 0.02`
  - execution cell controls:
    - `R0_NDVI_TIE_EPSILON = 0.02`
    - `R0_BUILD_TILES = tuple(tiles)`
  - `build_r0` execution is no longer hard-coded to a single `R0_BUILD_TILE`; it can now run for any configured subset of tiles
- focused validation from this session:
  - `module load miniforge && mamba run -n spipy14 python -m pytest -q tests/test_viirs_r0.py`
    - result: `15 passed`
  - `module load miniforge && mamba run -n spipy14 python -m pytest -q tests/test_curc_execution.py tests/test_viirs_r0.py`
    - result: `22 passed`

### Immediate Next Step

- use the notebook `R0_BUILD_TILES` control to start with a small tile subset while keeping the new `R0_NDVI_TIE_EPSILON = 0.02` default
- inspect the resulting `snpp_r0_<tile>_<year>.nc` files to confirm:
  - `r0_sensor_zenith` and `r0_sensor_azimuth` are present
  - candidate coverage and output quality remain acceptable under the new tie-break rule
- if that looks stable, broaden `R0_BUILD_TILES` to the full notebook tile set for the next full water-year pass

## Session Update (repo reconciliation after interrupted notebook work)

- repo reconciliation found one uncommitted file:
  - `examples/12_curc_sensor_workflow_planning.ipynb`
- no broader dirty tracked source tree was present during this check; the main undocumented state was notebook-side
- the notebook currently reflects a broader full-water-year execution posture than the prior "Immediate Next Step" language above:
  - title/scope is now a first-run `WY2024` `VNP09GA` flow across five tiles
  - default tile set remains:
    - `h08v04`
    - `h08v05`
    - `h09v04`
    - `h09v05`
    - `h10v04`
  - default water year remains `2024`
  - default source root remains `/pl/active/rittger_ops/vnp09ga.002`
- the notebook currently wires the full per-tile sequence in one place:
  - per-tile discovery
  - per-tile workflow-step planning
  - compact preview of `stage_reflectance`, `stage_ancillary`, and `build_r0`
  - executable `build_r0` with `R0_BUILD_TILES = tuple(tiles)`
  - one manifest-backed inversion-array submission cell per tile
  - one scanner cell per tile
  - log inspection cells for the resulting `run_inversion*` artifacts
- the notebook still uses the lower-memory CURC `Zarr` passthrough for `build_r0`:
  - `R0_BUILD_USE_ZARR = True`
  - `R0_BUILD_CHUNKS = {"time": 1, "y": 256, "x": 256, "band": -1}`
  - temporary stacks are directed under `/scratch/alpine/ropa5718/spipy/tmp/r0_zarr/`
- the current notebook config also surfaces inversion-runtime controls that matter for full-water-year submission behavior:
  - `apply_valid_inversion_mask = False`
  - `grouping_method = "chunk_bin_mean"`
  - Slurm profile currently renders with `mem = 8G`
  - the notebook also carries a large `--exclude=...` node list through `SlurmProfile.extra_args`
- the notebook’s planned inversion destinations currently align with the newer raw-output layout:
  - `/scratch/alpine/ropa5718/spipy/output/<sensor>/<platform>/<tile>/raw/wy<water_year>/`
- important documentation distinction after reconciliation:
  - the notebook is submission-ready for the full five-tile `WY2024` path
  - this note does not claim that the full five-tile `build_r0` or full five-tile inversion submission path was successfully executed in the interrupted session
  - it only records that the notebook now contains those cells/configurations and is ahead of the previous smaller-scope next-step note

### Updated Immediate Next Step

- decide whether to preserve the notebook in its current full-five-tile submission-ready state or narrow it back to a smaller validation subset before the next real run
- if keeping the broader notebook state, close the loop by recording actual execution outcomes for:
  - full-tile `build_r0`
  - per-tile inversion-array submission
  - per-tile scan summaries and any retry manifests

## Session Update (post-WY2024 R0 completion and logging refactor)

- notebook-driven `build_r0` for the full five-tile `WY2024` `VNP09GA` set now completed successfully after the memory-pressure cleanup changes:
  - `h08v04`
  - `h08v05`
  - `h09v04`
  - `h09v05`
  - `h10v04`
- validated scratch outputs now exist at:
  - `/scratch/alpine/ropa5718/spipy/input/viirs/snpp/ancillary/r0/<tile>/2023/snpp_r0_<tile>_2023.nc`
- spot checks confirmed the completed files currently report:
  - `build_status=complete`
  - `time_coverage_start=2023-06-01`
  - `time_coverage_end=2023-09-30`
  - `r0_sensor_zenith` present
  - `r0_sensor_azimuth` present
- the notebook then submitted inversion arrays for all five tiles for the full `WY2024` run
- an important logging-layout observation from that submission round:
  - all five Slurm submissions succeeded
  - only four timestamped log directories appeared because the last two tile submissions landed within the same second and therefore shared one timestamped directory
  - this confirmed the old timestamp-only directory scheme was too ambiguous for notebook batch launches

## Session Update (run-group / tile-summary refactor implemented in code)

- the CURC logging and summary layout has now been refactored in code toward an explicit run-group model
- the new intended layout is:
  - one `run_group_id` per user launch
  - one tile directory per tile beneath the run group
  - tile-local detailed artifacts under:
    - `<run_group>/<tile>/detailed_logs/`
  - tile-level summaries at:
    - `<run_group>/<tile>/run_inversion_<tile>_wy<water_year>_summary.csv`
    - `<run_group>/<tile>/run_inversion_<tile>_wy<water_year>_summary.txt`
  - group-level merged summaries at:
    - `<run_group>/run_inversion_wy<water_year>_summary.csv`
    - `<run_group>/run_inversion_wy<water_year>_summary.txt`
- aggregate water-year log writing has been removed from the new CURC flow:
  - do not expect new `run_inversion_wy<year>_aggregate.log` files from the refactored path
- `run_group_id` is now intended to be stored in manifest payloads as authoritative metadata for grouping
- summary behavior implemented in code is now:
  - write tile-level summaries only when a tile manifest family is terminal
  - write run-group merged summaries only when all tiles in the run group are terminal
- focused CURC validation after this refactor:
  - `module load miniforge && mamba run -n spipy14 python -m pytest -q tests/test_curc_planner.py tests/test_curc_status_reporting.py tests/test_curc_runtime_logging.py tests/test_curc_slurm_profile.py`
  - result: `15 passed`

### Immediate Next Step

- wire the notebook submission / scan cells in `examples/12_curc_sensor_workflow_planning.ipynb` to the new run-group model
- specifically:
  - generate one shared `RUN_GROUP_ID` once per batch submission / scan pass
  - pass that `RUN_GROUP_ID` through the per-tile manifest/submission path
  - print the resolved run-group directory and per-tile directories in notebook output so the new layout is obvious during use
- after notebook wiring is in place, rerun a small targeted submission/scan cycle to confirm:
  - manifests land under `<run_group>/<tile>/detailed_logs/`
  - tile-level summaries land at `<run_group>/<tile>/`
  - group-level summaries appear at `<run_group>/` only after all tiles in the run group are terminal
