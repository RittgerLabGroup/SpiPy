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
- active development branch for this session: `workflows/viirs-workflow`
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
