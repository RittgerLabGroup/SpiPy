# CURC Workflow Notes

## Purpose

This note is the current implementation summary for the CURC-specific workflow code in this repository. It is intended to replace the prior session-by-session narrative with a compact description of what exists now.

Archived historical notes live in `notes_curc_archive.md`.

## Scope

- The active CURC orchestration code lives in `workflows/curc/`.
- The main user-facing entrypoints live in `scripts/`.
- The exploratory notebook is `examples/12_curc_sensor_workflow_planning.ipynb`.
- The current step planner and runtime are specialized to `sensor="viirs"` and `platforms=("snpp",)`.

## Configuration Model

`workflows/curc/config.py`

- `CurcWorkflowConfig` defines scratch paths, source root, sensor/platform/tile/year selection, dry-run behavior, retry policy, inversion options, and Slurm settings.
- `canonicalized()` normalizes sensor and platform names through the SpiPy registry.
- `SlurmProfile` carries submission-time resource settings:
  - `account`
  - `qos`
  - `time`
  - `mem`
  - `cpus_per_task`
  - `output_dir`
  - `extra_args`

## Path Conventions

`workflows/curc/paths.py`

- Staged reflectance: `<scratch_root>/input/<sensor>/<platform>/reflectance/<tile>/<water_year>/`
- Ancillary root: `<scratch_root>/input/<sensor>/<platform>/ancillary/<tile>/`
- Annual R0 root: `<scratch_root>/input/<sensor>/<platform>/ancillary/r0/<tile>/<r0_year>/`
- Annual R0 dataset: `<scratch_root>/input/<sensor>/<platform>/ancillary/r0/<tile>/<r0_year>/<platform>_r0_<tile>_<r0_year>.nc`
- Raw inversion outputs: `<scratch_root>/output/<sensor>/<platform>/<tile>/raw/wy<water_year>/`
- Run-group logs: `<scratch_root>/logs/<run_group_id>/`
- Tile-local detailed logs: `<scratch_root>/logs/<run_group_id>/<tile>/detailed_logs/`

`build_run_group_id()` creates IDs like:

- full water year: `<timestamp>_viirs_snpp_wy2023_full`
- single date: `<timestamp>_viirs_snpp_wy2023_2023-03-16_single_date`
- subset: `<timestamp>_viirs_snpp_wy2023_<start>_<end>_date_subset`

## Planning Logic

`workflows/curc/planner.py`

- `plan_viirs_snpp_workflow_steps()` produces four explicit step plans:
  - `stage_reflectance`
  - `stage_ancillary`
  - `build_r0`
  - `run_inversion`
- Reflectance discovery uses `discover_viirs_snpp_reflectance_files()`.
- Water-year inversion scenes come from the requested water year or from explicit `target_dates`.
- `build_r0` uses summer scenes from the resolved `r0_year`.
- If `r0_year` is not supplied, the default comes from `default_r0_year_for_water_year()`.
- Tests currently assert the intended rule that water year `N` uses the previous summer for `R0`.

`plan_viirs_snpp_inversion_array()` converts the `run_inversion` portion into a `SlurmArrayPlan`.

- One logical task is created per acquisition date.
- Each task stores:
  - `task_index`
  - `date`
  - discovered source paths
  - output root
  - detailed log path
  - `r0_year`
  - `retry_count`
- The array plan also carries:
  - concurrency limit
  - retry limit
  - inversion mask/grouping options
  - full `SlurmProfile`

## Non-Array Step Execution

`workflows/curc/execution.py`

- `stage_reflectance`
  - preview mode renders `rsync -av --ignore-existing ...`
  - execute mode runs those `rsync` commands directly
- `stage_ancillary`
  - creates the expected ancillary, R0, and output directories
  - preview lists expected static files such as `canopy_fraction` and `glacier_ice_fraction`
- `build_r0`
  - calls `spires.sensors.viirs.r0.build_r0_from_sources(...)`
  - writes the annual dataset with the current CURC naming convention
- `run_inversion`
  - does not execute directly here
  - must go through the manifest-backed Slurm array path

## Manifest and Submission Flow

`workflows/curc/task_manifest.py`

- `write_inversion_array_manifest()` persists a `SlurmArrayPlan` as JSON.
- The manifest stores:
  - job metadata
  - run-group and tile log directories
  - retry limit
  - inversion options
  - Slurm profile
  - per-date tasks

`workflows/curc/slurm.py`

- `render_array_submission_payload_from_manifest()` reconstructs a compact submission payload from a manifest.
- `render_sbatch_command_for_array_payload()` builds the actual `sbatch` command.
- Slurm profile flags are rendered first; CLI `--sbatch-arg` values append after them as one-off overrides.
- Slurm stdout defaults to the manifest directory unless `slurm_profile.output_dir` is set.

## Runtime Execution

`workflows/curc/runtime.py`

- `build_viirs_snpp_inversion_runtime_context()` resolves one manifest task into concrete runtime paths.
- It infers `scratch_root` from the output path stored in the manifest task.
- It resolves:
  - staged reflectance path(s)
  - ancillary root
  - annual R0 path
  - platform LUT path
  - output dataset path
  - task log path
- The default output dataset name is:
  - `<platform>_raw_output_<tile>_<YYYYMMDD>.nc`

Important current runtime assumptions:

- The executor currently expects exactly one staged reflectance file per acquisition date.
- Missing reflectance, R0, or LUT inputs are classified before execution.
- `OSError`-style filesystem failures are treated as retryable.
- The runtime reads inversion options from the manifest unless explicitly overridden at execution time:
  - `apply_valid_inversion_mask`
  - `use_grouping`
  - `grouping_method`

`execute_viirs_snpp_inversion_task()` is the main manifest-backed runtime entrypoint.

- Dry-run mode resolves and validates context without running the inversion.
- Execute mode calls `spires.sensors.viirs.workflow.run_viirs_inversion(...)`.
- Structured task logs include Slurm metadata when present.
- Per-job log files use `run_inversion_<date>_job<SLURM_JOB_ID>.log` when a Slurm job ID exists.

## Status Scanning and Retry Logic

`workflows/curc/status.py`

- `scan_inversion_array_status()` inspects one manifest family and classifies each logical date task.
- Status is derived from:
  - output dataset validity
  - latest structured runtime summary log
  - retry counters stored in the manifest
- Current failure codes include:
  - `missing_staged_reflectance`
  - `missing_r0`
  - `missing_lut`
  - `missing_runtime_input`
  - `invalid_runtime_input`
  - `filesystem_error`
  - `python_exception`
  - `slurm_or_external_failure`

Retry behavior:

- `max_auto_retry_count` is stored in the manifest and defaults to `3`.
- `write_retry_manifest()` creates a new manifest containing failed tasks only.
- Retry tasks are renumbered from zero and get `retry_count + 1`.
- `should_auto_retry()` returns `True` only when at least one failed task is still eligible.

Summary artifacts:

- Tile-level summaries:
  - `run_inversion_<tile>_wy<water_year>_summary.csv`
  - `run_inversion_<tile>_wy<water_year>_summary.txt`
- Run-group summaries:
  - `run_inversion_wy<water_year>_summary.csv`
  - `run_inversion_wy<water_year>_summary.txt`
- `write_terminal_summary_artifacts()` writes tile summaries when a tile is terminal and run-group summaries once every tile in the run group is terminal.

## Script Surface

Current entrypoints:

- `scripts/run_curc_workflow_step.py`
  - preview or execute `stage_reflectance`, `stage_ancillary`, or `build_r0`
- `scripts/run_curc_inversion_array_task.py`
  - resolve or execute one manifest-backed inversion task
- `scripts/submit_curc_inversion_array.py`
  - preview or submit an initial array from an existing manifest
- `scripts/scan_curc_inversion_array.py`
  - scan status, write summaries, optionally emit a retry manifest
- `scripts/auto_retry_curc_inversion_array.py`
  - scan status, create the next retry manifest, optionally submit it
- `scripts/submit_curc_sensor_workflow.py`
  - renders older deterministic job-manifest payloads from `build_job_manifest()`

## Current Tests That Define Behavior

Relevant tests currently cover:

- planner behavior: `tests/test_curc_planner.py`
- Slurm profile persistence and sbatch rendering: `tests/test_curc_slurm_profile.py`
- runtime log naming and log path helpers: `tests/test_curc_runtime_logging.py`
- attempt-history summaries and run-group summaries: `tests/test_curc_status_reporting.py`

## Current Constraints

- CURC planning/runtime are presently VIIRS SNPP-specific.
- `run_inversion` is array-first; there is no direct non-Slurm step executor for it.
- Runtime currently assumes one reflectance file per date task.
- The newer CURC implementation of record is the `workflows/curc/` stack, not the historical notes or notebook-only logic.
