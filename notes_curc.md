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
