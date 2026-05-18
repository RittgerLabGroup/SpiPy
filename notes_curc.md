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
