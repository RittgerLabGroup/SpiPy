#!/usr/bin/env python3
"""Preview or execute one non-array CURC workflow step from a user config."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.curc.runner import run_viirs_snpp_step


def load_config(config_path: Path):
    spec = spec_from_file_location("curc_user_config", config_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load config module from {config_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "CONFIG"):
        raise ValueError(f"Config module {config_path} must define CONFIG")
    return module.CONFIG


def main(argv: list[str]) -> int:
    if len(argv) < 5:
        print(
            "usage: run_curc_workflow_step.py <config.py> <tile> <water_year> <step> "
            "[--date <YYYY-MM-DD> ...] [--r0-year <year>] [--execute] [--overwrite] "
            "[--rsync-exec <path>] [--lut-file <path>] [--ndvi-tie-epsilon <float>] [--show-progress]",
            file=sys.stderr,
        )
        return 2

    config_path = Path(argv[1]).expanduser().resolve()
    tile = argv[2]
    water_year = int(argv[3])
    step = argv[4]
    target_dates: list[str] = []
    r0_year = None
    execute = False
    overwrite = False
    rsync_executable = "rsync"
    lut_file = None
    ndvi_tie_epsilon = 0.02
    show_progress = False

    i = 5
    while i < len(argv):
        token = argv[i]
        if token == "--date":
            i += 1
            target_dates.append(argv[i])
        elif token == "--r0-year":
            i += 1
            r0_year = int(argv[i])
        elif token == "--execute":
            execute = True
        elif token == "--overwrite":
            overwrite = True
        elif token == "--rsync-exec":
            i += 1
            rsync_executable = argv[i]
        elif token == "--lut-file":
            i += 1
            lut_file = argv[i]
        elif token == "--ndvi-tie-epsilon":
            i += 1
            ndvi_tie_epsilon = float(argv[i])
        elif token == "--show-progress":
            show_progress = True
        else:
            raise ValueError(f"Unexpected argument: {token}")
        i += 1

    config = load_config(config_path)
    result = run_viirs_snpp_step(
        config,
        tile=tile,
        water_year=water_year,
        step=step,
        target_dates=tuple(target_dates),
        r0_year=r0_year,
        execute=execute,
        rsync_executable=rsync_executable,
        lut_file=lut_file,
        ndvi_tie_epsilon=ndvi_tie_epsilon,
        overwrite=overwrite,
        show_progress=show_progress,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
