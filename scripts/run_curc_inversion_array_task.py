#!/usr/bin/env python3
"""Resolve or execute one CURC inversion array task for Slurm runtime use."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.curc.runtime import execute_viirs_snpp_inversion_task


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(
            "usage: run_curc_inversion_array_task.py <manifest.json> [task_index] "
            "[--execute] [--overwrite] [--lut-file <path>] [--execution-profile <name>] "
            "[--apply-valid-inversion-mask <true|false>] [--use-grouping <true|false>] "
            "[--grouping-method <name>]",
            file=sys.stderr,
        )
        return 2

    manifest_path = Path(argv[1]).expanduser().resolve()
    task_index = None
    execute = False
    overwrite = False
    lut_file = None
    execution_profile = "cluster"
    apply_valid_inversion_mask = None
    use_grouping = None
    grouping_method = None

    i = 2
    while i < len(argv):
        token = argv[i]
        if token == "--execute":
            execute = True
        elif token == "--overwrite":
            overwrite = True
        elif token == "--lut-file":
            i += 1
            lut_file = argv[i]
        elif token == "--execution-profile":
            i += 1
            execution_profile = argv[i]
        elif token == "--apply-valid-inversion-mask":
            i += 1
            apply_valid_inversion_mask = argv[i].strip().lower() in {"1", "true", "yes", "y"}
        elif token == "--use-grouping":
            i += 1
            use_grouping = argv[i].strip().lower() in {"1", "true", "yes", "y"}
        elif token == "--grouping-method":
            i += 1
            grouping_method = argv[i]
        elif task_index is None:
            task_index = int(token)
        else:
            raise ValueError(f"Unexpected argument: {token}")
        i += 1

    context = execute_viirs_snpp_inversion_task(
        manifest_path,
        task_index=task_index,
        lut_file=lut_file,
        execution_profile=execution_profile,
        overwrite=overwrite,
        dry_run=not execute,
        apply_valid_inversion_mask=apply_valid_inversion_mask,
        use_grouping=use_grouping,
        grouping_method=grouping_method,
    )
    rendered = asdict(context) if is_dataclass(context) else context
    print(json.dumps(rendered, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
