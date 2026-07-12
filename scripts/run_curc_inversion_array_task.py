#!/usr/bin/env python3
"""Resolve or execute one CURC inversion array task for Slurm runtime use."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
import os
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
            "[--grouping-method <name>] [--mask-low-reflectance-for-inversion <true|false>] "
            "[--low-reflectance-threshold <value>] [--include-grouped-reflectance-rmse <true|false>]",
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
    mask_low_reflectance_for_inversion = None
    low_reflectance_threshold = None
    include_grouped_reflectance_rmse = None
    use_grouping = None
    grouping_method = None
    include_albedo = None
    include_radiative_forcing = None
    include_delta_vis = None
    albedo_lut_path = None
    radiative_forcing_lut_path = None
    terrain_ancillary_root = None

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
        elif token == "--mask-low-reflectance-for-inversion":
            i += 1
            mask_low_reflectance_for_inversion = argv[i].strip().lower() in {"1", "true", "yes", "y"}
        elif token == "--low-reflectance-threshold":
            i += 1
            low_reflectance_threshold = float(argv[i])
        elif token == "--include-grouped-reflectance-rmse":
            i += 1
            include_grouped_reflectance_rmse = argv[i].strip().lower() in {"1", "true", "yes", "y"}
        elif token == "--use-grouping":
            i += 1
            use_grouping = argv[i].strip().lower() in {"1", "true", "yes", "y"}
        elif token == "--grouping-method":
            i += 1
            grouping_method = argv[i]
        elif token == "--include-albedo":
            i += 1
            include_albedo = argv[i].strip().lower() in {"1", "true", "yes", "y"}
        elif token == "--include-radiative-forcing":
            i += 1
            include_radiative_forcing = argv[i].strip().lower() in {"1", "true", "yes", "y"}
        elif token == "--include-delta-vis":
            i += 1
            include_delta_vis = argv[i].strip().lower() in {"1", "true", "yes", "y"}
        elif token == "--albedo-lut-path":
            i += 1
            albedo_lut_path = argv[i]
        elif token == "--radiative-forcing-lut-path":
            i += 1
            radiative_forcing_lut_path = argv[i]
        elif token == "--terrain-ancillary-root":
            i += 1
            terrain_ancillary_root = argv[i]
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
        mask_low_reflectance_for_inversion=mask_low_reflectance_for_inversion,
        low_reflectance_threshold=low_reflectance_threshold,
        include_grouped_reflectance_rmse=include_grouped_reflectance_rmse,
        use_grouping=use_grouping,
        grouping_method=grouping_method,
        include_albedo=include_albedo,
        include_radiative_forcing=include_radiative_forcing,
        include_delta_vis=include_delta_vis,
        albedo_lut_path=albedo_lut_path,
        radiative_forcing_lut_path=radiative_forcing_lut_path,
        terrain_ancillary_root=terrain_ancillary_root,
    )
    rendered = asdict(context) if is_dataclass(context) else context
    if os.environ.get("SLURM_JOB_ID") is None:
        print(json.dumps(rendered, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
