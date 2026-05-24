#!/usr/bin/env python3
"""Finalize tile and run-group summaries for a CURC run-group directory."""

from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.curc.status import list_run_group_tile_manifests, write_run_group_summary_artifacts, write_tile_summary_artifacts


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: finalize_curc_run_group_summary.py <run_group_dir>", file=sys.stderr)
        return 2

    run_group_dir = Path(argv[1]).expanduser().resolve()
    manifests = list_run_group_tile_manifests(run_group_dir)
    if not manifests:
        raise ValueError(f"No tile manifests found under run group: {run_group_dir}")

    tile_summaries: list[dict[str, str]] = []
    for manifest_path in manifests:
        csv_path, txt_path = write_tile_summary_artifacts(manifest_path)
        tile_summaries.append(
            {
                "manifest_path": str(manifest_path),
                "tile_summary_csv_path": str(csv_path),
                "tile_summary_txt_path": str(txt_path),
            }
        )

    group_csv_path, group_txt_path = write_run_group_summary_artifacts(run_group_dir)
    print(
        json.dumps(
            {
                "run_group_dir": str(run_group_dir),
                "tile_summaries": tile_summaries,
                "run_group_summary_csv_path": str(group_csv_path),
                "run_group_summary_txt_path": str(group_txt_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
