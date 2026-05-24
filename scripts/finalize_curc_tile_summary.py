#!/usr/bin/env python3
"""Finalize one tile summary after its inversion array reaches terminal state."""

from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.curc.status import write_tile_summary_artifacts


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: finalize_curc_tile_summary.py <manifest.json>", file=sys.stderr)
        return 2

    manifest_path = Path(argv[1]).expanduser().resolve()
    csv_path, txt_path = write_tile_summary_artifacts(manifest_path)
    print(json.dumps({"tile_summary_csv_path": str(csv_path), "tile_summary_txt_path": str(txt_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
