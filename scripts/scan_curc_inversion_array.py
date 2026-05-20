#!/usr/bin/env python3
"""Scan CURC inversion array results and optionally write a retry manifest."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.curc.status import (
    scan_inversion_array_status,
    should_auto_retry,
    write_retry_manifest,
    write_status_summary_artifacts,
)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(
            "usage: scan_curc_inversion_array.py <manifest.json> [--write-retry-manifest] [--retry-all-failed]",
            file=sys.stderr,
        )
        return 2

    manifest_path = Path(argv[1]).expanduser().resolve()
    write_retry = "--write-retry-manifest" in argv[2:]
    retry_only = "--retry-all-failed" not in argv[2:]

    report = scan_inversion_array_status(manifest_path)
    rendered = asdict(report) if is_dataclass(report) else report
    rendered["should_auto_retry"] = should_auto_retry(manifest_path)
    task_attempts_csv_path, summary_txt_path = write_status_summary_artifacts(manifest_path, report=report)
    rendered["task_attempts_csv_path"] = str(task_attempts_csv_path)
    rendered["summary_txt_path"] = str(summary_txt_path)
    if write_retry:
        retry_manifest_path = write_retry_manifest(manifest_path, retry_only=retry_only)
        rendered["retry_manifest_path"] = str(retry_manifest_path)
    print(json.dumps(rendered, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
