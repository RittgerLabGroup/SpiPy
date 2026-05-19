#!/usr/bin/env python3
"""Prepare or submit an initial CURC inversion array from an existing manifest."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spires.logging_utils import configure_spires_file_logger, log_event
from workflows.curc.slurm import (
    render_array_submission_payload_from_manifest,
    render_sbatch_command_for_array_payload,
)
from workflows.curc.status import scan_inversion_array_status
from workflows.curc.task_manifest import load_inversion_array_manifest


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(
            "usage: submit_curc_inversion_array.py <manifest.json> "
            "[--submit] [--python-exec <python>] [--execution-profile <name>] [--sbatch-arg <arg> ...]",
            file=sys.stderr,
        )
        return 2

    manifest_path = Path(argv[1]).expanduser().resolve()
    submit = False
    python_exec = "python"
    execution_profile = "cluster"
    extra_sbatch_args: list[str] = []

    i = 2
    while i < len(argv):
        token = argv[i]
        if token == "--submit":
            submit = True
        elif token == "--python-exec":
            i += 1
            python_exec = argv[i]
        elif token == "--execution-profile":
            i += 1
            execution_profile = argv[i]
        elif token == "--sbatch-arg":
            i += 1
            extra_sbatch_args.append(argv[i])
        else:
            raise ValueError(f"Unexpected argument: {token}")
        i += 1

    manifest_payload = load_inversion_array_manifest(manifest_path)
    aggregate_log_path = manifest_path.parent / f"run_inversion_wy{manifest_payload['water_year']}_aggregate.log"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_log_path = manifest_path.parent / (manifest_path.stem + f"_initial_submission_{timestamp}.log")
    logger = configure_spires_file_logger(
        submission_log_path,
        logger_name=f"spires.curc.initial_submission.{manifest_path.stem}",
        mode="a",
        aggregate_log_path=aggregate_log_path,
    )

    report = scan_inversion_array_status(manifest_path)
    rendered_report = asdict(report) if is_dataclass(report) else report
    payload = render_array_submission_payload_from_manifest(manifest_path)
    sbatch_command = render_sbatch_command_for_array_payload(
        payload,
        repo_root=REPO_ROOT,
        python_executable=python_exec,
        execution_profile=execution_profile,
        extra_sbatch_args=tuple(extra_sbatch_args),
    )
    common_fields = {
        "manifest_path": str(manifest_path),
        "job_name": payload["job_name"],
        "sensor": payload["sensor"],
        "platform": payload["platform"],
        "tile": payload["tile"],
        "water_year": payload["water_year"],
        "submission_kind": "initial",
        "execution_profile": execution_profile,
        "python_executable": python_exec,
    }
    result: dict[str, object] = {
        "manifest_path": str(manifest_path),
        "report": rendered_report,
        "payload": payload,
        "sbatch_command": sbatch_command,
        "submitted": False,
    }

    if submit:
        completed = subprocess.run(
            sbatch_command,
            check=True,
            text=True,
            capture_output=True,
        )
        result["submitted"] = True
        result["sbatch_stdout"] = completed.stdout.strip()
        result["sbatch_stderr"] = completed.stderr.strip()
        log_event(
            logger,
            "curc_submit_inversion_array",
            stage="curc_submission",
            event_type="submission",
            status="submitted",
            scope=True,
            task_count=payload["task_count"],
            array_spec=payload["array_spec"],
            sbatch_command=sbatch_command,
            sbatch_stdout=completed.stdout.strip(),
            sbatch_stderr=completed.stderr.strip(),
            **common_fields,
        )

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
