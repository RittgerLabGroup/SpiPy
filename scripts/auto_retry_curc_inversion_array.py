#!/usr/bin/env python3
"""Scan a CURC inversion array and prepare or submit the next auto-retry array."""

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
from workflows.curc.status import scan_inversion_array_status, should_auto_retry, write_retry_manifest
from workflows.curc.task_manifest import load_inversion_array_manifest


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(
            "usage: auto_retry_curc_inversion_array.py <manifest.json> "
            "[--submit] [--retry-all-failed] [--python-exec <python>] "
            "[--execution-profile <name>] [--sbatch-arg <arg> ...]",
            file=sys.stderr,
        )
        return 2

    manifest_path = Path(argv[1]).expanduser().resolve()
    submit = False
    retry_only = True
    python_exec = "python"
    execution_profile = "cluster"
    extra_sbatch_args: list[str] = []

    i = 2
    while i < len(argv):
        token = argv[i]
        if token == "--submit":
            submit = True
        elif token == "--retry-all-failed":
            retry_only = False
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
    auto_retry_log_path = manifest_path.parent / (manifest_path.stem + f"_auto_retry_{timestamp}.log")
    logger = configure_spires_file_logger(
        auto_retry_log_path,
        logger_name=f"spires.curc.auto_retry.{manifest_path.stem}",
        mode="a",
        aggregate_log_path=aggregate_log_path,
    )

    report = scan_inversion_array_status(manifest_path)
    rendered_report = asdict(report) if is_dataclass(report) else report
    should_retry = should_auto_retry(manifest_path)
    common_fields = {
        "manifest_path": str(manifest_path),
        "job_name": manifest_path.stem,
        "sensor": manifest_payload["sensor"],
        "platform": manifest_payload["platform"],
        "tile": manifest_payload["tile"],
        "water_year": manifest_payload["water_year"],
        "submission_kind": "auto_retry",
        "retry_all_failed": not retry_only,
        "execution_profile": execution_profile,
        "python_executable": python_exec,
    }
    log_event(
        logger,
        "curc_auto_retry_inversion_array",
        stage="curc_auto_retry",
        event_type="start",
        status="started",
        scope=True,
        should_auto_retry=should_retry,
        auto_retry_log_path=str(auto_retry_log_path),
        **common_fields,
    )

    result: dict[str, object] = {
        "source_manifest_path": str(manifest_path),
        "report": rendered_report,
        "should_auto_retry": should_retry,
        "submitted": False,
    }
    if not should_retry:
        log_event(
            logger,
            "curc_auto_retry_inversion_array",
            stage="curc_auto_retry",
            event_type="summary",
            status="no_retry_needed",
            scope=True,
            auto_retry_complete=report.auto_retry_complete,
            auto_retry_eligible_count=report.auto_retry_eligible_count,
            retryable_count=report.retryable_count,
            **common_fields,
        )
        print(json.dumps(result, indent=2))
        return 0

    retry_manifest_path = write_retry_manifest(manifest_path, retry_only=retry_only)
    retry_payload = render_array_submission_payload_from_manifest(retry_manifest_path)
    sbatch_command = render_sbatch_command_for_array_payload(
        retry_payload,
        repo_root=REPO_ROOT,
        python_executable=python_exec,
        execution_profile=execution_profile,
        extra_sbatch_args=tuple(extra_sbatch_args),
    )

    result["retry_manifest_path"] = str(retry_manifest_path)
    result["retry_payload"] = retry_payload
    result["sbatch_command"] = sbatch_command
    log_event(
        logger,
        "curc_auto_retry_inversion_array",
        stage="curc_auto_retry",
        event_type="summary",
        status="retry_manifest_ready",
        scope=True,
        retry_manifest_path=str(retry_manifest_path),
        auto_retry_eligible_count=report.auto_retry_eligible_count,
        retry_payload=retry_payload,
        **common_fields,
    )

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
            "curc_auto_retry_inversion_array",
            stage="curc_auto_retry",
            event_type="submission",
            status="submitted",
            scope=True,
            source_manifest_path=str(manifest_path),
            retry_manifest_path=str(retry_manifest_path),
            sbatch_command=sbatch_command,
            sbatch_stdout=completed.stdout.strip(),
            sbatch_stderr=completed.stderr.strip(),
            **common_fields,
        )
    else:
        log_event(
            logger,
            "curc_auto_retry_inversion_array",
            stage="curc_auto_retry",
            event_type="summary",
            status="submission_preview_only",
            scope=True,
            source_manifest_path=str(manifest_path),
            retry_manifest_path=str(retry_manifest_path),
            sbatch_command=sbatch_command,
            **common_fields,
        )

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
