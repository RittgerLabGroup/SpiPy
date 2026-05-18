"""Post-run status scanning and retry planning for CURC inversion arrays."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

from spires.sensors.io import load_output_dataset_if_valid
from workflows.curc.steps import InversionTaskPlan
from workflows.curc.task_manifest import load_inversion_array_manifest


SUMMARY_EVENT_NAME = "curc_run_viirs_snpp_inversion_task"
_FIELD_RE = re.compile(r'([A-Za-z0-9_]+)=(".*?"|\{.*?\}|\[.*?\]|[^ ]+)')


@dataclass(frozen=True)
class InversionTaskStatus:
    """Observed status for one logical inversion task."""

    task_index: int
    date: str
    retry_count: int
    status: str
    failure_code: str
    retry_recommended: bool
    auto_retry_eligible: bool
    output_exists: bool
    output_valid: bool
    log_path: str
    output_path: str
    error_type: str | None = None
    error: str | None = None
    slurm_job_id: str | None = None
    slurm_array_task_id: str | None = None


@dataclass(frozen=True)
class InversionArrayStatusReport:
    """Summary report for one manifest-backed inversion array."""

    manifest_path: str
    task_count: int
    completed_count: int
    failed_count: int
    retryable_count: int
    auto_retry_eligible_count: int
    retry_exhausted_count: int
    missing_count: int
    max_auto_retry_count: int
    tasks: tuple[InversionTaskStatus, ...]
    auto_retry_complete: bool


def _parse_log_value(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _parse_structured_log_line(line: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, raw_value in _FIELD_RE.findall(line):
        result[key] = _parse_log_value(raw_value)
    return result


def _latest_summary_event(log_path: Path) -> dict[str, Any] | None:
    if not log_path.exists():
        return None
    latest: dict[str, Any] | None = None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if SUMMARY_EVENT_NAME not in line or 'event_type="summary"' not in line:
            continue
        parsed = _parse_structured_log_line(line)
        if parsed.get("event") == SUMMARY_EVENT_NAME:
            latest = parsed
    return latest


def _status_from_task(task: InversionTaskPlan, *, max_auto_retry_count: int) -> InversionTaskStatus:
    output_dataset_path = Path(task.output_path).expanduser().resolve() / "inversion.nc"
    output_exists = output_dataset_path.exists()
    output_valid = load_output_dataset_if_valid(output_dataset_path) is not None if output_exists else False
    log_path = Path(task.log_path).expanduser().resolve()
    summary = _latest_summary_event(log_path)

    if output_valid:
        status = "completed"
        failure_code = "none"
        retry_recommended = False
    elif summary is None:
        status = "missing_summary"
        failure_code = "slurm_or_external_failure"
        retry_recommended = True
    else:
        status = str(summary.get("status", "unknown"))
        failure_code = str(summary.get("failure_code", "unknown"))
        retry_recommended = bool(summary.get("retry_recommended", False))
    auto_retry_eligible = retry_recommended and task.retry_count < max_auto_retry_count

    return InversionTaskStatus(
        task_index=task.task_index,
        date=task.date,
        retry_count=task.retry_count,
        status=status,
        failure_code=failure_code,
        retry_recommended=retry_recommended,
        auto_retry_eligible=auto_retry_eligible,
        output_exists=output_exists,
        output_valid=output_valid,
        log_path=str(log_path),
        output_path=str(output_dataset_path),
        error_type=None if summary is None else summary.get("error_type"),
        error=None if summary is None else summary.get("error"),
        slurm_job_id=None if summary is None else summary.get("slurm_job_id"),
        slurm_array_task_id=None if summary is None else summary.get("slurm_array_task_id"),
    )


def scan_inversion_array_status(manifest_path: str | Path) -> InversionArrayStatusReport:
    """Scan a manifest-backed inversion array and classify task outcomes."""
    payload = load_inversion_array_manifest(manifest_path)
    max_auto_retry_count = int(payload.get("max_auto_retry_count", 3))
    tasks = tuple(
        InversionTaskPlan(
            task_index=raw["task_index"],
            sensor=raw["sensor"],
            platform=raw["platform"],
            tile=raw["tile"],
            water_year=raw["water_year"],
            date=raw["date"],
            source_paths=tuple(raw["source_paths"]),
            output_path=raw["output_path"],
            log_path=raw["log_path"],
            r0_year=raw["r0_year"],
            retry_count=raw.get("retry_count", 0),
        )
        for raw in payload["tasks"]
    )
    statuses = tuple(_status_from_task(task, max_auto_retry_count=max_auto_retry_count) for task in tasks)
    completed_count = sum(1 for task in statuses if task.status == "completed")
    failed_count = sum(1 for task in statuses if task.status not in {"completed", "loaded_existing"})
    retryable_count = sum(1 for task in statuses if task.retry_recommended)
    auto_retry_eligible_count = sum(1 for task in statuses if task.auto_retry_eligible)
    retry_exhausted_count = sum(
        1 for task in statuses if task.retry_recommended and task.retry_count >= max_auto_retry_count
    )
    missing_count = sum(1 for task in statuses if task.status == "missing_summary")
    return InversionArrayStatusReport(
        manifest_path=str(Path(manifest_path).expanduser().resolve()),
        task_count=len(statuses),
        completed_count=completed_count,
        failed_count=failed_count,
        retryable_count=retryable_count,
        auto_retry_eligible_count=auto_retry_eligible_count,
        retry_exhausted_count=retry_exhausted_count,
        missing_count=missing_count,
        max_auto_retry_count=max_auto_retry_count,
        tasks=statuses,
        auto_retry_complete=auto_retry_eligible_count == 0,
    )


def write_retry_manifest(
    manifest_path: str | Path,
    *,
    retry_only: bool = True,
    output_path: str | Path | None = None,
) -> Path:
    """Write a retry manifest containing failed tasks only."""
    original = load_inversion_array_manifest(manifest_path)
    report = scan_inversion_array_status(manifest_path)
    selected_indices = {
        task.task_index
        for task in report.tasks
        if task.status != "completed"
        and (
            task.auto_retry_eligible
            if retry_only
            else (task.retry_recommended or not retry_only)
        )
    }
    retry_tasks = [
        raw_task
        for raw_task in original["tasks"]
        if raw_task["task_index"] in selected_indices
    ]
    payload = {
        **{key: value for key, value in original.items() if key != "tasks"},
        "task_count": len(retry_tasks),
        "array_indices": list(range(len(retry_tasks))),
        "max_auto_retry_count": int(original.get("max_auto_retry_count", 3)),
        "tasks": [
            {
                **raw_task,
                "task_index": new_index,
                "retry_count": raw_task.get("retry_count", 0) + 1,
            }
            for new_index, raw_task in enumerate(retry_tasks)
        ],
    }
    if output_path is None:
        original_path = Path(manifest_path).expanduser().resolve()
        output_path = original_path.with_name(original_path.stem + "_retry.json")
    resolved_output_path = Path(output_path).expanduser().resolve()
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_path.write_text(json.dumps(payload, indent=2), encoding="ascii")
    return resolved_output_path


def should_auto_retry(manifest_path: str | Path) -> bool:
    """Return True when at least one failed task is still eligible for auto-retry."""
    report = scan_inversion_array_status(manifest_path)
    return not report.auto_retry_complete
