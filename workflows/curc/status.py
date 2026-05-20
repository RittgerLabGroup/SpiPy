"""Post-run status scanning and retry planning for CURC inversion arrays."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any

from spires.sensors.io import load_output_dataset_if_valid
from workflows.curc.paths import top_level_log_dir
from workflows.curc.steps import InversionTaskPlan
from workflows.curc.task_manifest import load_inversion_array_manifest


SUMMARY_EVENT_NAME = "curc_run_viirs_snpp_inversion_task"
_FIELD_RE = re.compile(r'([A-Za-z0-9_]+)=(".*?"|\{.*?\}|\[.*?\]|[^ ]+)')
_TIMESTAMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")


def _inversion_output_dataset_path(task: InversionTaskPlan) -> Path:
    date_token = task.date.replace("-", "")
    preferred = Path(task.output_path).expanduser().resolve() / f"{task.platform}_raw_output_{task.tile}_{date_token}.nc"
    legacy = Path(task.output_path).expanduser().resolve() / "inversion.nc"
    return preferred if preferred.exists() or not legacy.exists() else legacy


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


@dataclass(frozen=True)
class InversionTaskAttempt:
    """Observed status for one attempt of one logical inversion date."""

    water_year: int
    scene_date: str
    task_index: int
    attempt_ordinal: int
    retry_count: int
    submission_kind: str
    last_attempt_for_date: bool
    status: str
    failure_code: str
    retry_recommended: bool
    loaded_existing: bool
    submitted: bool
    started: bool
    completed: bool
    output_valid: bool
    output_path: str
    log_path: str
    manifest_path: str
    log_dir: str
    sensor: str
    platform: str
    tile: str
    slurm_array_job_id: str | None
    slurm_array_task_id: str | None
    slurm_job_id: str | None
    slurm_job_name: str | None
    slurm_cluster_name: str | None
    start_time_utc: str | None
    end_time_utc: str | None
    elapsed_seconds: float | None
    message: str


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


def _parse_log_timestamp(line: str) -> datetime | None:
    match = _TIMESTAMP_RE.match(line)
    if match is None:
        return None
    return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S,%f")


def _task_log_events(log_path: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if not log_path.exists():
        return None, None
    latest_start: dict[str, Any] | None = None
    latest_summary: dict[str, Any] | None = None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if SUMMARY_EVENT_NAME not in line:
            continue
        parsed = _parse_structured_log_line(line)
        if parsed.get("event") != SUMMARY_EVENT_NAME:
            continue
        timestamp = _parse_log_timestamp(line)
        if timestamp is not None:
            parsed["_timestamp"] = timestamp
        if parsed.get("event_type") == "start":
            latest_start = parsed
        elif parsed.get("event_type") == "summary":
            latest_summary = parsed
    return latest_start, latest_summary


def _latest_summary_event(log_path: Path) -> dict[str, Any] | None:
    _, latest_summary = _task_log_events(log_path)
    return latest_summary


def _resolve_task_log_path(task: InversionTaskPlan) -> Path:
    base = Path(task.log_path).expanduser().resolve()
    candidates = sorted(
        base.parent.glob(f"{base.stem}_job*{base.suffix}"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    return base


def _resolve_task_log_path_for_attempt(task: InversionTaskPlan) -> Path:
    base = Path(task.log_path).expanduser().resolve()
    candidates = sorted(
        base.parent.glob(f"{base.stem}_job*{base.suffix}"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        summary = _latest_summary_event(candidate)
        if summary is None:
            continue
        if str(summary.get("date")) != task.date:
            continue
        if int(summary.get("retry_count", -1)) == int(task.retry_count):
            return candidate
    return _resolve_task_log_path(task)


def _status_from_task(task: InversionTaskPlan, *, max_auto_retry_count: int) -> InversionTaskStatus:
    output_dataset_path = _inversion_output_dataset_path(task)
    output_exists = output_dataset_path.exists()
    validated_output = load_output_dataset_if_valid(output_dataset_path) is not None if output_exists else False
    log_path = _resolve_task_log_path(task)
    summary = _latest_summary_event(log_path)

    if summary is not None:
        status = str(summary.get("status", "unknown"))
        if status == "completed":
            output_valid = validated_output
        elif status == "loaded_existing":
            output_valid = validated_output
        else:
            output_valid = False
        failure_code = str(summary.get("failure_code", "unknown"))
        retry_recommended = bool(summary.get("retry_recommended", False))
    elif validated_output:
        status = "completed"
        output_valid = True
        failure_code = "none"
        retry_recommended = False
    else:
        status = "missing_summary"
        output_valid = False
        failure_code = "slurm_or_external_failure"
        retry_recommended = True
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


def _manifest_family_stem(stem: str) -> str:
    family_stem = stem
    while family_stem.endswith("_retry"):
        family_stem = family_stem[:-6]
    return family_stem


def _manifest_retry_depth(stem: str) -> int:
    depth = 0
    current = stem
    while current.endswith("_retry"):
        depth += 1
        current = current[:-6]
    return depth


def _related_manifest_paths(manifest_path: str | Path) -> list[Path]:
    resolved = Path(manifest_path).expanduser().resolve()
    family_stem = _manifest_family_stem(resolved.stem)
    candidates = [
        path
        for path in resolved.parent.glob(f"{family_stem}*.json")
        if path.stem == family_stem or path.stem.startswith(f"{family_stem}_retry")
    ]
    return sorted(candidates, key=lambda path: (_manifest_retry_depth(path.stem), path.name))


def _infer_submission_kind(manifest_stem: str) -> str:
    return "auto_retry" if manifest_stem.endswith("_retry") else "initial"


def _event_timestamp_iso(event: dict[str, Any] | None) -> str | None:
    if event is None:
        return None
    timestamp = event.get("_timestamp")
    if not isinstance(timestamp, datetime):
        return None
    return timestamp.isoformat(timespec="seconds") + "Z"


def _event_elapsed_seconds(start_event: dict[str, Any] | None, summary_event: dict[str, Any] | None) -> float | None:
    if start_event is None or summary_event is None:
        return None
    start_ts = start_event.get("_timestamp")
    end_ts = summary_event.get("_timestamp")
    if not isinstance(start_ts, datetime) or not isinstance(end_ts, datetime):
        return None
    return round((end_ts - start_ts).total_seconds(), 3)


def _attempt_message(status: str, failure_code: str) -> str:
    if status == "loaded_existing":
        return "reused prior successful output"
    if status == "completed":
        return "completed normally"
    if status == "missing_summary":
        return "missing runtime summary"
    if failure_code not in {"", "none", "unknown"}:
        return failure_code
    return status


def _collect_task_attempts(manifest_path: str | Path) -> tuple[InversionTaskAttempt, ...]:
    attempts_by_date: dict[str, list[InversionTaskAttempt]] = {}
    for family_index, path in enumerate(_related_manifest_paths(manifest_path)):
        payload = load_inversion_array_manifest(path)
        max_auto_retry_count = int(payload.get("max_auto_retry_count", 3))
        for raw_task in payload["tasks"]:
            task = InversionTaskPlan(
                task_index=raw_task["task_index"],
                sensor=raw_task["sensor"],
                platform=raw_task["platform"],
                tile=raw_task["tile"],
                water_year=raw_task["water_year"],
                date=raw_task["date"],
                source_paths=tuple(raw_task["source_paths"]),
                output_path=raw_task["output_path"],
                log_path=raw_task["log_path"],
                r0_year=raw_task["r0_year"],
                retry_count=raw_task.get("retry_count", 0),
            )
            output_dataset_path = _inversion_output_dataset_path(task)
            current_output_valid = output_dataset_path.exists() and load_output_dataset_if_valid(output_dataset_path) is not None
            log_path = _resolve_task_log_path_for_attempt(task)
            start_event, summary_event = _task_log_events(log_path)
            if summary_event is not None:
                status = str(summary_event.get("status", "unknown"))
                failure_code = str(summary_event.get("failure_code", "unknown"))
                retry_recommended = bool(summary_event.get("retry_recommended", False))
            elif current_output_valid:
                status = "completed"
                failure_code = "none"
                retry_recommended = False
            else:
                status = "missing_summary"
                failure_code = "slurm_or_external_failure"
                retry_recommended = True

            attempt = InversionTaskAttempt(
                water_year=task.water_year,
                scene_date=task.date,
                task_index=task.task_index,
                attempt_ordinal=family_index + 1,
                retry_count=task.retry_count,
                submission_kind=_infer_submission_kind(path.stem),
                last_attempt_for_date=False,
                status=status,
                failure_code=failure_code,
                retry_recommended=retry_recommended,
                loaded_existing=status == "loaded_existing",
                submitted=start_event is not None or summary_event is not None,
                started=start_event is not None,
                completed=status in {"completed", "loaded_existing"},
                output_valid=current_output_valid and status in {"completed", "loaded_existing"},
                output_path=str(output_dataset_path),
                log_path=str(log_path),
                manifest_path=str(path),
                log_dir=str(path.parent),
                sensor=task.sensor,
                platform=task.platform,
                tile=task.tile,
                slurm_array_job_id=None if summary_event is None else summary_event.get("slurm_array_job_id"),
                slurm_array_task_id=None if summary_event is None else summary_event.get("slurm_array_task_id"),
                slurm_job_id=None if summary_event is None else summary_event.get("slurm_job_id"),
                slurm_job_name=None if summary_event is None else summary_event.get("slurm_job_name"),
                slurm_cluster_name=None if summary_event is None else summary_event.get("slurm_cluster_name"),
                start_time_utc=_event_timestamp_iso(start_event),
                end_time_utc=_event_timestamp_iso(summary_event),
                elapsed_seconds=_event_elapsed_seconds(start_event, summary_event),
                message=_attempt_message(status, failure_code),
            )
            attempts_by_date.setdefault(task.date, []).append(attempt)

    all_attempts: list[InversionTaskAttempt] = []
    for date in sorted(attempts_by_date):
        ordered = sorted(
            attempts_by_date[date],
            key=lambda attempt: (attempt.retry_count, attempt.attempt_ordinal, attempt.task_index),
        )
        for index, attempt in enumerate(ordered, start=1):
            all_attempts.append(
                InversionTaskAttempt(
                    **{
                        **attempt.__dict__,
                        "attempt_ordinal": index,
                        "last_attempt_for_date": index == len(ordered),
                    }
                )
            )
    return tuple(all_attempts)


def _task_attempts_csv_path(manifest_path: str | Path, *, water_year: int) -> Path:
    resolved = top_level_log_dir(manifest_path)
    return resolved / f"run_inversion_wy{water_year}_task_attempts.csv"


def _summary_txt_path(manifest_path: str | Path, *, water_year: int) -> Path:
    resolved = top_level_log_dir(manifest_path)
    return resolved / f"run_inversion_wy{water_year}_summary.txt"


def write_status_summary_artifacts(
    manifest_path: str | Path,
    *,
    report: InversionArrayStatusReport | None = None,
) -> tuple[Path, Path]:
    """Write task-attempt CSV and per-date text summary for one manifest family."""
    resolved_manifest_path = Path(manifest_path).expanduser().resolve()
    manifest_payload = load_inversion_array_manifest(resolved_manifest_path)
    attempts = _collect_task_attempts(resolved_manifest_path)
    csv_path = _task_attempts_csv_path(resolved_manifest_path, water_year=int(manifest_payload["water_year"]))
    txt_path = _summary_txt_path(resolved_manifest_path, water_year=int(manifest_payload["water_year"]))

    csv_fields = [
        "water_year",
        "scene_date",
        "task_index",
        "attempt_ordinal",
        "submission_kind",
        "retry_count",
        "last_attempt_for_date",
        "status",
        "failure_code",
        "retry_recommended",
        "loaded_existing",
        "submitted",
        "started",
        "completed",
        "output_valid",
        "output_path",
        "log_path",
        "manifest_path",
        "log_dir",
        "sensor",
        "platform",
        "tile",
        "slurm_array_job_id",
        "slurm_array_task_id",
        "slurm_job_id",
        "slurm_job_name",
        "slurm_cluster_name",
        "start_time_utc",
        "end_time_utc",
        "elapsed_seconds",
        "message",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for attempt in attempts:
            writer.writerow({field: getattr(attempt, field) for field in csv_fields})

    final_attempts = [attempt for attempt in attempts if attempt.last_attempt_for_date]
    final_attempts.sort(key=lambda attempt: attempt.scene_date)
    completed_count = sum(1 for attempt in final_attempts if attempt.status == "completed")
    loaded_existing_count = sum(1 for attempt in final_attempts if attempt.status == "loaded_existing")
    failed_count = sum(1 for attempt in final_attempts if attempt.status not in {"completed", "loaded_existing"})
    missing_output_count = sum(1 for attempt in final_attempts if not attempt.output_valid)
    auto_retried_dates = sum(1 for attempt in final_attempts if attempt.retry_count > 0)

    attempts_per_date: dict[str, int] = {}
    for attempt in attempts:
        attempts_per_date[attempt.scene_date] = attempts_per_date.get(attempt.scene_date, 0) + 1

    lines = [
        f"WATER YEAR {manifest_payload['water_year']}",
        f"sensor={manifest_payload['sensor']} platform={manifest_payload['platform']} tile={manifest_payload['tile']}",
        f"log_dir={top_level_log_dir(resolved_manifest_path)}",
        f"manifest={resolved_manifest_path}",
        "",
        "TOTALS",
        f"dates={len(final_attempts)}",
        f"completed={completed_count}",
        f"loaded_existing={loaded_existing_count}",
        f"failed={failed_count}",
        f"missing_output={missing_output_count}",
        f"auto_retried_dates={auto_retried_dates}",
        f"total_attempt_rows={len(attempts)}",
        "",
        "scene_date   final_status     attempts  retry_count  failure_code  output  slurm_job  note",
    ]
    for attempt in final_attempts:
        output_flag = "yes" if attempt.output_valid else "no"
        slurm_job = attempt.slurm_job_id or "-"
        lines.append(
            f"{attempt.scene_date:<12} "
            f"{attempt.status:<16} "
            f"{attempts_per_date.get(attempt.scene_date, 0):<9} "
            f"{attempt.retry_count:<12} "
            f"{attempt.failure_code:<13} "
            f"{output_flag:<7} "
            f"{slurm_job:<10} "
            f"{attempt.message}"
        )
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, txt_path


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
