"""Manifest serialization helpers for CURC workflow task plans."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from workflows.curc.steps import InversionTaskPlan, SlurmArrayPlan


def default_inversion_array_manifest_path(
    plan: SlurmArrayPlan,
    *,
    root_dir: str | Path | None = None,
) -> Path:
    """Return a stable manifest path for one inversion array plan."""
    if root_dir is None:
        root_dir = Path(plan.tasks[0].log_path).parent if plan.tasks else Path.cwd()
    root = Path(root_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{plan.job_name}_array_manifest.json"


def write_inversion_array_manifest(
    plan: SlurmArrayPlan,
    *,
    manifest_path: str | Path | None = None,
) -> Path:
    """Write an inversion array manifest to disk and return its resolved path."""
    resolved_path = (
        default_inversion_array_manifest_path(plan)
        if manifest_path is None
        else Path(manifest_path).expanduser().resolve()
    )
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "job_name": plan.job_name,
        "step": plan.step,
        "sensor": plan.sensor,
        "platform": plan.platform,
        "tile": plan.tile,
        "water_year": plan.water_year,
        "task_count": plan.task_count,
        "array_indices": list(plan.array_indices),
        "max_concurrent_tasks": plan.max_concurrent_tasks,
        "max_auto_retry_count": plan.max_auto_retry_count,
        "apply_valid_inversion_mask": plan.apply_valid_inversion_mask,
        "use_grouping": plan.use_grouping,
        "grouping_method": plan.grouping_method,
        "r0_year": plan.r0_year,
        "slurm_profile": plan.slurm_profile.to_payload(),
        "tasks": [asdict(task) for task in plan.tasks],
    }
    resolved_path.write_text(json.dumps(payload, indent=2), encoding="ascii")
    return resolved_path


def load_inversion_array_manifest(manifest_path: str | Path) -> dict[str, object]:
    """Load an inversion array manifest from disk."""
    resolved_path = Path(manifest_path).expanduser().resolve()
    return json.loads(resolved_path.read_text(encoding="ascii"))


def resolve_inversion_task_from_manifest(
    manifest_path: str | Path,
    task_index: int,
) -> InversionTaskPlan:
    """Return one logical inversion task from a manifest by task index."""
    payload = load_inversion_array_manifest(manifest_path)
    for raw_task in payload["tasks"]:
        if raw_task["task_index"] == task_index:
            return InversionTaskPlan(
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
    raise KeyError(f"No inversion task with task_index={task_index} in manifest {manifest_path}")
