"""Slurm payload rendering for CURC-specific SpiPy workflows."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import shlex

from workflows.curc.config import SlurmProfile
from workflows.curc.manifest import PlannedJob
from workflows.curc.steps import SlurmArrayPlan
from workflows.curc.task_manifest import load_inversion_array_manifest


def render_submission_payload(job: PlannedJob) -> dict[str, object]:
    """Render a stable submission payload for a planned CURC job."""
    payload = asdict(job)
    payload["job_name"] = f"spipy-{job.sensor}-{job.platform}-{job.tile}-{job.year}"
    return payload


def render_array_submission_payload(
    plan: SlurmArrayPlan,
    *,
    manifest_path: str | Path | None = None,
) -> dict[str, object]:
    """Render a compact payload for a Slurm array submission."""
    payload = {
        "step": plan.step,
        "job_name": plan.job_name,
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
        "notes": list(plan.notes),
        "r0_year": plan.r0_year,
        "slurm_profile": plan.slurm_profile.to_payload(),
    }
    if plan.array_indices:
        start = min(plan.array_indices)
        end = max(plan.array_indices)
        array_spec = f"{start}-{end}"
        if plan.max_concurrent_tasks is not None:
            array_spec = f"{array_spec}%{plan.max_concurrent_tasks}"
    else:
        array_spec = ""
    payload["array_spec"] = array_spec
    payload["manifest_path"] = str(Path(manifest_path).expanduser().resolve()) if manifest_path is not None else None
    return payload


def render_array_submission_payload_from_manifest(manifest_path: str | Path) -> dict[str, object]:
    """Render a compact array submission payload directly from a manifest file."""
    payload = load_inversion_array_manifest(manifest_path)
    array_indices = tuple(int(index) for index in payload.get("array_indices", ()))
    max_concurrent_tasks = payload.get("max_concurrent_tasks")
    if array_indices:
        start = min(array_indices)
        end = max(array_indices)
        array_spec = f"{start}-{end}"
        if max_concurrent_tasks is not None:
            array_spec = f"{array_spec}%{max_concurrent_tasks}"
    else:
        array_spec = ""
    return {
        "step": payload["step"],
        "job_name": payload["job_name"],
        "sensor": payload["sensor"],
        "platform": payload["platform"],
        "tile": payload["tile"],
        "water_year": payload["water_year"],
        "task_count": payload["task_count"],
        "array_indices": list(array_indices),
        "max_concurrent_tasks": max_concurrent_tasks,
        "max_auto_retry_count": payload.get("max_auto_retry_count", 3),
        "apply_valid_inversion_mask": bool(payload.get("apply_valid_inversion_mask", False)),
        "use_grouping": bool(payload.get("use_grouping", True)),
        "grouping_method": str(payload.get("grouping_method", "chunk_bin_mean")),
        "r0_year": payload.get("r0_year"),
        "manifest_path": str(Path(manifest_path).expanduser().resolve()),
        "array_spec": array_spec,
        "slurm_profile": SlurmProfile.from_payload(payload.get("slurm_profile")).to_payload(),
    }


def render_sbatch_command_for_array_payload(
    payload: dict[str, object],
    *,
    repo_root: str | Path,
    python_executable: str = "python",
    execution_profile: str = "cluster",
    extra_sbatch_args: tuple[str, ...] = (),
) -> list[str]:
    """Render an `sbatch` command for one manifest-backed inversion array payload."""
    repo_root = Path(repo_root).expanduser().resolve()
    manifest_path = Path(str(payload["manifest_path"])).expanduser().resolve()
    slurm_profile = SlurmProfile.from_payload(payload.get("slurm_profile"))
    stdout_dir = manifest_path.parent if slurm_profile.output_dir is None else slurm_profile.output_dir
    slurm_stdout_path = stdout_dir.expanduser().resolve() / f"{payload['job_name']}_%A_%a.out"
    task_script = repo_root / "scripts" / "run_curc_inversion_array_task.py"
    wrapped_command = " ".join(
        [
            shlex.quote(str(python_executable)),
            shlex.quote(str(task_script)),
            shlex.quote(str(manifest_path)),
            "--execute",
            "--execution-profile",
            shlex.quote(str(execution_profile)),
            "--apply-valid-inversion-mask",
            str(bool(payload.get("apply_valid_inversion_mask", False))).lower(),
            "--use-grouping",
            str(bool(payload.get("use_grouping", True))).lower(),
            "--grouping-method",
            shlex.quote(str(payload.get("grouping_method", "chunk_bin_mean"))),
        ]
    )
    slurm_args: list[str] = []
    if slurm_profile.account:
        slurm_args.extend(["--account", slurm_profile.account])
    if slurm_profile.qos:
        slurm_args.extend(["--qos", slurm_profile.qos])
    if slurm_profile.time:
        slurm_args.extend(["--time", slurm_profile.time])
    if slurm_profile.mem:
        slurm_args.extend(["--mem", slurm_profile.mem])
    if slurm_profile.cpus_per_task is not None:
        slurm_args.extend(["--cpus-per-task", str(slurm_profile.cpus_per_task)])
    slurm_args.extend(slurm_profile.extra_args)
    slurm_args.extend(extra_sbatch_args)
    command = [
        "sbatch",
        "--parsable",
        "--job-name",
        str(payload["job_name"]),
        "--array",
        str(payload["array_spec"]),
        "--output",
        str(slurm_stdout_path),
        *slurm_args,
        "--wrap",
        wrapped_command,
    ]
    return command


def render_sbatch_command_for_finalize_wrap(
    *,
    job_name: str,
    wrapped_command: str,
    stdout_path: str | Path,
    slurm_profile: SlurmProfile | None = None,
    dependencies: tuple[str, ...] = (),
    extra_sbatch_args: tuple[str, ...] = (),
) -> list[str]:
    """Render an `sbatch` command for a single finalize-style wrapped command."""
    profile = SlurmProfile() if slurm_profile is None else slurm_profile
    slurm_args: list[str] = []
    if profile.account:
        slurm_args.extend(["--account", profile.account])
    if profile.qos:
        slurm_args.extend(["--qos", profile.qos])
    if profile.time:
        slurm_args.extend(["--time", profile.time])
    if profile.mem:
        slurm_args.extend(["--mem", profile.mem])
    if profile.cpus_per_task is not None:
        slurm_args.extend(["--cpus-per-task", str(profile.cpus_per_task)])
    slurm_args.extend(profile.extra_args)
    slurm_args.extend(extra_sbatch_args)
    if dependencies:
        slurm_args.extend(["--dependency", "afterany:" + ":".join(dependencies)])

    return [
        "sbatch",
        "--parsable",
        "--job-name",
        job_name,
        "--output",
        str(Path(stdout_path).expanduser().resolve()),
        *slurm_args,
        "--wrap",
        wrapped_command,
    ]
