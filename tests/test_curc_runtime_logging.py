from pathlib import Path

from workflows.curc.runtime import resolve_runtime_task_log_path, resolve_water_year_aggregate_log_path
from workflows.curc.status import _resolve_task_log_path
from workflows.curc.steps import InversionTaskPlan


def _task(tmp_path: Path) -> InversionTaskPlan:
    return InversionTaskPlan(
        task_index=0,
        sensor="viirs",
        platform="snpp",
        tile="h08v05",
        water_year=2023,
        date="2023-03-16",
        source_paths=("/tmp/input.h5",),
        output_path="/tmp/output/2023-03-16",
        log_path=str(tmp_path / "run_inversion_2023-03-16.log"),
        r0_year=2022,
        retry_count=0,
    )


def test_runtime_log_path_includes_jobid_when_present(tmp_path: Path) -> None:
    task = _task(tmp_path)
    resolved = resolve_runtime_task_log_path(task, slurm_job_id="26199999")
    assert resolved.name == "run_inversion_2023-03-16_job26199999.log"


def test_runtime_log_path_uses_legacy_name_without_jobid(tmp_path: Path) -> None:
    task = _task(tmp_path)
    resolved = resolve_runtime_task_log_path(task, slurm_job_id=None)
    assert resolved.name == "run_inversion_2023-03-16.log"


def test_water_year_aggregate_log_path(tmp_path: Path) -> None:
    task = _task(tmp_path)
    resolved = resolve_water_year_aggregate_log_path(task)
    assert resolved.name == "run_inversion_wy2023_aggregate.log"
    assert resolved.parent == Path(task.log_path).parent


def test_status_prefers_newest_job_specific_log(tmp_path: Path) -> None:
    task = _task(tmp_path)
    base = Path(task.log_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    first = base.with_name("run_inversion_2023-03-16_job111.log")
    second = base.with_name("run_inversion_2023-03-16_job222.log")
    first.write_text("old", encoding="utf-8")
    second.write_text("new", encoding="utf-8")
    first.touch()
    second.touch()

    resolved = _resolve_task_log_path(task)
    assert resolved.name == "run_inversion_2023-03-16_job222.log"


def test_status_falls_back_to_legacy_log(tmp_path: Path) -> None:
    task = _task(tmp_path)
    legacy = Path(task.log_path)
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text("legacy", encoding="utf-8")

    resolved = _resolve_task_log_path(task)
    assert resolved == legacy
