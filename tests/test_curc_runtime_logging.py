import json
import logging
from pathlib import Path

from spires.logging_utils import configure_spires_file_logger, log_event, remove_empty_log_file
from workflows.curc.paths import detailed_log_dir, top_level_log_dir
from workflows.curc.runtime import (
    resolve_runtime_task_log_path,
    resolve_slurm_stdout_path,
    resolve_water_year_aggregate_log_path,
)
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


def test_resolve_slurm_stdout_path_from_manifest(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "job_name": "spipy-viirs-snpp-h08v05-wy2023",
                "slurm_profile": {"output_dir": str(tmp_path / "slurm_out")},
            }
        ),
        encoding="utf-8",
    )

    resolved = resolve_slurm_stdout_path(
        manifest_path,
        slurm_job_name="spipy-viirs-snpp-h08v05-wy2023",
        slurm_array_job_id="26186675",
        slurm_array_task_id="350",
    )

    assert resolved == (tmp_path / "slurm_out" / "spipy-viirs-snpp-h08v05-wy2023_26186675_350.out")


def test_detailed_and_top_level_log_dir_helpers(tmp_path: Path) -> None:
    top = tmp_path / "20260519_143044"
    detailed = detailed_log_dir(top)

    assert detailed == top / "detailed_logs"
    assert top_level_log_dir(detailed) == top
    assert top_level_log_dir(detailed / "run_inversion_2023-03-16.log") == top


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


def test_remove_empty_log_file_deletes_only_zero_byte_files(tmp_path: Path) -> None:
    empty = tmp_path / "empty.out"
    empty.write_text("", encoding="utf-8")
    nonempty = tmp_path / "nonempty.out"
    nonempty.write_text("content", encoding="utf-8")

    assert remove_empty_log_file(empty) is True
    assert empty.exists() is False
    assert remove_empty_log_file(nonempty) is False
    assert nonempty.exists() is True


def test_plain_file_logger_keeps_separator_before_start_and_after_summary(tmp_path: Path) -> None:
    log_path = tmp_path / "runtime.log"
    logger = configure_spires_file_logger(
        log_path,
        logger_name="spires.test.runtime.separators",
        log_to_stdout=False,
        aggregate_log_path=None,
    )

    log_event(
        logger,
        "curc_run_viirs_snpp_inversion_task",
        level=logging.INFO,
        stage="curc_runtime",
        event_type="start",
        status="started",
        scope=True,
        date="2023-03-16",
    )
    log_event(
        logger,
        "curc_run_viirs_snpp_inversion_task",
        level=logging.INFO,
        stage="curc_runtime",
        event_type="summary",
        status="completed",
        scope=True,
        date="2023-03-16",
    )

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4
    assert set(lines[0]) == {"="}
    assert "====== START ======" in lines[1]
    assert "====== SUMMARY ======" in lines[2]
    assert set(lines[3]) == {"="}
