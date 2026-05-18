from pathlib import Path

from workflows.curc.config import CurcWorkflowConfig, SlurmProfile
from workflows.curc.slurm import (
    render_array_submission_payload_from_manifest,
    render_sbatch_command_for_array_payload,
)
from workflows.curc.task_manifest import write_inversion_array_manifest
from workflows.curc.steps import InversionTaskPlan, SlurmArrayPlan


def test_manifest_round_trip_preserves_slurm_profile(tmp_path: Path) -> None:
    plan = SlurmArrayPlan(
        step="run_inversion",
        job_name="spipy-viirs-snpp-h09v05-wy2024",
        sensor="viirs",
        platform="snpp",
        tile="h09v05",
        water_year=2024,
        task_count=1,
        array_indices=(0,),
        max_concurrent_tasks=1,
        max_auto_retry_count=3,
        tasks=(
            InversionTaskPlan(
                task_index=0,
                sensor="viirs",
                platform="snpp",
                tile="h09v05",
                water_year=2024,
                date="2024-04-03",
                source_paths=("/tmp/input.h5",),
                output_path="/tmp/output/2024-04-03",
                log_path=str(tmp_path / "logs" / "run_inversion_2024-04-03.log"),
                r0_year=2024,
                retry_count=0,
            ),
        ),
        slurm_profile=SlurmProfile(
            account="blanca-rittger",
            qos="preemptable",
            time="15:00:00",
            mem="44G",
            cpus_per_task=1,
            output_dir=tmp_path / "slurm_out",
            extra_args=("--mail-type=FAIL",),
        ),
        notes=(),
        r0_year=2024,
    )

    manifest_path = write_inversion_array_manifest(plan, manifest_path=tmp_path / "manifest.json")
    payload = render_array_submission_payload_from_manifest(manifest_path)

    assert payload["slurm_profile"] == {
        "account": "blanca-rittger",
        "qos": "preemptable",
        "time": "15:00:00",
        "mem": "44G",
        "cpus_per_task": 1,
        "output_dir": str(tmp_path / "slurm_out"),
        "extra_args": ["--mail-type=FAIL"],
    }


def test_render_sbatch_command_uses_profile_and_cli_overrides(tmp_path: Path) -> None:
    payload = {
        "job_name": "spipy-viirs-snpp-h09v05-wy2024",
        "array_spec": "0-4%2",
        "manifest_path": str(tmp_path / "manifest.json"),
        "slurm_profile": {
            "account": "blanca-rittger",
            "qos": "preemptable",
            "time": "15:00:00",
            "mem": "44G",
            "cpus_per_task": 1,
            "output_dir": str(tmp_path / "slurm_out"),
            "extra_args": ["--mail-type=FAIL"],
        },
    }

    command = render_sbatch_command_for_array_payload(
        payload,
        repo_root=tmp_path,
        python_executable="python",
        execution_profile="cluster",
        extra_sbatch_args=("--qos=debug",),
    )

    assert command[:9] == [
        "sbatch",
        "--parsable",
        "--job-name",
        "spipy-viirs-snpp-h09v05-wy2024",
        "--array",
        "0-4%2",
        "--output",
        str((tmp_path / "slurm_out" / "spipy-viirs-snpp-h09v05-wy2024_%A_%a.out").resolve()),
        "--account",
    ]
    assert "--qos" in command
    assert "preemptable" in command
    assert "--mail-type=FAIL" in command
    assert "--qos=debug" in command
    assert command[-2] == "--wrap"
    assert "run_curc_inversion_array_task.py" in command[-1]


def test_curc_workflow_config_canonicalized_preserves_slurm_profile() -> None:
    config = CurcWorkflowConfig(
        scratch_root=Path("/scratch/alpine/ropa5718/spipy"),
        input_source_root=Path("/pl/active/rittger_ops/vnp09ga.002"),
        sensor="viirs",
        platforms=("snpp",),
        tiles=("h09v05",),
        years=(),
        water_years=(2024,),
        slurm_profile=SlurmProfile(account="blanca-rittger", cpus_per_task=1),
    )

    canonical = config.canonicalized()

    assert canonical.slurm_profile == config.slurm_profile
