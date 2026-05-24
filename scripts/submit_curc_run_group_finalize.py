#!/usr/bin/env python3
"""Submit or preview a CURC run-group finalizer Slurm job."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import shlex
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.curc.slurm import render_sbatch_command_for_finalize_wrap
from workflows.curc.status import list_run_group_tile_manifests
from workflows.curc.task_manifest import load_inversion_array_manifest
from workflows.curc.config import SlurmProfile


def _parse_dependencies(text: str | None) -> tuple[str, ...]:
    if not text:
        return ()
    return tuple(token.strip() for token in text.split(",") if token.strip())


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(
            "usage: submit_curc_run_group_finalize.py <run_group_dir> "
            "[--dependencies <jobid,jobid,...>] [--submit] [--python-exec <python>] [--sbatch-arg <arg> ...]",
            file=sys.stderr,
        )
        return 2

    run_group_dir = Path(argv[1]).expanduser().resolve()
    dependencies: tuple[str, ...] = ()
    submit = False
    python_exec = "python"
    extra_sbatch_args: list[str] = []

    i = 2
    while i < len(argv):
        token = argv[i]
        if token == "--dependencies":
            i += 1
            dependencies = _parse_dependencies(argv[i])
        elif token == "--submit":
            submit = True
        elif token == "--python-exec":
            i += 1
            python_exec = argv[i]
        elif token == "--sbatch-arg":
            i += 1
            extra_sbatch_args.append(argv[i])
        else:
            raise ValueError(f"Unexpected argument: {token}")
        i += 1

    manifests = list_run_group_tile_manifests(run_group_dir)
    if not manifests:
        raise ValueError(f"No tile manifests found under run group: {run_group_dir}")
    first_payload = load_inversion_array_manifest(manifests[0])
    profile = first_payload.get("slurm_profile", {})
    job_name = f"spipy-curc-finalize-{run_group_dir.name}"
    stdout_path = run_group_dir / f"{job_name}_%j.out"
    finalize_script = REPO_ROOT / "scripts" / "finalize_curc_run_group_summary.py"
    wrapped = " ".join([shlex.quote(str(python_exec)), shlex.quote(str(finalize_script)), shlex.quote(str(run_group_dir))])
    sbatch_command = render_sbatch_command_for_finalize_wrap(
        job_name=job_name,
        wrapped_command=wrapped,
        stdout_path=stdout_path,
        slurm_profile=SlurmProfile.from_payload(profile),
        dependencies=dependencies,
        extra_sbatch_args=tuple(extra_sbatch_args),
    )

    result: dict[str, object] = {
        "run_group_dir": str(run_group_dir),
        "job_name": job_name,
        "dependencies": list(dependencies),
        "sbatch_command": sbatch_command,
        "submitted": False,
    }

    if submit:
        completed = subprocess.run(sbatch_command, check=True, text=True, capture_output=True)
        result["submitted"] = True
        result["submitted_at_utc"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        result["sbatch_stdout"] = completed.stdout.strip()
        result["sbatch_stderr"] = completed.stderr.strip()

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
