"""Example CURC workflow config for VIIRS SNPP tile h09v05 and water year 2024."""

from pathlib import Path

from workflows.curc.config import CurcWorkflowConfig, SlurmProfile


CONFIG = CurcWorkflowConfig(
    scratch_root=Path("/scratch/alpine/ropa5718/spipy"),
    input_source_root=Path("/pl/active/rittger_ops/vnp09ga.002"),
    sensor="viirs",
    platforms=("snpp",),
    tiles=("h09v05",),
    years=(),
    water_years=(2024,),
    dates=(),
    date_glob="*",
    dry_run=True,
    max_auto_retry_count=3,
    slurm_profile=SlurmProfile(
        account="blanca-rittger",
        qos="preemptable",
        time="15:00:00",
        mem="44G",
        cpus_per_task=1,
        output_dir=Path("/projects/ropa5718/slurm_out"),
    ),
)
