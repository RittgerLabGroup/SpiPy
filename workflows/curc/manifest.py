"""Deterministic job manifest generation for CURC SpiPy workflows."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from workflows.curc.config import CurcWorkflowConfig
from workflows.curc.paths import ancillary_dir, job_log_dir, output_tile_root, r0_dir, reflectance_dir


@dataclass(frozen=True)
class PlannedJob:
    """A single planned workflow unit suitable for future Slurm submission."""

    sensor: str
    platform: str
    tile: str
    year: int
    reflectance_dir: str
    ancillary_dir: str
    r0_dir: str
    output_root: str
    log_dir: str
    input_source_root: str | None
    date_glob: str


def build_job_manifest(config: CurcWorkflowConfig) -> list[PlannedJob]:
    """Expand a config into a stable list of `(platform, tile, year)` jobs."""
    canonical = config.canonicalized()
    jobs: list[PlannedJob] = []
    for platform, tile, year in product(canonical.platforms, canonical.tiles, canonical.years):
        jobs.append(
            PlannedJob(
                sensor=canonical.sensor,
                platform=platform,
                tile=tile,
                year=year,
                reflectance_dir=str(reflectance_dir(canonical, platform)),
                ancillary_dir=str(ancillary_dir(canonical, platform)),
                r0_dir=str(r0_dir(canonical, platform)),
                output_root=str(output_tile_root(canonical, platform, tile)),
                log_dir=str(job_log_dir(canonical, platform, tile, year)),
                input_source_root=(
                    str(canonical.input_source_root) if canonical.input_source_root is not None else None
                ),
                date_glob=canonical.date_glob,
            )
        )
    return jobs
