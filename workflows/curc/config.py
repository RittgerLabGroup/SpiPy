"""Configuration models for CURC-specific SpiPy workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from spires.sensors.registry import normalize_platform_name, normalize_sensor_name


@dataclass(frozen=True)
class SlurmProfile:
    """User-configurable Slurm resource settings for CURC submissions."""

    account: str | None = None
    qos: str | None = None
    time: str | None = None
    mem: str | None = None
    cpus_per_task: int | None = None
    output_dir: Path | None = None
    extra_args: tuple[str, ...] = ()

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-serializable representation for manifests and previews."""
        return {
            "account": self.account,
            "qos": self.qos,
            "time": self.time,
            "mem": self.mem,
            "cpus_per_task": self.cpus_per_task,
            "output_dir": None if self.output_dir is None else str(self.output_dir),
            "extra_args": list(self.extra_args),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, object] | None) -> "SlurmProfile":
        """Build a profile from manifest or preview payload data."""
        if payload is None:
            return cls()
        output_dir = payload.get("output_dir")
        return cls(
            account=payload.get("account"),
            qos=payload.get("qos"),
            time=payload.get("time"),
            mem=payload.get("mem"),
            cpus_per_task=payload.get("cpus_per_task"),
            output_dir=None if output_dir is None else Path(str(output_dir)),
            extra_args=tuple(str(arg) for arg in payload.get("extra_args", ())),
        )


@dataclass(frozen=True)
class CurcWorkflowConfig:
    """User-editable settings for CURC sensor workflow planning and submission."""

    scratch_root: Path
    input_source_root: Path | None
    sensor: str
    platforms: tuple[str, ...]
    tiles: tuple[str, ...]
    years: tuple[int, ...]
    water_years: tuple[int, ...] = ()
    dates: tuple[str, ...] = ()
    date_glob: str = "*"
    dry_run: bool = True
    max_auto_retry_count: int = 3
    apply_valid_inversion_mask: bool = False
    use_grouping: bool = True
    grouping_method: str = "chunk_bin_mean"
    slurm_profile: SlurmProfile = field(default_factory=SlurmProfile)

    def canonicalized(self) -> "CurcWorkflowConfig":
        """Return a copy using canonical SpiPy sensor/platform names."""
        sensor = normalize_sensor_name(self.sensor)
        platforms = tuple(normalize_platform_name(sensor, platform) for platform in self.platforms)
        return CurcWorkflowConfig(
            scratch_root=self.scratch_root,
            input_source_root=self.input_source_root,
            sensor=sensor,
            platforms=platforms,
            tiles=self.tiles,
            years=self.years,
            water_years=self.water_years,
            dates=self.dates,
            date_glob=self.date_glob,
            dry_run=self.dry_run,
            max_auto_retry_count=self.max_auto_retry_count,
            apply_valid_inversion_mask=self.apply_valid_inversion_mask,
            use_grouping=self.use_grouping,
            grouping_method=self.grouping_method,
            slurm_profile=self.slurm_profile,
        )
