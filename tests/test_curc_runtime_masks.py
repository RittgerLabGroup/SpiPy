from pathlib import Path

import pytest

from workflows.curc.config import SlurmProfile
from workflows.curc.runtime import (
    build_viirs_snpp_inversion_runtime_context,
    summarize_viirs_snpp_runtime_requirements,
)
from workflows.curc.steps import InversionTaskPlan, SlurmArrayPlan
from workflows.curc.task_manifest import write_inversion_array_manifest


@pytest.mark.parametrize(
    ("platform", "scene_name"),
    (
        ("snpp", "VNP09GA.A2023274.h08v04.002.2023277125049.h5"),
        ("noaa20", "VJ109GA.A2023274.h08v04.002.2023277125049.h5"),
        ("noaa21", "VJ209GA.A2023274.h08v04.002.2023277125049.h5"),
    ),
)
def test_runtime_context_resolves_curc_mask_inputs(tmp_path: Path, platform: str, scene_name: str) -> None:
    scratch = tmp_path / "scratch"
    tile = "h08v04"
    water_year = 2024
    date = "2023-10-01"
    date_token = date.replace("-", "")

    reflectance_dir = scratch / "input" / "viirs" / platform / "reflectance" / tile / str(water_year)
    ancillary_root = scratch / "input" / "viirs" / platform / "ancillary" / tile
    cloud_dir = scratch / "input" / "viirs" / platform / "ancillary" / "cloud" / tile / str(water_year)
    r0_dir = scratch / "input" / "viirs" / platform / "ancillary" / "r0" / tile / "2023"
    output_dir = scratch / "output" / "viirs" / platform / tile / "raw" / f"wy{water_year}"
    log_dir = scratch / "logs" / "run_group" / tile / "detailed_logs"

    for directory in (reflectance_dir, ancillary_root, cloud_dir, r0_dir, output_dir, log_dir):
        directory.mkdir(parents=True)
    (reflectance_dir / scene_name).touch()
    (r0_dir / f"{platform}_r0_{tile}_2023.nc").touch()
    (ancillary_root / "canopy_fraction.tif").touch()
    (ancillary_root / f"{tile}_water_mod44_50.tif").touch()
    (ancillary_root / f"{tile}_ice_rgi60_202309.tif").touch()
    (ancillary_root / f"{tile}_stc_false_positive_manual_20230920_mod44_50.tif").touch()
    (ancillary_root / f"{tile}_slope_gmted_med075.tif").touch()
    (ancillary_root / f"{tile}_aspect_gmted_med075_ccw_from_south.tif").touch()
    (cloud_dir / f"{platform}_{tile}_{date_token}_cloud_mask.tif").touch()

    task = InversionTaskPlan(
        task_index=0,
        sensor="viirs",
        platform=platform,
        tile=tile,
        water_year=water_year,
        date=date,
        source_paths=(f"/source/{scene_name}",),
        output_path=str(output_dir),
        log_path=str(log_dir / f"run_inversion_{date}.log"),
        r0_year=2023,
    )
    plan = SlurmArrayPlan(
        step="run_inversion",
        job_name=f"spipy-viirs-{platform}-{tile}-wy{water_year}",
        sensor="viirs",
        platform=platform,
        tile=tile,
        water_year=water_year,
        task_count=1,
        array_indices=(0,),
        max_concurrent_tasks=None,
        max_auto_retry_count=3,
        apply_valid_inversion_mask=True,
        mask_low_reflectance_for_inversion=False,
        low_reflectance_threshold=0.1,
        include_grouped_reflectance_rmse=False,
        use_grouping=True,
        grouping_method="chunk_bin_mean",
        tasks=(task,),
        slurm_profile=SlurmProfile(),
        r0_year=2023,
    )
    manifest_path = write_inversion_array_manifest(plan, manifest_path=log_dir / "manifest.json")

    context = build_viirs_snpp_inversion_runtime_context(manifest_path, task_index=0)
    missing = summarize_viirs_snpp_runtime_requirements(context)

    assert context.cloud_mask_path == str(cloud_dir / f"{platform}_{tile}_{date_token}_cloud_mask.tif")
    assert context.water_mask_path == str(ancillary_root / f"{tile}_water_mod44_50.tif")
    assert context.ice_fraction_path == str(ancillary_root / f"{tile}_ice_rgi60_202309.tif")
    assert context.playa_mask_path == str(ancillary_root / f"{tile}_stc_false_positive_manual_20230920_mod44_50.tif")
    assert context.terrain_ancillary_root == str(scratch / "input" / "viirs" / platform / "ancillary")
    assert context.slope_path == str(ancillary_root / f"{tile}_slope_gmted_med075.tif")
    assert context.aspect_path == str(ancillary_root / f"{tile}_aspect_gmted_med075_ccw_from_south.tif")
    assert not any(missing.values())
