from pathlib import Path

import pytest

from workflows.curc.config import CurcWorkflowConfig
from workflows.curc.discovery import discover_viirs_water_year_reflectance_files
from workflows.curc.planner import plan_viirs_snpp_inversion_array, plan_viirs_snpp_workflow_steps


def _touch_viirs_file(
    root: Path,
    *,
    tile: str,
    year: int,
    doy: int,
    product: str = "VNP09GA",
) -> Path:
    path = root / "input" / tile / str(year) / f"{product}.A{year}{doy:03d}.{tile}.002.2023101231908.h5"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def _build_config(tmp_path: Path, *, platform: str = "snpp") -> CurcWorkflowConfig:
    return CurcWorkflowConfig(
        scratch_root=tmp_path / "scratch",
        input_source_root=tmp_path / "source",
        sensor="viirs",
        platforms=(platform,),
        tiles=("h08v05",),
        years=(2023,),
        water_years=(2023,),
    )


def test_single_date_workflow_plan_uses_previous_summer_for_r0(tmp_path):
    config = _build_config(tmp_path)
    source_root = Path(config.input_source_root)
    summer_june = _touch_viirs_file(source_root, tile="h08v05", year=2022, doy=152)
    summer_september = _touch_viirs_file(source_root, tile="h08v05", year=2022, doy=273)
    _touch_viirs_file(source_root, tile="h08v05", year=2022, doy=274)
    rerun_scene = _touch_viirs_file(source_root, tile="h08v05", year=2023, doy=75)

    steps = plan_viirs_snpp_workflow_steps(
        config,
        tile="h08v05",
        water_year=2023,
        target_dates=("2023-03-16",),
    )

    stage_reflectance, _, build_r0, run_inversion = steps

    assert stage_reflectance.dates == ("2023-03-16",)
    assert stage_reflectance.source_paths == (str(rerun_scene),)

    assert build_r0.r0_year == 2022
    assert build_r0.dates == ("2022-06-01", "2022-09-30")
    assert build_r0.source_paths == (str(summer_june), str(summer_september))
    assert build_r0.destination_path.endswith("/ancillary/r0/h08v05/2022")
    assert "2022-06-01 through 2022-09-30" in build_r0.notes[0]

    assert run_inversion.r0_year == 2022
    assert run_inversion.dates == ("2023-03-16",)


def test_water_year_workflow_and_array_plan_use_previous_summer_r0(tmp_path):
    config = _build_config(tmp_path)
    source_root = Path(config.input_source_root)
    summer_june = _touch_viirs_file(source_root, tile="h08v05", year=2022, doy=166)
    summer_august = _touch_viirs_file(source_root, tile="h08v05", year=2022, doy=227)
    water_year_october = _touch_viirs_file(source_root, tile="h08v05", year=2022, doy=274)
    water_year_march = _touch_viirs_file(source_root, tile="h08v05", year=2023, doy=75)

    steps = plan_viirs_snpp_workflow_steps(
        config,
        tile="h08v05",
        water_year=2023,
    )
    build_r0 = next(step for step in steps if step.step == "build_r0")
    run_inversion = next(step for step in steps if step.step == "run_inversion")

    assert build_r0.r0_year == 2022
    assert build_r0.dates == ("2022-06-15", "2022-08-15")
    assert build_r0.source_paths == (str(summer_june), str(summer_august))
    assert run_inversion.source_paths == (str(water_year_october), str(water_year_march))
    assert run_inversion.r0_year == 2022

    array_plan = plan_viirs_snpp_inversion_array(
        config,
        tile="h08v05",
        water_year=2023,
    )

    assert array_plan.r0_year == 2022
    assert tuple(task.date for task in array_plan.tasks) == ("2022-10-01", "2023-03-16")
    assert all(task.r0_year == 2022 for task in array_plan.tasks)
    assert all(Path(task.log_path).parent.name == "detailed_logs" for task in array_plan.tasks)


@pytest.mark.parametrize(
    ("platform", "product"),
    (
        ("noaa20", "VJ109GA"),
        ("noaa21", "VJ209GA"),
    ),
)
def test_viirs_non_snpp_platforms_use_platform_specific_products_and_paths(tmp_path, platform, product):
    config = _build_config(tmp_path, platform=platform)
    source_root = Path(config.input_source_root)
    water_year_scene = _touch_viirs_file(source_root, tile="h08v05", year=2022, doy=274, product=product)
    r0_scene = _touch_viirs_file(source_root, tile="h08v05", year=2022, doy=166, product=product)

    discovered = discover_viirs_water_year_reflectance_files(
        config,
        tile="h08v05",
        water_year=2023,
    )
    assert discovered == [water_year_scene]

    steps = plan_viirs_snpp_workflow_steps(
        config,
        tile="h08v05",
        water_year=2023,
    )
    stage_reflectance = next(step for step in steps if step.step == "stage_reflectance")
    build_r0 = next(step for step in steps if step.step == "build_r0")

    assert stage_reflectance.platform == platform
    assert stage_reflectance.source_paths == (str(water_year_scene),)
    assert stage_reflectance.destination_path.endswith(f"/input/viirs/{platform}/reflectance/h08v05/2023")
    assert product in stage_reflectance.notes[0]

    assert build_r0.platform == platform
    assert build_r0.source_paths == (str(r0_scene),)
    assert build_r0.destination_path.endswith(f"/input/viirs/{platform}/ancillary/r0/h08v05/2022")

    array_plan = plan_viirs_snpp_inversion_array(
        config,
        tile="h08v05",
        water_year=2023,
    )
    assert array_plan.platform == platform
    assert array_plan.job_name == f"spipy-viirs-{platform}-h08v05-wy2023"
    assert array_plan.tasks[0].output_path.endswith(f"/output/viirs/{platform}/h08v05/raw/wy2023")
