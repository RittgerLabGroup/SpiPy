from pathlib import Path

import xarray as xr

from workflows.curc.config import CurcWorkflowConfig
from workflows.curc.execution import execute_viirs_snpp_workflow_step, preview_viirs_snpp_workflow_step_execution
from workflows.curc.paths import r0_dataset_path
from workflows.curc.steps import WorkflowStepPlan


def _build_config(tmp_path: Path) -> CurcWorkflowConfig:
    return CurcWorkflowConfig(
        scratch_root=tmp_path / "scratch",
        input_source_root=tmp_path / "source",
        sensor="viirs",
        platforms=("snpp",),
        tiles=("h08v05",),
        years=(2023,),
        water_years=(2023,),
    )


def test_preview_stage_reflectance_renders_rsync_commands(tmp_path):
    config = _build_config(tmp_path)
    source_a = tmp_path / "source" / "input" / "h08v05" / "2022" / "VNP09GA.A2022274.h08v05.002.2023101231908.h5"
    source_b = tmp_path / "source" / "input" / "h08v05" / "2023" / "VNP09GA.A2023075.h08v05.002.2023101231908.h5"
    step = WorkflowStepPlan(
        step="stage_reflectance",
        sensor="viirs",
        platform="snpp",
        tile="h08v05",
        water_year=2023,
        date_count=2,
        dates=("2022-10-01", "2023-03-16"),
        source_paths=(str(source_a), str(source_b)),
        destination_path=str(tmp_path / "scratch" / "input" / "viirs" / "snpp" / "reflectance" / "h08v05" / "2023"),
    )

    preview = preview_viirs_snpp_workflow_step_execution(config, step)

    assert preview["mode"] == "direct_rsync"
    assert len(preview["commands"]) == 2
    assert preview["commands"][0][0] == "rsync"
    assert preview["shell_commands"][0].startswith("rsync -av --ignore-existing")


def test_execute_stage_ancillary_creates_expected_directories(tmp_path):
    config = _build_config(tmp_path)
    step = WorkflowStepPlan(
        step="stage_ancillary",
        sensor="viirs",
        platform="snpp",
        tile="h08v05",
        water_year=2023,
        date_count=1,
        dates=("2023-03-16",),
        source_paths=(),
        destination_path=str(tmp_path / "scratch" / "input" / "viirs" / "snpp" / "ancillary" / "h08v05"),
        r0_year=2022,
    )

    result = execute_viirs_snpp_workflow_step(config, step, execute=True)

    assert result["executed"] is True
    for path in result["created_paths"]:
        assert Path(path).exists()


def test_execute_build_r0_calls_builder(monkeypatch, tmp_path):
    config = _build_config(tmp_path)
    source_path = tmp_path / "source" / "input" / "h08v05" / "2022" / "VNP09GA.A2022152.h08v05.002.2023101231908.h5"
    step = WorkflowStepPlan(
        step="build_r0",
        sensor="viirs",
        platform="snpp",
        tile="h08v05",
        water_year=2023,
        date_count=1,
        dates=("2022-06-01",),
        source_paths=(str(source_path),),
        destination_path=str(tmp_path / "scratch" / "input" / "viirs" / "snpp" / "ancillary" / "r0" / "h08v05" / "2022"),
        r0_year=2022,
    )
    captured = {}

    def fake_build_r0_from_sources(sources, **kwargs):
        captured["sources"] = sources
        captured["kwargs"] = kwargs
        return xr.Dataset(
            data_vars={"r0_reflectance": (("y", "x", "band"), [[[0.1]]])},
            coords={"y": [0], "x": [0], "band": ["I1"]},
            attrs={"build_status": "complete"},
        )

    monkeypatch.setattr("workflows.curc.execution.build_r0_from_sources", fake_build_r0_from_sources)

    result = execute_viirs_snpp_workflow_step(config, step, execute=True, overwrite=True)
    expected_output_path = r0_dataset_path(config, "snpp", "h08v05", 2022).resolve()

    assert result["executed"] is True
    assert captured["sources"] == [str(source_path)]
    assert captured["kwargs"]["overwrite"] is True
    assert captured["kwargs"]["r0_path"] == expected_output_path
    assert result["output_dataset_path"] == str(expected_output_path)


def test_preview_build_r0_includes_optional_zarr_settings(tmp_path):
    config = _build_config(tmp_path)
    source_path = tmp_path / "source" / "input" / "h08v05" / "2022" / "VNP09GA.A2022152.h08v05.002.2023101231908.h5"
    step = WorkflowStepPlan(
        step="build_r0",
        sensor="viirs",
        platform="snpp",
        tile="h08v05",
        water_year=2023,
        date_count=1,
        dates=("2022-06-01",),
        source_paths=(str(source_path),),
        destination_path=str(tmp_path / "scratch" / "input" / "viirs" / "snpp" / "ancillary" / "r0" / "h08v05" / "2022"),
        r0_year=2022,
    )
    zarr_path = tmp_path / "scratch" / "tmp" / "viirs_r0_stack.zarr"
    chunks = {"time": 1, "y": 256, "x": 256, "band": -1}

    preview = preview_viirs_snpp_workflow_step_execution(config, step, zarr_path=zarr_path, chunks=chunks)

    assert preview["mode"] == "python_r0_builder"
    assert preview["ndvi_tie_epsilon"] == 0.02
    assert preview["zarr_path"] == str(zarr_path.resolve())
    assert preview["chunks"] == chunks


def test_preview_build_r0_includes_custom_ndvi_tie_epsilon(tmp_path):
    config = _build_config(tmp_path)
    source_path = tmp_path / "source" / "input" / "h08v05" / "2022" / "VNP09GA.A2022152.h08v05.002.2023101231908.h5"
    step = WorkflowStepPlan(
        step="build_r0",
        sensor="viirs",
        platform="snpp",
        tile="h08v05",
        water_year=2023,
        date_count=1,
        dates=("2022-06-01",),
        source_paths=(str(source_path),),
        destination_path=str(tmp_path / "scratch" / "input" / "viirs" / "snpp" / "ancillary" / "r0" / "h08v05" / "2022"),
        r0_year=2022,
    )

    preview = preview_viirs_snpp_workflow_step_execution(config, step, ndvi_tie_epsilon=0.015)

    assert preview["mode"] == "python_r0_builder"
    assert preview["ndvi_tie_epsilon"] == 0.015


def test_execute_build_r0_passes_optional_zarr_settings(monkeypatch, tmp_path):
    config = _build_config(tmp_path)
    source_path = tmp_path / "source" / "input" / "h08v05" / "2022" / "VNP09GA.A2022152.h08v05.002.2023101231908.h5"
    step = WorkflowStepPlan(
        step="build_r0",
        sensor="viirs",
        platform="snpp",
        tile="h08v05",
        water_year=2023,
        date_count=1,
        dates=("2022-06-01",),
        source_paths=(str(source_path),),
        destination_path=str(tmp_path / "scratch" / "input" / "viirs" / "snpp" / "ancillary" / "r0" / "h08v05" / "2022"),
        r0_year=2022,
    )
    zarr_path = tmp_path / "scratch" / "tmp" / "viirs_r0_stack.zarr"
    chunks = {"time": 1, "y": 256, "x": 256, "band": -1}
    captured = {}

    def fake_build_r0_from_sources(sources, **kwargs):
        captured["sources"] = sources
        captured["kwargs"] = kwargs
        return xr.Dataset(
            data_vars={"r0_reflectance": (("y", "x", "band"), [[[0.1]]])},
            coords={"y": [0], "x": [0], "band": ["I1"]},
            attrs={"build_status": "complete"},
        )

    monkeypatch.setattr("workflows.curc.execution.build_r0_from_sources", fake_build_r0_from_sources)

    result = execute_viirs_snpp_workflow_step(config, step, execute=True, zarr_path=zarr_path, chunks=chunks)

    assert result["executed"] is True
    assert captured["kwargs"]["zarr_path"] == zarr_path
    assert captured["kwargs"]["chunks"] == chunks


def test_execute_build_r0_passes_ndvi_tie_epsilon(monkeypatch, tmp_path):
    config = _build_config(tmp_path)
    source_path = tmp_path / "source" / "input" / "h08v05" / "2022" / "VNP09GA.A2022152.h08v05.002.2023101231908.h5"
    step = WorkflowStepPlan(
        step="build_r0",
        sensor="viirs",
        platform="snpp",
        tile="h08v05",
        water_year=2023,
        date_count=1,
        dates=("2022-06-01",),
        source_paths=(str(source_path),),
        destination_path=str(tmp_path / "scratch" / "input" / "viirs" / "snpp" / "ancillary" / "r0" / "h08v05" / "2022"),
        r0_year=2022,
    )
    captured = {}

    def fake_build_r0_from_sources(sources, **kwargs):
        captured["sources"] = sources
        captured["kwargs"] = kwargs
        return xr.Dataset(
            data_vars={"r0_reflectance": (("y", "x", "band"), [[[0.1]]])},
            coords={"y": [0], "x": [0], "band": ["I1"]},
            attrs={"build_status": "complete"},
        )

    monkeypatch.setattr("workflows.curc.execution.build_r0_from_sources", fake_build_r0_from_sources)

    result = execute_viirs_snpp_workflow_step(config, step, execute=True, ndvi_tie_epsilon=0.015)

    assert result["executed"] is True
    assert captured["kwargs"]["ndvi_tie_epsilon"] == 0.015
