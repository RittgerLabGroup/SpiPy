import numpy as np
import pytest
import xarray as xr

from spires import invert


class DummyInterpolator:
    def __init__(self):
        self.bands = np.array([1.0, 2.0], dtype=np.float64)
        self.solar_angles = np.array([30.0], dtype=np.float64)
        self.dust_concentrations = np.array([0.0], dtype=np.float64)
        self.grain_sizes = np.array([250.0], dtype=np.float64)
        self.reflectances = np.ones((2, 1, 1, 1), dtype=np.float64)


def test_speedy_invert_dask_passes_spectrum_shade_to_array2d(monkeypatch):
    pytest.importorskip("dask.array")
    captured = {}

    def fake_speedy_invert_array2d(
        *,
        spectra_targets,
        spectra_backgrounds,
        obs_solar_angles,
        spectrum_shade,
        bands,
        solar_angles,
        dust_concentrations,
        grain_sizes,
        reflectances,
        max_eval,
        x0,
        algorithm,
    ):
        captured["spectrum_shade"] = spectrum_shade
        return np.zeros(spectra_targets.shape[:2] + (4,), dtype=np.float32)

    monkeypatch.setattr(invert, "speedy_invert_array2d", fake_speedy_invert_array2d)

    spectra_targets = xr.DataArray(
        np.full((2, 2, 2), 0.2, dtype=np.float32),
        dims=("y", "x", "band"),
        coords={"y": [0, 1], "x": [0, 1], "band": ["b1", "b2"]},
    ).chunk({"y": 1, "x": 1, "band": -1})
    spectra_backgrounds = xr.DataArray(
        np.full((2, 2, 2), 0.1, dtype=np.float32),
        dims=("y", "x", "band"),
        coords=spectra_targets.coords,
    ).chunk({"y": 1, "x": 1, "band": -1})
    obs_solar_angles = xr.DataArray(
        np.full((2, 2), 30.0, dtype=np.float32),
        dims=("y", "x"),
        coords={"y": [0, 1], "x": [0, 1]},
    ).chunk({"y": 1, "x": 1})
    spectrum_shade = np.array([0.0, 0.0], dtype=np.float64)

    result = invert.speedy_invert_dask(
        spectra_targets=spectra_targets,
        spectra_backgrounds=spectra_backgrounds,
        obs_solar_angles=obs_solar_angles,
        interpolator=DummyInterpolator(),
        spectrum_shade=spectrum_shade,
        scatter_lut=False,
    )
    computed = result.compute()

    np.testing.assert_array_equal(captured["spectrum_shade"], spectrum_shade)
    assert computed["fsca"].shape == (2, 2)
