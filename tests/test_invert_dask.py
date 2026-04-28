import numpy as np
import pytest
import xarray as xr

from spires import invert
from spires.speedy_utol import group_spectra_rows


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
        valid_mask,
        use_grouping,
        grouping_method,
        grouping_tolerance,
        grouping_reflectance_tol,
        grouping_background_tol,
        grouping_solar_zenith_tol,
    ):
        captured["spectrum_shade"] = spectrum_shade
        captured["valid_mask"] = valid_mask
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
    np.testing.assert_array_equal(captured["valid_mask"], np.ones((1, 1), dtype=bool))
    assert computed["fsca"].shape == (2, 2)


def test_speedy_invert_array2d_grouping_broadcasts_results_and_skips_invalid(monkeypatch):
    captured = {}

    def fake_speedy_invert_array1d(
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
        captured["targets"] = spectra_targets.copy()
        captured["backgrounds"] = spectra_backgrounds.copy()
        captured["solar"] = obs_solar_angles.copy()
        n = spectra_targets.shape[0]
        return np.column_stack(
            [
                np.arange(n, dtype=np.float64),
                np.arange(n, dtype=np.float64) + 10.0,
                np.arange(n, dtype=np.float64) + 20.0,
                np.arange(n, dtype=np.float64) + 30.0,
            ]
        )

    monkeypatch.setattr(invert, "speedy_invert_array1d", fake_speedy_invert_array1d)

    spectra_targets = np.array(
        [
            [[0.20, 0.30], [0.20, 0.30]],
            [[0.80, 0.90], [0.50, 0.60]],
        ],
        dtype=np.float64,
    )
    spectra_backgrounds = np.array(
        [
            [[0.10, 0.10], [0.10, 0.10]],
            [[0.40, 0.40], [0.20, 0.20]],
        ],
        dtype=np.float64,
    )
    obs_solar_angles = np.array([[30.0, 30.0], [40.0, 50.0]], dtype=np.float64)
    valid_mask = np.array([[True, True], [True, False]])

    result = invert.speedy_invert_array2d(
        spectra_targets=spectra_targets,
        spectra_backgrounds=spectra_backgrounds,
        obs_solar_angles=obs_solar_angles,
        bands=np.array([1.0, 2.0]),
        solar_angles=np.array([30.0]),
        dust_concentrations=np.array([0.0]),
        grain_sizes=np.array([250.0]),
        reflectances=np.ones((2, 1, 1, 1), dtype=np.float64),
        valid_mask=valid_mask,
        use_grouping=True,
        grouping_method="first",
        grouping_tolerance=0.02,
    )

    assert captured["targets"].shape[0] == 2
    np.testing.assert_allclose(captured["targets"], np.array([[0.2, 0.3], [0.8, 0.9]]))
    np.testing.assert_allclose(captured["backgrounds"], np.array([[0.1, 0.1], [0.4, 0.4]]))
    np.testing.assert_allclose(captured["solar"], np.array([30.0, 40.0]))
    np.testing.assert_allclose(result[0, 0], np.array([0.0, 10.0, 20.0, 30.0]))
    np.testing.assert_allclose(result[0, 1], np.array([0.0, 10.0, 20.0, 30.0]))
    np.testing.assert_allclose(result[1, 0], np.array([1.0, 11.0, 21.0, 31.0]))
    assert np.isnan(result[1, 1]).all()


def test_group_spectra_rows_first_uses_first_member_representative():
    targets = np.array(
        [
            [0.201, 0.301],
            [0.209, 0.309],
            [0.700, 0.800],
        ],
        dtype=np.float64,
    )
    backgrounds = np.array(
        [
            [0.101, 0.111],
            [0.109, 0.119],
            [0.400, 0.500],
        ],
        dtype=np.float64,
    )
    solar = np.array([30.0, 31.0, 40.0], dtype=np.float64)

    grouped = group_spectra_rows(
        targets,
        backgrounds,
        solar,
        representative_method="first",
        tolerance=0.05,
    )

    assert grouped.n_groups == 2
    np.testing.assert_allclose(grouped.representative_targets[0], targets[0])
    np.testing.assert_allclose(grouped.representative_backgrounds[0], backgrounds[0])
    np.testing.assert_allclose(grouped.representative_solar_zenith[0], solar[0])
    np.testing.assert_array_equal(grouped.inverse_indices, np.array([0, 0, 1]))
