from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from spires.albedo import (
    ALBEDO_PRODUCT_NAMES,
    DELTA_VIS_PRODUCT_NAME,
    RADIATIVE_FORCING_PRODUCT_NAME,
    AlbedoLutFunction,
    AlbedoLuts,
    AlbedoProductFlags,
    RadiativeForcingLuts,
    derived_product_attrs,
    dust_ppm_to_lut_fraction,
    generate_albedo_products,
    grain_size_micrometers_to_lut_gs,
    solar_azimuth_to_phi0,
    sunslope,
)


def test_solar_azimuth_to_phi0_maps_to_sunslope_convention():
    source = np.array([0.0, 90.0, 180.0, 270.0, 360.0], dtype=np.float32)

    converted = solar_azimuth_to_phi0(source)

    np.testing.assert_allclose(converted, np.array([180.0, 90.0, 0.0, -90.0, -180.0]))


def test_solar_azimuth_to_phi0_preserves_xarray_metadata():
    source = xr.DataArray(
        np.array([[180.0, 270.0]], dtype=np.float32),
        dims=("y", "x"),
        coords={"y": [10], "x": [20, 21]},
    )

    converted = solar_azimuth_to_phi0(source)

    assert converted.dims == ("y", "x")
    assert list(converted.coords["x"].values) == [20, 21]
    np.testing.assert_allclose(converted.values, np.array([[0.0, -90.0]]))


def test_dust_conversion_uses_physical_ppm_only():
    dust_ppm = xr.DataArray(np.array([[0.0, 10.0, 250.0]], dtype=np.float32), dims=("y", "x"))

    dust_fraction = dust_ppm_to_lut_fraction(dust_ppm)

    np.testing.assert_allclose(dust_fraction.values, np.array([[0.0, 0.01, 0.25]]))


def test_grain_size_conversion_uses_square_root_coordinate():
    grain_size = np.array([30.0, 400.0, 900.0], dtype=np.float32)

    gs = grain_size_micrometers_to_lut_gs(grain_size)

    np.testing.assert_allclose(gs, np.array([np.sqrt(30.0), 20.0, 30.0], dtype=np.float32))


def test_sunslope_returns_mu0_for_flat_terrain():
    mu0 = np.array([0.25, 0.5, 0.75], dtype=np.float32)

    mu_z = sunslope(mu0, phi0=30.0, slope=0.0, aspect_ccw_from_south=120.0)

    np.testing.assert_allclose(mu_z, mu0)


def test_sunslope_simple_slope_cases_and_clipping():
    mu0 = np.cos(np.deg2rad(30.0))

    facing_sun = sunslope(mu0, phi0=45.0, slope=30.0, aspect_ccw_from_south=45.0)
    away_from_sun = sunslope(np.cos(np.deg2rad(75.0)), phi0=0.0, slope=75.0, aspect_ccw_from_south=180.0)
    above_one = sunslope(1.000001, phi0=0.0, slope=0.0, aspect_ccw_from_south=0.0)

    assert facing_sun == pytest.approx(1.0)
    assert away_from_sun == pytest.approx(0.0)
    assert above_one == pytest.approx(1.0)


def test_product_flags_default_to_all_requested_variables():
    flags = AlbedoProductFlags()

    assert flags.requested_variables == (
        *ALBEDO_PRODUCT_NAMES,
        RADIATIVE_FORCING_PRODUCT_NAME,
        DELTA_VIS_PRODUCT_NAME,
    )
    assert flags.requires_albedo_lut
    assert flags.requires_radiative_forcing_lut
    assert flags.requires_terrain


def test_product_flags_can_disable_individual_lut_requirements():
    flags = AlbedoProductFlags(include_albedo=False, include_radiative_forcing=False, include_delta_vis=True)

    assert flags.requested_variables == (DELTA_VIS_PRODUCT_NAME,)
    assert not flags.requires_albedo_lut
    assert flags.requires_radiative_forcing_lut
    assert not flags.requires_terrain


def test_product_attrs_define_units_for_outputs():
    assert derived_product_attrs("albedo_clean_flat")["units"] == "1"
    assert derived_product_attrs("delta_vis")["units"] == "1"
    assert derived_product_attrs("radiative_forcing")["units"] == "W m-2"

    with pytest.raises(ValueError, match="Unknown albedo product"):
        derived_product_attrs("not_a_product")


def test_requested_lut_path_validation_follows_enabled_flags(tmp_path):
    from spires.albedo import validate_requested_lut_paths

    rf_lut = tmp_path / "rf.mat"
    rf_lut.write_bytes(b"placeholder")
    flags = AlbedoProductFlags(include_albedo=False, include_radiative_forcing=False, include_delta_vis=True)

    validate_requested_lut_paths(flags, radiative_forcing_lut_path=rf_lut)

    with pytest.raises(FileNotFoundError, match="albedo LUT"):
        validate_requested_lut_paths(
            AlbedoProductFlags(include_albedo=True, include_radiative_forcing=False, include_delta_vis=False),
            albedo_lut_path=tmp_path / "missing.mat",
        )

    with pytest.raises(ValueError, match="LUT path"):
        validate_requested_lut_paths(AlbedoProductFlags(include_albedo=False, include_delta_vis=True))


def _lut_function(name, dimensions, points, scale):
    mesh = np.meshgrid(*points, indexing="ij")
    values = np.zeros_like(mesh[0], dtype=np.float32)
    for i, coordinate in enumerate(mesh):
        values = values + (i + 1) * scale * coordinate.astype(np.float32)
    return AlbedoLutFunction(
        name=name,
        points=tuple(np.asarray(point, dtype=np.float32) for point in points),
        values=values.astype(np.float32),
        dimensions=dimensions,
    )


def _synthetic_luts():
    mu0 = np.array([0.5, 1.0], dtype=np.float32)
    muz = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    gs = np.array([10.0, 20.0], dtype=np.float32)
    dust = np.array([0.0, 0.01], dtype=np.float32)
    soot = np.array([0.0, 1.0e-6], dtype=np.float32)
    albedo = AlbedoLuts(
        clean=_lut_function("Fclean", ("mu0", "muZ", "gs"), (mu0, muz, gs), 0.01),
        dirty=_lut_function("Fdirty", ("mu0", "muZ", "gs", "dust", "soot"), (mu0, muz, gs, dust, soot), 0.02),
    )
    rf = RadiativeForcingLuts(
        darken=_lut_function("Fdarken", ("mu0", "muZ", "gs", "dust", "soot"), (mu0, muz, gs, dust, soot), 0.03),
        force=_lut_function("Fforce", ("mu0", "muZ", "gs", "dust", "soot"), (mu0, muz, gs, dust, soot), 0.04),
    )
    return albedo, rf


def _inversion_dataset():
    coords = {"y": [0], "x": [0]}
    return xr.Dataset(
        data_vars={
            "grain_size": xr.DataArray(np.array([[100.0]], dtype=np.float32), dims=("y", "x"), coords=coords),
            "dust_concentration": xr.DataArray(np.array([[10.0]], dtype=np.float32), dims=("y", "x"), coords=coords),
            "solar_zenith": xr.DataArray(np.array([[0.0]], dtype=np.float32), dims=("y", "x"), coords=coords),
            "solar_azimuth": xr.DataArray(np.array([[180.0]], dtype=np.float32), dims=("y", "x"), coords=coords),
        }
    )


def test_lut_function_interpolates_and_returns_nan_outside_bounds():
    function = _lut_function(
        "test",
        ("x", "y"),
        (np.array([0.0, 1.0], dtype=np.float32), np.array([0.0, 1.0], dtype=np.float32)),
        1.0,
    )
    x = xr.DataArray(np.array([[0.5, 2.0]], dtype=np.float32), dims=("y", "x"))
    y = xr.DataArray(np.array([[0.25, 0.25]], dtype=np.float32), dims=("y", "x"))

    result = function.evaluate(x, y)

    assert result.dims == ("y", "x")
    assert result.values[0, 0] == pytest.approx(1.0)
    assert np.isnan(result.values[0, 1])


def test_generate_albedo_products_maps_requested_luts_and_attrs():
    albedo, rf = _synthetic_luts()
    dataset = _inversion_dataset()
    slope = xr.zeros_like(dataset["grain_size"])
    aspect = xr.zeros_like(dataset["grain_size"])

    products = generate_albedo_products(
        dataset,
        albedo_luts=albedo,
        radiative_forcing_luts=rf,
        slope=slope,
        aspect=aspect,
    )

    assert set(products.data_vars) == {
        *ALBEDO_PRODUCT_NAMES,
        RADIATIVE_FORCING_PRODUCT_NAME,
        DELTA_VIS_PRODUCT_NAME,
    }
    np.testing.assert_allclose(products["albedo_clean_flat"], [[0.33]])
    np.testing.assert_allclose(products["albedo_dirty_flat"], [[0.6608]], rtol=1e-6)
    np.testing.assert_allclose(products["delta_vis"], [[0.9912]], rtol=1e-6)
    np.testing.assert_allclose(products["radiative_forcing"], [[1.3216]], rtol=1e-6)
    assert products["radiative_forcing"].attrs["units"] == "W m-2"
    assert products["delta_vis"].attrs["units"] == "1"


def test_generate_albedo_products_respects_disabled_product_flags():
    _, rf = _synthetic_luts()

    products = generate_albedo_products(
        _inversion_dataset(),
        flags=AlbedoProductFlags(include_albedo=False, include_radiative_forcing=False, include_delta_vis=True),
        radiative_forcing_luts=rf,
    )

    assert list(products.data_vars) == [DELTA_VIS_PRODUCT_NAME]


def test_generate_albedo_products_requires_terrain_for_albedo():
    albedo, _ = _synthetic_luts()

    with pytest.raises(ValueError, match="Terrain-corrected albedo"):
        generate_albedo_products(
            _inversion_dataset(),
            flags=AlbedoProductFlags(include_radiative_forcing=False, include_delta_vis=False),
            albedo_luts=albedo,
        )
