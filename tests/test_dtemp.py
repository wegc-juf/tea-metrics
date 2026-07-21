import numpy as np
import pandas as pd
import xarray as xr
import pytest

from teametrics.TEA import TEAIndicators


EXPECTED_CRS = "EPSG:4326"


def _make_input_grid():
    y = np.array([47.0, 48.0])
    x = np.array([15.0, 16.0])
    values = np.ones((1, len(y), len(x)))
    data = xr.DataArray(
        values,
        coords={"time": [pd.Timestamp("1980-01-01")], "lat": y, "lon": x},
        dims=("time", "lat", "lon"),
        name="input",
    )
    data.attrs["coordinate_sys"] = EXPECTED_CRS

    threshold = xr.DataArray(
        np.full((len(y), len(x)), 0.5),
        coords={"lat": y, "lon": x},
        dims=("lat", "lon"),
        name="threshold",
    )
    threshold.attrs["coordinate_sys"] = EXPECTED_CRS
    return data, threshold


def _make_daily_input():
    times = pd.date_range("1980-01-01", "1994-12-31", freq="D")
    y = np.array([47.0, 48.0])
    x = np.array([15.0, 16.0])

    data = xr.DataArray(
        np.full((len(times), len(y), len(x)), 10.0),
        coords={"time": times, "lat": y, "lon": x},
        dims=("time", "lat", "lon"),
        name="input",
    )
    data.attrs["coordinate_sys"] = EXPECTED_CRS
    return data


def _make_population_grid():
    y = np.array([47.0, 48.0])
    x = np.array([15.0, 16.0])
    pop = xr.DataArray(
        np.array([[100.0, 200.0], [300.0, 400.0]]),
        coords={"lat": y, "lon": x},
        dims=("lat", "lon"),
        name="population",
    )
    pop.attrs["coordinate_sys"] = EXPECTED_CRS
    return pop


def test_calc_DTEMP():
    input_data = _make_daily_input()
    _, threshold = _make_input_grid()
    population_grid = _make_population_grid()
    tea = TEAIndicators(
        input_data=input_data,
        threshold=threshold,
        population_grid=population_grid,
        unit="K",
    )

    tea.calc_daily_basis_vars(grid=True, gr=False)

    assert 'DTEMP' in tea.daily_results
    assert 'DTEM' in tea.daily_results
    assert 'DTEP' in tea.daily_results

    expected = tea.daily_results.DTEM.values * tea.daily_results.DTEP.values
    np.testing.assert_array_equal(tea.daily_results.DTEMP.values, expected)

    assert tea.daily_results.DTEMP.attrs['long_name'] == \
        'daily threshold exceedance magnitude * population (auxiliary)'


def test_calc_DTEP_GR():
    input_data = _make_daily_input()
    _, threshold = _make_input_grid()
    population_grid = _make_population_grid()
    tea = TEAIndicators(
        input_data=input_data,
        threshold=threshold,
        population_grid=population_grid,
        unit="K",
    )

    tea.calc_daily_basis_vars(grid=True, gr=True)

    assert 'DTEP_GR' in tea.daily_results
    assert 'DTEP' in tea.daily_results

    expected = tea.daily_results.DTEP.sum(
        dim=(tea.xdim, tea.ydim), skipna=True
    ).values
    np.testing.assert_array_equal(tea.daily_results.DTEP_GR.values, expected)


def test_calc_DTEMP_GR():
    input_data = _make_daily_input()
    _, threshold = _make_input_grid()
    population_grid = _make_population_grid()
    tea = TEAIndicators(
        input_data=input_data,
        threshold=threshold,
        population_grid=population_grid,
        unit="K",
    )

    tea.calc_daily_basis_vars(grid=True, gr=True)

    assert 'DTEMP_GR' in tea.daily_results
    assert 'DTEM_GR' in tea.daily_results
    assert 'DTEP_GR' in tea.daily_results

    expected = tea.daily_results.DTEM_GR.values * tea.daily_results.DTEP_GR.values
    np.testing.assert_array_equal(tea.daily_results.DTEMP_GR.values, expected)

    assert tea.daily_results.DTEMP_GR.attrs['long_name'] == \
        'daily threshold exceedance magnitude * population (auxiliary) (GR)'


def test_DTEMP_idempotent():
    input_data = _make_daily_input()
    _, threshold = _make_input_grid()
    population_grid = _make_population_grid()
    tea = TEAIndicators(
        input_data=input_data,
        threshold=threshold,
        population_grid=population_grid,
        unit="K",
    )

    tea.calc_daily_basis_vars(grid=True, gr=True)
    dtemp_values = tea.daily_results.DTEMP.values.copy()
    dtemp_gr_values = tea.daily_results.DTEMP_GR.values.copy()

    tea.calc_daily_basis_vars(grid=True, gr=True)
    np.testing.assert_array_equal(tea.daily_results.DTEMP.values, dtemp_values)
    np.testing.assert_array_equal(tea.daily_results.DTEMP_GR.values, dtemp_gr_values)


def test_DTEP_skipped_without_population():
    input_data = _make_daily_input()
    _, threshold = _make_input_grid()
    tea = TEAIndicators(
        input_data=input_data,
        threshold=threshold,
        unit="K",
    )

    tea.calc_daily_basis_vars(grid=True, gr=True)

    assert 'DTEP' not in tea.daily_results
    assert 'DTEMP' not in tea.daily_results
    assert 'DTEP_GR' not in tea.daily_results
    assert 'DTEMP_GR' not in tea.daily_results
