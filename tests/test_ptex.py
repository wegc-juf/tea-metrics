import numpy as np
import pandas as pd
import xarray as xr

from teametrics.TEA import TEAIndicators


EXPECTED_CRS = "EPSG:4326"


def _make_daily_input():
    times = pd.date_range("1980-01-01", "1994-12-31", freq="D")
    y = np.array([47.0, 48.0])
    x = np.array([15.0, 16.0])
    doy = np.array(times.dayofyear)
    seasonal = 22 + 14 * np.cos(2 * np.pi * (doy - 200) / 365.25)
    spatial = np.array([[1.0, 3.0], [-2.0, 0.0]])
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 2, size=(len(times), 2, 2))
    field = seasonal[:, np.newaxis, np.newaxis] + spatial[np.newaxis, :, :] + noise
    data = xr.DataArray(
        field,
        coords={"time": times, "lat": y, "lon": x},
        dims=("time", "lat", "lon"),
        name="input",
    )
    data.attrs["coordinate_sys"] = EXPECTED_CRS
    return data


def _make_threshold():
    y = np.array([47.0, 48.0])
    x = np.array([15.0, 16.0])
    threshold = xr.DataArray(
        np.full((len(y), len(x)), 30.0),
        coords={"lat": y, "lon": x},
        dims=("lat", "lon"),
        name="threshold",
    )
    threshold.attrs["coordinate_sys"] = EXPECTED_CRS
    return threshold


def _make_population_grid():
    y = np.array([47.0, 48.0])
    x = np.array([15.0, 16.0])
    pop = xr.DataArray(
        np.array([[50.0, 120.0], [5.0, 15.0]]),
        coords={"lat": y, "lon": x},
        dims=("lat", "lon"),
        name="population",
    )
    pop.attrs["coordinate_sys"] = EXPECTED_CRS
    return pop


def _make_tea(population_grid=None):
    data = _make_daily_input()
    threshold = _make_threshold()
    return TEAIndicators(
        input_data=data,
        threshold=threshold,
        population_grid=population_grid,
        unit="K",
    )


def test_pTEX_values():
    tea = _make_tea(population_grid=_make_population_grid())
    tea.calc_daily_basis_vars(grid=True, gr=True)
    tea.calc_annual_ctp_indicators(ctp="annual")

    assert 'pTEX' in tea.ctp_results
    assert 'pTEX_GR' in tea.ctp_results

    np.testing.assert_array_equal(
        tea.ctp_results['pTEX'].values,
        tea._CTP_resample_sum.DTEMP.values)

    np.testing.assert_array_equal(
        tea.ctp_results['pTEX_GR'].values,
        tea._CTP_resample_sum.DTEMP_GR.values)


def test_pTEX_attrs():
    tea = _make_tea(population_grid=_make_population_grid())
    tea.calc_daily_basis_vars(grid=True, gr=True)
    tea.calc_annual_ctp_indicators(ctp="annual")

    assert tea.ctp_results['pTEX'].attrs['long_name'] == \
        'total events population-extremity'
    assert tea.ctp_results['pTEX'].attrs['units'] == '10^4 person K d'

    assert tea.ctp_results['pTEX_GR'].attrs['long_name'] == \
        'total events population-extremity (GR)'
    assert tea.ctp_results['pTEX_GR'].attrs['units'] == '10^4 person K d'


def test_pTEX_idempotent():
    tea = _make_tea(population_grid=_make_population_grid())
    tea.calc_daily_basis_vars(grid=True, gr=True)
    tea.calc_annual_ctp_indicators(ctp="annual")
    ptex_values = tea.ctp_results['pTEX'].values.copy()
    ptex_gr_values = tea.ctp_results['pTEX_GR'].values.copy()

    tea.calc_annual_ctp_indicators(ctp="annual")
    np.testing.assert_array_equal(tea.ctp_results['pTEX'].values, ptex_values)
    np.testing.assert_array_equal(tea.ctp_results['pTEX_GR'].values, ptex_gr_values)


def test_pTEX_skipped_without_population():
    tea = _make_tea(population_grid=None)
    tea.calc_daily_basis_vars(grid=True, gr=True)
    tea.calc_annual_ctp_indicators(ctp="annual")

    assert 'pTEX' not in tea.ctp_results
    assert 'pTEX_GR' not in tea.ctp_results
