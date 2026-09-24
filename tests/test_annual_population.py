import numpy as np
import pandas as pd
import pytest
import xarray as xr

from teametrics.TEA import TEAIndicators


def _annual_tea(population, times=None):
    if times is None:
        times = pd.to_datetime(['2001-12-31', '2002-01-01'])
    data = xr.DataArray(
        np.full((len(times), 2, 2), 2.0),
        dims=('time', 'lat', 'lon'),
        coords={'time': times, 'lat': [47.0, 48.0], 'lon': [15.0, 16.0]},
    )
    threshold = xr.zeros_like(data.isel(time=0))
    return TEAIndicators(input_data=data, threshold=threshold, population_grid=population, unit='K')


def _annual_population(lat=(47.0, 48.0), lon=(15.0, 16.0), years=(2001, 2002)):
    values = np.stack([np.full((2, 2), year - 1900.0) for year in years])
    return xr.DataArray(
        values, dims=('year', 'lat', 'lon'),
        coords={'year': np.asarray(years), 'lat': np.asarray(lat), 'lon': np.asarray(lon)}, name='population',
        attrs={'units': '1', 'population_representation': 'count_per_grid_cell'},
    )


def test_annual_population_is_selected_by_calendar_year():
    tea = _annual_tea(_annual_population())
    tea.calc_daily_basis_vars(grid=True, gr=True)

    np.testing.assert_array_equal(
        tea.daily_results.DTEP.isel(lat=0, lon=0).values, [0.0101, 0.0102]
    )
    np.testing.assert_array_equal(
        tea.daily_results.DTEP_GR.values, [0.0404, 0.0408]
    )
    assert tea.daily_results.attrs['population_time_basis'] == 'calendar_year'
    assert tea.daily_results.population_grid.dims == ('year', 'lat', 'lon')


def test_annual_population_is_used_by_ctp_ptex():
    tea = _annual_tea(_annual_population())
    tea.calc_daily_basis_vars(grid=True, gr=True)
    tea.calc_annual_ctp_indicators(ctp='annual')

    np.testing.assert_allclose(tea.ctp_results.pTEX_GR.values, [0.0808, 0.0816])
    assert tea.ctp_results.pTEX.attrs['population_time_basis'] == 'calendar_year'
    assert tea.ctp_results.pTEX_GR.attrs['population_year_start'] == 2001
    assert tea.ctp_results.pTEX_GR.attrs['population_year_end'] == 2002


def test_annual_population_grid_is_retained_in_ctp_results():
    population = _annual_population()
    tea = _annual_tea(population)
    tea.calc_daily_basis_vars(grid=True, gr=True)
    tea.calc_annual_ctp_indicators(ctp='annual')

    xr.testing.assert_equal(
        tea.ctp_results.population_grid,
        population.sel(year=xr.DataArray(
            [2001, 2002], dims='time', coords={'time': tea.ctp_results.time}
        )).drop_vars('year'),
    )
    assert tea.ctp_results.population_grid.dims == ('time', 'lat', 'lon')
    assert tea.ctp_results.population_grid.attrs['population_time_basis'] == 'calendar_year'
    assert tea.ctp_results.population_grid.attrs['population_year_start'] == 2001
    assert tea.ctp_results.population_grid.attrs['population_year_end'] == 2002


def test_annual_population_ctp_chunks_combine_by_time(tmp_path):
    tea = _annual_tea(_annual_population())
    tea.calc_daily_basis_vars(grid=True, gr=True)
    tea.calc_annual_ctp_indicators(ctp='annual')
    first = tmp_path / 'ctp_2001.nc'
    second = tmp_path / 'ctp_2002.nc'
    ctp = tea.ctp_results[['EM_avg', 'pTEX_GR', 'population_grid']]
    ctp.isel(time=[0]).to_netcdf(first)
    ctp.isel(time=[1]).to_netcdf(second)

    loaded = TEAIndicators(unit='K')
    loaded.load_ctp_results([first, second], use_dask=False)

    assert loaded.ctp_results.sizes['time'] == 2
    assert loaded.ctp_results.population_grid.dims == ('time', 'lat', 'lon')
    np.testing.assert_array_equal(loaded.ctp_results.population_grid.time, tea.ctp_results.time)


def test_annual_population_grid_uses_decadal_rolling_mean():
    time = pd.date_range('2001-01-01', periods=15, freq='YS')
    population = xr.DataArray(
        np.arange(15 * 2 * 2, dtype=float).reshape(15, 2, 2),
        dims=('time', 'lat', 'lon'),
        coords={'time': time, 'lat': [47.0, 48.0], 'lon': [15.0, 16.0]},
        attrs={
            'long_name': 'annual population count per grid cell',
            'metric_type': 'metadata',
            'population_time_basis': 'calendar_year',
        },
    )
    tea = object.__new__(TEAIndicators)
    tea.unit = 'K'
    tea.ctp_results = xr.Dataset({'population_grid': population})
    tea.decadal_results = xr.Dataset()

    tea._calc_decadal_mean((10, 5, 4))

    expected = population.rolling(time=10, center=True, min_periods=1).mean()
    expected[:5] = np.nan
    expected[-4:] = np.nan
    expected.attrs = population.attrs.copy()
    expected.attrs['long_name'] = 'decadal-mean annual population count per grid cell'
    expected.attrs['population_year_start'] = 2001
    expected.attrs['population_year_end'] = 2015
    expected = expected.rename('population_grid')
    xr.testing.assert_identical(tea.decadal_results.population_grid, expected)


def test_annual_population_is_used_by_ctp_population_exposure():
    tea = _annual_tea(_annual_population())
    tea.calc_daily_basis_vars(grid=True, gr=True)
    tea.calc_annual_ctp_indicators(ctp='annual')

    np.testing.assert_allclose(
        tea.ctp_results.EP.isel(lat=0, lon=0).values, [0.0101, 0.0102]
    )
    assert tea.ctp_results.EP.attrs['population_time_basis'] == 'calendar_year'
    assert tea.ctp_results.EP.attrs['population_year_start'] == 2001
    assert tea.ctp_results.EP.attrs['population_year_end'] == 2002


def test_annual_population_is_used_by_ctp_average_population_exposure():
    tea = _annual_tea(_annual_population())
    tea.calc_daily_basis_vars(grid=True, gr=True)
    tea.calc_annual_ctp_indicators(ctp='annual')

    np.testing.assert_allclose(tea.ctp_results.EP_avg_GR.values, [0.0404, 0.0408])
    assert tea.ctp_results.EP_avg_GR.attrs['population_time_basis'] == 'calendar_year'
    assert tea.ctp_results.EP_avg_GR.attrs['population_year_start'] == 2001
    assert tea.ctp_results.EP_avg_GR.attrs['population_year_end'] == 2002


def test_annual_population_is_used_by_ctp_population_weighted_magnitudes():
    tea = _annual_tea(_annual_population())
    tea.calc_daily_basis_vars(grid=True, gr=True)
    tea.calc_annual_ctp_indicators(ctp='annual')

    np.testing.assert_allclose(tea.ctp_results.pEM_GR.values, [2.0, 2.0])
    np.testing.assert_allclose(tea.ctp_results.pEM_avg_GR.values, [2.0, 2.0])
    np.testing.assert_allclose(tea.ctp_results.pEM_avg_GR_Md.values, [2.0, 2.0])
    np.testing.assert_allclose(tea.ctp_results.pEM_GR_Md.values, [2.0, 2.0])
    assert tea.ctp_results.pEM_GR.attrs['population_time_basis'] == 'calendar_year'
    assert tea.ctp_results.pEM_GR.attrs['population_year_start'] == 2001
    assert tea.ctp_results.pEM_GR.attrs['population_year_end'] == 2002


def test_annual_population_rejects_missing_calendar_year():
    with pytest.raises(ValueError, match=r'missing calendar years: \[2002\]'):
        _annual_tea(_annual_population(years=(2001,)))


def test_annual_population_rejects_shifted_coordinates():
    with pytest.raises(ValueError, match='coordinates must exactly match'):
        _annual_tea(_annual_population(lat=(47.1, 48.1)))


def test_annual_population_survives_daily_results_reload(tmp_path):
    tea = _annual_tea(_annual_population())
    tea.calc_daily_basis_vars(grid=True, gr=True)
    path = tmp_path / 'annual_population_daily.nc'
    tea.save_daily_results(path)

    loaded = TEAIndicators(unit='K')
    loaded.load_daily_results(path)
    assert loaded.population_grid.dims == ('year', 'lat', 'lon')
    loaded.update_min_area(0)
    np.testing.assert_allclose(loaded.daily_results.DTEP_GR.values, [0.0404, 0.0408])


def test_annual_population_rejects_static_daily_results_cache(tmp_path):
    static = _annual_population().isel(year=0, drop=True)
    static_tea = _annual_tea(static)
    static_tea.calc_daily_basis_vars(grid=True, gr=True)
    path = tmp_path / 'static_population_daily.nc'
    static_tea.save_daily_results(path)

    annual_tea = _annual_tea(_annual_population())
    with pytest.raises(ValueError, match='do not use annual population'):
        annual_tea.load_daily_results(path)
