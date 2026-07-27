import numpy as np
import pandas as pd
import pytest
import xarray as xr
from teametrics.TEA import TEAIndicators
from conftest import EXPECTED_CRS


@pytest.fixture
def hourly_tea():
    daily_times = pd.date_range("1980-01-01", "1980-01-10", freq="D")
    y = np.array([47.0, 48.0])
    x = np.array([15.0, 16.0])
    daily_data = xr.DataArray(
        np.full((len(daily_times), 2, 2), 32.0),
        coords={"time": daily_times, "lat": y, "lon": x},
        dims=("time", "lat", "lon"),
        name="input",
    )
    daily_data.attrs["coordinate_sys"] = EXPECTED_CRS

    threshold = xr.DataArray(
        np.full((2, 2), 30.0),
        coords={"lat": y, "lon": x},
        dims=("lat", "lon"),
        name="threshold",
    )
    threshold.attrs["coordinate_sys"] = EXPECTED_CRS

    pop = xr.DataArray(
        np.array([[50.0, 120.0], [5.0, 15.0]]),
        coords={"lat": y, "lon": x},
        dims=("lat", "lon"),
        name="population",
        attrs={"coordinate_sys": EXPECTED_CRS},
    )

    tea = TEAIndicators(
        input_data=daily_data, threshold=threshold,
        population_grid=pop, unit="K")
    tea.calc_daily_basis_vars(grid=True, gr=True)

    hourly_times = pd.date_range("1980-01-01", "1980-01-10", freq="h",
                                 inclusive="left")
    hourly_data = xr.DataArray(
        np.full((len(hourly_times), 2, 2), 32.0),
        coords={"time": hourly_times, "lat": y, "lon": x},
        dims=("time", "lat", "lon"),
        name="input",
    )
    hourly_data.attrs["coordinate_sys"] = EXPECTED_CRS
    tea._hourly_input_data = hourly_data
    return tea


class TestCalcDET:
    def test_DTED_and_Nhours_exist(self, hourly_tea):
        hourly_tea.calc_hourly_indicators(hourly_tea._hourly_input_data)
        assert "DTED" in hourly_tea.daily_results
        assert "Nhours" in hourly_tea.daily_results

    def test_Nhours_max_24(self, hourly_tea):
        hourly_tea.calc_hourly_indicators(hourly_tea._hourly_input_data)
        nhours = hourly_tea.daily_results.Nhours.values
        valid = nhours[~np.isnan(nhours)]
        if len(valid) > 0:
            assert np.all(valid <= 24)
            assert np.all(valid >= 0)

    def test_DTED_attrs(self, hourly_tea):
        hourly_tea.calc_hourly_indicators(hourly_tea._hourly_input_data)
        assert "units" in hourly_tea.daily_results.DTED.attrs


class TestCalcDETGR:
    def test_DTED_GR_exists(self, hourly_tea):
        hourly_tea.calc_hourly_indicators(hourly_tea._hourly_input_data)
        assert "DTED_GR" in hourly_tea.daily_results

    def test_Nhours_GR_exists(self, hourly_tea):
        hourly_tea.calc_hourly_indicators(hourly_tea._hourly_input_data)
        assert "Nhours_GR" in hourly_tea.daily_results


class TestCalcDEH:
    def test_hourly_vars_exist(self, hourly_tea):
        hourly_tea.calc_hourly_indicators(hourly_tea._hourly_input_data)
        for var in ["t_hfirst", "t_hlast", "t_hmax"]:
            assert var in hourly_tea.daily_results


class TestCalcDEHGR:
    def test_hourly_GR_vars_exist(self, hourly_tea):
        hourly_tea.calc_hourly_indicators(hourly_tea._hourly_input_data)
        for var in ["t_hfirst_GR", "t_hlast_GR", "t_hmax_GR"]:
            assert var in hourly_tea.daily_results


class TestCalcHourlyCTPVars:
    def test_h_avg_exists(self, hourly_tea):
        hourly_tea.calc_hourly_indicators(hourly_tea._hourly_input_data)
        hourly_tea.calc_annual_ctp_indicators(ctp="annual")
        assert "h_avg" in hourly_tea.ctp_results
        assert "h_avg_GR" in hourly_tea.ctp_results
