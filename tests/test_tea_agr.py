import numpy as np
import xarray as xr
import pandas as pd
import pytest
from teametrics.TEA_AGR import TEAAgr
from conftest import EXPECTED_CRS


class TestTEAAgrInit:
    def test_init_defaults(self):
        tea = TEAAgr()
        assert tea.gr_grid_res == 0.5
        assert tea.land_frac_min == 0.25

    def test_init_with_land_sea_mask(self):
        y = np.arange(46.0, 50.0, 0.5)
        x = np.arange(14.0, 18.0, 0.5)
        times = pd.date_range("1980-01-01", "1982-12-31", freq="D")
        data = xr.DataArray(
            np.full((len(times), len(y), len(x)), 32.0),
            coords={"time": times, "lat": y, "lon": x},
            dims=("time", "lat", "lon"),
            attrs={"coordinate_sys": EXPECTED_CRS},
        )
        threshold = xr.DataArray(
            np.full((len(y), len(x)), 30.0),
            coords={"lat": y, "lon": x},
            dims=("lat", "lon"),
            attrs={"coordinate_sys": EXPECTED_CRS},
        )
        lsm = xr.DataArray(
            np.ones((len(y), len(x))),
            coords={"lat": y, "lon": x},
            dims=("lat", "lon"),
        )
        tea = TEAAgr(
            input_data=data, threshold=threshold,
            land_sea_mask=lsm, unit="K", gr_grid_res=1.0)
        assert tea.unit == "K"
        assert tea.land_sea_mask is not None


class TestCalcAreaWeightedMean:
    @pytest.fixture
    def tea_agr_basic(self):
        y = np.arange(46.0, 50.0, 0.5)
        x = np.arange(14.0, 18.0, 0.5)
        times = pd.date_range("1980-01-01", "1982-12-31", freq="D")
        data = xr.DataArray(
            np.full((len(times), len(y), len(x)), 32.0),
            coords={"time": times, "lat": y, "lon": x},
            dims=("time", "lat", "lon"),
            attrs={"coordinate_sys": EXPECTED_CRS},
        )
        threshold = xr.DataArray(
            np.full((len(y), len(x)), 30.0),
            coords={"lat": y, "lon": x},
            dims=("lat", "lon"),
            attrs={"coordinate_sys": EXPECTED_CRS},
        )
        lsm = xr.DataArray(
            np.ones((len(y), len(x))),
            coords={"lat": y, "lon": x},
            dims=("lat", "lon"),
        )
        return TEAAgr(
            input_data=data, threshold=threshold,
            land_sea_mask=lsm, unit="K", gr_grid_res=1.0)

    def test_calc_area_weighted_mean(self, tea_agr_basic):
        data = xr.DataArray(
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            dims=("lat", "lon"),
            coords={"lat": [47.0, 48.0], "lon": [15.0, 16.0]},
        )
        area = xr.DataArray(
            np.array([[1.0, 1.0], [1.0, 1.0]]),
            dims=("lat", "lon"),
            coords={"lat": [47.0, 48.0], "lon": [15.0, 16.0]},
        )
        result = tea_agr_basic.calc_area_weighted_mean(area, data)
        assert np.isclose(result.values, 2.5)

    def test_calc_area_weighted_mean_all_nan(self, tea_agr_basic):
        data = xr.DataArray(
            np.full((2, 2), np.nan),
            dims=("lat", "lon"),
            coords={"lat": [47.0, 48.0], "lon": [15.0, 16.0]},
        )
        area = xr.DataArray(
            np.ones((2, 2)),
            dims=("lat", "lon"),
            coords={"lat": [47.0, 48.0], "lon": [15.0, 16.0]},
        )
        result = tea_agr_basic.calc_area_weighted_mean(area, data)
        assert np.isnan(result.values)

    def test_calc_area_weighted_mean_with_weights(self, tea_agr_basic):
        data = xr.DataArray(
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            dims=("lat", "lon"),
            coords={"lat": [47.0, 48.0], "lon": [15.0, 16.0]},
        )
        area = xr.DataArray(
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            dims=("lat", "lon"),
            coords={"lat": [47.0, 48.0], "lon": [15.0, 16.0]},
        )
        result = tea_agr_basic.calc_area_weighted_mean(area, data)
        expected = (1*1 + 2*2 + 3*3 + 4*4) / (1 + 2 + 3 + 4)
        assert np.isclose(result.values, expected)


class TestGetCTPResults:
    def test_get_ctp_results_returns_empty(self):
        from teametrics.TEA_AGR import TEAAgr
        y = np.arange(46.0, 50.0, 0.5)
        x = np.arange(14.0, 18.0, 0.5)
        times = pd.date_range("1980-01-01", "1982-12-31", freq="D")
        data = xr.DataArray(
            np.full((len(times), len(y), len(x)), 32.0),
            coords={"time": times, "lat": y, "lon": x},
            dims=("time", "lat", "lon"),
            attrs={"coordinate_sys": EXPECTED_CRS},
        )
        threshold = xr.DataArray(
            np.full((len(y), len(x)), 30.0),
            coords={"lat": y, "lon": x},
            dims=("lat", "lon"),
            attrs={"coordinate_sys": EXPECTED_CRS},
        )
        lsm = xr.DataArray(
            np.ones((len(y), len(x))),
            coords={"lat": y, "lon": x},
            dims=("lat", "lon"),
        )
        tea = TEAAgr(
            input_data=data, threshold=threshold,
            land_sea_mask=lsm, unit="K", gr_grid_res=1.0)
        assert tea.get_ctp_results() is tea.ctp_results


class TestCalcWeightedPerc:
    def test_calc_weighted_perc_single_uniform(self):
        from teametrics.TEA_AGR import TEAAgr
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        p5, p95 = TEAAgr._calc_weighted_perc_single(values, weights)
        assert p5 <= p95

    def test_calc_weighted_perc_single_skewed(self):
        from teametrics.TEA_AGR import TEAAgr
        values = np.array([1.0, 10.0, 100.0])
        weights = np.array([10.0, 1.0, 0.1])
        p5, p95 = TEAAgr._calc_weighted_perc_single(values, weights)
        assert p5 <= p95
        assert p5 >= 0
        assert p95 <= 100

    def test_calc_weighted_perc_single_single_value(self):
        from teametrics.TEA_AGR import TEAAgr
        values = np.array([42.0])
        weights = np.array([1.0])
        p5, p95 = TEAAgr._calc_weighted_perc_single(values, weights)
        assert np.isclose(p5, 42.0)
        assert np.isclose(p95, 42.0)
