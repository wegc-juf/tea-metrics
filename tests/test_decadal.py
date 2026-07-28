import numpy as np
import pandas as pd
import pytest
import xarray as xr
from teametrics.TEA import TEAIndicators


class TestCalcDecadalIndicators:
    def test_decadal_results_populated(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=True)
        tea_constant.calc_annual_ctp_indicators(ctp="annual")
        tea_constant.calc_decadal_indicators(
            decadal_window=(10, 5, 4), drop_annual_results=False,
            calc_annual_ref=False)
        assert len(tea_constant.decadal_results.data_vars) > 0

    def test_decadal_mean_values(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=True)
        tea_constant.calc_annual_ctp_indicators(ctp="annual")
        tea_constant.calc_decadal_indicators(
            decadal_window=(10, 5, 4), drop_annual_results=False,
            calc_annual_ref=False)
        for v in ["EF_GR", "EM_GR", "ED_GR"]:
            assert v in tea_constant.decadal_results

    def test_decadal_compound_vars(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=True)
        tea_constant.calc_annual_ctp_indicators(ctp="annual")
        tea_constant.calc_decadal_indicators(
            decadal_window=(10, 5, 4), drop_annual_results=False,
            calc_annual_ref=False)
        for v in ["TEX_GR", "ES_avg_GR", "EM_GR"]:
            assert v in tea_constant.decadal_results

    def test_decadal_spread_estimators(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=True)
        tea_constant.calc_annual_ctp_indicators(ctp="annual")
        tea_constant.calc_decadal_indicators(
            decadal_window=(10, 5, 4), calc_spread=True,
            drop_annual_results=False, calc_annual_ref=False)
        spread_vars = [v for v in tea_constant.decadal_results.data_vars
                       if v.endswith("_supp") or v.endswith("_slow")]
        assert len(spread_vars) > 0

    def test_decadal_spread_estimators_dask(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=True)
        tea_constant.calc_annual_ctp_indicators(ctp="annual")
        tea_constant.ctp_results = tea_constant.ctp_results.chunk({"time": 5})
        tea_constant.use_dask = True
        tea_constant.calc_decadal_indicators(
            decadal_window=(10, 5, 4), calc_spread=True,
            drop_annual_results=False, calc_annual_ref=False)

        spread_vars = [v for v in tea_constant.decadal_results.data_vars
                       if v.endswith("_supp") or v.endswith("_slow")]
        assert spread_vars
        assert tea_constant.decadal_results[spread_vars[0]].chunks is not None


class TestCalcCompoundVars:
    def _make_tea_with_unit(self):
        tea = object.__new__(TEAIndicators)
        tea.unit = "K"
        tea.null_val = 0
        return tea

    def test_calc_cumulative_events_duration(self):
        tea = self._make_tea_with_unit()
        f = xr.DataArray(np.array([3.0, 5.0]), dims="time")
        d = xr.DataArray(np.array([2.0, 1.0]), dims="time")
        ced = tea._calc_cumulative_events_duration(f, d)
        np.testing.assert_array_equal(ced.values, [6.0, 5.0])

    def test_calc_temporal_events_extremity(self):
        tea = self._make_tea_with_unit()
        f = xr.DataArray(np.array([3.0, 5.0]), dims="time")
        d = xr.DataArray(np.array([2.0, 1.0]), dims="time")
        ed = xr.DataArray(np.array([6.0, 5.0]), dims="time")
        m = xr.DataArray(np.array([4.0, 2.0]), dims="time")
        tex = tea._calc_temporal_events_extremity(f, d, ed, m)
        np.testing.assert_array_equal(tex.values, [24.0, 10.0])

    def test_calc_event_severity(self):
        tea = self._make_tea_with_unit()
        d = xr.DataArray(np.array([2.0, 3.0]), dims="time")
        m = xr.DataArray(np.array([4.0, 5.0]), dims="time")
        a = xr.DataArray(np.array([1.5, 2.0]), dims="time")
        es = tea._calc_event_severity(d, m, a)
        np.testing.assert_array_equal(es.values, [12.0, 30.0])

    def test_calc_total_events_extremity(self):
        tea = self._make_tea_with_unit()
        f = xr.DataArray(np.array([3.0, 5.0]), dims="time")
        s = xr.DataArray(np.array([12.0, 30.0]), dims="time")
        tex = tea._calc_total_events_extremity(f, s=s)
        np.testing.assert_array_equal(tex.values, [36.0, 150.0])

    def test_calc_hourly_event_severity(self):
        tea = self._make_tea_with_unit()
        es = xr.DataArray(np.array([12.0, 30.0]), dims="time")
        h_avg = xr.DataArray(np.array([8.0, 10.0]), dims="time")
        hes = tea._calc_hourly_event_severity(es, h_avg)
        np.testing.assert_array_equal(hes.values, [96.0, 300.0])

    def test_calc_hourly_total_events_extremity(self):
        tea = self._make_tea_with_unit()
        tex = xr.DataArray(np.array([36.0, 150.0]), dims="time")
        h_avg = xr.DataArray(np.array([8.0, 10.0]), dims="time")
        htex = tea._calc_hourly_total_events_extremity(tex, h_avg)
        np.testing.assert_array_equal(htex.values, [288.0, 1500.0])


class TestStaticHelpers:
    def test_calc_doy_adjustment(self):
        doy_first = xr.DataArray(np.array([50.0, 100.0]), dims="time")
        doy_last = xr.DataArray(np.array([300.0, 350.0]), dims="time")
        aep = xr.DataArray(np.array([6.0, 12.0]), dims="time")
        f, l = TEAIndicators._calc_doy_adjustment(doy_first, doy_last, aep)
        assert f is not None
        assert l is not None

    def test_calc_h_rise_set_adjustment(self):
        h_rise = xr.DataArray(np.array([2.0, 3.0]), dims="time")
        h_set = xr.DataArray(np.array([4.0, 5.0]), dims="time")
        h_avg = xr.DataArray(np.array([6.0, 8.0]), dims="time")
        rise, set_ = TEAIndicators._calc_h_rise_set_adjustment(
            h_rise, h_set, h_avg)
        assert rise is not None
        assert set_ is not None

    def test_calc_maximum_event_extremity_1d_no_min_duration(self):
        dtec = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0])
        dtema = np.array([0.0, 2.0, 3.0, 0.0, 1.0, 0.0, 0.0, 4.0, 5.0, 6.0, 0.0])
        time = np.arange(len(dtec))
        val, start, end = TEAIndicators._calc_maximum_event_extremity_1d(
            dtec, dtema, time, min_duration=1)
        assert val == 15.0
        assert start == 7 or start == pd.Timestamp("1970-01-01 00:00:00.000000007")
        assert end == 9 or end == pd.Timestamp("1970-01-01 00:00:00.000000009")

    def test_calc_maximum_event_extremity_1d_min_duration_filter(self):
        dtec = np.array([0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0])
        dtema = np.array([0.0, 2.0, 3.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        time = np.arange(len(dtec))
        val, start, end = TEAIndicators._calc_maximum_event_extremity_1d(
            dtec, dtema, time, min_duration=3)
        assert val == 0.0 or val == 0 or np.isnan(val) or np.isclose(val, 0.0)

    def test_gmean_custom_basic(self):
        data = xr.DataArray(np.array([1.0, 2.0, 4.0]), dims="x")
        gmean = TEAIndicators._gmean_custom(data, dim="x")
        expected = np.exp(np.mean(np.log([1.0, 2.0, 4.0])))
        assert np.isclose(gmean.values, expected, rtol=1e-10)

    def test_gmean_with_nan(self):
        data = xr.DataArray(np.array([1.0, np.nan, 4.0]), dims="x")
        gmean = TEAIndicators._gmean_custom(data, dim="x", skipna=True)
        expected = np.exp(np.mean(np.log([1.0, 4.0])))
        assert np.isclose(gmean.values, expected, rtol=1e-10)

    def test_gmean_all_nan(self):
        data = xr.DataArray(np.array([np.nan, np.nan]), dims="x")
        gmean = TEAIndicators._gmean_custom(data, dim="x", skipna=True)
        assert np.isnan(gmean.values)

    def test_apply_min_duration_basic(self):
        ds = xr.Dataset({"ED": xr.DataArray(
            np.array([0.5, 3.0, 1.0]),
            dims="time", coords={"time": [0, 1, 2]})})
        TEAIndicators._apply_min_duration(ds, min_duration=1.0)
        assert np.isnan(ds.ED.values[0])
        assert not np.isnan(ds.ED.values[1])
        assert not np.isnan(ds.ED.values[2])

    def test_duplicate_vars(self):
        ds = xr.Dataset({"EM": xr.DataArray(np.array([1.0, 2.0]), dims="time")})
        result = TEAIndicators._duplicate_vars(ds)
        assert "tEX" in result
        np.testing.assert_array_equal(result.tEX.values, result.EM.values)
