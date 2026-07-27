import numpy as np
import xarray as xr
from teametrics.TEA import TEAIndicators


class TestCalcCTP:
    def test_ctp_results_populated(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=True)
        tea_constant.calc_annual_ctp_indicators(ctp="annual")
        assert len(tea_constant.ctp_results.data_vars) > 0


class TestCalcEventFrequency:
    def test_EF_exists(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=True)
        tea_constant.calc_annual_ctp_indicators(ctp="annual")
        assert "EF" in tea_constant.ctp_results
        assert "EF_GR" in tea_constant.ctp_results

    def test_EF_nonnegative(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=True)
        tea_constant.calc_annual_ctp_indicators(ctp="annual")
        assert np.all(tea_constant.ctp_results.EF.values >= 0)
        assert np.all(tea_constant.ctp_results.EF_GR.values >= 0)


class TestCalcSupplementaryEventVars:
    def test_doy_first_last_exist(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=True)
        tea_constant.calc_annual_ctp_indicators(ctp="annual")
        for v in ["doy_first", "doy_last", "AEP"]:
            assert v in tea_constant.ctp_results

    def test_AEP_between_0_and_12(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=True)
        tea_constant.calc_annual_ctp_indicators(ctp="annual")
        aep = tea_constant.ctp_results.AEP.values
        valid = aep[~np.isnan(aep)]
        if len(valid) > 0:
            assert np.all(valid >= 0) and np.all(valid <= 12)


class TestCalcEventDuration:
    def test_ED_exists(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=True)
        tea_constant.calc_annual_ctp_indicators(ctp="annual")
        for v in ["ED", "ED_avg", "ED_GR"]:
            assert v in tea_constant.ctp_results

    def test_ED_nonnegative(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=True)
        tea_constant.calc_annual_ctp_indicators(ctp="annual")
        for v in ["ED", "ED_avg", "ED_GR", "ED_avg_GR"]:
            vals = tea_constant.ctp_results[v].values
            valid = vals[~np.isnan(vals)]
            if len(valid) > 0:
                assert np.all(valid >= 0)


class TestCalcExceedanceMagnitude:
    def test_EM_exists(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=True)
        tea_constant.calc_annual_ctp_indicators(ctp="annual")
        for v in ["EM", "EM_avg", "EM_GR", "EM_avg_GR"]:
            assert v in tea_constant.ctp_results

    def test_EM_nonnegative(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=True)
        tea_constant.calc_annual_ctp_indicators(ctp="annual")
        for v in ["EM", "EM_avg", "EM_GR", "EM_avg_GR"]:
            vals = tea_constant.ctp_results[v].values
            valid = vals[~np.isnan(vals)]
            if len(valid) > 0:
                assert np.all(valid >= 0)


class TestCalcAvgDurationMagnitude:
    def test_DM_avg_exists(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=True)
        tea_constant.calc_annual_ctp_indicators(ctp="annual")
        assert "DM_avg" in tea_constant.ctp_results





class TestCalcTEX:
    def test_TEX_exists(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=True)
        tea_constant.calc_annual_ctp_indicators(ctp="annual")
        assert "TEX" in tea_constant.ctp_results
        assert "TEX_GR" in tea_constant.ctp_results

    def test_TEX_attrs(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=True)
        tea_constant.calc_annual_ctp_indicators(ctp="annual")
        assert tea_constant.ctp_results.TEX.attrs["long_name"] == \
            "total events extremity"
        assert tea_constant.ctp_results.TEX_GR.attrs["long_name"] == \
            "total events extremity (GR)"


class TestCalcPTEX:
    def test_pTEX_exists(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=True)
        tea_constant.calc_annual_ctp_indicators(ctp="annual")
        assert "pTEX" in tea_constant.ctp_results
        assert "pTEX_GR" in tea_constant.ctp_results

    def test_pTEX_values(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=True)
        tea_constant.calc_annual_ctp_indicators(ctp="annual")
        np.testing.assert_array_equal(
            tea_constant.ctp_results["pTEX"].values,
            tea_constant._CTP_resample_sum.DTEMP.values)
        np.testing.assert_array_equal(
            tea_constant.ctp_results["pTEX_GR"].values,
            tea_constant._CTP_resample_sum.DTEMP_GR.values)

    def test_pTEX_attrs(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=True)
        tea_constant.calc_annual_ctp_indicators(ctp="annual")
        assert tea_constant.ctp_results["pTEX"].attrs["long_name"] == \
            "total events population-extremity"
        assert tea_constant.ctp_results["pTEX"].attrs["units"] == \
            "10^4 person K d"
        assert tea_constant.ctp_results["pTEX_GR"].attrs["long_name"] == \
            "total events population-extremity (GR)"

    def test_pTEX_idempotent(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=True)
        tea_constant.calc_annual_ctp_indicators(ctp="annual")
        ptex_values = tea_constant.ctp_results["pTEX"].values.copy()
        ptex_gr_values = tea_constant.ctp_results["pTEX_GR"].values.copy()

        tea_constant.calc_annual_ctp_indicators(ctp="annual")
        np.testing.assert_array_equal(
            tea_constant.ctp_results["pTEX"].values, ptex_values)
        np.testing.assert_array_equal(
            tea_constant.ctp_results["pTEX_GR"].values, ptex_gr_values)

    def test_pTEX_skipped_without_population(self, tea_constant_no_pop):
        tea_constant_no_pop.calc_daily_basis_vars(grid=True, gr=True)
        tea_constant_no_pop.calc_annual_ctp_indicators(ctp="annual")
        assert "pTEX" not in tea_constant_no_pop.ctp_results
        assert "pTEX_GR" not in tea_constant_no_pop.ctp_results


class TestCalcExceedanceArea:
    def test_EA_exists(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=True)
        tea_constant.calc_annual_ctp_indicators(ctp="annual")
        assert "EA" in tea_constant.ctp_results
        assert "EA_avg_GR" in tea_constant.ctp_results


class TestCalcEventSeverity:
    def test_ES_exists(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=True)
        tea_constant.calc_annual_ctp_indicators(ctp="annual")
        assert "ES_avg_GR" in tea_constant.ctp_results


class TestCalcExceedanceHeatContent:
    def test_H_AEHC_exists(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=True)
        tea_constant.calc_annual_ctp_indicators(ctp="annual")
        assert "H_AEHC_avg_GR" in tea_constant.ctp_results
        assert "H_AEHC_GR" in tea_constant.ctp_results


class TestCalcMaxEventExtremity:
    def test_TEX_max_exists(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=True)
        tea_constant.calc_annual_ctp_indicators(ctp="annual")
        assert "TEX_max_GR" in tea_constant.ctp_results
        assert "TEX_max_interval_start_GR" in tea_constant.ctp_results
        assert "TEX_max_interval_end_GR" in tea_constant.ctp_results


class TestCalcMaxHeatwaveExtremity:
    def test_TEX_HW_max_exists(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=True)
        tea_constant.calc_annual_ctp_indicators(ctp="annual")
        assert "TEX_HW_max_GR" in tea_constant.ctp_results
        assert "TEX_HW_max_interval_start_GR" in tea_constant.ctp_results
        assert "TEX_HW_max_interval_end_GR" in tea_constant.ctp_results


class TestCalcStaticHelpers:
    def test_calc_exceedance_heat_content(self):
        s_avg = 5.0
        d_avg = 2.0
        tex_val = 10.0
        h_aehc_avg, h_aehc = TEAIndicators._calc_exceedance_heat_content(
            s_avg, d_avg, tex_val)
        assert h_aehc_avg > 0
        assert h_aehc > 0

    def test_duplicate_vars(self):
        ds = xr.Dataset({"EM": xr.DataArray(np.array([1.0, 2.0]), dims="time")})
        result = TEAIndicators._duplicate_vars(ds)
        assert "tEX" in result
        np.testing.assert_array_equal(result.tEX.values, result.EM.values)

    def test_gmean_custom(self):
        data = xr.DataArray(np.array([1.0, 2.0, 4.0]), dims="x")
        gmean = TEAIndicators._gmean_custom(data, dim="x")
        expected = np.exp(np.mean(np.log([1.0, 2.0, 4.0])))
        assert np.isclose(gmean.values, expected, rtol=1e-10)
