import numpy as np
import xarray as xr
from teametrics.TEA import TEAIndicators


class TestCalcAmplificationFactors:
    def test_amplification_factors_populated(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=True)
        tea_constant.calc_annual_ctp_indicators(ctp="annual")
        tea_constant.calc_decadal_indicators(
            decadal_window=(10, 5, 4), drop_annual_results=False,
            calc_annual_ref=False)
        tea_constant.calc_amplification_factors(
            ref_period=(1980, 1984), cc_period=(1990, 1994))
        assert len(tea_constant.amplification_factors.data_vars) > 0

    def test_AF_vars_exist(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=True)
        tea_constant.calc_annual_ctp_indicators(ctp="annual")
        tea_constant.calc_decadal_indicators(
            decadal_window=(10, 5, 4), drop_annual_results=False,
            calc_annual_ref=False)
        tea_constant.calc_amplification_factors(
            ref_period=(1980, 1984), cc_period=(1990, 1994))
        af_vars = tea_constant.amplification_factors.data_vars
        af_names = [v for v in af_vars if v.endswith("_AF")]
        assert len(af_names) > 0

    def test_AF_CC_vars_exist(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=True)
        tea_constant.calc_annual_ctp_indicators(ctp="annual")
        tea_constant.calc_decadal_indicators(
            decadal_window=(10, 5, 4), drop_annual_results=False,
            calc_annual_ref=False)
        tea_constant.calc_amplification_factors(
            ref_period=(1980, 1984), cc_period=(1990, 1994))
        af_cc_vars = [v for v in
                      tea_constant.amplification_factors.data_vars
                      if v.endswith("_AF_CC")]
        assert len(af_cc_vars) > 0

    def test_amplification_factors_nonnegative(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=True)
        tea_constant.calc_annual_ctp_indicators(ctp="annual")
        tea_constant.calc_decadal_indicators(
            decadal_window=(10, 5, 4), drop_annual_results=False,
            calc_annual_ref=False)
        tea_constant.calc_amplification_factors(
            ref_period=(1980, 1984), cc_period=(1990, 1994))
        for v in tea_constant.amplification_factors.data_vars:
            vals = tea_constant.amplification_factors[v].values
            valid = vals[~np.isnan(vals)]
            if len(valid) > 0:
                assert np.all(valid >= 0), f"Negative in {v}"


class TestGmean:
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
