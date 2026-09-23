import numpy as np
import xarray as xr

from teametrics.common.var_attrs import get_attrs
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

    def test_pTEX_GR_is_spatial_sum_of_pTEX(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=True)
        tea_constant.calc_annual_ctp_indicators(ctp="annual")
        xr.testing.assert_allclose(
            tea_constant.ctp_results["pTEX_GR"],
            tea_constant.ctp_results["pTEX"].sum(
                dim=(tea_constant.xdim, tea_constant.ydim), skipna=True
            ),
        )

    def test_pTEX_attrs(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=True)
        tea_constant.calc_annual_ctp_indicators(ctp="annual")
        assert tea_constant.ctp_results["pTEX"].attrs["long_name"] == \
            "total events population-extremity"
        assert tea_constant.ctp_results["pTEX"].attrs["units"] == \
            "10^4 person K d"
        assert tea_constant.ctp_results["pTEX_GR"].attrs["long_name"] == \
            "total events population-extremity (GR)"

class TestCalcExceedancePopulation:
    def test_EP_is_temporal_sum_of_DTEP(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=True)
        tea_constant.calc_annual_ctp_indicators(ctp="annual")

        xr.testing.assert_allclose(
            tea_constant.ctp_results['EP'],
            tea_constant._CTP_resample_sum['DTEP'],
        )
        assert tea_constant.ctp_results['EP'].attrs['units'] == '10^4 person d'

    def test_EP_is_skipped_without_population(self, tea_constant_no_pop):
        tea_constant_no_pop.calc_daily_basis_vars(grid=True, gr=True)
        tea_constant_no_pop.calc_annual_ctp_indicators(ctp="annual")

        assert 'EP' not in tea_constant_no_pop.ctp_results

    def test_EP_avg_GR_uses_population_weighted_magnitude(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=True)
        tea_constant.calc_annual_ctp_indicators(ctp="annual")

        pem_gr = tea_constant.ctp_results['pEM_GR']
        expected = (tea_constant.ctp_results['pTEX_GR'] / pem_gr).where(
            pem_gr > 0, 0
        )
        xr.testing.assert_allclose(tea_constant.ctp_results['EP_avg_GR'], expected)
        assert tea_constant.ctp_results['EP_avg_GR'].attrs['units'] == '10^4 person'
        assert tea_constant.ctp_results['EP_avg_GR'].attrs['metric_type'] == 'basic'

    def test_population_weighted_magnitude_family(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=True)
        tea_constant.calc_annual_ctp_indicators(ctp="annual")

        xr.testing.assert_allclose(
            tea_constant.ctp_results['pEM_GR'],
            tea_constant._CTP_resample_sum['DTEM_PW_GR'],
        )
        xr.testing.assert_allclose(
            tea_constant.ctp_results['pEM_avg_GR'],
            (tea_constant.ctp_results['pEM_GR'] / tea_constant.ctp_results['ED_GR']).where(
                tea_constant.ctp_results['ED_GR'] > 0, 0
            ),
        )
        xr.testing.assert_allclose(
            tea_constant.ctp_results['pEM_avg_GR_Md'],
            tea_constant._CTP_resample_median['DTEM_PW_GR'],
        )
        xr.testing.assert_allclose(
            tea_constant.ctp_results['pEM_GR_Md'],
            tea_constant.ctp_results['ED_GR'] * tea_constant.ctp_results['pEM_avg_GR_Md'],
        )

    def test_EP_avg_GR_is_skipped_without_population(self, tea_constant_no_pop):
        tea_constant_no_pop.calc_daily_basis_vars(grid=True, gr=True)
        tea_constant_no_pop.calc_annual_ctp_indicators(ctp="annual")

        assert 'EP_avg_GR' not in tea_constant_no_pop.ctp_results

    def test_population_weighted_magnitudes_are_skipped_without_population(self, tea_constant_no_pop):
        tea_constant_no_pop.calc_daily_basis_vars(grid=True, gr=True)
        tea_constant_no_pop.calc_annual_ctp_indicators(ctp="annual")

        for variable in ('pEM_GR', 'pEM_avg_GR', 'pEM_avg_GR_Md', 'pEM_GR_Md'):
            assert variable not in tea_constant_no_pop.ctp_results


class TestPopulationAndAreaExtremityUnits:
    def test_population_and_area_extremity_units(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=True)
        tea_constant.calc_annual_ctp_indicators(ctp="annual")
        assert tea_constant.daily_results.DTEMA.attrs['units'] == '100 km^2 K'
        assert tea_constant.daily_results.DTEMP.attrs['units'] == '10^4 person K'
        assert tea_constant.daily_results.DTEMP_GR.attrs['units'] == '10^4 person K'
        assert tea_constant.ctp_results.TEX.attrs['units'] == '100 km^2 K d'
        assert tea_constant.ctp_results.TEX_GR.attrs['units'] == '100 km^2 K d'
        assert tea_constant.ctp_results.pTEX.attrs['units'] == '10^4 person K d'
        assert tea_constant.ctp_results.pTEX_GR.attrs['units'] == '10^4 person K d'


def test_cumulative_area_and_event_count_units():
    assert get_attrs('EA')['units'] == '100 km^2 d'
    assert get_attrs('EA')['long_name'] == 'cumulative exceedance area'
    assert get_attrs('EF')['units'] == '1'
    assert get_attrs('EF')['long_name'] == 'event count in climatic time period'

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

    def test_returns_when_required_ctp_variables_cannot_be_created(self, tea_constant, monkeypatch):
        tea_constant.ctp_results = xr.Dataset()
        monkeypatch.setattr(tea_constant, '_calc_annual_total_events_extremity', lambda: None)

        tea_constant._calc_annual_exceedance_area()

        assert not tea_constant.ctp_results.data_vars


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
