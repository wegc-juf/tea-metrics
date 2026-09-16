import numpy as np
import xarray as xr
from teametrics.TEA import TEAIndicators


class TestCalcDTEC:
    def test_DTEC_exists(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=False)
        assert "DTEC" in tea_constant.daily_results

    def test_DTEC_zero_one_values(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=False)
        dt = tea_constant.daily_results.DTEC.values
        assert np.all((dt == 0) | (dt == 1))


class TestCalcDTEA:
    def test_DTEA_exists(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=False)
        assert "DTEA" in tea_constant.daily_results

    def test_DTEA_attrs(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=False)
        assert "daily threshold exceedance area" in \
            tea_constant.daily_results.DTEA.attrs["long_name"]


class TestCalcDTEP:
    def test_DTEP_exists(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=False)
        assert "DTEP" in tea_constant.daily_results

    def test_DTEP_formula(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=False)
        expected = tea_constant.daily_results.DTEC.values * \
            tea_constant.population_grid.values / 10000
        np.testing.assert_array_equal(
            tea_constant.daily_results.DTEP.values, expected)

    def test_DTEP_attrs(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=False)
        assert tea_constant.daily_results.DTEP.attrs["long_name"] == \
            "daily threshold exceedance population"

    def test_DTEP_skipped_without_population(self, tea_constant_no_pop):
        tea_constant_no_pop.calc_daily_basis_vars(grid=True, gr=False)
        assert "DTEP" not in tea_constant_no_pop.daily_results


class TestCalcDTEM:
    def test_DTEM_exists(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=False)
        assert "DTEM" in tea_constant.daily_results

    def test_DTEM_zero_below_threshold(self, tea):
        tea.calc_daily_basis_vars(grid=True, gr=False)
        dtem = tea.daily_results.DTEM.values
        data = tea.input_data.values
        thresh = tea.threshold_grid.values
        mask = data <= thresh
        if np.any(mask):
            assert np.all(dtem[mask] == 0)

    def test_DTEM_positive_above_threshold(self, tea):
        tea.calc_daily_basis_vars(grid=True, gr=False)
        dtem = tea.daily_results.DTEM.values
        data = tea.input_data.values
        thresh = tea.threshold_grid.values
        mask = data > thresh
        if np.any(mask):
            assert np.all(dtem[mask] > 0)

    def test_DTEM_attrs(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=False)
        assert "DTEM" in tea_constant.daily_results
        assert "units" in tea_constant.daily_results.DTEM.attrs


class TestCalcDTEEC:
    def test_DTEEC_nd_matches_1d_reference(self, tea_constant):
        dtec = np.array([
            [[1, 0, np.nan], [0, 1, 1]],
            [[1, 1, np.nan], [0, 1, 0]],
            [[0, 1, 1], [1, 1, 0]],
            [[1, 1, 1], [1, 0, 1]],
        ])
        expected = np.apply_along_axis(tea_constant._calc_dteec_1d, 0, dtec)

        result = tea_constant._calc_dteec_nd(dtec)

        np.testing.assert_array_equal(result, expected)

    def test_DTEEC_legacy_preserves_all_nan_rows(self, tea_constant):
        tea_constant._calc_DTEM()
        tea_constant._calc_DTEC()
        tea_constant.daily_results['DTEC'][{tea_constant.ydim: 0}] = np.nan

        tea_constant._calc_DTEEC_legacy()

        assert tea_constant.daily_results.DTEEC.isel({tea_constant.ydim: 0}).isnull().all()

    def test_DTEEC_legacy_accepts_integer_data(self, tea_constant):
        tea_constant.daily_results['DTEC'] = xr.ones_like(tea_constant.input_data, dtype=np.int16)

        tea_constant._calc_DTEEC_legacy()

        assert tea_constant.daily_results.DTEEC.dtype == np.dtype('int16')
        assert tea_constant.daily_results.DTEEC.sum() == tea_constant.input_data.isel(
            {tea_constant.tdim: 0}).size

        eager = tea_constant.daily_results.DTEEC.copy()
        tea_constant.daily_results['DTEC'] = tea_constant.daily_results.DTEC.chunk(
            {tea_constant.ydim: 1, tea_constant.xdim: 1})
        tea_constant._calc_DTEEC_parallel()
        parallel = tea_constant.daily_results.DTEEC.compute()

        assert parallel.dtype == np.dtype('int16')
        xr.testing.assert_equal(parallel, eager)

    def test_DTEEC_parallel_matches_legacy(self, tea_constant):
        tea_constant._calc_DTEM()
        tea_constant._calc_DTEC()
        tea_constant.daily_results['DTEC'][{tea_constant.ydim: 0}] = np.nan
        tea_constant._calc_DTEEC_legacy()
        legacy = tea_constant.daily_results.DTEEC.copy()

        tea_constant.daily_results = tea_constant.daily_results.drop_vars('DTEEC')
        tea_constant.use_dask = True
        tea_constant.daily_results['DTEC'] = tea_constant.daily_results.DTEC.chunk(
            {tea_constant.tdim: -1, tea_constant.ydim: 1, tea_constant.xdim: 1})
        tea_constant._calc_DTEEC_parallel()
        parallel = tea_constant.daily_results.DTEEC.compute()

        xr.testing.assert_equal(parallel, legacy)

    def test_DTEEC_exists(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=False)
        assert "DTEEC" in tea_constant.daily_results

    def test_DTEEC_event_count_properties(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=False)
        dteec = tea_constant.daily_results.DTEEC.values
        dtec = tea_constant.daily_results.DTEC.values
        assert np.all(dteec >= 0)
        assert np.all(dteec <= dtec)


class TestCalcDTEMA:
    def test_DTEMA_exists(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=True)
        assert "DTEMA" in tea_constant.daily_results

    def test_DTEMA_formula(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=True)
        expected = tea_constant.daily_results.DTEM.values * \
            tea_constant.daily_results.DTEA.values
        np.testing.assert_array_equal(
            tea_constant.daily_results.DTEMA.values, expected)


class TestCalcDTEMP:
    def test_DTEMP_exists(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=False)
        assert "DTEMP" in tea_constant.daily_results

    def test_DTEMP_formula(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=False)
        expected = tea_constant.daily_results.DTEM.values * \
            tea_constant.daily_results.DTEP.values
        np.testing.assert_array_equal(
            tea_constant.daily_results.DTEMP.values, expected)

    def test_DTEMP_attrs(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=False)
        assert tea_constant.daily_results.DTEMP.attrs["long_name"] == \
            "daily threshold exceedance magnitude * population (auxiliary)"

    def test_DTEMP_skipped_without_population(self, tea_constant_no_pop):
        tea_constant_no_pop.calc_daily_basis_vars(grid=True, gr=False)
        assert "DTEMP" not in tea_constant_no_pop.daily_results


class TestCalcDTEPGR:
    def test_DTEP_GR_exists(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=True)
        assert "DTEP_GR" in tea_constant.daily_results

    def test_DTEP_GR_is_sum_over_space(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=True)
        expected = (tea_constant.daily_results.DTEC * tea_constant.population_grid).sum(
            dim=(tea_constant.xdim, tea_constant.ydim), skipna=True).values / 10000
        np.testing.assert_array_equal(
            tea_constant.daily_results.DTEP_GR.values, expected)

    def test_DTEP_GR_dask_matches_eager_at_rounding_boundary(self, tea_constant):
        tea_constant.population_grid[:] = [[5278382, 1], [1, 1]]
        tea_constant.calc_daily_basis_vars(grid=True, gr=False)
        tea_constant._calc_DTEP_GR()
        eager = tea_constant.daily_results.DTEP_GR.copy()

        tea_constant.daily_results = tea_constant.daily_results.drop_vars('DTEP_GR')
        tea_constant.daily_results['DTEC'] = tea_constant.daily_results.DTEC.chunk(
            {tea_constant.ydim: 1, tea_constant.xdim: 1})
        tea_constant._calc_DTEP_GR()

        xr.testing.assert_identical(tea_constant.daily_results.DTEP_GR.compute(), eager)

    def test_DTEP_GR_skipped_without_population(self, tea_constant_no_pop):
        tea_constant_no_pop.calc_daily_basis_vars(grid=True, gr=True)
        assert "DTEP_GR" not in tea_constant_no_pop.daily_results


class TestCalcDTEMPGR:
    def test_DTEMP_GR_exists(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=True)
        assert "DTEMP_GR" in tea_constant.daily_results

    def test_DTEMP_GR_formula(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=True)
        expected = tea_constant.daily_results.DTEM_GR.values * \
            tea_constant.daily_results.DTEP_GR.values
        np.testing.assert_array_equal(
            tea_constant.daily_results.DTEMP_GR.values, expected)

    def test_DTEMP_GR_attrs(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=True)
        assert tea_constant.daily_results.DTEMP_GR.attrs["long_name"] == \
            "daily threshold exceedance magnitude * population (auxiliary) (GR)"

    def test_DTEMP_GR_skipped_without_population(self, tea_constant_no_pop):
        tea_constant_no_pop.calc_daily_basis_vars(grid=True, gr=True)
        assert "DTEMP_GR" not in tea_constant_no_pop.daily_results


class TestCalcDTEAGR:
    def test_DTEA_GR_exists(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=True)
        assert "DTEA_GR" in tea_constant.daily_results

    def test_DTEA_GR_attrs(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=True)
        assert "GR" in tea_constant.daily_results.DTEA_GR.attrs["long_name"]


class TestUpdateMinArea:
    def test_recalculates_cached_gr_compounds(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=True)
        assert np.any(tea_constant.daily_results.DTEMA_GR.values > 0)
        assert np.any(tea_constant.daily_results.DTEMP_GR.values > 0)

        tea_constant.update_min_area(min_area=1e100)

        assert np.all(tea_constant.daily_results.DTEC_GR.values == 0)
        assert np.all(tea_constant.daily_results.DTEM_GR.values == 0)
        assert np.all(tea_constant.daily_results.DTEMA_GR.values == 0)
        assert np.all(tea_constant.daily_results.DTEMP_GR.values == 0)
        np.testing.assert_array_equal(
            tea_constant.daily_results.DTEMA_GR.values,
            (tea_constant.daily_results.DTEM_GR * tea_constant.daily_results.DTEA_GR).values,
        )
        np.testing.assert_array_equal(
            tea_constant.daily_results.DTEMP_GR.values,
            (tea_constant.daily_results.DTEM_GR * tea_constant.daily_results.DTEP_GR).values,
        )

    def test_invalidates_hourly_and_ctp_caches(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=True)
        for variable in (
            'Nhours_GR', 'DTED_GR', 't_hfirst_GR', 't_hlast_GR', 't_hmax_GR',
            'h_rise_GR', 'h_set_GR',
        ):
            tea_constant.daily_results[variable] = tea_constant.daily_results.DTEC_GR.copy()
        tea_constant._daily_results_filtered = tea_constant.daily_results.copy()
        tea_constant._CTP_resample_sum = xr.Dataset({'stale': xr.DataArray(1)})
        tea_constant._CTP_resample_mean = xr.Dataset({'stale': xr.DataArray(1)})
        tea_constant.ctp_results = xr.Dataset({'stale': xr.DataArray(1)})
        tea_constant.decadal_results = xr.Dataset({'stale': xr.DataArray(1)})
        tea_constant._decadal_ED = xr.Dataset({'stale': xr.DataArray(1)})

        tea_constant.update_min_area(min_area=1e100)

        for variable in (
            'Nhours_GR', 'DTED_GR', 't_hfirst_GR', 't_hlast_GR', 't_hmax_GR',
            'h_rise_GR', 'h_set_GR',
        ):
            assert variable not in tea_constant.daily_results
        assert tea_constant._daily_results_filtered is None
        assert tea_constant._CTP_resample_sum is None
        assert tea_constant._CTP_resample_mean is None
        assert not tea_constant.ctp_results.data_vars
        assert not tea_constant.decadal_results.data_vars
        assert not tea_constant._decadal_ED.data_vars


class TestCalcDTECGR:
    def test_DTEC_GR_exists(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=True)
        assert "DTEC_GR" in tea_constant.daily_results


class TestCalcDteecGR:
    def test_DTEEC_GR_exists(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=True)
        assert "DTEEC_GR" in tea_constant.daily_results


class TestCalcDTEMGR:
    def test_DTEM_GR_exists(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=True)
        assert "DTEM_GR" in tea_constant.daily_results

    def test_DTEM_Max_GR_exists(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=True)
        assert "DTEM_Max_GR" in tea_constant.daily_results


class TestCalcDTEMAGR:
    def test_DTEMA_GR_exists(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=True)
        assert "DTEMA_GR" in tea_constant.daily_results


class TestCalcAvgThresholdGR:
    def test_threshold_avg_GR_exists(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=True)
        assert "threshold_avg_GR" in tea_constant.daily_results


class TestIdempotent:
    def test_DTEMP_idempotent(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=True)
        dtemp_values = tea_constant.daily_results.DTEMP.values.copy()
        dtemp_gr_values = tea_constant.daily_results.DTEMP_GR.values.copy()

        tea_constant.calc_daily_basis_vars(grid=True, gr=True)
        np.testing.assert_array_equal(
            tea_constant.daily_results.DTEMP.values, dtemp_values)
        np.testing.assert_array_equal(
            tea_constant.daily_results.DTEMP_GR.values, dtemp_gr_values)

    def test_all_vars_idempotent(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=True)
        vars1 = {k: v.values.copy()
                 for k, v in tea_constant.daily_results.data_vars.items()}

        tea_constant.calc_daily_basis_vars(grid=True, gr=True)
        for k in vars1:
            np.testing.assert_array_equal(
                tea_constant.daily_results[k].values, vars1[k],
                err_msg=f"{k} not idempotent")
