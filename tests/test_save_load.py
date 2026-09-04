import numpy as np
import pandas as pd
import pytest
import xarray as xr
import dask
import dask.array as da
import pytest
from teametrics.TEA import TEAIndicators
from conftest import EXPECTED_CRS


class TestSaveLoadDaily:
    def test_significant_digits_alias(self):
        with pytest.warns(DeprecationWarning):
            tea = TEAIndicators(significant_digits=4)
        assert tea.rounding_decimal_places == 4

        with pytest.warns(DeprecationWarning):
            tea.significant_digits = 2
        assert tea.rounding_decimal_places == 2

    def test_rounding_alias_conflict_is_rejected(self):
        with pytest.raises(ValueError, match="Set only rounding_decimal_places"):
            TEAIndicators(significant_digits=2, rounding_decimal_places=3)

    def test_netcdf_compression_level(self, tmp_path, monkeypatch):
        tea = TEAIndicators(compression_level=1, rounding_decimal_places=3, zlib_compression=True)
        dataset = xr.Dataset({"value": xr.DataArray([1.2345, 2.3456], dims="time")})
        calls = []
        saved_datasets = []

        def capture_to_netcdf(self, filepath, encoding=None):
            calls.append(encoding)
            saved_datasets.append(self)

        monkeypatch.setattr(xr.Dataset, "to_netcdf", capture_to_netcdf)
        tea._to_netcdf(dataset, tmp_path / "compressed.nc")
        assert calls[-1]["value"]["complevel"] == 1
        assert calls[-1]["value"]["zlib"] is True

        tea.zlib_compression = False
        tea._to_netcdf(dataset, tmp_path / "uncompressed.nc")
        assert calls[-1] is None
        np.testing.assert_allclose(saved_datasets[-1].value.values, [1.234, 2.346])

        tea.rounding_decimal_places = -1
        tea.zlib_compression = True
        tea._to_netcdf(dataset, tmp_path / "unrounded_compressed.nc")
        assert calls[-1]["value"]["zlib"] is True

    def test_spatial_float_output_uses_safe_float32_encoding(self, tmp_path, monkeypatch):
        tea = TEAIndicators(rounding_decimal_places=3, zlib_compression=False)
        dataset = xr.Dataset({"value": xr.DataArray(
            [[1.2345, 2.3456]], dims=("lat", "lon"))})
        calls = []

        def capture_to_netcdf(self, filepath, encoding=None):
            calls.append((encoding, self.value.dtype))

        monkeypatch.setattr(xr.Dataset, "to_netcdf", capture_to_netcdf)
        tea._to_netcdf(dataset, tmp_path / "float32.nc")

        assert calls == [(None, np.dtype("float32"))]

    def test_spatial_float_output_keeps_float64_when_precision_is_unsafe(self, tmp_path, monkeypatch):
        tea = TEAIndicators(rounding_decimal_places=3, zlib_compression=False)
        dataset = xr.Dataset({"value": xr.DataArray(
            [[16384.001, 16384.002]], dims=("lat", "lon"))})
        calls = []

        def capture_to_netcdf(self, filepath, encoding=None):
            calls.append(encoding)

        monkeypatch.setattr(xr.Dataset, "to_netcdf", capture_to_netcdf)
        tea._to_netcdf(dataset, tmp_path / "float64.nc")

        assert calls == [None]

    def test_netcdf_computes_dask_data_before_serialization(self, tmp_path, monkeypatch):
        tea = TEAIndicators(use_dask=True)
        dataset = xr.Dataset({"value": xr.DataArray(
            da.ones((4, 2), chunks=(2, 1)), dims=("time", "x"))})
        observed = []

        def capture_to_netcdf(self, filepath, encoding=None):
            observed.append(any(getattr(data.data, "chunks", None) is not None
                                for data in self.data_vars.values()))

        monkeypatch.setattr(xr.Dataset, "to_netcdf", capture_to_netcdf)
        computed = tea._to_netcdf(dataset, tmp_path / "computed.nc")

        assert observed == [False]
        assert all(getattr(data.data, "chunks", None) is None for data in computed.data_vars.values())

    def test_save_and_load_daily_roundtrip(self, tea_constant, tmp_path):
        tea_constant.calc_daily_basis_vars(grid=True, gr=True)
        path = tmp_path / "daily.nc"
        tea_constant.save_daily_results(path)
        tea2 = TEAIndicators(unit="K")
        tea2.load_daily_results(path)
        for var in tea_constant.daily_results.data_vars:
            assert var in tea2.daily_results
        assert tea2.daily_results.DTEM.dtype == np.dtype("float32")
        assert tea2.daily_results.DTEM_GR.dtype == np.dtype("float64")

    def test_save_daily_with_tiff(self, tea_constant, tmp_path):
        tea_constant.calc_daily_basis_vars(grid=True, gr=False)
        path = str(tmp_path / "daily_tiff.nc")
        try:
            tea_constant.save_daily_results(path, save_tiff=True)
        except Exception:
            pass

    def test_save_daily_variables_subset(self, tea_constant, tmp_path):
        tea_constant.calc_daily_basis_vars(grid=True, gr=True)
        path = tmp_path / "daily_subset.nc"
        tea_constant.save_daily_results(
            path, variables=["DTEM", "DTEM_GR"])


class TestSaveLoadCTP:
    def test_save_and_load_ctp_roundtrip(self, tea_constant, tmp_path):
        tea_constant.calc_daily_basis_vars(grid=True, gr=True)
        tea_constant.calc_annual_ctp_indicators(ctp="annual")
        path = tmp_path / "ctp.nc"
        tea_constant.save_ctp_results(path)
        tea2 = TEAIndicators(unit="K")
        tea2.load_ctp_results(path, use_dask=False)
        assert tea2._crs == EXPECTED_CRS
        for var in tea_constant.ctp_results.data_vars:
            assert var in tea2.ctp_results

    def test_ctp_interval_vars_datetime(self, tea_constant, tmp_path):
        tea_constant.calc_daily_basis_vars(grid=True, gr=True)
        tea_constant.calc_annual_ctp_indicators(ctp="annual")
        path = tmp_path / "ctp_interval.nc"
        tea_constant.save_ctp_results(path)
        tea2 = TEAIndicators(unit="K")
        tea2.load_ctp_results(path, use_dask=False)
        for v in ["TEX_max_interval_start_GR", "TEX_max_interval_end_GR",
                   "TEX_HW_max_interval_start_GR",
                   "TEX_HW_max_interval_end_GR"]:
            if v in tea2.ctp_results:
                assert np.issubdtype(
                    tea2.ctp_results[v].dtype, np.datetime64)

    def test_load_split_ctp_results_without_dask(self, tea_constant, tmp_path):
        tea_constant.calc_daily_basis_vars(grid=True, gr=True)
        tea_constant.calc_annual_ctp_indicators(ctp="annual")
        first = tmp_path / "ctp_first.nc"
        second = tmp_path / "ctp_second.nc"
        tea_constant.ctp_results.isel(time=slice(None, 5)).to_netcdf(first)
        tea_constant.ctp_results.isel(time=slice(5, None)).to_netcdf(second)

        loaded = TEAIndicators(unit="K")
        loaded.load_ctp_results([first, second], use_dask=False)

        for variable in tea_constant.ctp_results.data_vars:
            assert variable in loaded.ctp_results
        assert loaded.ctp_results.sizes["time"] == tea_constant.ctp_results.sizes["time"]

    @pytest.mark.parametrize("use_dask", [False, True])
    def test_load_split_ctp_results_with_different_history(self, tea_constant, tmp_path, use_dask):
        tea_constant.calc_daily_basis_vars(grid=True, gr=True)
        tea_constant.calc_annual_ctp_indicators(ctp="annual")
        first = tmp_path / "ctp_first.nc"
        second = tmp_path / "ctp_second.nc"
        first_results = tea_constant.ctp_results.isel(time=slice(None, 5)).copy()
        first_results.attrs["history"] = "first CTP calculation"
        second_results = tea_constant.ctp_results.isel(time=slice(5, None)).copy()
        second_results.attrs["history"] = "second CTP calculation"
        first_results.to_netcdf(first)
        second_results.to_netcdf(second)

        loaded = TEAIndicators(unit="K")
        loaded.load_ctp_results([first, second], use_dask=use_dask)

        assert loaded.ctp_results.sizes["time"] == tea_constant.ctp_results.sizes["time"]
        assert "history" not in loaded.ctp_results.attrs

    @pytest.mark.parametrize("use_dask", [False, True])
    def test_load_split_ctp_results_rejects_different_spatial_extent(self, tea_constant, tmp_path, use_dask):
        tea_constant.calc_daily_basis_vars(grid=True, gr=True)
        tea_constant.calc_annual_ctp_indicators(ctp="annual")
        first = tmp_path / "ctp_first.nc"
        second = tmp_path / "ctp_second.nc"
        tea_constant.ctp_results.isel(time=slice(None, 5)).isel(lon=slice(None, -1)).to_netcdf(first)
        tea_constant.ctp_results.isel(time=slice(5, None)).to_netcdf(second)

        loaded = TEAIndicators(unit="K")
        with pytest.raises(ValueError, match="join='exact'"):
            loaded.load_ctp_results([first, second], use_dask=use_dask)

    def test_load_split_ctp_results_configures_dask(self, tea_constant, tmp_path):
        tea_constant.calc_daily_basis_vars(grid=True, gr=True)
        tea_constant.calc_annual_ctp_indicators(ctp="annual")
        first = tmp_path / "ctp_first.nc"
        second = tmp_path / "ctp_second.nc"
        tea_constant.ctp_results.isel(time=slice(None, 5)).to_netcdf(first)
        tea_constant.ctp_results.isel(time=slice(5, None)).to_netcdf(second)

        loaded = TEAIndicators(unit="K")
        loaded.load_ctp_results([first, second], use_dask=True)

        assert dask.config.get("scheduler") == "threads"
        assert loaded.ctp_results.EM_avg.chunks is not None


class TestSaveLoadDecadal:
    def test_save_and_load_decadal_roundtrip(self, tea_constant, tmp_path):
        tea_constant.calc_daily_basis_vars(grid=True, gr=True)
        tea_constant.calc_annual_ctp_indicators(ctp="annual")
        tea_constant.calc_decadal_indicators(
            decadal_window=(10, 5, 4), drop_annual_results=False,
            calc_annual_ref=False)
        path = tmp_path / "decadal.nc"
        tea_constant.save_decadal_results(path)
        tea2 = TEAIndicators(unit="K")
        tea2.load_decadal_results(path)
        assert tea2._crs == EXPECTED_CRS
        for var in tea_constant.decadal_results.data_vars:
            assert var in tea2.decadal_results


class TestSaveLoadAmplification:
    def test_save_and_load_amplification_roundtrip(
            self, tea_constant, tmp_path):
        tea_constant.calc_daily_basis_vars(grid=True, gr=True)
        tea_constant.calc_annual_ctp_indicators(ctp="annual")
        tea_constant.calc_decadal_indicators(
            decadal_window=(10, 5, 4), drop_annual_results=False,
            calc_annual_ref=False)
        tea_constant.calc_amplification_factors(
            ref_period=(1980, 1984), cc_period=(1990, 1994))
        path = tmp_path / "amplification.nc"
        tea_constant.save_amplification_factors(path)
        tea2 = TEAIndicators(unit="K")
        tea2.load_amplification_factors(path)
        assert tea2._crs == EXPECTED_CRS
        for var in tea_constant.amplification_factors.data_vars:
            assert var in tea2.amplification_factors


class TestCRSPropagation:
    def test_crs_propagates_through_full_pipeline(
            self, tea_constant, tmp_path):
        tea_constant.calc_daily_basis_vars(grid=True, gr=True)
        tea_constant.calc_annual_ctp_indicators(ctp="annual")
        assert tea_constant.ctp_results.attrs.get(
            "coordinate_sys") == EXPECTED_CRS
        assert tea_constant.ctp_results.EF_GR.attrs.get(
            "coordinate_sys") == EXPECTED_CRS

        tea_constant.calc_decadal_indicators(
            decadal_window=(10, 5, 4), drop_annual_results=False,
            calc_annual_ref=False)
        assert tea_constant.decadal_results.attrs.get(
            "coordinate_sys") == EXPECTED_CRS
        assert tea_constant.decadal_results.EF_GR.attrs.get(
            "coordinate_sys") == EXPECTED_CRS

        tea_constant.calc_amplification_factors(
            ref_period=(1980, 1984), cc_period=(1990, 1994))
        assert tea_constant.amplification_factors.attrs.get(
            "coordinate_sys") == EXPECTED_CRS
        assert tea_constant.amplification_factors.EF_GR_AF.attrs.get(
            "coordinate_sys") == EXPECTED_CRS

    def test_crs_roundtrip_saveload(self, tea_constant, tmp_path):
        tea_constant.calc_daily_basis_vars(grid=True, gr=True)
        tea_constant.calc_annual_ctp_indicators(ctp="annual")
        ctp_path = tmp_path / "ctp.nc"
        tea_constant.save_ctp_results(ctp_path)
        loaded = TEAIndicators(unit="K")
        loaded.load_ctp_results(ctp_path, use_dask=False)
        assert loaded._crs == EXPECTED_CRS
