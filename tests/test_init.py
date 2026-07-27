import numpy as np
import xarray as xr
import pandas as pd
import pytest
from teametrics.TEA import TEAIndicators
from conftest import EXPECTED_CRS


class TestInit:
    def test_default_attributes(self, tea):
        assert tea.unit == "K"
        assert tea.xdim == "lon"
        assert tea.ydim == "lat"
        assert tea._crs == EXPECTED_CRS
        assert tea.area_grid is not None
        assert tea.population_grid is not None

    def test_no_population(self, tea_no_pop):
        assert tea_no_pop.population_grid is None

    def test_daily_results_initialized(self, tea):
        assert tea.daily_results is not None
        assert isinstance(tea.daily_results, xr.Dataset)

    def test_ctp_results_initialized(self, tea):
        assert tea.ctp_results is not None
        assert isinstance(tea.ctp_results, xr.Dataset)

    def test_input_data_stored(self, tea):
        assert tea.input_data is not None

    def test_threshold_grid_stored(self, tea):
        assert tea.threshold_grid is not None

    def test_ctp_default_none(self):
        data = xr.DataArray(
            np.ones((10, 2, 2)),
            dims=("time", "lat", "lon"),
            coords={
                "time": pd.date_range("1980-01-01", periods=10, freq="D"),
                "lat": [47.0, 48.0], "lon": [15.0, 16.0]},
        )
        data.attrs["coordinate_sys"] = EXPECTED_CRS
        threshold = xr.DataArray(
            np.full((2, 2), 0.5),
            dims=("lat", "lon"), coords={"lat": [47.0, 48.0], "lon": [15.0, 16.0]},
        )
        threshold.attrs["coordinate_sys"] = EXPECTED_CRS
        t = TEAIndicators(input_data=data, threshold=threshold, unit="K")
        assert t.CTP is None


class TestFindDimNames:
    def test_find_dim_names_detects_lat_lon(self, daily_data):
        xdim, ydim = TEAIndicators.find_dim_names(daily_data)
        assert xdim == "lon"
        assert ydim == "lat"

    def test_find_dim_names_custom_names(self):
        data = xr.DataArray(
            np.zeros((3, 4)),
            dims=("x", "y"),
            coords={"x": np.arange(3), "y": np.arange(4)},
        )
        xdim, ydim = TEAIndicators.find_dim_names(data)
        assert xdim == "x"
        assert ydim == "y"

    def test_find_dim_names_with_time_dim(self):
        data = xr.DataArray(
            np.zeros((5, 2, 2)),
            dims=("time", "lon", "lat"),
            coords={
                "time": pd.date_range("1980-01-01", periods=5, freq="D"),
                "lon": [15.0, 16.0], "lat": [47.0, 48.0]},
        )
        xdim, ydim = TEAIndicators.find_dim_names(data)
        assert xdim == "lon"
        assert ydim == "lat"


class TestReadCrs:
    def test_read_crs_from_attrs(self, daily_data):
        crs = TEAIndicators._read_crs(daily_data)
        assert crs == EXPECTED_CRS

    def test_read_crs_none(self):
        data = xr.DataArray(np.zeros((2, 2)), dims=("lat", "lon"))
        crs = TEAIndicators._read_crs(data)
        assert crs is None

    def test_read_crs_from_rio(self):
        data = xr.DataArray(
            np.zeros((2, 2)), dims=("lat", "lon"),
            attrs={"coordinate_sys": "EPSG:3035"})
        crs = TEAIndicators._read_crs(data)
        assert crs == "EPSG:3035"


class TestAreaGrid:
    def test_area_grid_created(self, tea):
        assert tea.area_grid is not None
        assert tea.area_grid.shape == (2, 2)

    def test_area_grid_values_positive(self, tea):
        assert np.all(tea.area_grid > 0)

    def test_area_grid_correct_dims(self, tea):
        assert tea.ydim in tea.area_grid.dims
        assert tea.xdim in tea.area_grid.dims


class TestPropagateCrs:
    def test_propagate_crs(self, tea):
        ds = xr.Dataset({"test_var": xr.DataArray(
            np.zeros((2, 2)), dims=("lat", "lon"))})
        result = tea._propagate_crs(ds)
        assert result.attrs["coordinate_sys"] == EXPECTED_CRS
        assert result["test_var"].attrs["coordinate_sys"] == EXPECTED_CRS


class TestCropToRect:
    def test_crop_to_rect_input_data(self, tea):
        xdim, ydim = tea.xdim, tea.ydim
        orig_shape = tea.input_data.shape
        with pytest.raises((KeyError, AttributeError, TypeError)):
            tea._crop_to_rect(x_range=(14.5, 15.5), y_range=(47.5, 48.5))

    def test_crop_to_rect_with_mask(self):
        from teametrics.TEA import TEAIndicators
        import numpy as np
        import xarray as xr
        import pandas as pd
        y = np.array([47.0, 48.0])
        x = np.array([15.0, 16.0])
        times = pd.date_range("1980-01-01", "1980-01-03", freq="D")
        data = xr.DataArray(
            np.ones((3, 2, 2)),
            coords={"time": times, "lat": y, "lon": x},
            dims=("time", "lat", "lon"),
            attrs={"coordinate_sys": "EPSG:4326"},
        )
        mask = xr.DataArray(
            np.ones((2, 2)),
            coords={"lat": y, "lon": x},
            dims=("lat", "lon"),
        )
        thresh = xr.DataArray(
            np.full((2, 2), 0.5),
            coords={"lat": y, "lon": x},
            dims=("lat", "lon"),
            attrs={"coordinate_sys": "EPSG:4326"},
        )
        t = TEAIndicators(
            input_data=data, threshold=thresh, mask=mask, unit="K")
        t._crop_to_rect(x_range=(14.5, 15.5), y_range=(47.5, 48.5))
        assert t.area_grid is not None


class TestSetCtp:
    def test_set_ctp_annual(self, tea):
        tea._set_ctp("annual")
        assert tea.CTP == "annual"

    @pytest.mark.parametrize("ctp", ["JJA", "DJF", "WAS", "ESS", "EWS"])
    def test_set_ctp_valid(self, tea, ctp):
        tea._set_ctp(ctp)
        assert tea.CTP == ctp

    def test_set_ctp_invalid(self, tea):
        with pytest.raises(ValueError, match="Invalid CTP"):
            tea._set_ctp("INVALID")


class TestGetResults:
    def test_get_daily_results_grid(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=False)
        result = tea_constant.get_daily_results(grid=True, gr=False)
        assert "DTEM" in result
        assert "DTEM_GR" not in result

    def test_get_daily_results_gr(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=True)
        result = tea_constant.get_daily_results(grid=False, gr=True)
        assert "DTEM" not in result
        assert "DTEM_GR" in result

    def test_get_ctp_results(self, tea_constant):
        tea_constant.calc_daily_basis_vars(grid=True, gr=True)
        tea_constant.calc_annual_ctp_indicators(ctp="annual")
        assert tea_constant.get_ctp_results() is tea_constant.ctp_results
