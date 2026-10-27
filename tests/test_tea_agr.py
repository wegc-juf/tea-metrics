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


class TestGRGridGeneration:
    def test_center_lattice_does_not_shift_with_cell_size(self):
        lat = np.arange(51.0, 49.75, -0.25)
        lon = np.arange(7.25, 8.5, 0.25)
        mask = xr.DataArray(
            np.ones((len(lat), len(lon))),
            coords={"lat": lat, "lon": lon},
            dims=("lat", "lon"),
        )
        generated = []

        for cell_size in (0.2, 0.5):
            tea = TEAAgr(mask=mask, gr_grid_res=0.25, cell_size_y=cell_size, land_frac_min=0)
            xcoords, ycoords = tea._get_xy()
            tea.generate_gr_grid_mask()
            generated.append((xcoords, ycoords, tea.gr_grid_mask, tea.gr_grid_areas))

        for xcoords, ycoords, grid_mask, grid_areas in generated:
            assert np.allclose(xcoords, lon)
            assert np.allclose(ycoords, lat)
            assert np.isfinite(grid_mask).all()
            assert np.isfinite(grid_areas).all()
            assert (grid_mask > 0).all()

        assert np.array_equal(generated[0][0], generated[1][0])
        assert np.array_equal(generated[0][1], generated[1][1])

    def test_sub_native_cell_size_still_generates_finite_grg_mask(self):
        lat = np.arange(51.0, 49.75, -0.25)
        lon = np.arange(7.25, 8.5, 0.25)
        mask = xr.DataArray(
            np.ones((len(lat), len(lon))),
            coords={"lat": lat, "lon": lon},
            dims=("lat", "lon"),
        )
        tea = TEAAgr(mask=mask, gr_grid_res=0.25, cell_size_y=0.2, land_frac_min=0)

        tea.generate_gr_grid_mask()

        assert np.isfinite(tea.gr_grid_mask).all()
        assert (tea.gr_grid_mask > 0).all()
        assert np.isfinite(tea.gr_grid_areas).all()


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


class TestSelectSubGr:
    def test_sub_gr_selection_uses_daily_grid_coordinates(self):
        lat = np.arange(4.0, 0.0, -0.5)
        lon = np.arange(0.0, 4.0, 0.5)
        times = pd.date_range("2000-01-01", periods=2, freq="D")
        data = xr.DataArray(
            np.full((len(times), len(lat), len(lon)), 10.0),
            coords={"time": times, "lat": lat, "lon": lon},
            dims=("time", "lat", "lon"),
            attrs={"coordinate_sys": EXPECTED_CRS},
        )
        threshold = xr.zeros_like(data.isel(time=0))
        tea = TEAAgr(input_data=data, threshold=threshold, unit="K", cell_size_y=0.25,
                     land_frac_min=0)
        tea.calc_daily_basis_vars(grid=True, gr=True)

        # Model an area grid with a wider extent than this cached data chunk.
        wider_lon = np.arange(-0.5, 4.5, 0.5)
        tea.area_grid = xr.DataArray(
            np.ones((len(lat), len(wider_lon))),
            coords={"lat": lat, "lon": wider_lon},
            dims=("lat", "lon"),
            attrs={"coordinate_sys": EXPECTED_CRS},
        )

        tea_sub = tea.select_sub_gr(ycoord=2.0, xcoord=3.5)

        assert tea_sub is not None
        assert tea_sub.area_grid.sizes == {"lat": 1, "lon": 1}

    def test_land_mask_is_aligned_by_coordinates_before_indexing(self):
        full_mask = xr.DataArray(
            np.zeros((4, 4)),
            coords={"lat": [3.0, 2.0, 1.0, 0.0], "lon": [0.0, 1.0, 2.0, 3.0]},
            dims=("lat", "lon"),
        )
        full_mask.loc[dict(lat=[2.0, 1.0], lon=[1.0, 2.0])] = 1
        source_grid = xr.DataArray(
            np.ones((2, 2)),
            coords={"lat": [2.0, 1.0], "lon": [1.0, 2.0]},
            dims=("lat", "lon"),
        )

        aligned = TEAAgr._align_spatial_grid(full_mask, source_grid)

        xr.testing.assert_equal(aligned, source_grid)

    def test_small_cell_uses_fractional_pixel_overlap(self):
        lat = np.arange(4.0, 0.0, -0.5)
        lon = np.arange(0.0, 4.0, 0.5)
        times = pd.date_range("2000-01-01", periods=2, freq="D")
        data = xr.DataArray(
            np.full((len(times), len(lat), len(lon)), 10.0),
            coords={"time": times, "lat": lat, "lon": lon},
            dims=("time", "lat", "lon"),
            attrs={"coordinate_sys": EXPECTED_CRS},
        )
        threshold = xr.zeros_like(data.isel(time=0))
        population = xr.full_like(threshold, 100.0)
        tea = TEAAgr(
            input_data=data,
            threshold=threshold,
            population_grid=population,
            unit="K",
            gr_grid_res=1.0,
            cell_size_y=0.25,
            land_frac_min=0,
        )
        tea.calc_daily_basis_vars(grid=True, gr=True)

        tea_sub = tea.select_sub_gr(ycoord=3.0, xcoord=1.0)

        assert tea_sub is not None
        assert tea_sub.area_grid.sizes == {"lat": 1, "lon": 1}
        original_area = tea.area_grid.sel(lat=3.0, lon=1.0)
        area_fraction = (tea_sub.area_grid / original_area).item()
        assert 0 < area_fraction < 1
        assert np.isclose(area_fraction, 0.25, atol=0.002)
        assert np.isclose(tea_sub.population_grid.item(), 100 * area_fraction)
        xr.testing.assert_allclose(tea_sub.daily_results.area_grid, tea_sub.area_grid)
        xr.testing.assert_allclose(
            tea_sub.daily_results.DTEA,
            tea_sub.daily_results.DTEC * tea_sub.area_grid,
        )
        xr.testing.assert_allclose(
            tea_sub.daily_results.DTEMA,
            tea_sub.daily_results.DTEM * tea_sub.daily_results.DTEA,
        )
        xr.testing.assert_allclose(
            tea_sub.daily_results.DTEP,
            tea_sub.daily_results.DTEC * tea_sub.population_grid / 10000,
        )
        xr.testing.assert_allclose(
            tea_sub.daily_results.DTEMP,
            tea_sub.daily_results.DTEM * tea_sub.daily_results.DTEP,
        )

    def test_small_cell_recomputes_annual_population_fields_by_year(self):
        lat = np.arange(4.0, 0.0, -0.5)
        lon = np.arange(0.0, 4.0, 0.5)
        times = pd.to_datetime(["2000-01-01", "2001-01-01"])
        data = xr.DataArray(
            np.full((len(times), len(lat), len(lon)), 10.0),
            coords={"time": times, "lat": lat, "lon": lon},
            dims=("time", "lat", "lon"),
            attrs={"coordinate_sys": EXPECTED_CRS},
        )
        threshold = xr.zeros_like(data.isel(time=0))
        population = xr.DataArray(
            np.stack([np.full((len(lat), len(lon)), 100.0),
                      np.full((len(lat), len(lon)), 200.0)]),
            coords={"year": [2000, 2001], "lat": lat, "lon": lon},
            dims=("year", "lat", "lon"),
            attrs={"units": "persons", "population_representation": "count_per_grid_cell"},
        )
        tea = TEAAgr(
            input_data=data,
            threshold=threshold,
            population_grid=population,
            unit="K",
            cell_size_y=0.25,
            land_frac_min=0,
        )
        tea.calc_daily_basis_vars(grid=True, gr=True)

        tea_sub = tea.select_sub_gr(ycoord=3.0, xcoord=1.0)
        daily_population = tea_sub._population_for_time(tea_sub.daily_results.time)
        original_area = tea.area_grid.sel(lat=3.0, lon=1.0)
        area_fraction = (tea_sub.area_grid / original_area).item()

        expected_population = population.sel(lat=[3.0], lon=[1.0]) * area_fraction
        xr.testing.assert_allclose(tea_sub.daily_results.population_grid, expected_population)
        assert tea_sub.population_grid.attrs == population.attrs
        xr.testing.assert_allclose(
            tea_sub.daily_results.DTEP,
            tea_sub.daily_results.DTEC * daily_population / 10000,
        )
        xr.testing.assert_allclose(
            tea_sub.daily_results.DTEMP,
            tea_sub.daily_results.DTEM * tea_sub.daily_results.DTEP,
        )

    def test_axis_overlap_is_independent_of_coordinate_direction(self):
        ascending = TEAAgr._get_axis_overlap([0.0, 1.0, 2.0], 1.0, 0.5)
        descending = TEAAgr._get_axis_overlap([2.0, 1.0, 0.0], 1.0, 0.5)

        assert np.array_equal(ascending[1], descending[1])
        assert np.array_equal(ascending[1], [0.5])

    def test_recalculates_aggregate_variables_for_selected_cell(self):
        lat = np.arange(4.0, 0.0, -0.5)
        lon = np.arange(0.0, 4.0, 0.5)
        times = pd.date_range("2000-01-01", periods=2, freq="D")
        values = np.full((len(times), len(lat), len(lon)), 10.0)
        values[:, :4, :4] = 1.0
        data = xr.DataArray(
            values,
            coords={"time": times, "lat": lat, "lon": lon},
            dims=("time", "lat", "lon"),
            attrs={"coordinate_sys": EXPECTED_CRS},
        )
        threshold = xr.zeros_like(data.isel(time=0))
        population = xr.full_like(threshold, 100.0)
        population.loc[{"lat": lat[:4], "lon": lon[:4]}] = 10.0
        tea = TEAAgr(
            input_data=data,
            threshold=threshold,
            population_grid=population,
            unit="K",
            land_frac_min=0,
        )
        tea.calc_daily_basis_vars(grid=True, gr=True)

        tea_sub = tea.select_sub_gr(ycoord=3.0, xcoord=1.0)

        assert tea_sub is not None
        assert "DTEMA_GR" not in tea_sub.daily_results
        assert "DTEMP_GR" not in tea_sub.daily_results

        tea_sub.calc_daily_basis_vars(grid=False, gr=True)

        xr.testing.assert_allclose(
            tea_sub.daily_results.DTEMA_GR,
            tea_sub.daily_results.DTEM_GR * tea_sub.daily_results.DTEA_GR,
        )
        xr.testing.assert_allclose(
            tea_sub.daily_results.DTEMP_GR,
            tea_sub.daily_results.DTEMP.sum(
                dim=(tea_sub.xdim, tea_sub.ydim), skipna=True
            ),
        )
        assert not np.allclose(
            tea_sub.daily_results.DTEMA_GR.values,
            tea.daily_results.DTEMA_GR.values,
        )
        assert not np.allclose(
            tea_sub.daily_results.DTEMP_GR.values,
            tea.daily_results.DTEMP_GR.values,
        )


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
