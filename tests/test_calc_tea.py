import argparse
import pytest
import numpy as np
import xarray as xr
from types import SimpleNamespace

_import_error = None
try:
    from teametrics.calc_TEA import (
        _getopts, _get_ctp_filepath, _calc_x_y_range, _reduce_region,
        _daily_cache_covers_grid, _gr_grid_static_is_usable, _gr_grid_static_covers_reference,
        _get_threshold, _load_mask_file, _load_gr_grid_static,
        _compare_to_ctp_ref, _load_population_grid, _get_chunk_workers, calc_dbv_indicators,
    )
    HAS_CALC_TEA = True
except (ImportError, FileNotFoundError) as e:
    HAS_CALC_TEA = False
    _import_error = str(e)


pytestmark = pytest.mark.skipif(
    not HAS_CALC_TEA,
    reason=f"calc_TEA not importable (likely udunits2 missing): {_import_error}")


class TestGetopts:
    def test_getopts_defaults(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["calc_tea", "--config-file",
                                         "test.yaml"])
        opts = _getopts()
        assert opts.config_file == "test.yaml"
        assert opts.loglevel == "INFO"

    def test_getopts_loglevel(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["calc_tea", "--loglevel", "warning"])
        opts = _getopts()
        assert opts.loglevel == "WARNING"

    def test_getopts_version(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["calc_tea", "--config-file",
                                         "test.yaml", "--version"])
        with pytest.raises(SystemExit) as exc_info:
            _getopts()
        assert exc_info.value.code == 0


class TestParallelChunks:
    def test_chunk_workers_are_bounded(self, monkeypatch):
        monkeypatch.setattr("teametrics.calc_TEA.os.cpu_count", lambda: 96)
        monkeypatch.setattr(
            "teametrics.calc_TEA.psutil.virtual_memory",
            lambda: type("Memory", (), {"available": 128 * 1024 ** 3})(),
        )
        assert _get_chunk_workers(20) == 4
        assert _get_chunk_workers(20, max_workers=2) == 2

    def test_grg_tile_workers_have_no_fixed_upper_limit(self):
        from teametrics.common.config import check_type

        check_type('grg_tile_workers', 40)
        with pytest.raises(argparse.ArgumentTypeError, match='grg_tile_workers must be at least 1'):
            check_type('grg_tile_workers', 0)


class TestSubregionGrgStaticCoverage:
    @staticmethod
    def _grids():
        from teametrics.TEA_AGR import TEAAgr

        lat = np.arange(0.0, 15.0, 0.25)
        lon = np.arange(0.0, 20.0, 0.25)
        coords = {"lat": lat, "lon": lon}
        mask_values = np.zeros((len(lat), len(lon)))
        mask_values[20:22, 40:42] = 1.0
        region_mask = xr.DataArray(mask_values, coords=coords, dims=("lat", "lon"))
        reference = xr.Dataset(
            {"ED": (("lat", "lon"), np.ones_like(mask_values))},
            coords=coords,
        )

        # The region cache is only a subset of the larger CTP coordinate grid,
        # but it covers the region mask extent plus the two-degree tile margin.
        cached_lat = lat[(lat >= 3.0) & (lat <= 8.0)]
        cached_lon = lon[(lon >= 8.0) & (lon <= 13.0)]
        cached_coords = {"lat": cached_lat, "lon": cached_lon}
        cached_shape = (len(cached_lat), len(cached_lon))
        cached_mask = xr.DataArray(
            np.ones(cached_shape), coords=cached_coords, dims=("lat", "lon"), name="mask"
        )
        cached_full_mask = cached_mask.rename("mask_full")
        cached_areas = xr.DataArray(
            np.ones(cached_shape), coords=cached_coords, dims=("lat", "lon"), name="area_grid"
        )
        tea = TEAAgr(mask=region_mask, full_mask=region_mask, gr_grid_res=0.25,
                     cell_size_y=2.0, land_frac_min=0)
        tea._ref_mean = reference
        return tea, reference, region_mask, cached_mask, cached_full_mask, cached_areas

    def test_subset_static_cache_covers_only_the_region_extent(self):
        from teametrics.calc_TEA import _gr_grid_static_covers_reference

        tea, reference, region_mask, cached_mask, _, _ = self._grids()

        assert not _gr_grid_static_covers_reference(cached_mask, reference)
        assert _gr_grid_static_covers_reference(
            cached_mask, reference, region_mask=region_mask, cell_size_y=2.0
        )

    def test_subset_cache_is_aligned_without_regenerating(self, monkeypatch):
        import teametrics.calc_TEA as calc_tea_module

        tea, reference, _, cached_mask, cached_full_mask, cached_areas = self._grids()
        monkeypatch.setattr(
            calc_tea_module,
            "_load_gr_grid_static",
            lambda opts: (cached_mask, cached_areas, cached_full_mask),
        )
        monkeypatch.setattr(
            tea,
            "generate_gr_grid_mask",
            lambda: pytest.fail("A valid subregion cache should not be regenerated."),
        )

        calc_tea_module._load_or_generate_gr_grid_static(
            SimpleNamespace(grg_cell_size=2.0, recalc_decadal=False), tea
        )

        assert tea.gr_grid_mask.sizes == reference.sizes
        assert tea.gr_grid_areas.sizes == reference.sizes
        assert np.isnan(tea.gr_grid_mask.sel(lat=0.0, lon=0.0))
        assert tea.gr_grid_mask.sel(lat=5.0, lon=10.0).item() == 1.0


class TestGetCTPFilepath:
    def test_get_ctp_filepath_creates_dir(self, tmp_path):
        from argparse import Namespace
        output_path = str(tmp_path / "output" / "test")
        opts = Namespace(outpath=output_path, region="test", param_str="Tx99p",
                         period="annual", dataset="ERA5")
        path = _get_ctp_filepath(1980, 1990, opts)
        assert path.endswith(".nc")
        assert "test" in path

    def test_get_ctp_filepath_agr(self, tmp_path):
        from argparse import Namespace
        output_path = str(tmp_path / "output")
        opts = Namespace(outpath=output_path, region="AUT", agr="EUR",
                         grg_grid_spacing=0.5, grg_cell_size=2,
                         param_str="Tx99p", period="annual",
                         dataset="ERA5")
        path = _get_ctp_filepath(1980, 1990, opts, annual_agr=True)
        assert path.endswith(".nc")
        assert "agr" in path.lower()
        assert "AGR_0p5_cell2-EUR" in path

    def test_get_ctp_filepath_grg_includes_spacing_and_cell_size(self, tmp_path):
        from argparse import Namespace
        opts = Namespace(outpath=str(tmp_path), region="AUT", agr="SEA",
                         grg_grid_spacing=0.25, grg_cell_size=0.5,
                         param_str="Tx25.0degC", period="annual", dataset="ERA5")

        path = _get_ctp_filepath(1961, 1970, opts)

        assert "GRG_0p25_cell0p5-AUT" in path


class TestDailyCacheCoverage:
    def test_daily_cache_with_larger_extent_is_reusable(self, tmp_path):
        cached = xr.DataArray(
            np.ones((5, 5)), coords={"lat": [3, 2, 1, 0, -1], "lon": [0, 1, 2, 3, 4]},
            dims=("lat", "lon"), name="area_grid")
        cache_path = tmp_path / "dbv.nc"
        cached.to_dataset().to_netcdf(cache_path)
        required = xr.DataArray(
            np.ones((2, 2)), coords={"lat": [2, 1], "lon": [1, 2]},
            dims=("lat", "lon"), name="mask")

        assert _daily_cache_covers_grid(cache_path, required)

    def test_daily_cache_missing_required_boundary_coordinate_is_rejected(self, tmp_path):
        cached = xr.DataArray(
            np.ones((3, 3)), coords={"lat": [2, 1, 0], "lon": [1, 2, 3]},
            dims=("lat", "lon"), name="area_grid")
        cache_path = tmp_path / "dbv.nc"
        cached.to_dataset().to_netcdf(cache_path)
        required = xr.DataArray(
            np.ones((2, 2)), coords={"lat": [2, 1], "lon": [0, 1]},
            dims=("lat", "lon"), name="mask")

        assert not _daily_cache_covers_grid(cache_path, required)


class TestCalcXYRange:
    def test_calc_x_y_range_basic(self):
        cell_size_y = 0.5
        mask = xr.DataArray(
            np.array([[0, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 0]]),
            dims=("y", "x"),
            coords={"y": [47.0, 47.5, 48.0, 48.5],
                    "x": [15.0, 15.5, 16.0]},
        )
        x_min, y_min, x_max, y_max = _calc_x_y_range(cell_size_y, mask)
        assert x_min < x_max
        assert y_min < y_max

    def test_lat_lon_crop_uses_two_degree_padding_per_side(self):
        mask = xr.DataArray(
            np.zeros((7, 9)), dims=("lat", "lon"),
            coords={"lat": np.arange(45.0, 52.0), "lon": np.arange(10.0, 19.0)})
        mask.loc[dict(lat=[47.0, 48.0], lon=[13.0, 14.0])] = 1

        x_min, y_min, x_max, y_max = _calc_x_y_range(0.5, mask)

        assert (x_min, y_min, x_max, y_max) == (11, 45, 16, 50)

    def test_projected_crop_retains_half_cell_padding(self):
        mask = xr.DataArray(
            np.zeros((7, 7)), dims=("y", "x"),
            coords={"y": np.arange(0.0, 700.0, 100.0), "x": np.arange(0.0, 700.0, 100.0)})
        mask.loc[dict(y=[200.0, 300.0], x=[200.0, 300.0])] = 1

        x_min, y_min, x_max, y_max = _calc_x_y_range(100.0, mask)

        assert (x_min, y_min, x_max, y_max) == (150, 150, 350, 350)

    def test_calc_x_y_range_no_valid_cells(self):
        mask = xr.DataArray(
            np.zeros((4, 3)),
            dims=("y", "x"),
            coords={"y": range(4), "x": range(3)},
        )
        with pytest.raises(ValueError):
            _calc_x_y_range(0.5, mask)


class TestReduceRegion:
    def test_reduce_region_no_crop(self):
        from types import SimpleNamespace
        opts = SimpleNamespace(region="test", grg_cell_size=0.5, threshold_type="perc")
        data = xr.DataArray(
            np.ones((5, 4, 3)),
            dims=("time", "y", "x"),
            coords={"time": range(5), "y": [47.0, 47.5, 48.0, 48.5],
                    "x": [15.0, 15.5, 16.0]},
        )
        mask = xr.DataArray(
            np.ones((4, 3)),
            dims=("y", "x"),
            coords={"y": [47.0, 47.5, 48.0, 48.5],
                    "x": [15.0, 15.5, 16.0]},
        )
        result = _reduce_region(opts, data, mask)
        assert result is not None


class TestGetThreshold:
    def test_get_threshold_absolute(self):
        from types import SimpleNamespace
        opts = SimpleNamespace(
            threshold_type="abs", threshold=30.0, unit="degC")
        result = _get_threshold(opts)
        assert result == 30.0


class TestLoadMaskFile:
    def test_load_mask_file_ignores_area_variables(self, tmp_path):
        mask = xr.DataArray(np.ones((2, 2)), dims=('y', 'x'), name='mask')
        ds = mask.to_dataset()
        ds['mask_full'] = xr.ones_like(mask).rename('mask_full')
        ds['area_grid_full'] = xr.ones_like(mask)
        ds['area_full'] = xr.DataArray(4.0)
        ds['area_grid'] = xr.ones_like(mask)
        ds['area'] = xr.DataArray(4.0)
        mask_dir = tmp_path / 'masks'
        mask_dir.mkdir()
        ds.to_netcdf(mask_dir / 'AUT_mask_ERA5_1500.nc')
        opts = SimpleNamespace(gr_type='polygon', maskpath=str(tmp_path), mask_sub='masks',
                               region='AUT', dataset='ERA5', altitude_threshold=1500)

        result, area_grid, full_mask = _load_mask_file(opts, include_area=True)

        xr.testing.assert_equal(result, mask)
        xr.testing.assert_equal(area_grid, ds.area_grid)
        xr.testing.assert_equal(full_mask, ds.mask_full)

    def test_altitude_filtered_agr_mask_requires_pre_altitude_mask(self, tmp_path):
        mask = xr.DataArray(np.ones((2, 2)), dims=('y', 'x'), name='mask')
        mask_dir = tmp_path / 'masks'
        mask_dir.mkdir()
        mask.to_dataset().to_netcdf(mask_dir / 'AUT_mask_ERA5_1500.nc')
        opts = SimpleNamespace(gr_type='polygon', maskpath=str(tmp_path), mask_sub='masks',
                               region='AUT', dataset='ERA5', altitude_threshold=1500, agr=True)

        with pytest.raises(ValueError, match='pre-altitude mask_full'):
            _load_mask_file(opts, include_area=True)

    def test_load_mask_file_missing(self, tmp_path):
        from types import SimpleNamespace
        opts = SimpleNamespace(gr_type="polygon", maskpath=str(tmp_path), mask_sub="masks",
                               region="AUT", dataset="ERA5", altitude_threshold=1500)
        with pytest.raises(FileNotFoundError):
            _load_mask_file(opts)


class TestLoadPopulationGrid:
    @pytest.mark.filterwarnings("ignore:numpy.ndarray size changed:RuntimeWarning")
    def test_load_named_population_variable(self, tmp_path):
        from types import SimpleNamespace
        population = xr.DataArray(
            np.ones((2, 2)), dims=("y", "x"), coords={"y": [1, 2], "x": [3, 4]},
            name="population")
        path = tmp_path / "population.nc"
        population.to_dataset().to_netcdf(path)

        result = _load_population_grid(SimpleNamespace(population_grid_path=str(path)))

        xr.testing.assert_equal(result, population)

    def test_no_population_path(self):
        from types import SimpleNamespace
        assert _load_population_grid(SimpleNamespace(population_grid_path=None)) is None


class TestReduceRegionPopulation:
    def test_population_grid_is_reduced_with_region(self):
        from types import SimpleNamespace
        opts = SimpleNamespace(region="test", grg_cell_size=0.5, threshold_type="perc")
        coords = {"y": [47.0, 47.5, 48.0, 48.5], "x": [15.0, 15.5, 16.0]}
        mask = xr.DataArray(
            np.array([[0, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 0]]),
            dims=("y", "x"), coords=coords)
        population = xr.DataArray(
            np.ones((4, 3)), dims=("y", "x"), coords=coords)

        result = _reduce_region(opts, None, mask, population_grid=population)

        assert len(result) == 4
        assert result[3].sizes == result[1].sizes


class TestLoadGrGridStatic:
    def test_static_grid_cache_rejects_all_nan_data(self):
        mask = xr.DataArray(np.full((2, 2), np.nan), dims=("y", "x"))
        areas = xr.DataArray(np.full((2, 2), np.nan), dims=("y", "x"))

        assert not _gr_grid_static_is_usable(mask, areas)

    def test_static_grid_cache_accepts_valid_cells(self):
        mask = xr.DataArray(np.ones((2, 2)), dims=("y", "x"))
        areas = xr.DataArray(np.ones((2, 2)), dims=("y", "x"))

        assert _gr_grid_static_is_usable(mask, areas)

    def test_static_grid_cache_with_larger_axes_covers_reference(self):
        cached = xr.DataArray(
            np.ones((4, 4)), coords={"lat": [3, 2, 1, 0], "lon": [0, 1, 2, 3]},
            dims=("lat", "lon"), name="mask")
        reference = xr.DataArray(
            np.ones((2, 2)), coords={"lat": [2, 1], "lon": [1, 2]},
            dims=("lat", "lon"), name="EM_avg")

        assert _gr_grid_static_covers_reference(cached, reference)

    def test_static_grid_cache_with_shifted_lattice_does_not_cover_reference(self):
        cached = xr.DataArray(
            np.ones((2, 2)), coords={"lat": [2.1, 1.1], "lon": [1.1, 2.1]},
            dims=("lat", "lon"), name="mask")
        reference = xr.DataArray(
            np.ones((2, 2)), coords={"lat": [2.0, 1.0], "lon": [1.0, 2.0]},
            dims=("lat", "lon"), name="EM_avg")

        assert not _gr_grid_static_covers_reference(cached, reference)

    def test_load_gr_grid_static_not_found(self, tmp_path):
        from types import SimpleNamespace
        statpath = str(tmp_path / "stats")
        opts = SimpleNamespace(statpath=statpath, maskpath=str(tmp_path / "masks"), mask_sub="masks",
                               region="test", dataset="ERA5", grg_grid_spacing=0.5, grg_cell_size=2,
                               altitude_threshold=1500, decadal_only=True)
        mask, areas, full_mask = _load_gr_grid_static(opts)
        assert mask is None
        assert areas is None
        assert full_mask is None

    def test_load_gr_grid_static_reads_full_and_filtered_masks(self, tmp_path):
        mask_dir = tmp_path / "masks" / "masks"
        mask_dir.mkdir(parents=True)
        statpath = tmp_path / "stats"
        statpath.mkdir()
        coords = {"lat": [2.0, 1.0], "lon": [3.0, 4.0]}
        mask = xr.DataArray(np.full((2, 2), 0.5), coords=coords,
                            dims=("lat", "lon"), name="mask")
        full_mask = xr.DataArray(np.ones((2, 2)), coords=coords,
                                 dims=("lat", "lon"), name="mask_full")
        areas = xr.DataArray(np.ones((2, 2)), coords=coords,
                             dims=("lat", "lon"), name="area_grid")
        xr.Dataset({"mask": mask, "mask_full": full_mask}).to_netcdf(
            mask_dir / "test_mask_0p5_cell0p5_ERA5_1500.nc")
        areas.to_dataset().to_netcdf(statpath / "area_grid_0p5_cell0p5_test_ERA5.nc")
        opts = SimpleNamespace(statpath=str(statpath), maskpath=str(tmp_path / "masks"), mask_sub="masks",
                               region="test", dataset="ERA5", grg_grid_spacing=0.5, grg_cell_size=0.5,
                               altitude_threshold=1500, decadal_only=True)

        loaded_mask, loaded_areas, loaded_full_mask = _load_gr_grid_static(opts)

        xr.testing.assert_equal(loaded_mask, mask)
        xr.testing.assert_equal(loaded_full_mask, full_mask)
        xr.testing.assert_equal(loaded_areas, areas)

    def test_load_gr_grid_static_ignores_untagged_legacy_cache(self, tmp_path):
        from argparse import Namespace

        mask_dir = tmp_path / "masks" / "masks"
        mask_dir.mkdir(parents=True)
        statpath = tmp_path / "stats"
        statpath.mkdir()
        legacy_mask = xr.DataArray(
            np.ones((2, 2)), coords={"lat": [2.0, 1.0], "lon": [3.0, 4.0]},
            dims=("lat", "lon"), name="mask")
        legacy_mask.to_dataset().to_netcdf(mask_dir / "test_mask_0p5_ERA5_1500.nc")
        xr.ones_like(legacy_mask).rename("area_grid").to_dataset().to_netcdf(
            statpath / "area_grid_0p5_test_ERA5.nc")
        opts = Namespace(statpath=str(statpath), maskpath=str(tmp_path / "masks"), mask_sub="masks",
                         region="test", dataset="ERA5", grg_grid_spacing=0.5, grg_cell_size=0.5,
                         altitude_threshold=1500, decadal_only=True)

        mask, areas, full_mask = _load_gr_grid_static(opts)

        assert mask is None
        assert areas is None
        assert full_mask is None


class TestCompareToCTPRef:
    def test_compare_to_ctp_ref_no_file(self, tea_constant, tmp_path):
        ref_path = str(tmp_path / "nonexistent.nc")
        _compare_to_ctp_ref(tea_constant, ref_path)
