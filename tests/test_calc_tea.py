import pytest
import numpy as np
import xarray as xr
from types import SimpleNamespace

_import_error = None
try:
    from teametrics.calc_TEA import (
        _getopts, _get_ctp_filepath, _calc_x_y_range, _reduce_region,
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
        assert opts.loglevel == "DEBUG"

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
                         grg_grid_spacing=0.5, param_str="Tx99p", period="annual",
                         dataset="ERA5")
        path = _get_ctp_filepath(1980, 1990, opts, annual_agr=True)
        assert path.endswith(".nc")
        assert "agr" in path.lower()


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
        opts = SimpleNamespace(region="test", agr_cell_size=0.5, threshold_type="perc")
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
        ds['area_grid_full'] = xr.ones_like(mask)
        ds['area_full'] = xr.DataArray(4.0)
        ds['area_grid'] = xr.ones_like(mask)
        ds['area'] = xr.DataArray(4.0)
        mask_dir = tmp_path / 'masks'
        mask_dir.mkdir()
        ds.to_netcdf(mask_dir / 'AUT_mask_ERA5_1500.nc')
        opts = SimpleNamespace(gr_type='polygon', maskpath=str(tmp_path), mask_sub='masks',
                               region='AUT', dataset='ERA5', altitude_threshold=1500)

        result, area_grid = _load_mask_file(opts, include_area=True)

        xr.testing.assert_equal(result, mask)
        xr.testing.assert_equal(area_grid, ds.area_grid)

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
        opts = SimpleNamespace(region="test", agr_cell_size=0.5, threshold_type="perc")
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
    def test_load_gr_grid_static_not_found(self, tmp_path):
        from types import SimpleNamespace
        statpath = str(tmp_path / "stats")
        opts = SimpleNamespace(statpath=statpath, maskpath=str(tmp_path / "masks"), mask_sub="masks",
                               region="test", dataset="ERA5", grg_grid_spacing=0.5,
                               altitude_threshold=1500, decadal_only=True)
        mask, areas = _load_gr_grid_static(opts)
        assert mask is None
        assert areas is None


class TestCompareToCTPRef:
    def test_compare_to_ctp_ref_no_file(self, tea_constant, tmp_path):
        ref_path = str(tmp_path / "nonexistent.nc")
        _compare_to_ctp_ref(tea_constant, ref_path)
