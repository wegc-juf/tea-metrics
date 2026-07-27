import pytest
import numpy as np
import xarray as xr

_import_error = None
try:
    from teametrics.calc_TEA import (
        _getopts, _get_ctp_filepath, _calc_x_y_range, _reduce_region,
        _get_threshold, _load_mask_file, _load_gr_grid_static,
        _compare_to_ctp_ref, _load_population_grid, _reduce_region, calc_dbv_indicators,
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

    def test_getopts_version(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["calc_tea", "--config-file",
                                         "test.yaml", "--version"])
        opts = _getopts()
        assert opts.version


class TestGetCTPFilepath:
    def test_get_ctp_filepath_creates_dir(self, tmp_path):
        from types import SimpleNamespace
        output_path = str(tmp_path / "output" / "test")
        opts = SimpleNamespace(output_path=output_path, reg_name="test",
                               version="v1", variable="tas")
        path = _get_ctp_filepath(1980, 1990, opts)
        assert path.endswith(".nc")
        assert "test" in path

    def test_get_ctp_filepath_agr(self, tmp_path):
        from types import SimpleNamespace
        output_path = str(tmp_path / "output")
        opts = SimpleNamespace(output_path=output_path, reg_name="at",
                               version="v1", variable="tas")
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
        x_range, y_range = _calc_x_y_range(cell_size_y, mask)
        assert len(x_range) == 2
        assert len(y_range) == 2

    def test_calc_x_y_range_no_valid_cells(self):
        mask = xr.DataArray(
            np.zeros((4, 3)),
            dims=("y", "x"),
            coords={"y": range(4), "x": range(3)},
        )
        x_range, y_range = _calc_x_y_range(0.5, mask)
        assert x_range is None
        assert y_range is None


class TestReduceRegion:
    def test_reduce_region_no_crop(self):
        from types import SimpleNamespace
        opts = SimpleNamespace(region=None, full_region=False)
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
            threshold_type="abs", threshold_value=30.0,
            statpath="/tmp", variable="tas", reg_name="test",
            cell_size_y=0.5, start_year=1980, end_year=1994)
        result = _get_threshold(opts)
        assert result == 30.0


class TestLoadMaskFile:
    def test_load_mask_file_none(self):
        from types import SimpleNamespace
        opts = SimpleNamespace(mask_type=None, maskpath="/tmp",
                               reg_name="test")
        result = _load_mask_file(opts)
        assert result is None


class TestLoadPopulationGrid:
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
        opts = SimpleNamespace(statpath=statpath, reg_name="test",
                               variable="tas")
        mask, areas = _load_gr_grid_static(opts)
        assert mask is None
        assert areas is None


class TestCompareToCTPRef:
    def test_compare_to_ctp_ref_no_file(self, tea_constant, tmp_path):
        ref_path = str(tmp_path / "nonexistent.nc")
        _compare_to_ctp_ref(tea_constant, ref_path)
