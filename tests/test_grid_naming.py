from argparse import Namespace

import pytest

from teametrics.common.grid_naming import grg_grid_token
from teametrics.utils.calc_decadal_indicators import (
    get_amplification_outpath,
    get_decadal_outpath,
    load_ctp_data,
)


def test_grg_grid_token_includes_spacing_and_cell_size():
    opts = Namespace(grg_grid_spacing=0.25, grg_cell_size=0.5)

    assert grg_grid_token(opts) == '0p25_cell0p5'


def test_grg_grid_token_normalizes_integer_float_values():
    opts = Namespace(grg_grid_spacing=0.5, grg_cell_size=2.0)

    assert grg_grid_token(opts) == '0p5_cell2'


def test_grg_grid_token_uses_config_defaults_when_optional_values_are_absent():
    assert grg_grid_token(Namespace()) == '0p5_cell2'
    assert grg_grid_token(Namespace(precip=True)) == '0p5_cell1'


def test_decadal_and_amplification_paths_include_cell_size(tmp_path):
    opts = Namespace(
        outpath=str(tmp_path), agr='SEA', grg_grid_spacing=0.25, grg_cell_size=0.5,
        param_str='Tx25.0degC', period='annual', dataset='ERA5', start=1961, end=2025)

    decadal_path = get_decadal_outpath(opts, 'SEA')
    amplification_path = get_amplification_outpath(opts, 'SEA')

    assert 'AGR_0p25_cell0p5-SEA' in str(decadal_path)
    assert 'AGR_0p25_cell0p5-SEA' in str(amplification_path)


def test_ctp_loader_does_not_fall_back_to_untagged_legacy_files(tmp_path):
    class FakeAgr:
        pass

    opts = Namespace(
        outpath=str(tmp_path), param_str='Tx25.0degC', grg_grid_spacing=0.25,
        grg_cell_size=0.5, region='AUT', period='annual', dataset='ERA5',
        start=1961, end=2025)

    with pytest.raises(FileNotFoundError, match='grg_grid_spacing=0.25 and grg_cell_size=0.5'):
        load_ctp_data(opts, FakeAgr())
