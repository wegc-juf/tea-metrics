import argparse

import pytest

from teametrics.common.config import _get_default_opts


def test_legacy_agr_grid_spacing_is_mapped_to_grg():
    opts = argparse.Namespace(agr_grid_spacing=0.25)

    with pytest.warns(DeprecationWarning, match='agr_grid_spacing'):
        _get_default_opts('test', opts)

    assert opts.grg_grid_spacing == 0.25
    assert not hasattr(opts, 'agr_grid_spacing')


def test_canonical_grid_spacing_takes_precedence_when_values_match():
    opts = argparse.Namespace(grg_grid_spacing=0.25, agr_grid_spacing=0.25)

    with pytest.warns(DeprecationWarning, match='agr_grid_spacing'):
        _get_default_opts('test', opts)

    assert opts.grg_grid_spacing == 0.25
    assert not hasattr(opts, 'agr_grid_spacing')


def test_conflicting_grid_spacing_names_are_rejected():
    opts = argparse.Namespace(grg_grid_spacing=0.25, agr_grid_spacing=0.5)

    with pytest.raises(ValueError, match='grid spacing'):
        _get_default_opts('test', opts)


def test_legacy_agr_cell_size_is_mapped_to_grg():
    opts = argparse.Namespace(agr_cell_size=0.5)

    with pytest.warns(DeprecationWarning, match='agr_cell_size'):
        _get_default_opts('test', opts)

    assert opts.grg_cell_size == 0.5
    assert not hasattr(opts, 'agr_cell_size')


def test_canonical_cell_size_takes_precedence_when_values_match():
    opts = argparse.Namespace(grg_cell_size=0.5, agr_cell_size=0.5)

    with pytest.warns(DeprecationWarning, match='agr_cell_size'):
        _get_default_opts('test', opts)

    assert opts.grg_cell_size == 0.5
    assert not hasattr(opts, 'agr_cell_size')


def test_conflicting_cell_size_names_are_rejected():
    opts = argparse.Namespace(grg_cell_size=0.5, agr_cell_size=1)

    with pytest.raises(ValueError, match='cell size'):
        _get_default_opts('test', opts)


@pytest.mark.parametrize(('precip', 'expected'), [(True, 1), (False, 2)])
def test_grg_cell_size_default(precip, expected):
    opts = argparse.Namespace(agr='AUT', precip=precip, agr_range=None)

    _get_default_opts('test', opts)

    assert opts.grg_cell_size == expected
