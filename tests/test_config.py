import argparse

import pytest

from teametrics.common.config import _get_default_opts


def test_legacy_agr_grid_spacing_is_mapped_to_grg():
    opts = argparse.Namespace(agr_grid_spacing=0.25)

    _get_default_opts('test', opts)

    assert opts.grg_grid_spacing == 0.25
    assert not hasattr(opts, 'agr_grid_spacing')


def test_canonical_grid_spacing_takes_precedence_when_values_match():
    opts = argparse.Namespace(grg_grid_spacing=0.25, agr_grid_spacing=0.25)

    _get_default_opts('test', opts)

    assert opts.grg_grid_spacing == 0.25
    assert not hasattr(opts, 'agr_grid_spacing')


def test_conflicting_grid_spacing_names_are_rejected():
    opts = argparse.Namespace(grg_grid_spacing=0.25, agr_grid_spacing=0.5)

    with pytest.raises(ValueError, match='grid spacing'):
        _get_default_opts('test', opts)
