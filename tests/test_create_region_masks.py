from types import SimpleNamespace

import argparse
import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from shapely.geometry import box

from teametrics.utils import create_region_masks as masks
from teametrics.common.config import check_type


def _opts():
    return SimpleNamespace(
        region='TEST',
        dataset='ERA5',
        xy_name='lon,lat',
        target_sys=4326,
        orofile='orography.nc',
        altitude_threshold=0,
        calculate_area=False,
        parallel_workers=1,
        mask_parallel_workers=1,
        start=2000,
        script='create_region_masks.py',
    )


def test_mask_parallel_workers_allows_ceiling_100():
    check_type('mask_parallel_workers', 100)

    try:
        check_type('mask_parallel_workers', 101)
    except argparse.ArgumentTypeError:
        pass
    else:
        raise AssertionError('mask_parallel_workers should be capped at 100')


def test_create_mask_file_includes_last_row_and_column(monkeypatch):
    opts = _opts()
    template = xr.Dataset(coords={'lon': [0.0, 1.0, 2.0], 'lat': [0.0, 1.0, 2.0]})
    captured = {}
    messages = []
    shape = gpd.GeoDataFrame(geometry=[box(1.5, 1.5, 2.5, 2.5)], crs='EPSG:4326')

    monkeypatch.setattr(masks.logger, 'info', messages.append)
    monkeypatch.setattr(masks, 'get_gridded_data', lambda *args, **kwargs: template)
    monkeypatch.setattr(masks, '_load_shp', lambda opts: shape)
    monkeypatch.setattr(masks, '_save_output', lambda ds, opts, out_region=None: captured.update(ds=ds))

    masks.create_mask_file(opts)

    result = captured['ds'].mask.values
    assert result.shape == (3, 3)
    assert result[2, 2] == 1
    assert np.isnan(result[0, 0])
    assert any('Restricted intersection calculation to 2 x 2' in message for message in messages)
    assert any('Calculated cell intersections' in message for message in messages)


def test_create_mask_file_unions_multiple_features(monkeypatch):
    opts = _opts()
    template = xr.Dataset(coords={'lon': [0.0, 1.0, 2.0], 'lat': [0.0, 1.0, 2.0]})
    captured = {}
    shape = gpd.GeoDataFrame(
        geometry=[box(-0.5, -0.5, 0.5, 0.5), box(1.5, 1.5, 2.5, 2.5)],
        crs='EPSG:4326',
    )

    monkeypatch.setattr(masks, 'get_gridded_data', lambda *args, **kwargs: template)
    monkeypatch.setattr(masks, '_load_shp', lambda opts: shape)
    monkeypatch.setattr(masks, '_save_output', lambda ds, opts, out_region=None: captured.update(ds=ds))

    masks.create_mask_file(opts)

    result = captured['ds'].mask.values
    assert result[0, 0] == 1
    assert result[2, 2] == 1
    assert np.isnan(result[1, 1])


def test_create_mask_file_writes_full_and_filtered_areas(monkeypatch):
    opts = _opts()
    opts.calculate_area = True
    opts.altitude_threshold = 100
    template = xr.Dataset(coords={'lon': [0.0, 1.0], 'lat': [0.0, 1.0]})
    orography = xr.Dataset({'altitude': (('lat', 'lon'), [[0, 200], [0, 0]])},
                           coords=template.coords)
    captured = {}
    shape = gpd.GeoDataFrame(geometry=[box(-0.5, -0.5, 1.5, 1.5)], crs='EPSG:4326')

    monkeypatch.setattr(masks, 'get_gridded_data', lambda *args, **kwargs: template)
    monkeypatch.setattr(masks, '_load_shp', lambda opts: shape)
    monkeypatch.setattr(masks.xr, 'open_dataset', lambda *args, **kwargs: orography)
    monkeypatch.setattr(masks, '_save_output', lambda ds, opts, out_region=None: captured.update(ds=ds))

    masks.create_mask_file(opts)

    result = captured['ds']
    assert {'mask', 'mask_full', 'area_grid_full', 'area_full', 'area_grid', 'area'} <= set(result.data_vars)
    assert result.area_full.item() > result.area.item()
    assert np.isnan(result.area_grid.values[0, 1])
    assert np.isfinite(result.area_grid_full.values[0, 1])
    assert np.isfinite(result.mask_full.values[0, 1])
    assert np.isnan(result.mask.values[0, 1])


def test_create_sea_mask_preserves_pre_altitude_region_and_areas(monkeypatch, tmp_path):
    opts = _opts()
    opts.region = 'SEA'
    opts.maskpath = str(tmp_path)
    opts.mask_sub = 'masks'
    opts.altitude_threshold = 1500
    opts.calculate_area = True
    coords = {'lat': [1.0, 0.0], 'lon': [0.0, 1.0]}
    aut = xr.Dataset({
        'mask': (('lat', 'lon'), [[1.0, np.nan], [1.0, 1.0]]),
        'mask_full': (('lat', 'lon'), np.ones((2, 2))),
    }, coords=coords)
    sar = xr.Dataset({
        'mask': (('lat', 'lon'), [[1.0, 1.0], [np.nan, 1.0]]),
        'mask_full': (('lat', 'lon'), np.ones((2, 2))),
    }, coords=coords)
    captured = {}

    def open_mask(path):
        return aut if 'AUT_mask_' in str(path) else sar

    monkeypatch.setattr(masks.xr, 'open_dataset', open_mask)
    monkeypatch.setattr(masks, 'create_history_from_cfg', lambda **kwargs: None)
    monkeypatch.setattr(masks, '_save_output', lambda ds, opts, out_region=None: captured.update(ds=ds))

    masks.create_sea_mask(opts)

    result = captured['ds']
    assert {'mask', 'mask_full', 'area_grid_full', 'area_grid'} <= set(result.data_vars)
    assert np.isnan(result.mask.values[0, 1])
    assert np.isnan(result.mask.values[1, 0])
    assert np.all(result.mask_full.values == 1)
    assert np.isfinite(result.area_grid_full.values).all()
    assert np.isnan(result.area_grid.values[0, 1])
    assert np.isnan(result.area_grid.values[1, 0])


def test_create_sea_mask_requires_full_aut_sar_masks_for_altitude_filter(monkeypatch, tmp_path):
    opts = _opts()
    opts.region = 'SEA'
    opts.maskpath = str(tmp_path)
    opts.mask_sub = 'masks'
    opts.altitude_threshold = 1500
    mask = xr.DataArray(np.ones((2, 2)), dims=('lat', 'lon'), name='mask')
    mask_ds = mask.to_dataset()
    monkeypatch.setattr(masks.xr, 'open_dataset', lambda *args, **kwargs: mask_ds)

    with pytest.raises(ValueError, match='mask_full'):
        masks.create_sea_mask(opts)
