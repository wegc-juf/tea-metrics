from types import SimpleNamespace

import argparse
import geopandas as gpd
import numpy as np
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
    assert {'mask', 'area_grid_full', 'area_full', 'area_grid', 'area'} <= set(result.data_vars)
    assert result.area_full.item() > result.area.item()
    assert np.isnan(result.area_grid.values[0, 1])
    assert np.isfinite(result.area_grid_full.values[0, 1])
