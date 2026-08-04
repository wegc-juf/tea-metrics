from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import xarray as xr
from shapely.geometry import box

from teametrics.utils import create_region_masks as masks


def _opts():
    return SimpleNamespace(
        region='TEST',
        dataset='ERA5',
        xy_name='lon,lat',
        target_sys=4326,
        altitude_threshold=0,
        parallel_workers=1,
        start=2000,
        script='create_region_masks.py',
    )


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
