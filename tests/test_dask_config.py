import dask
import dask.array as da
import xarray as xr

from teametrics.common.dask_config import configure_dask_data


def test_auto_keeps_small_data_eager():
    data = xr.DataArray(da.zeros((10, 2, 2), chunks=(10, 2, 2)), dims=('time', 'y', 'x'))

    result, use_dask = configure_dask_data(data, use_dask='auto')

    assert use_dask is False
    assert not hasattr(result.data, 'dask')


def test_true_uses_resource_aware_spatial_chunks(monkeypatch):
    monkeypatch.setattr('teametrics.common.dask_config.os.cpu_count', lambda: 4)
    monkeypatch.setattr(
        'teametrics.common.dask_config.psutil.virtual_memory',
        lambda: type('Memory', (), {'available': 8 * 1024 ** 3})(),
    )
    data = xr.DataArray(
        da.zeros((3650, 100, 200), chunks=(365, 100, 200)),
        dims=('time', 'y', 'x'),
    )

    result, use_dask = configure_dask_data(data, use_dask=True)

    assert use_dask is True
    assert result.chunksizes['time'] == (3650,)
    assert result.chunksizes['y'][0] < 100 or result.chunksizes['x'][0] < 200
    assert dask.config.get('num_workers') == 4


def test_explicit_dask_workers_bounds_thread_count(monkeypatch):
    monkeypatch.setattr('teametrics.common.dask_config.os.cpu_count', lambda: 32)
    monkeypatch.setattr(
        'teametrics.common.dask_config.psutil.virtual_memory',
        lambda: type('Memory', (), {'available': 64 * 1024 ** 3})(),
    )
    data = xr.DataArray(
        da.zeros((3650, 100, 200), chunks=(365, 100, 200)),
        dims=('time', 'y', 'x'),
    )

    _, use_dask = configure_dask_data(data, use_dask=True, dask_workers=4)

    assert use_dask is True
    assert dask.config.get('num_workers') == 4


def test_auto_uses_dask_when_working_set_does_not_fit(monkeypatch):
    monkeypatch.setattr('teametrics.common.dask_config.os.cpu_count', lambda: 4)
    monkeypatch.setattr(
        'teametrics.common.dask_config.psutil.virtual_memory',
        lambda: type('Memory', (), {'available': 1 * 1024 ** 3})(),
    )
    data = xr.DataArray(
        da.zeros((3650, 100, 200), chunks=(365, 100, 200)),
        dims=('time', 'y', 'x'),
    )

    result, use_dask = configure_dask_data(data, use_dask='auto')

    assert use_dask is True
    assert hasattr(result.data, 'dask')
