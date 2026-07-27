"""Resource-aware configuration for local Dask calculations."""

import math
import os

import dask
import psutil

from .TEA_logger import logger


_MIB = 1024 ** 2
_GIB = 1024 ** 3


def configure_dask(data, use_dask='auto'):
    """
    Resolve Dask usage and configure the local threaded scheduler.

    ``use_dask`` may be ``True``, ``False``, or ``'auto'``. Auto mode keeps
    small datasets eager and limits workers to the available CPU and memory.
    """
    if use_dask is False:
        return False
    if use_dask is not True and use_dask != 'auto':
        raise ValueError("use_dask must be True, False, or 'auto'")

    cpu_count = max(1, os.cpu_count() or 1)
    available_memory = max(1, psutil.virtual_memory().available)
    data_size = data.nbytes if data.nbytes is not None else 0

    if use_dask == 'auto' and (cpu_count == 1 or data_size <= 256 * _MIB):
        logger.info("Dask auto mode: using eager NumPy execution for small input data.")
        return False

    workers_by_memory = max(1, available_memory // _GIB)
    workers = min(cpu_count, workers_by_memory, 32)
    dask.config.set(scheduler='threads', num_workers=workers)
    logger.info(f"Dask enabled with {workers} threaded workers "
                f"({available_memory / _GIB:.1f} GiB available, "
                f"{data_size / _MIB:.1f} MiB input).")
    return True


def configure_dask_data(data, use_dask='auto'):
    """Return data with resource-aware Dask settings applied."""
    if use_dask is False:
        data.load()
        return data, False

    resolved = configure_dask(data, use_dask=use_dask)
    if not resolved:
        data.load()
        return data, False

    available_memory = max(1, psutil.virtual_memory().available)
    workers = dask.config.get('num_workers') or 1
    target_chunk_bytes = min(256 * _MIB, max(32 * _MIB, available_memory // (workers * 8)))
    chunks = _spatial_chunks(data, target_chunk_bytes)
    if chunks:
        data = data.chunk(chunks)
        logger.info(f"Dask input chunks: {chunks}")
    return data, True


def _spatial_chunks(data, target_bytes):
    spatial_dims = [dim for dim in data.dims if dim != 'time']
    if len(spatial_dims) != 2 or 'time' not in data.dims:
        return {}

    dtype_size = data.dtype.itemsize
    time_size = data.sizes['time']
    cells_per_chunk = max(1, target_bytes // max(1, dtype_size * time_size))
    ydim, xdim = spatial_dims[-2:]
    y_size, x_size = data.sizes[ydim], data.sizes[xdim]
    aspect = x_size / max(1, y_size)
    y_chunk = max(1, min(y_size, int(math.sqrt(cells_per_chunk / max(aspect, 1e-12)))))
    x_chunk = max(1, min(x_size, int(cells_per_chunk / y_chunk)))
    return {"time": -1, ydim: y_chunk, xdim: x_chunk}
