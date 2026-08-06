#!/usr/bin/env python3
"""
create region masks for TEA indicator calculation
author: hst
"""

import geopandas as gpd
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import time
from shapely import area as shapely_area
from shapely import box as shapely_box
from shapely import intersection as shapely_intersection
import xarray as xr

from ..common.general_functions import create_history_from_cfg, get_gridded_data
from ..common.config import load_opts
from ..common.TEA_logger import logger
from ..calc_TEA import _getopts
from ..TEA import TEAIndicators


def _load_shp(opts):
    """
    load shp file for given region
    Args:
        opts: CLI parameter

    Returns:
        shp: geopandas df of shp file

    """

    start = time.perf_counter()
    logger.info(f'Loading shape file {opts.shpfile}')
    shp = gpd.read_file(opts.shpfile)

    if opts.subreg:
        try:
            shp = shp[(shp.CNTR_ID == opts.region)]
        except AttributeError:
            try:
                shp = shp[(shp.LAND_NAME == opts.region)]
            except AttributeError:
                raise AttributeError('The given shape file has neither CNTR_ID nor '
                                     'LAND_NAME information.')

    # Transfer it to the wanted coordinate system
    shp = shp.to_crs(epsg=opts.target_sys)
    logger.info(f'Loaded and reprojected {len(shp)} shape features in '
                f'{time.perf_counter() - start:.2f}s')

    return shp


def _intersect_cells(cells, poly, workers):
    """Return intersection areas, optionally evaluating geometry chunks in parallel."""
    start = time.perf_counter()
    parallel = workers > 1 and len(cells) >= 10_000
    logger.info(f'Calculating intersections for {len(cells):,} candidate cells '
                f'using {workers if parallel else 1} worker thread(s)')
    if workers <= 1 or len(cells) < 10_000:
        result = shapely_area(shapely_intersection(cells, poly))
    else:
        chunks = np.array_split(cells, workers)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            areas = executor.map(
                lambda chunk: shapely_area(shapely_intersection(chunk, poly)), chunks
            )
            result = np.concatenate(tuple(areas))

    logger.info(f'Calculated cell intersections in {time.perf_counter() - start:.2f}s')
    return result


def _create_cell_polygons(opts, xvals, yvals, offset, x_indices=None, y_indices=None):
    """
    create list of polygons for each cell
    Args:
        opts: CLI parameter
        xvals: x-coordinates
        yvals: y-coordinates
        offset: half of grid spacing

    Returns:
        cells: list of cell polygons

    """

    xvals = np.asarray(xvals)
    yvals = np.asarray(yvals)
    if x_indices is None:
        x_indices = np.arange(len(xvals))
    if y_indices is None:
        y_indices = np.arange(len(yvals))
    x_grid, y_grid = np.meshgrid(xvals, yvals)
    ix_grid, iy_grid = np.meshgrid(y_indices, x_indices, indexing='ij')
    return {
        'ix': ix_grid.ravel(),
        'iy': iy_grid.ravel(),
        'geometry': shapely_box(
            (x_grid - offset).ravel(), (y_grid - offset).ravel(),
            (x_grid + offset).ravel(), (y_grid + offset).ravel(),
        ),
    }


def create_sea_mask(opts):
    """
    create SEA mask (part of SAR that's within AUT)
    Args:
        opts: Options as defined in config file

    Returns:

    """

    start = time.perf_counter()
    logger.info(f'Creating SEA mask from AUT and SAR masks for {opts.dataset}')
    try:
        suffix = f'_{opts.altitude_threshold}'
        aut = xr.open_dataset(Path(opts.maskpath) / opts.mask_sub / f'AUT_mask_{opts.dataset}{suffix}.nc')
        sar = xr.open_dataset(Path(opts.maskpath) / opts.mask_sub / f'SAR_mask_{opts.dataset}{suffix}.nc')
    except FileNotFoundError:
        raise FileNotFoundError('For SEA mask, run create_region_masks.py for AUT and SAR first.')

    logger.info(f'Loaded AUT and SAR masks in {time.perf_counter() - start:.2f}s')

    mask = aut['mask'].where(sar['mask'].notnull())
    mask = mask.rename('mask')
    mask.attrs = {'long_name': 'weighted mask', 'coordinate_sys': f'EPSG:{opts.target_sys}'}
    ds = mask.to_dataset()
    if getattr(opts, 'calculate_area', False):
        ds = _add_area_variables(ds, mask, mask, opts)

    create_history_from_cfg(cfg_params=opts, ds=ds)

    _save_output(ds, opts)


def _prep_lsm(opts):
    """
    load Land-Sea-Mask and convert coordinates
    Args:
        opts: CLI parameter

    Returns:
        lsm: land sea mask
    """
    start = time.perf_counter()
    logger.info(f'Loading land-sea mask {opts.lsmfile}')
    lsm_raw = xr.open_dataset(opts.lsmfile)

    logger.info(f'Loading orography grid {opts.orofile} for coordinate alignment')
    data = xr.open_dataset(opts.orofile)
    data = data.altitude

    if opts.region != 'GLO':
        # get lat resolution
        lat_resolution = round(abs(data.lat.values[0] - data.lat.values[1]), 4)

        step = lat_resolution

        # split to eastern and western hemisphere (from 0 ... 360 to -180 .. 180)
        lsm_e = lsm_raw.sel(longitude=slice(180 + step, 360))
        lsm_w = lsm_raw.sel(longitude=slice(0, 180))
        lsm_values = np.concatenate((lsm_e.lsm.values[0, :, :], lsm_w.lsm.values[0, :, :]), axis=1)

        lsm_lon = np.arange(-180, 180, step).astype('float32')
        lsm_lat = lsm_raw.latitude.values

        lsm = xr.DataArray(data=lsm_values, dims=('lat', 'lon'), coords={
            'lon': (['lon'], lsm_lon), 'lat': (['lat'], lsm_lat)})

        if opts.dataset == 'ERA5Land':
            lsm = lsm.interp(lon=(np.arange(-1800, 1800, step * 10) / 10),
                             lat=(np.arange(-900, 900, step * 10) / 10)[::-1])
    else:
        lsm = lsm_raw.lsm.sel(time=lsm_raw.time[0])
        lsm = lsm.rename({'latitude': 'lat', 'longitude': 'lon'})

    lsm = lsm.sel(lat=data.lat.values, lon=data.lon.values)

    logger.info(f'Prepared land-sea mask with dimensions {dict(lsm.sizes)} in '
                f'{time.perf_counter() - start:.2f}s')

    return lsm


def create_agr_mask(opts):
    """
    create mask for AGRs
    Args:
        opts: CLI parameter

    Returns:

    """
    if 'ERA5' not in opts.dataset:
        raise AttributeError('AGR mask can only be created for ERA5(Land) data.')

    logger.info(f'Creating AGR mask for {opts.region} / {opts.dataset}')
    mask = _prep_lsm(opts=opts)
    mask = mask.where(mask > opts.land_frac_min)
    mask = mask.rename('mask')
    mask.attrs = {'long_name': 'weighted mask', 'coordinate_sys': f'EPSG:{opts.target_sys}'}
    full_mask = mask.copy()

    # apply altitude threshold if set
    if opts.altitude_threshold != 0:
        mask = _apply_altitude_threshold(mask, opts)
    ds = mask.to_dataset()
    if getattr(opts, 'calculate_area', False):
        ds = _add_area_variables(ds, full_mask, mask, opts)
    
    create_history_from_cfg(cfg_params=opts, ds=ds)

    _save_output(ds, opts)


def _apply_altitude_threshold(mask, opts):
    """
    apply altitude threshold to mask
    Args:
        mask: mask DataArray
        opts: Config parameters

    Returns:
        mask: mask DataArray with altitude threshold applied

    """
    start = time.perf_counter()
    logger.info(f'Applying altitude threshold of {opts.altitude_threshold} to mask')
    # load orography
    orog = xr.open_dataset(opts.orofile)
    if 'altitude' in orog.data_vars:
        orog = orog.altitude
    elif 'elevation' in orog.data_vars:
        orog = orog.elevation
    else:
        orog = orog.orog

    mask = mask.where(orog < opts.altitude_threshold)
    logger.info(f'Applied altitude threshold in {time.perf_counter() - start:.2f}s')
    return mask


def _create_area_grid(mask):
    """Create a mask-weighted area grid using the TEA implementation."""
    return TEAIndicators(mask=mask).area_grid


def _add_area_variables(ds, full_mask, mask, opts):
    """Add full and altitude-filtered area grids and their summed sizes."""
    area_grid_full = _create_area_grid(full_mask)
    area_grid = _create_area_grid(mask)
    area_attrs = {
        'long_name': 'mask-weighted grid-cell area',
        'units': 'areal (100 m2)',
        'coordinate_sys': f'EPSG:{opts.target_sys}',
    }
    area_size_attrs = {
        'long_name': 'summed mask area',
        'units': 'areal (100 m2)',
    }
    area_grid_full.attrs = area_attrs
    area_grid.attrs = area_attrs
    ds['area_grid_full'] = area_grid_full
    ds['area_full'] = (area_grid_full.sum(skipna=True).rename('area_full')
                       .assign_attrs(area_size_attrs))
    ds['area_grid'] = area_grid
    ds['area'] = area_grid.sum(skipna=True).rename('area').assign_attrs(area_size_attrs)
    return ds


def _find_closest(coords, corner_val, direction):
    """
    Find the closest value in sorted_list to the target with a given direction.
    direction=-1 means the closest on the left (smaller than target)
    direction=1 means the closest on the right (larger than target)
    """
    if direction == 1:
        for i in range(len(coords)):
            if coords[i] > corner_val:
                return coords[i]
        return coords[-1]  # If no larger value found, return the last value
    elif direction == -1:
        for i in reversed(range(len(coords))):
            if coords[i] < corner_val:
                return coords[i]
        return coords[0]  # If no smaller value found, return the first value
    else:
        raise ValueError('Direction must be either -1 or 1.')


def _save_output(ds, opts, out_region=None):
    """
    save output to netcdf files
    Args:
        ds: dataset
        opts: options
        out_region: region acronym (default: opts.region)

    Returns:

    """
    if out_region is None:
        out_region = opts.region
    outpath = Path(opts.maskpath) / opts.mask_sub / f'{out_region}_mask_{opts.dataset}_{opts.altitude_threshold}.nc'
    start = time.perf_counter()
    logger.info(f'Saving mask file to {outpath}')
    ds.to_netcdf(outpath)
    logger.info(f'Saved mask file in {time.perf_counter() - start:.2f}s')


def create_rectangular_gr(opts):
    """
    create rectangular grid mask using either corners or center coordinates + extent
    Args:
        opts: options as defined in the config file

    Returns:

    """
    start = time.perf_counter()
    logger.info(f'Creating rectangular mask for {opts.dataset}')
    # load template file
    template_file = get_gridded_data(opts.start, opts.start + 1, opts)
    xy = opts.xy_name.split(',')
    x, y = xy[0], xy[1]
    logger.info(f'Template grid loaded with {x}={len(template_file[x])}, '
                f'{y}={len(template_file[y])}')
    dx = template_file[x][1] - template_file[x][0]
    dy = abs(template_file[y][1] - template_file[y][0])

    # get corners from CFG file
    if opts.gr_type == 'corners':
        sw_coords = opts.sw_corner.split(',')
        ne_coords = opts.ne_corner.split(',')
        sw_coords = [float(ii) for ii in sw_coords]
        ne_coords = [float(ii) for ii in ne_coords]
    elif opts.gr_type == 'center':
        center_coords = opts.center.split(',')
        center_coords = [float(ii) for ii in center_coords]
        sw_coords = [center_coords[0] - float(opts.we_len) / 2, center_coords[1] - float(opts.ns_len) / 2]
        ne_coords = [center_coords[0] + float(opts.we_len) / 2, center_coords[1] + float(opts.ns_len) / 2]
    else:
        raise ValueError('gr_type must be either "polygon", "corners", or "center".')

    if 'ERA5' in opts.dataset:
        yidxn, yidxx = -1, 0
    else:
        yidxn, yidxx = 0, -1

    xn, xx = sw_coords[0], ne_coords[0]
    yn, yx = sw_coords[1], ne_coords[1]

    # check if corners are within grid
    if any(xv < template_file[x][0] for xv in [xn, xx]) or any(xv > template_file[x][-1] for xv in [xn, xx]):
        raise KeyError('Passed corner(s) are outside of target grid!')
    if any(yv < template_file[y][yidxn] for yv in [yn, yx]) or any(yv > template_file[y][yidxx] for yv in [yn, yx]):
        raise KeyError('Passed corner(s) are outside of target grid!')

    # create non weighted mask array
    nw_mask_arr = np.full((len(template_file[y]), len(template_file[x])), np.nan)
    da_nwmask = xr.DataArray(data=nw_mask_arr, coords={y: ([y], template_file[y].data),
                                                       x: ([x], template_file[x].data)},
                             attrs={'long_name': 'non weighted mask',
                                    'coordinate_sys': f'EPSG:{opts.target_sys}'},
                             name='nw_mask')

    # check if corners are identical with grid points on target grid
    xvals_check = all(xv in template_file[x] for xv in [xn, xx])
    yvals_check = all(yv in template_file[y] for yv in [yn, yx])

    # set values in non-weighted mask within GR to 1 and create weighted mask
    if xvals_check and yvals_check:
        da_nwmask.loc[yn:yx, xn:xx] = 1
        da_mask = da_nwmask.copy()
        da_mask = da_mask.rename('mask')
        da_mask.attrs['long_name'] = 'non weighted mask'
    else:
        # Find the closest x and y for the corners and calculate fractions of cell area
        if 'ERA5' in opts.dataset:
            closest_sw_y = _find_closest(template_file[y][::-1], yn, direction=1)
            closest_ne_y = _find_closest(template_file[y][::-1], yx, direction=-1)
            s_frac = (closest_sw_y - yn) / dy
            n_frac = (yx - closest_ne_y) / dy
        else:
            closest_sw_y = _find_closest(template_file[y], yn, direction=-1)
            closest_ne_y = _find_closest(template_file[y], yx, direction=1)
            s_frac = (yn - closest_sw_y) / dy
            n_frac = (closest_ne_y - yx) / dy
        closest_sw_x = _find_closest(template_file[x], xn, direction=-1)
        closest_ne_x = _find_closest(template_file[x], xx, direction=1)
        w_frac = (xn - closest_sw_x) / dx
        e_frac = (closest_ne_x - xx) / dx

        # set values in non-weighted mask within GR to 1
        if 'ERA5' in opts.dataset:
            da_nwmask.loc[closest_ne_y:closest_sw_y, closest_sw_x:closest_ne_x] = 1
        else:
            da_nwmask.loc[closest_sw_y:closest_ne_y, closest_sw_x:closest_ne_x] = 1

        # create weighted mask
        da_mask = da_nwmask.copy()
        da_mask = da_mask.rename('mask')
        da_mask.attrs['long_name'] = 'non weighted mask'

        # apply fractions to mask to get weighted mask
        da_mask.loc[:, closest_sw_x] = da_mask.loc[:, closest_sw_x] * w_frac
        da_mask.loc[closest_sw_y, :] = da_mask.loc[closest_sw_y, :] * s_frac
        da_mask.loc[:, closest_ne_x] = da_mask.loc[:, closest_ne_x] * e_frac
        da_mask.loc[closest_ne_y, :] = da_mask.loc[closest_ne_y, :] * n_frac

    if opts.altitude_threshold != 0:
        full_mask = da_mask.copy()
        da_mask = _apply_altitude_threshold(da_mask, opts)
    else:
        full_mask = da_mask.copy()
    ds_mask = da_mask.to_dataset()
    if getattr(opts, 'calculate_area', False):
        ds_mask = _add_area_variables(ds_mask, full_mask, da_mask, opts)

    create_history_from_cfg(cfg_params=opts, ds=ds_mask)

    out_region = f'SW_{xn:.1f}_{yn:.1f}-NE_{xx:.1f}_{yx:.1f}'
    logger.info(f'Created rectangular mask in {time.perf_counter() - start:.2f}s')
    _save_output(ds_mask, opts, out_region)


def create_mask_file(opts):
    """
    create mask file for given region
    Args:
        opts: Options as defined in config file

    Returns:

    """
    start = time.perf_counter()
    logger.info(f'Creating polygon mask for {opts.region} / {opts.dataset}')
    # Load template file
    template_file = get_gridded_data(opts.start, opts.start + 1, opts)
    xy = opts.xy_name.split(',')
    x, y = xy[0], xy[1]
    logger.info(f'Template grid loaded with {x}={len(template_file[x])}, '
                f'{y}={len(template_file[y])}')

    # Load shp file and transform it to desired coordinate system
    shp = _load_shp(opts=opts)
    logger.info(f'Preparing geometry from {len(shp)} shape feature(s)')
    # Define the cell grid
    xvals, yvals = template_file[x], template_file[y]

    # Get grid spacing
    # Coordinates of ERA5(Land) have some precision trouble
    if opts.dataset in ['ERA5', 'ERA5Land']:
        dx = set(abs(np.round(xvals[1:].values - xvals[:-1].values, 2)))
        dy = set(abs(np.round(yvals[1:].values - yvals[:-1].values, 2)))
    else:
        dx = set(abs(xvals[1:].values - xvals[:-1].values))
        dy = set(abs(yvals[1:].values - yvals[:-1].values))
    if len(dx) > 1 or len(dy) > 1 or dx != dy:
        raise ValueError('The given test file does not have a regular grid. '
                         'Provide a file with a regular grid.')
    dx, dy = list(dx)[0], list(dy)[0]
    offset = dx / 2

    # Initialize mask array
    mask = np.zeros(shape=(len(yvals), len(xvals)), dtype='float32')

    if len(shp) == 0:
        raise ValueError(f'No geometry found for region {opts.region!r}.')
    poly = shp.geometry.iloc[0] if len(shp) == 1 else shp.geometry.union_all()
    logger.info(f'Prepared region geometry with bounds {poly.bounds}')

    if poly.is_empty:
        raise ValueError(f'No geometry found for region {opts.region!r}.')

    min_x, min_y, max_x, max_y = poly.bounds
    x_indices = np.flatnonzero((xvals >= min_x - offset) & (xvals <= max_x + offset))
    y_indices = np.flatnonzero((yvals >= min_y - offset) & (yvals <= max_y + offset))
    if not len(x_indices) or not len(y_indices):
        raise ValueError(f'Region {opts.region!r} does not overlap the target grid.')

    logger.info(f'Restricted intersection calculation to {len(x_indices)} x '
                f'{len(y_indices)} = {len(x_indices) * len(y_indices):,} candidate cells')

    cells = _create_cell_polygons(
        opts=opts,
        xvals=xvals[x_indices],
        yvals=yvals[y_indices],
        offset=offset,
        x_indices=x_indices,
        y_indices=y_indices,
    )
    areas = _intersect_cells(
        cells['geometry'], poly, max(1, getattr(opts, 'mask_parallel_workers', 1))
    )
    mask[cells['ix'], cells['iy']] = np.clip(areas / (4 * offset * offset), 0, 1)
    logger.info(f'Calculated coverage for {np.count_nonzero(areas > 0):,} cells')

    # Set cells outside of region to nan
    mask[np.where(mask == 0)] = np.nan

    # Create non-weighted mask
    nw_mask = mask.copy()
    nw_mask[np.where(mask > 0)] = 1

    # Convert to da
    da_mask = xr.DataArray(data=mask, coords={y: ([y], yvals.data), x: ([x], xvals.data)},
                           attrs={'long_name': 'weighted mask',
                                  'coordinate_sys': f'EPSG:{opts.target_sys}'},
                           name='mask')
    full_mask = da_mask.copy()
    if opts.altitude_threshold != 0:
        da_mask = _apply_altitude_threshold(da_mask, opts)
        
    ds_mask = da_mask.to_dataset()
    if getattr(opts, 'calculate_area', False):
        ds_mask = _add_area_variables(ds_mask, full_mask, da_mask, opts)
    create_history_from_cfg(cfg_params=opts, ds=ds_mask)
    out_region = opts.region
    _save_output(ds_mask, opts, out_region)
    logger.info(f'Created polygon mask in {time.perf_counter() - start:.2f}s')


def run():
    cmd_opts = _getopts()
    
    # load CFG parameter
    opts = load_opts(fname=__file__, config_file=cmd_opts.config_file)
    logger.info(f'Starting create_region_masks for region {opts.region}, '
                f'dataset {opts.dataset}, gr_type {opts.gr_type}')
    
    if opts.gr_type != 'polygon':
        create_rectangular_gr(opts=opts)
    elif opts.region == 'SEA':
        create_sea_mask(opts=opts)
    elif opts.region in ['EUR', 'AFR', 'GLO']:
        create_agr_mask(opts=opts)
    else:
        create_mask_file(opts)

    logger.info('create_region_masks completed successfully')


if __name__ == '__main__':
    run()
