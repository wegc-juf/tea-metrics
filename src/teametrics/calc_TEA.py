#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: hst, juf
"""

import os
import gc
import math
import warnings
from copy import deepcopy

import argparse
import numpy as np
from pathlib import Path
import xarray as xr

from .common.general_functions import (create_history_from_cfg, create_tea_history, compare_to_ref,
                                       get_gridded_data,
                                       get_csv_data, create_threshold_grid)
from .common.config import load_opts
from .common.TEA_logger import logger
from .utils.calc_decadal_indicators import (calc_decadal_indicators, calc_amplification_factors,
                                            get_decadal_outpath, get_amplification_outpath)
from .TEA import TEAIndicators
from .TEA_AGR import TEAAgr
from . import __version__ as TEA_VERSION


def calc_tea_indicators(opts):
    """
    calculate TEA indicators as defined in https://doi.org/10.48550/arXiv.2504.18964 and
    Methods as defined in
    Kirchengast, G., Haas, S. J. & Fuchsberger, J. Compound event metrics detect and explain ten-fold
    increase of extreme heat over Europe—Supplementary Note: Detailed methods description for
    computing threshold-exceedance-amount (TEA) indicators. Supplementary Information (SI) to
    Preprint – April 2025. 40 pp. Wegener Center, University of Graz, Graz, Austria, 2025.

    Args:
        opts: options as defined in CFG-PARAMS-doc.md and TEA_CFG_DEFAULT.yaml
    """

    # load mask if needed
    if 'maskpath' in opts and 'station' not in opts:
        mask = _load_mask_file(opts)
    else:
        mask = None

    # calculate daily and annual climatic time period indicators
    if not opts.decadal_only:

        # load threshold grid or set threshold value
        if 'station' not in opts:
            gridded = True
            threshold_grid = _get_threshold(opts)

            # do calcs in chunks of 10 years for gridded data
            starts = np.arange(opts.start, opts.end, 10)
            ends = np.append(np.arange(opts.start + 10 - 1, opts.end, 10), opts.end)
        else:
            gridded = False
            threshold_grid = None
            starts = [opts.start]
            ends = [opts.end]

        for p_start, p_end in zip(starts, ends):
            # calculate daily basis variables
            tea = calc_dbv_indicators(mask=mask, opts=opts, start=p_start, end=p_end,
                                      gridded=gridded, threshold=threshold_grid)

            # for aggregate GeoRegion calculation, load GR grid files
            if 'agr' in opts:
                _load_or_generate_gr_grid(opts, tea)

            # calculate CTP indicators
            calc_annual_ctp_indicators(tea=tea, opts=opts, start=p_start, end=p_end)

            # collect garbage
            gc.collect()

    # calculate decadal indicators and amplification factors
    if opts.decadal or opts.decadal_only or opts.recalc_decadal:
        if 'agr' in opts:
            tea = TEAAgr(mask=mask)
        else:
            tea = TEAIndicators()

        # calculate decadal-mean ctp indicator variables
        calc_decadal_indicators(opts=opts, tea=tea)

        # calculate amplification factors
        calc_amplification_factors(opts=opts, tea=tea)

        # calculate AGR variables
        if 'agr' in opts:
            _load_or_generate_gr_grid(opts, tea)
            _calc_agr_mean_and_spread(opts=opts, tea=tea)


def calc_dbv_indicators(start, end, threshold, opts, mask=None, gridded=True):
    """
    calculate daily basis variables for a given time period
    Args:
        start: start year
        end: end year
        threshold: either gridded threshold values (xarray DataArray) or a constant threshold value (int, float)
        opts: options as defined in CFG-PARAMS-doc.md and TEA_CFG_DEFAULT.yaml
        mask: mask grid for input data containing nan values for cells that should be masked. Fractions of 1 are
        interpreted as area fractions for the given cell. (optional)
        gridded: if True, load gridded data, else load station timeseries (default: True)

    Returns:
        tea: TEA object with daily basis variables

    """
    # check and create output path
    dbv_outpath = f'{opts.outpath}/daily_basis_variables'
    if not os.path.exists(dbv_outpath):
        os.makedirs(dbv_outpath)

    logger.info(f'Calculating TEA indicators for years {start}-{end}.')

    # use either TEAIndicators or TEAAgr class depending on the options
    if 'agr' in opts:
        agr_str = 'AGR-'
        TEA_class_obj = TEAAgr
    else:
        agr_str = ''
        TEA_class_obj = TEAIndicators

    # load land-sea mask for AGR
    if 'agr' in opts and 'maskpath' in opts:
        # load land-sea mask for AGR
        lsm = _load_lsm_file(opts)
    else:
        lsm = None

    # DBV can't be the same for AGR and non-AGR (AGR is always without mask and has margins) so optionally add agr_str
    if gridded:
        name = opts.region
    else:
        name = opts.station
    dbv_filename = (f'{dbv_outpath}/'
                    f'DBV_{opts.param_str}_{agr_str}{name}_annual_{opts.dataset}'
                    f'_{start}to{end}.nc')

    # recalculate daily basis variables if needed
    if opts.recalc_daily or not os.path.exists(dbv_filename):

        # always calculate annual basis variables to later extract sub-annual values
        period = 'annual'
        if gridded:
            data = get_gridded_data(start=start, end=end, opts=opts, period=period)
        else:
            data = get_csv_data(opts)
            threshold = create_threshold_grid(opts, data=data)

        # reduce extent of data to the region of interest
        # TODO: use this also for non-AGR and test
        if 'agr' in opts:
            data, mask, threshold = _reduce_region(opts, data, mask, threshold)

        if opts.primary_threshold is not None:
            logger.info(f'Applying primary threshold of {opts.primary_threshold} to input data')
            if opts.low_extreme:
                data = data.where(data <= opts.primary_threshold, opts.threshold)
            else:
                data = data.where(data >= opts.primary_threshold, opts.threshold)

        if not os.path.exists(dbv_filename):
            logger.info(f'Daily basis variable file {dbv_filename} not found. '
                        'Daily basis variables will be calculated.')
        else:
            logger.info('recalc_daily is set: Daily basis variables will be recalculated. Period set to annual.')

        # set min area to < 1 grid cell area so that all exceedance days are considered
        min_area = 0.0001

        # initialize TEA object
        if 'agr' in opts:
            tea = TEA_class_obj(input_data=data, threshold=threshold, mask=mask,
                                min_area=min_area, low_extreme=opts.low_extreme,
                                unit=opts.unit, land_sea_mask=lsm, gr_grid_res=opts.grg_grid_spacing,
                                cell_size_lat=opts.agr_cell_size, land_frac_min=opts.land_frac_min)
        else:
            tea = TEA_class_obj(input_data=data, threshold=threshold, mask=mask,
                                min_area=min_area, low_extreme=opts.low_extreme,
                                unit=opts.unit, land_sea_mask=lsm)

        # computation of daily basis variables (Methods chapter 3)
        if gridded:
            gr = opts.hourly
            if opts.primary_threshold is not None:
                gr = True
            else:
                gr = opts.hourly
        else:
            gr = False
        tea.calc_daily_basis_vars(gr=gr)

        # calculate hourly indicators
        if opts.hourly:
            _calc_hourly_indicators(tea=tea, opts=opts, start=start, end=end)

        # save results
        create_tea_history(cfg_params=opts, tea=tea, dataset='daily_results')
        tea.save_daily_results(filepath=dbv_filename)
    else:
        # load existing results
        if 'agr' in opts:
            tea = TEA_class_obj(threshold=threshold, mask=mask, low_extreme=opts.low_extreme,
                                unit=opts.unit, land_sea_mask=lsm, gr_grid_res=opts.grg_grid_spacing,
                                cell_size_lat=opts.agr_cell_size, land_frac_min=opts.land_frac_min)
        else:
            tea = TEA_class_obj(threshold=threshold, mask=mask, low_extreme=opts.low_extreme,
                                unit=opts.unit,
                                land_sea_mask=lsm)
        logger.info(
            f'Loading daily basis variables from {dbv_filename}; if you want to recalculate them, '
            'set --recalc-daily.')
        tea.load_daily_results(dbv_filename)
    return tea


def calc_annual_ctp_indicators(tea, opts, start, end):
    """
    calculate the TEA indicators for the annual climatic time period
    Args:
        tea: TEA object
        opts: options as defined in CFG-PARAMS-doc.md and TEA_CFG_DEFAULT.yaml
        start: start year
        end: end year
    """

    if 'station' not in opts:
        # apply criterion that DTEA_GR > DTEA_min and all GR variables use same dates,
        # dtea_min is given in areals (1 areal = 100 km2)
        dtea_min = opts.min_exceedance_area  # according to equation 03
        tea.update_min_area(dtea_min)

    if 'agr' in opts:
        tea.land_frac_min = opts.land_frac_min

    # calculate annual climatic time period indicators
    logger.info('Calculating annual CTP indicators')
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in multiply")
        tea.calc_annual_ctp_indicators(opts.period, drop_daily_results=True)

    # save output
    _save_ctp_output(opts=opts, tea=tea, start=start, end=end)


def _get_threshold(opts):
    """
    load threshold grid or set threshold value
    Args:
        opts: options as defined in CFG-PARAMS-doc.md and TEA_CFG_DEFAULT.yaml

    Returns:
        threshold_grid: threshold grid (xarray DataArray) or constant threshold value (int, float)

    """
    if opts.threshold_type == 'abs':
        threshold_grid = opts.threshold
    else:
        threshold_file = f'{opts.statpath}/threshold_{opts.param_str}_{opts.period}_{opts.region}_{opts.dataset}.nc'
        if opts.recalc_threshold or not os.path.exists(threshold_file):
            if not os.path.exists(threshold_file):
                logger.info(f'Threshold file {threshold_file} not found. Calculating percentiles...')
            else:
                logger.info('Calculating percentiles...')
            threshold_grid = create_threshold_grid(opts=opts)
            logger.info(f'Saving threshold grid to {threshold_file}')
            threshold_grid.to_netcdf(threshold_file)
        else:
            logger.info(f'Loading threshold grid from {threshold_file}')
            threshold_grid = xr.open_dataset(threshold_file).threshold

    # from now on, only deviations from the threshold are considered ==> set unit to Kelvin
    if opts.unit == 'degC':
        opts.unit = 'K'

    return threshold_grid


def _load_mask_file(opts):
    """
    load GR mask
    Args:
        opts: options as defined in CFG-PARAMS-doc.md and TEA_CFG_DEFAULT.yaml

    Returns:
        mask: GR mask (Xarray DataArray)

    """
    if opts.gr_type == 'polygon':
        maskpath = (Path(opts.maskpath) / opts.mask_sub /
                    f'{opts.region}_mask_{opts.dataset}_{opts.altitude_threshold}.nc')
    elif opts.gr_type == 'corners':
        sw_coords = opts.sw_corner.split(',')
        sw_coords = '_'.join([f'{float(coord):.1f}' for coord in sw_coords])
        ne_coords = opts.ne_corner.split(',')
        ne_coords = '_'.join([f'{float(coord):.1f}' for coord in ne_coords])
        maskpath = Path(opts.maskpath) / opts.mask_sub / f'SW_{sw_coords}-NE_{ne_coords}_mask_{opts.dataset}.nc'
    else:
        center_coords = opts.center.split(',')
        center_coords = [float(ii) for ii in center_coords]
        sw_coords = [center_coords[0] - float(opts.we_len) / 2,
                     center_coords[1] - float(opts.ns_len) / 2]
        sw_coords = '_'.join([f'{float(coord):.1f}' for coord in sw_coords])
        ne_coords = [center_coords[0] + float(opts.we_len) / 2,
                     center_coords[1] + float(opts.ns_len) / 2]
        ne_coords = '_'.join([f'{float(coord):.1f}' for coord in ne_coords])
        maskpath = Path(opts.maskpath) / opts.mask_sub / f'SW_{sw_coords}-NE_{ne_coords}_mask_{opts.dataset}.nc'
    logger.info(f'Loading mask from {maskpath}')
    mask_file = xr.open_dataset(maskpath)

    return mask_file.mask


def _load_lsm_file(opts):
    """
    load land-sea-mask for AGR
    Args:
        opts: options as defined in CFG-PARAMS-doc.md and TEA_CFG_DEFAULT.yaml

    Returns:
        mask: mask (Xarray DataArray)

    """
    new_opts = deepcopy(opts)
    if 'EUR' in opts.region:
        # for all EUR sub-regions, use the EUR mask
        new_opts.region = 'EUR'
    return _load_mask_file(new_opts)


def _compare_to_ctp_ref(tea, ctp_filename_ref):
    """
    compare results to reference file
    TODO: move this to test routine
    Args:
        tea: TEA object
        ctp_filename_ref: reference file
    """

    if os.path.exists(ctp_filename_ref):
        logger.info(f'Comparing results to reference file {ctp_filename_ref}')
        tea_ref = TEAIndicators()
        tea_ref.load_ctp_results(ctp_filename_ref)
        tea_result = tea.ctp_results
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid value encountered in multiply")
            compare_to_ref(tea_result, tea_ref.ctp_results)
    else:
        logger.warning(f'Reference file {ctp_filename_ref} not found.')


def _save_ctp_output(opts, tea, start, end):
    """
    save annual CTP results to netcdf file
    Args:
        opts: options as defined in CFG-PARAMS-doc.md and TEA_CFG_DEFAULT.yaml
        tea: TEA object
        start: start year
        end: end year
    """
    create_tea_history(cfg_params=opts, tea=tea, dataset='ctp_results')

    path = Path(f'{opts.outpath}/ctp_indicator_variables/')
    path.mkdir(parents=True, exist_ok=True)

    if 'agr' in opts:
        grg_str = 'GRG-'
    else:
        grg_str = ''

    if 'station' in opts:
        name = opts.station
    else:
        name = opts.region
    outpath = (f'{opts.outpath}/ctp_indicator_variables/'
               f'CTP_{opts.param_str}_{grg_str}{name}_{opts.period}_{opts.dataset}'
               f'_{start}to{end}.nc')

    path_ref = outpath.replace('.nc', '_ref.nc')

    logger.info(f'Saving CTP indicators to {outpath}')
    tea.save_ctp_results(filepath=outpath)

    if opts.compare_to_ref:
        _compare_to_ctp_ref(tea, path_ref)


def _save_grg_mask(opts, grg_mask, grg_areas):
    """
    save grg mask to netcdf file
    Args:
        opts: CLI parameter
        grg_mask: mask on grg grid
        grg_areas: area grid on grg grid
    """
    res = str(opts.grg_grid_spacing)
    res_str = res.replace('.', 'p')
    create_history_from_cfg(cfg_params=opts, ds=grg_areas)
    area_grid_file = Path(opts.statpath) / f'area_grid_{res_str}_{opts.region}_{opts.dataset}.nc'
    logger.info(f'Saving GR area grid to {area_grid_file}')
    try:
        grg_areas.to_netcdf(area_grid_file)
    except PermissionError:
        os.remove(area_grid_file)
        grg_areas.to_netcdf(area_grid_file)

    # save GRG mask
    create_history_from_cfg(cfg_params=opts, ds=grg_mask)
    mask_file = (Path(opts.maskpath) / opts.mask_sub /
                 f'{opts.region}_mask_{res_str}_{opts.dataset}_{opts.altitude_threshold}.nc')
    logger.info(f'Saving GR mask to {mask_file}')
    try:
        grg_mask.to_netcdf(mask_file)
    except PermissionError:
        os.remove(mask_file)
        grg_mask.to_netcdf(mask_file)


def _load_or_generate_gr_grid(opts, tea):
    """
    load or generate grid of GRs mask and area grid for AGR calculation
    Args:
        opts: options as defined in CFG-PARAMS-doc.md and TEA_CFG_DEFAULT.yaml
        tea: TEA object

    Returns:

    """
    # load static GR grid files
    gr_grid_mask, gr_grid_areas = _load_gr_grid_static(opts)
    # generate GR grid mask and area if necessary
    if gr_grid_mask is None or gr_grid_areas is None:
        tea.generate_gr_grid_mask()
        _save_grg_mask(opts, tea.gr_grid_mask, tea.gr_grid_areas)
    else:
        # set GR grid mask and area grid
        tea.gr_grid_mask = gr_grid_mask
        tea.gr_grid_areas = gr_grid_areas

    # set cell_size
    tea.cell_size_lat = opts.agr_cell_size


def _calc_lat_lon_range(cell_size_lat, mask):
    """
    calculate latitude and longitude range for selected region
    Args:
        cell_size_lat: size of the grid cell in latitude
        mask: mask grid

    Returns:
        min_lat: minimum latitude
        min_lon: minimum longitude
        max_lat: maximum latitude
        max_lon: maximum longitude

    """
    valid_cells = mask.where(mask > 0, drop=True)
    min_lat = math.floor(valid_cells.lat.min().values - cell_size_lat / 2)
    if min_lat < mask.lat.min().values:
        min_lat = float(mask.lat.min().values)
    max_lat = math.ceil(valid_cells.lat.max().values + cell_size_lat / 2)
    if max_lat > mask.lat.max().values:
        max_lat = float(mask.lat.max().values)
    cell_size_lon = 1 / np.cos(np.deg2rad(max_lat)) * cell_size_lat
    min_lon = math.floor(valid_cells.lon.min().values - cell_size_lon / 2)
    if min_lon < mask.lon.min().values:
        min_lon = float(mask.lon.min().values)
    max_lon = math.ceil(valid_cells.lon.max().values + cell_size_lon / 2)
    if max_lon > mask.lon.max().values:
        max_lon = float(mask.lon.max().values)
    return min_lat, min_lon, max_lat, max_lon


def _reduce_region(opts, data, mask, threshold=None, full_region=False):
    """
    reduce data to the region of interest
    Args:
        opts: options as defined in CFG-PARAMS-doc.md and TEA_CFG_DEFAULT.yaml
        data: input data
        mask: mask grid
        threshold: threshold grid
        full_region: if True, use the full region

    Returns:
        data: reduced data
        mask: reduced mask grid
        threshold: reduced threshold grid

    """
    cell_size_lat = opts.agr_cell_size

    # preselect region to reduce computation time (incl. some margins to avoid boundary effects)
    if full_region:
        min_lat = mask.lat.min().values
        max_lat = mask.lat.max().values
        min_lon = mask.lon.min().values
        max_lon = mask.lon.max().values
    else:
        min_lat, min_lon, max_lat, max_lon = _calc_lat_lon_range(cell_size_lat, mask)

    if opts.region == 'EUR':
        # hardcoded extent for EUR region
        lons = np.arange(-12, 40.5, opts.grg_grid_spacing)
        cell_size_lon = 1 / np.cos(np.deg2rad(max_lat)) * cell_size_lat
        min_lon = math.floor(lons[0] - cell_size_lon / 2)
        max_lon = math.ceil(lons[-1] + cell_size_lon / 2)

    proc_data = data.sel(lat=slice(max_lat, min_lat), lon=slice(min_lon, max_lon))
    proc_mask = mask.sel(lat=slice(max_lat, min_lat), lon=slice(min_lon, max_lon))
    if threshold is not None:
        threshold = threshold.sel(lat=slice(max_lat, min_lat), lon=slice(min_lon, max_lon))

    return proc_data, proc_mask, threshold


def _getopts():
    """
    get command line arguments

    Returns:
        opts: command line parameters
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--config-file', '-cf',
                        dest='config_file',
                        type=str,
                        default='../TEA_CFG.yaml',
                        help='TEA configuration file (default: TEA_CFG.yaml)')

    parser.add_argument('--version', '-v',
                        action='version',
                        version=TEA_VERSION,
                        help='show version and exit')

    myopts = parser.parse_args()

    return myopts


def _calc_hourly_indicators(tea, opts, start, end):
    """
    calculate hourly indicators for a given time period
    Args:
        tea: TEA object with daily basis variables
        opts: options
        start: start year
        end: end year

    Returns:
        tea: TEA object with hourly indicators

    """
    # load data
    data = get_gridded_data(start=start, end=end, opts=opts, hourly=True)

    if 'agr' in opts:
        # reduce data to the region of interest
        data, _, _ = _reduce_region(opts, data, tea.mask, full_region=True)

    if not _check_data_extent(data, tea.input_data):
        logger.warning('Hourly data extent is not the same as daily data extent. '
                       'Please check your data and the region you want to calculate.')

    logger.info('Calculating hourly basis variables.')
    # calculate hourly indicators
    tea.calc_hourly_indicators(input_data=data)


def _check_data_extent(data, ref_data):
    """
    check if data extent is the same as the TEA data extent
    Args:
        data: input data
        ref_data: reference data

    Returns:
        True if data extent is the same, False otherwise

    """
    if not np.array_equal(data.lat.values, ref_data.lat.values) or not np.array_equal(
            data.lon.values,
            ref_data.lon.values):
        return False
    else:
        return True


def _calc_agr_mean_and_spread(opts, tea):
    """
    calculate aggregate GeoRegion means and spread estimators

    Args:
        opts: options as defined in CFG-PARAMS-doc.md and TEA_CFG_DEFAULT.yaml
        tea: teaAgr object

    Returns:

    """
    crop_to_shp = False
    agr_lat_range = None
    agr_lon_range = None

    if opts.agr_range is not None:
        agr_lat_range = opts.agr_range[:2]
        agr_lon_range = opts.agr_range[-2:]
    elif opts.agr != opts.region:
        # use region shape for AGR calculation
        # load mask for the specified region
        agr_opts = deepcopy(opts)
        agr_opts.region = opts.agr
        mask_file = _load_mask_file(agr_opts)
        tea.mask = mask_file
        _load_or_generate_gr_grid(agr_opts, tea)
        crop_to_shp = True

    tea.calc_agr_vars(lat_range=agr_lat_range, lon_range=agr_lon_range, spreads=opts.spreads, crop_to_shp=crop_to_shp)

    # save results
    outpath_decadal = get_decadal_outpath(opts, opts.agr)
    outpath_ampl = get_amplification_outpath(opts, opts.agr)
    logger.info(f'Saving AGR decadal results to {outpath_decadal}')
    # remove outpath_decadal if it exists
    if os.path.exists(outpath_decadal):
        os.remove(outpath_decadal)
    create_tea_history(cfg_params=opts, tea=tea, dataset='decadal_results')
    tea.save_decadal_results(filepath=outpath_decadal)
    logger.info(f'Saving AGR amplification factors to {outpath_ampl}')
    create_tea_history(cfg_params=opts, tea=tea, dataset='amplification_factors')
    tea.save_amplification_factors(filepath=outpath_ampl)


def _load_gr_grid_static(opts):
    """
    load grid of GRs mask and area grid for AGR calculation

    Args:
        opts: options as defined in CFG-PARAMS-doc.md and TEA_CFG_DEFAULT.yaml

    Returns:
        gr_grid_areas: area grid (xarray DataArray)
        gr_grid_mask: mask grid (xarray DataArray)

    """
    res = str(opts.grg_grid_spacing)
    res_str = res.replace('.', 'p')
    gr_grid_mask_file = (Path(opts.maskpath) / opts.mask_sub /
                         f'{opts.region}_mask_{res_str}_{opts.dataset}_{opts.altitude_threshold}.nc')
    logger.info(f'Loading GR mask from {gr_grid_mask_file}')
    try:
        gr_grid_mask = xr.open_dataset(gr_grid_mask_file)
        gr_grid_mask = gr_grid_mask.mask
    except FileNotFoundError:
        if opts.decadal_only:
            logger.warning(f'No GR mask found at {gr_grid_mask_file}.')
        gr_grid_mask = None
    gr_grid_areas_file = Path(opts.statpath) / f'area_grid_{res_str}_{opts.region}_{opts.dataset}.nc'
    logger.info(f'Loading GR area grid from {gr_grid_areas_file}')
    try:
        gr_grid_areas = xr.open_dataset(gr_grid_areas_file)
        gr_grid_areas = gr_grid_areas.area_grid
    except FileNotFoundError:
        if opts.decadal_only:
            logger.info(
                f'No GR area grid found at {gr_grid_areas_file}. Trying to generate one')
        gr_grid_areas = None
    return gr_grid_mask, gr_grid_areas


def run():
    """
    run the script
    Returns:

    """

    # suppress warnings
    warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')
    warnings.filterwarnings(action='ignore', message='divide by zero encountered in divide')
    warnings.filterwarnings(action='ignore', message='invalid value encountered in divide')

    # get command line parameters
    cmd_opts = _getopts()

    # load CFG parameters
    opts = load_opts(fname=__file__, config_file=cmd_opts.config_file)

    # download example file if specified in config
    if 'example' in opts.input_data_path:
        from .TEA_example import dl_example_file
        eRA5_file, example_path = dl_example_file()
        opts.input_data_path = example_path

    # calculate TEA indicators
    calc_tea_indicators(opts)


if __name__ == '__main__':
    run()
