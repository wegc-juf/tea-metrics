#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: hst, juf
"""

import os
import gc
import math
import time
import warnings
from copy import deepcopy

import argparse
import numpy as np
import psutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import xarray as xr

from .common.general_functions import (create_history_from_cfg, create_tea_history, compare_to_ref,
                                       get_gridded_data,
                                       get_csv_data, create_threshold_grid)
from .common.config import load_opts
from .common.grid_naming import grg_grid_token
from .common.TEA_logger import logger, set_log_level
from .utils.calc_decadal_indicators import (calc_decadal_indicators, calc_amplification_factors,
                                            get_decadal_outpath, get_amplification_outpath)
from .TEA import TEAIndicators
from .TEA import DEBUG
from .TEA_AGR import TEAAgr
from . import __version__ as TEA_VERSION


def calc_tea_indicators(opts):
    """
    calculate TEA indicators as defined in https://doi.org/10.1016/j.wace.2026.100855 and
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
        mask, area_grid, full_mask = _load_mask_file(opts, include_area=True)
    else:
        mask = None
        area_grid = None
        full_mask = None
    
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
        
        chunks = list(zip(starts, ends))
        if (getattr(opts, 'parallel_workers', 1) > 1 and gridded and len(chunks) > 1
                and 'agr' not in opts):
            workers = _get_chunk_workers(len(chunks), opts.parallel_workers)
            logger.info(f'Calculating {len(chunks)} daily/CTP chunks with {workers} worker processes.')
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(_calculate_chunk_worker, opts, int(p_start), int(p_end))
                           for p_start, p_end in chunks]
                for future in futures:
                    future.result()
        else:
            logger.info('Only one decade requested. Calculating daily/CTP indicators in one process.')
            for p_start, p_end in chunks:
                _calculate_chunk(opts, mask, threshold_grid, gridded, int(p_start), int(p_end), area_grid,
                                 full_mask)
    
    # calculate decadal indicators and amplification factors
    if opts.decadal or opts.decadal_only or opts.recalc_decadal:
        if 'agr' in opts:
            tea = TEAAgr(mask=mask, full_mask=full_mask, gr_grid_res=opts.grg_grid_spacing,
                         rounding_decimal_places=opts.rounding_decimal_places, ref_period=opts.ref_period,
                         agr_frac_min=opts.agr_frac_min, min_duration=opts.min_duration,
                         land_frac_min=opts.land_frac_min, zlib_compression=opts.zlib_compression)
        else:
            tea = TEAIndicators(rounding_decimal_places=opts.rounding_decimal_places, ref_period=opts.ref_period,
                                use_dask=opts.use_dask, compression_level=opts.compression_level,
                                zlib_compression=opts.zlib_compression)
        
        # calculate decadal-mean ctp indicator variables
        calc_decadal_indicators(opts=opts, tea=tea)
        
        # calculate amplification factors
        calc_amplification_factors(opts=opts, tea=tea)
        
        # calculate AGR variables
        if 'agr' in opts:
            _load_or_generate_gr_grid_static(opts, tea)
            # TODO calc also for annual data
            _calc_agr_mean_and_spread(opts=opts, tea=tea)


def _calculate_chunk_worker(opts, start, end):
    """Calculate one independent gridded chunk in a worker process."""
    worker_opts = deepcopy(opts)
    worker_opts.recalc_threshold = False
    # Chunk-level processes provide the parallelism; avoid nested Dask pools.
    worker_opts.use_dask = False
    mask, area_grid, full_mask = _load_mask_file(worker_opts, include_area=True)
    threshold = _get_threshold(worker_opts)
    _calculate_chunk(worker_opts, mask, threshold, True, start, end, area_grid, full_mask)


def _calculate_chunk(opts, mask, threshold, gridded, start, end, area_grid=None, full_mask=None):
    """Calculate and save daily and CTP results for one time chunk."""
    tea = calc_dbv_indicators(mask=mask, opts=opts, start=start, end=end,
                              gridded=gridded, threshold=threshold, area_grid=area_grid,
                              full_mask=full_mask)
    if 'agr' in opts:
        _load_or_generate_gr_grid_static(opts, tea)
    calc_annual_ctp_indicators(tea=tea, opts=opts, start=start, end=end)
    gc.collect()


def _get_chunk_workers(chunk_count, max_workers=4):
    """Choose a conservative process count for independent time chunks."""
    cpu_count = max(1, os.cpu_count() or 1)
    available_memory = max(1, psutil.virtual_memory().available)
    workers_by_memory = max(1, available_memory // (32 * 1024 ** 3))
    return min(chunk_count, max(1, cpu_count // 4), workers_by_memory, max_workers, 4)


def calc_dbv_indicators(start, end, threshold, opts, mask=None, gridded=True, area_grid=None, full_mask=None):
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
        agr_str = f'AGR-'
        TEA_class_obj = TEAAgr
    else:
        agr_str = ''
        TEA_class_obj = TEAIndicators
    
    population_grid = _load_population_grid(opts)
    
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
    
    # Reuse AGR daily caches only when their coordinates cover the current
    # crop. A cache with a larger spatial extent is valid for a smaller crop.
    cached_crop = None
    recalculate_daily = opts.recalc_daily or not os.path.exists(dbv_filename)
    if not recalculate_daily and 'agr' in opts:
        cached_crop = _reduce_region(opts, None, mask, threshold, population_grid=population_grid)
        required_mask = cached_crop[1]
        if not _daily_cache_covers_grid(dbv_filename, required_mask):
            logger.info(f'Daily basis cache {dbv_filename} does not cover the crop for '
                        f'grg_cell_size={opts.grg_cell_size}; recalculating daily data.')
            recalculate_daily = True

    # recalculate daily basis variables if needed
    if recalculate_daily:
        
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
            reduced = _reduce_region(opts, data, mask, threshold, population_grid=population_grid)
            if population_grid is None:
                data, mask, threshold = reduced
            else:
                data, mask, threshold, population_grid = reduced
        
        if opts.primary_threshold is not None:
            logger.info(f'Applying primary threshold of {opts.primary_threshold} to input data')
            if opts.low_extreme:
                data = data.where(data <= opts.primary_threshold, opts.threshold)
            else:
                data = data.where(data >= opts.primary_threshold, opts.threshold)
        
        if not os.path.exists(dbv_filename):
            logger.info(f'Daily basis variable file {dbv_filename} not found. '
                        'Daily basis variables will be calculated.')
        elif opts.recalc_daily:
            logger.info('recalc_daily is set: Daily basis variables will be recalculated. Period set to annual.')
        
        # set min area to < 1 grid cell area so that all exceedance days are considered
        min_area = 0.0001
        
        # initialize TEA object
        if 'agr' in opts:
            tea = TEA_class_obj(input_data=data, threshold=threshold, mask=mask, full_mask=full_mask,
                                population_grid=population_grid,
                                min_area=min_area, low_extreme=opts.low_extreme,
                                unit=opts.unit, land_sea_mask=lsm, gr_grid_res=opts.grg_grid_spacing,
                                use_dask=opts.use_dask,
                                compression_level=opts.compression_level,
                                cell_size_y=opts.grg_cell_size,
                                rounding_decimal_places=opts.rounding_decimal_places, agr_frac_min=opts.agr_frac_min,
                                land_frac_min=opts.land_frac_min, min_duration=opts.min_duration,
                                grg_tile_workers=opts.grg_tile_workers,
                                zlib_compression=opts.zlib_compression)
        else:
            tea = TEA_class_obj(input_data=data, threshold=threshold, mask=mask,
                                population_grid=population_grid,
                                min_area=min_area, low_extreme=opts.low_extreme,
                                unit=opts.unit, area_grid=area_grid, land_sea_mask=lsm,
                                rounding_decimal_places=opts.rounding_decimal_places,
                                use_dask=opts.use_dask, compression_level=opts.compression_level,
                                zlib_compression=opts.zlib_compression)
        
        # computation of daily basis variables (Methods chapter 3)
        if gridded:
            gr = True
        else:
            gr = False
        start_calc = time.perf_counter()
        tea.calc_daily_basis_vars(gr=gr)
        logger.info(f"Calculated daily basis variables in {time.perf_counter() - start_calc:.2f}s")
        
        # calculate hourly indicators
        if opts.hourly:
            _calc_hourly_indicators(tea=tea, opts=opts, start=start, end=end)
        
        # Daily calculations no longer need the full input time series. Release
        # it before serializing daily results, which creates rounded copies.
        tea.input_data = None
        gc.collect()
        
        # save results
        create_tea_history(cfg_params=opts, tea=tea, dataset='daily_results')
        tea.save_daily_results(filepath=dbv_filename, save_tiff=opts.file_format == 'GeoTiff')
        if 'agr' in opts and os.path.isfile(dbv_filename):
            logger.info('Reloading serialized daily basis variables before AGR CTP calculation.')
            tea = _reload_daily_results_from_cache(tea, dbv_filename)
    else:
        # load existing results
        if 'agr' in opts:
            reduced = cached_crop
            if population_grid is None:
                data, mask, threshold = reduced
            else:
                data, mask, threshold, population_grid = reduced
            tea = TEA_class_obj(threshold=threshold, mask=mask, full_mask=full_mask,
                                low_extreme=opts.low_extreme,
                                population_grid=population_grid,
                                unit=opts.unit, land_sea_mask=lsm, gr_grid_res=opts.grg_grid_spacing,
                                cell_size_y=opts.grg_cell_size,
                                rounding_decimal_places=opts.rounding_decimal_places,
                                agr_frac_min=opts.agr_frac_min,
                                compression_level=opts.compression_level,
                                  grg_tile_workers=opts.grg_tile_workers,
                                zlib_compression=opts.zlib_compression)
        else:
            tea = TEA_class_obj(threshold=threshold, mask=mask, low_extreme=opts.low_extreme,
                                 population_grid=population_grid,
                                 unit=opts.unit,
                                 land_sea_mask=lsm, rounding_decimal_places=opts.rounding_decimal_places,
                                 compression_level=opts.compression_level,
                                 zlib_compression=opts.zlib_compression)
        logger.info(
            f'Loading daily basis variables from {dbv_filename}; if you want to recalculate them, '
            'set --recalc-daily.')
        tea.load_daily_results(dbv_filename)
    tea._daily_cache_path = dbv_filename
    tea._daily_cache_is_source = os.path.isfile(dbv_filename)
    return tea


def _reload_daily_results_from_cache(tea, dbv_filename):
    """Use the serialized daily fields as the canonical input to AGR CTPs."""
    if tea.daily_results is not None:
        tea.daily_results.close()
        tea.daily_results = None
        gc.collect()
    tea.load_daily_results(dbv_filename)
    tea._daily_cache_path = dbv_filename
    tea._daily_cache_is_source = True
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
        if dtea_min > 0:
            logger.info(f'Applying minimum exceedance area of {dtea_min} areal (100 km2) to all GR variables')
            tea.update_min_area(dtea_min)
    
    if 'agr' in opts:
        tea.land_frac_min = opts.land_frac_min
        tea.min_duration = opts.min_duration
    
    # calculate annual climatic time period indicators
    logger.info('Calculating annual CTP indicators...')
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


def _load_mask_file(opts, include_area=False):
    """
    load GR mask
    Args:
        opts: options as defined in CFG-PARAMS-doc.md and TEA_CFG_DEFAULT.yaml

    Returns:
        mask: altitude-filtered GR mask. With ``include_area=True``, returns
            ``(mask, area_grid, mask_full)`` where ``mask_full`` is the region
            mask before altitude filtering.

    """
    if opts.gr_type == 'polygon':
        maskpath = (Path(opts.maskpath) / opts.mask_sub /
                    f'{opts.region}_mask_{opts.dataset}_{opts.altitude_threshold}.nc')
    elif opts.gr_type == 'corners':
        sw_coords = opts.sw_corner.split(',')
        sw_coords = '_'.join([f'{float(coord):.1f}' for coord in sw_coords])
        ne_coords = opts.ne_corner.split(',')
        ne_coords = '_'.join([f'{float(coord):.1f}' for coord in ne_coords])
        maskpath = (Path(opts.maskpath) / opts.mask_sub /
                    f'SW_{sw_coords}-NE_{ne_coords}_mask_{opts.dataset}_{opts.altitude_threshold}.nc')
    else:
        center_coords = opts.center.split(',')
        center_coords = [float(ii) for ii in center_coords]
        sw_coords = [center_coords[0] - float(opts.we_len) / 2,
                     center_coords[1] - float(opts.ns_len) / 2]
        sw_coords = '_'.join([f'{float(coord):.1f}' for coord in sw_coords])
        ne_coords = [center_coords[0] + float(opts.we_len) / 2,
                     center_coords[1] + float(opts.ns_len) / 2]
        ne_coords = '_'.join([f'{float(coord):.1f}' for coord in ne_coords])
        maskpath = (Path(opts.maskpath) / opts.mask_sub /
                    f'SW_{sw_coords}-NE_{ne_coords}_mask_{opts.dataset}_{opts.altitude_threshold}.nc')
    logger.info(f'Loading mask from {maskpath}')
    if not maskpath.is_file():
        raise FileNotFoundError(maskpath)
    mask_file = xr.open_dataset(maskpath)
    
    mask = mask_file.mask
    if include_area:
        full_mask = mask_file.data_vars.get('mask_full')
        if full_mask is None:
            is_agr = ('agr' in opts if hasattr(opts, '__contains__') else getattr(opts, 'agr', False))
            if opts.altitude_threshold and is_agr:
                raise ValueError(
                    f'Mask file {maskpath} does not contain the pre-altitude mask_full. '
                    'Regenerate the region mask before calculating AGR areas.')
            full_mask = mask
        return mask, mask_file.data_vars.get('area_grid'), full_mask
    return mask


def _load_population_grid(opts):
    """
    Load the optional population grid configured for population-dependent TEA variables.

    The preferred variable name is ``population``. A file with exactly one data variable
    is also accepted to support population datasets using a different variable name.
    """
    population_path = getattr(opts, 'population_grid_path', None)
    if population_path is None:
        return None
    
    logger.info(f'Loading population grid from {population_path}')
    population_file = xr.open_dataset(population_path)
    if 'population' in population_file.data_vars:
        population = population_file.population
    elif len(population_file.data_vars) == 1:
        population = population_file[next(iter(population_file.data_vars))]
    else:
        raise ValueError(f'Population grid file {population_path} must contain a "population" '
                         'variable or exactly one data variable.')
    population.attrs = {
        **population.attrs,
        'population_source_path': str(population_path),
        'population_source_product': population_file.attrs.get('source_population_product', ''),
        'population_source_hierarchy': population_file.attrs.get('source_hierarchy', ''),
        'allocation_grid': population_file.attrs.get('allocation_grid', ''),
    }
    return population


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
    
    outpath = _get_ctp_filepath(start, end, opts)
    
    path_ref = outpath.replace('.nc', '_ref.nc')
    
    logger.info(f'Saving CTP indicators to {outpath}')
    tea.save_ctp_results(filepath=outpath, save_tiff=opts.file_format == 'GeoTiff',
                         )
    
    if opts.compare_to_ref:
        _compare_to_ctp_ref(tea, path_ref)


def _get_ctp_filepath(start, end, opts, annual_agr=False) -> str:
    """
    get filepath for CTP results
    Args:
        start: start year
        end: end year
        opts: options as defined in CFG-PARAMS-doc
        annual_agr: set if annual agr results should be saved

    Returns:
        outpath: filepath for CTP results

    """
    path = Path(f'{opts.outpath}/ctp_indicator_variables/')
    path.mkdir(parents=True, exist_ok=True)
    
    region = opts.region
    
    if annual_agr:
        grg_str = f'AGR_{grg_grid_token(opts)}-'
        region = opts.agr
    elif 'agr' in opts:
        grg_str = f'GRG_{grg_grid_token(opts)}-'
    else:
        grg_str = ''
    
    if 'station' in opts:
        region = opts.station
    
    outpath = (f'{opts.outpath}/ctp_indicator_variables/'
               f'CTP_{opts.param_str}_{grg_str}{region}_{opts.period}_{opts.dataset}'
               f'_{start}to{end}.nc')
    return outpath


def _save_grg_mask(opts, grg_mask, grg_areas, grg_mask_full):
    """
    save grg mask to netcdf file
    Args:
        opts: CLI parameter
        grg_mask: altitude-filtered mask on GRG grid
        grg_areas: area grid on grg grid
        grg_mask_full: pre-altitude region mask on GRG grid
    """
    grid_token = grg_grid_token(opts)
    create_history_from_cfg(cfg_params=opts, ds=grg_areas)
    area_grid_file = Path(opts.statpath) / f'area_grid_{grid_token}_{opts.region}_{opts.dataset}.nc'
    logger.info(f'Saving GR area grid to {area_grid_file}')
    try:
        grg_areas.to_netcdf(area_grid_file)
    except PermissionError:
        os.remove(area_grid_file)
        grg_areas.to_netcdf(area_grid_file)
    
    # save GRG mask
    grg_masks = xr.Dataset({'mask': grg_mask, 'mask_full': grg_mask_full})
    create_history_from_cfg(cfg_params=opts, ds=grg_masks)
    mask_file = (Path(opts.maskpath) / opts.mask_sub /
                 f'{opts.region}_mask_{grid_token}_{opts.dataset}_{opts.altitude_threshold}.nc')
    logger.info(f'Saving GR mask to {mask_file}')
    try:
        grg_masks.to_netcdf(mask_file)
    except PermissionError:
        os.remove(mask_file)
        grg_masks.to_netcdf(mask_file)


# noinspection PyProtectedMember
def _load_or_generate_gr_grid_static(opts, tea):
    """
    load or generate grid of GRs mask and area grid for AGR calculation
    Args:
        opts: options as defined in CFG-PARAMS-doc.md and TEA_CFG_DEFAULT.yaml
        tea: TEA object

    Returns:

    """
    # load static GR grid files
    gr_grid_mask, gr_grid_areas, gr_grid_mask_full = _load_gr_grid_static(opts)
    if (gr_grid_mask is not None and gr_grid_areas is not None
            and gr_grid_mask_full is not None
            and not _gr_grid_static_is_usable(gr_grid_mask, gr_grid_areas)):
        logger.warning('Cached GR grid mask/area contain no valid cells; regenerating the static grid.')
        gr_grid_mask.close()
        gr_grid_areas.close()
        gr_grid_mask = None
        gr_grid_areas = None
        gr_grid_mask_full = None
        if hasattr(opts, 'recalc_decadal'):
            opts.recalc_decadal = True

    reference_grid = tea._ref_mean
    cache_covers_reference = True
    if (reference_grid is not None and gr_grid_mask is not None and gr_grid_areas is not None
            and gr_grid_mask_full is not None):
        # A subregion's static GRG grid is a valid subset of a wider CTP grid.
        # Check coverage only over the tile-coordinate extent needed by its mask,
        # then align the cached subset onto the reference axes below.
        cache_covers_reference = _gr_grid_static_covers_reference(
            gr_grid_mask, reference_grid, region_mask=tea.mask,
            cell_size_y=getattr(opts, 'grg_cell_size', tea.cell_size_y),
        )
        if cache_covers_reference:
            gr_grid_mask = tea._align_spatial_grid(gr_grid_mask, reference_grid)
            gr_grid_areas = tea._align_spatial_grid(gr_grid_areas, reference_grid)
            gr_grid_mask_full = tea._align_spatial_grid(gr_grid_mask_full, reference_grid)

    if (gr_grid_mask is None or gr_grid_areas is None or gr_grid_mask_full is None
            or not cache_covers_reference):
        if reference_grid is not None:
            # Static region masks may cover a much broader domain than the
            # CTP reference data. Generate the cached GRG grid on the reference
            # coordinates so later runs can reuse it without extent mismatch.
            tea.mask = tea._align_spatial_grid(tea.mask, reference_grid)
            tea.full_mask = tea._align_spatial_grid(tea.full_mask, reference_grid)
            tea.area_grid = tea._align_spatial_grid(tea.area_grid, reference_grid)
        tea.gr_grid_mask = None
        tea.generate_gr_grid_mask()
        _save_grg_mask(opts, tea.gr_grid_mask, tea.gr_grid_areas, tea.gr_grid_mask_full)
    else:
        # set GR grid mask and area grid
        tea.gr_grid_mask = gr_grid_mask
        tea.gr_grid_areas = gr_grid_areas
        tea.gr_grid_mask_full = gr_grid_mask_full
    
    # set cell_size
    tea.cell_size_y = opts.grg_cell_size


def _gr_grid_static_is_usable(gr_grid_mask, gr_grid_areas):
    """Return whether cached GRG masks and areas contain any valid grid cells."""
    mask_has_cells = bool((gr_grid_mask > 0).any().compute().item())
    area_has_cells = bool((np.isfinite(gr_grid_areas) & (gr_grid_areas > 0)).any().compute().item())
    return mask_has_cells and area_has_cells


def _gr_grid_static_covers_reference(gr_grid_mask, reference_grid, region_mask=None, cell_size_y=None):
    """Check cached GRG axes cover the reference area needed by a region mask.

    When no region mask is supplied, retain the full-reference coverage check.
    A subregion grid may use fewer coordinates than the CTP reference grid, as
    long as it covers the region's mask extent and shares the reference lattice.
    """
    xdim, ydim = TEAIndicators.find_dim_names(reference_grid)
    required_coords = {xdim: reference_grid[xdim].values, ydim: reference_grid[ydim].values}
    if region_mask is not None and cell_size_y is not None:
        try:
            x_min, y_min, x_max, y_max = _calc_x_y_range(cell_size_y, region_mask)
        except (ValueError, TypeError, IndexError):
            return False
        required_coords = {
            xdim: required_coords[xdim][(required_coords[xdim] >= x_min)
                                        & (required_coords[xdim] <= x_max)],
            ydim: required_coords[ydim][(required_coords[ydim] >= y_min)
                                        & (required_coords[ydim] <= y_max)],
        }
    return all(
        dim in gr_grid_mask.coords
        and _coordinates_are_covered(gr_grid_mask[dim].values, required_coords[dim])
        for dim in (xdim, ydim)
    )


def _coordinates_are_covered(available_coords, required_coords):
    """Return whether every required coordinate is represented in the cache."""
    available = np.sort(np.asarray(available_coords, dtype=float))
    required = np.asarray(required_coords, dtype=float)
    if required.size == 0:
        return True
    if available.size == 0:
        return False

    steps = np.diff(available)
    steps = steps[steps > 0]
    if steps.size == 0:
        tolerance = 1e-7
    else:
        tolerance = max(float(np.median(steps)) * 1e-5,
                        np.finfo(np.float32).eps * float(np.max(np.abs(available))) * 2)

    right_idx = np.clip(np.searchsorted(available, required), 0, available.size - 1)
    left_idx = np.clip(right_idx - 1, 0, available.size - 1)
    nearest_delta = np.minimum(np.abs(available[right_idx] - required),
                               np.abs(available[left_idx] - required))
    return bool(np.all(nearest_delta <= tolerance))


def _daily_cache_covers_grid(dbv_filename, required_grid):
    """Check that a cached daily-basis file includes every required grid coordinate."""
    xdim, ydim = TEAIndicators.find_dim_names(required_grid)
    try:
        with xr.open_dataset(dbv_filename) as cached:
            return all(
                dim in cached.coords
                and _coordinates_are_covered(cached[dim].values, required_grid[dim].values)
                for dim in (xdim, ydim)
            )
    except (OSError, KeyError, ValueError):
        return False


def _calc_x_y_range(cell_size_y, mask):
    """
    calculate x and y coord range for selected region
    Args:
        cell_size_y: size of the grid cell in latitude/y direction
        mask: mask grid

    Returns:
        y_min: minimum y coordinate
        x_min: minimum x coordinate
        y_max: maximum y coordinate
        x_max: maximum x coordinate

    """
    x_dim, y_dim = TEAIndicators.find_dim_names(mask)
    valid_cells = mask.where(mask > 0, drop=True)

    geographic = x_dim.lower() in {'lon', 'longitude'}
    y_margin = max(2.0, cell_size_y / 2) if geographic else cell_size_y / 2
    y_min = math.floor(valid_cells[y_dim].min().values - y_margin)
    if y_min < float(mask[y_dim].min().values):
        y_min = float(mask[y_dim].min().values)
    y_max = math.ceil(valid_cells[y_dim].max().values + y_margin)
    if y_max > float(mask[y_dim].max().values):
        y_max = float(mask[y_dim].max().values)
    
    if geographic:
        cell_size_x = 1 / np.cos(np.deg2rad(y_max)) * cell_size_y
        x_margin = max(2.0, cell_size_x / 2)
    else:
        # Projected grids use the configured cell size in their coordinate units.
        x_margin = cell_size_y / 2

    x_min = math.floor(valid_cells[x_dim].min().values - x_margin)
    if x_min < float(mask[x_dim].min().values):
        x_min = float(mask[x_dim].min().values)
    x_max = math.ceil(valid_cells[x_dim].max().values + x_margin)
    if x_max > float(mask[x_dim].max().values):
        x_max = float(mask[x_dim].max().values)
    
    return x_min, y_min, x_max, y_max


def _reduce_region(opts, data, mask, threshold=None, full_region=False, population_grid=None):
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
    cell_size_y = opts.grg_cell_size
    xdim, ydim = TEAIndicators.find_dim_names(mask)
    
    # preselect region to reduce computation time (incl. some margins to avoid boundary effects)
    if full_region:
        y_min = float(mask[ydim].min().values)
        y_max = float(mask[ydim].max().values)
        x_min = float(mask[xdim].min().values)
        x_max = float(mask[xdim].max().values)
    else:
        x_min, y_min, x_max, y_max = _calc_x_y_range(cell_size_y, mask)
    
    if opts.region == 'EUR':
        # hardcoded extent for EUR region
        lons = np.arange(-12, 40.5, opts.grg_grid_spacing)
        cell_size_lon = 1 / np.cos(np.deg2rad(y_max)) * cell_size_y
        if xdim.lower() in {'lon', 'longitude'}:
            lon_margin = max(2.0, cell_size_lon / 2)
        else:
            lon_margin = cell_size_lon / 2
        x_min = math.floor(lons[0] - lon_margin)
        x_max = math.ceil(lons[-1] + lon_margin)
    
    # get order of y coordinates
    y_ascending = mask[ydim][1] > mask[ydim][0]
    if not y_ascending:
        # swap min and max if values are in descending order
        y_min, y_max = y_max, y_min
    proc_data = None
    if data is not None:
        proc_data = data.sel({ydim: slice(y_min, y_max), xdim: slice(x_min, x_max)})
    proc_mask = mask.sel({ydim: slice(y_min, y_max), xdim: slice(x_min, x_max)})
    if threshold is not None and opts.threshold_type != 'abs':
        threshold = threshold.sel({ydim: slice(y_min, y_max), xdim: slice(x_min, x_max)})
    
    if population_grid is not None:
        dim_map = {}
        if 'easting' in population_grid.dims:
            dim_map['easting'] = xdim
        if 'northing' in population_grid.dims:
            dim_map['northing'] = ydim
        if dim_map:
            population_grid = population_grid.rename(dim_map)
        population_grid = population_grid.sel({ydim: slice(y_min, y_max), xdim: slice(x_min, x_max)})
    
    if population_grid is not None:
        return proc_data, proc_mask, threshold, population_grid
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
    
    parser.add_argument('--loglevel', '-ll',
                        type=str.upper,
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default='INFO',
                        help='set logging level (default: INFO)')

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
    
    logger.info('Calculating hourly basis variables.')
    # calculate hourly indicators
    tea.calc_hourly_indicators(input_data=data)


def _calc_agr_mean_and_spread(opts, tea):
    """
    calculate aggregate GeoRegion means and spread estimators

    Args:
        opts: options as defined in CFG-PARAMS-doc.md and TEA_CFG_DEFAULT.yaml
        tea: teaAgr object

    Returns:

    """
    crop_to_shp = False
    agr_x_range = None
    agr_y_range = None
    
    if opts.agr_range is not None:
        agr_x_range = opts.agr_range[-2:]
        agr_y_range = opts.agr_range[:2]
    elif opts.agr != opts.region:
        # use region shape for AGR calculation
        # load mask for the specified region
        agr_opts = deepcopy(opts)
        agr_opts.region = opts.agr
        mask_file = _load_mask_file(agr_opts)
        tea.mask = mask_file
        _load_or_generate_gr_grid_static(agr_opts, tea)
        crop_to_shp = True
    
    tea.calc_agr_vars(y_range=agr_y_range, x_range=agr_x_range, spreads=opts.spreads, crop_to_shp=crop_to_shp,
                      calc_annual=opts.annual_spreads)
    
    # save results
    # # decadal
    outpath_decadal = get_decadal_outpath(opts, opts.agr)
    logger.info(f'Saving decadal AGR results to {outpath_decadal}')
    # remove outpath_decadal if it exists
    if os.path.exists(outpath_decadal):
        try:
            os.remove(outpath_decadal)
        except PermissionError as err:
            if not DEBUG:
                raise err
            logger.info(f"Permission denied for {outpath_decadal}. Trying to save file to /tmp/ instead.")
            filename = os.path.basename(outpath_decadal)
            outpath_decadal = os.path.join('/tmp/', filename)
            if os.path.exists(outpath_decadal):
                os.remove(outpath_decadal)
    create_tea_history(cfg_params=opts, tea=tea, dataset='decadal_results')
    tea.save_decadal_results(filepath=outpath_decadal, save_tiff=opts.file_format == 'GeoTiff',
                             )
    
    # # annual
    if opts.annual_spreads:
        filepath_annual = _get_ctp_filepath(opts.start, opts.end, opts, annual_agr=True)
        logger.info(f'Saving annual AGR results to {filepath_annual}')
        create_tea_history(cfg_params=opts, tea=tea, dataset='ctp_results')
        # drop 3D vars already saved in GRG files
        tea.ctp_results = tea.ctp_results.drop_vars(
            [var for var in tea.ctp_results.data_vars if 'AGR' not in var])
        tea.save_ctp_results(filepath=filepath_annual, save_tiff=opts.file_format == 'GeoTiff',
                             )
    
    # # amplification factors
    outpath_ampl = get_amplification_outpath(opts, opts.agr)
    logger.info(f'Saving AGR amplification factors to {outpath_ampl}')
    create_tea_history(cfg_params=opts, tea=tea, dataset='amplification_factors')
    tea.save_amplification_factors(filepath=outpath_ampl)


def _load_gr_grid_static(opts):
    """
    load grid of GRs mask and area grid for AGR calculation

    Args:
        opts: options as defined in CFG-PARAMS-doc.md and TEA_CFG_DEFAULT.yaml

    Returns:
        gr_grid_mask: altitude-filtered coverage grid (xarray DataArray)
        gr_grid_areas: full geometric tile area grid (xarray DataArray)
        gr_grid_mask_full: pre-altitude region mask grid (xarray DataArray),
            or ``None`` for older cached files that require regeneration.

    """
    grid_token = grg_grid_token(opts)
    gr_grid_mask_file = (Path(opts.maskpath) / opts.mask_sub /
                         f'{opts.region}_mask_{grid_token}_{opts.dataset}_{opts.altitude_threshold}.nc')
    logger.info(f'Loading GR mask from {gr_grid_mask_file}')
    try:
        mask_ds = xr.open_dataset(gr_grid_mask_file)
        gr_grid_mask = mask_ds.data_vars.get('mask')
        gr_grid_mask_full = mask_ds.data_vars.get('mask_full')
    except FileNotFoundError:
        if opts.decadal_only:
            logger.warning(f'No GR mask found at {gr_grid_mask_file}.')
        gr_grid_mask = None
        gr_grid_mask_full = None
    gr_grid_areas_file = Path(opts.statpath) / f'area_grid_{grid_token}_{opts.region}_{opts.dataset}.nc'
    logger.info(f'Loading GR area grid from {gr_grid_areas_file}')
    try:
        gr_grid_areas = xr.open_dataset(gr_grid_areas_file)
        gr_grid_areas = gr_grid_areas.area_grid
    except (FileNotFoundError, AttributeError):
        if opts.decadal_only:
            logger.info(
                f'No GR area grid found at {gr_grid_areas_file}. Trying to generate one')
        gr_grid_areas = None
    return gr_grid_mask, gr_grid_areas, gr_grid_mask_full


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
    set_log_level(cmd_opts.loglevel)
    logger.info(f'Running teametrics version {TEA_VERSION}')
    
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
