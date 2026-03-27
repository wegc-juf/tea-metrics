"""
scripts for general stuff (e.g. nc-history)
"""

import os
import glob
import datetime as dt

import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path

from .. import __version__
from .TEA_logger import logger
from .var_attrs import get_global_attrs


def create_history_from_cli_params(cli_params, ds, dsname):
    """
    add history to dataset
    :param cli_params: CLI parameter
    :param ds: dataset
    :param dsname: name of the dataset (e.g. 'SPARTACUS', 'ERA5', etc.)
    :return: ds with history in attrs
    """

    script = cli_params[0].split('/')[-1]
    cli_params = cli_params[1:]

    _create_history_for_dataset(ds, cli_params, script, dsname=dsname)


def create_history_from_cfg(cfg_params, ds):
    """
    add history to dataset
    :param cfg_params: CFG parameter
    :param ds: dataset
    :return: ds with history in attrs
    """

    parts = []
    for key, value in vars(cfg_params).items():
        if key != 'script':
            part = f"--{key} {value}"
            parts.append(part)
    params = ' '.join(parts)

    script = cfg_params.script.split('/')[-1]

    _create_history_for_dataset(ds, params, script, dsname=cfg_params.dataset)


def create_tea_history(cfg_params, tea, dataset):
    """
    add history and version to dataset

    :param cfg_params: yaml config parameters
    :param tea: TEA object
    :param dataset: dataset (e.g. 'CTP_results')
    """
    ds = getattr(tea, f'{dataset}')

    # create history from CFG parameters
    create_history_from_cfg(cfg_params, ds)

    # get global attributes
    glob_attrs = get_global_attrs(level=dataset, period=cfg_params.period)
    glob_attrs.update(ds.attrs)

    # update dataset attributes
    ds.attrs = glob_attrs


def _create_history_for_dataset(ds, params, script, dsname):
    """
    Helper function to create history and version for a dataset.
    Args:
        ds: Xarray Dataset
        params: configuration parameters as a string
        script: name of the script that generated the dataset
        dsname: name of the dataset (e.g. 'SPARTACUS', 'ERA5', etc.)

    Returns:

    """
    new_hist = f'{dt.datetime.now():%FT%H:%M:%S} {script} {params}; teametrics v{__version__}'
    if 'history' in ds.attrs:
        ds.attrs['history'] = ds.attrs['history'] + new_hist
    else:
        ds.attrs['history'] = new_hist
    if 'version' not in ds.attrs:
        ds.attrs['version'] = __version__
    if 'source' not in ds.attrs:
        ds.attrs['source'] = dsname


def create_natvar_history(cfg_params, nv):
    """
    add history to dataset
    :param cfg_params: yaml config parameters
    :param nv: NatVar object
    """
    ds = getattr(nv, f'nv')

    parts = []
    for key, value in vars(cfg_params).items():
        if key != 'script':
            part = f"--{key} {value}"
            parts.append(part)
    params = ' '.join(parts)

    script = cfg_params.script.split('/')[-1]

    if 'history' in ds.attrs:
        new_hist = f'{ds.history}; {dt.datetime.now():%FT%H:%M:%S} {script} {params}'
    else:
        new_hist = f'{dt.datetime.now():%FT%H:%M:%S} {script} {params}'

    nv.create_history_from_cli_params(new_hist, dsname=cfg_params.dataset)


def ref_cc_params():
    params = {'REF': {'start': '1961-01-01', 'end': '1990-12-31',
                      'start_cy': '1966-01-01', 'end_cy': '1986-12-31',
                      'ref_str': 'REF1961-1990'},
              'CC': {'start': '2010-01-01', 'end': '2024-12-31',
                     'start_cy': '2015-01-01', 'end_cy': '2020-12-31',
                     'cc_str': 'CC2010-2024'}}

    return params


def compare_to_ref(tea_result, tea_ref, relative=False):
    for vvar in tea_result.data_vars:
        if vvar in tea_ref.data_vars:
            if relative:
                diff = (tea_result[vvar] - tea_ref[vvar]) / tea_ref[vvar]
                diff = diff.where(np.isfinite(diff), 0)
                threshold = .05
            else:
                diff = tea_result[vvar] - tea_ref[vvar]
                threshold = 5e-5
            max_diff = diff.max(skipna=True).values
            if max_diff > threshold:
                print(f'Maximum difference in {vvar} is {max_diff}')
        else:
            print(f'{vvar} not found in reference file.')


def get_input_filenames(start, end, inpath, param_str, ds_name, period='annual', hourly=False, hourly_path=None):
    """
    get input filenames

    :param start: start year
    :type start: int
    :param end: end year
    :type end: int
    :param inpath: input path
    :param param_str: parameter string
    :param ds_name: name of dataset of input data
    :param period: period of interest. Default is 'annual'
    :type period: str
    :param hourly: if True, return hourly data filenames
    :type hourly: bool
    :param hourly_path: path to hourly data

    :return: list of filenames
    """

    if ds_name == 'EOBS':
        filenames = Path.glob(f'{inpath}{param_str}*.nc')

    else:
        # check if inpath is file
        if os.path.isfile(inpath):
            return inpath
        elif '*' in inpath and glob.glob(inpath):
            return sorted(glob.glob(inpath))

        if hourly:
            inpath = hourly_path

        # select only files of interest, if chosen period is 'seasonal' append one year in the
        # beginning to have the first winter fully included
        filenames = []
        if period == 'seasonal' and start != '1961':
            yrs = np.arange(start - 1, end + 1)
        else:
            yrs = np.arange(start, end + 1)
        for yr in yrs:
            file_mask = f'{inpath}/*{param_str}*{yr}*.nc'
            year_files = sorted(glob.glob(file_mask))
            if not year_files:
                logger.warning(f'No input files found for year {yr} with mask {file_mask}.')
            filenames.extend(year_files)
    return filenames


def extract_period(ds, period, start_year=None, end_year=None):
    """
    select only times of interest

    Args:
        ds: Xarray DataArray or Pandas DataFrame
        period: period of interest (annual, seasonal, ESS, WAS, JJA)
        start_year: start year (in case of seasonal: start year of first winter season (optional)
        end_year: end year (optional)

    Returns:
        ds: Data with selected time period

    """
    if period == 'seasonal':
        first_year = ds.time[0].dt.year
        if start_year is not None and start_year > first_year:
            start = f'{start_year - 1}-12-01 00:00'
            end = f'{end_year}-11-30 23:59'
        else:
            # if first year is first year of record, exclude first winter (data of Dec 1960 missing)
            start = f'{first_year}-03-01 00:00'
            end = f'{last_year}-11-30 23:59'
        ds = ds.loc[start:end]
    elif start_year is not None and end_year is not None:
        start = f'{start_year}-01-01 00:00'
        end = f'{end_year}-12-31 23:59'
        ds = ds.loc[start:end]
    if period in ['ESS', 'WAS', 'JJA']:
        months = {'ESS': np.arange(5, 10), 'WAS': np.arange(4, 11), 'JJA': np.arange(6, 9)}
        if isinstance(ds, xr.DataArray):
            season = ds['time.month'].isin(months[period])
            ds = ds.sel(time=season)
        elif isinstance(ds, pd.DataFrame):
            season = ds.index.month.isin(months[period])
            ds = ds.loc[season]
        else:
            raise ValueError('ds must be either xarray DataArray or pandas DataFrame')
    return ds


def get_gridded_data(start, end, opts, period='annual', hourly=False):
    """
    loads data for parameter and period
    :param start: start year
    :ptype start: int
    :param end: end year
    :ptype end: int
    :param opts: options
    :param period: period to load (annual, seasonal, ESS, WAS, JJA); default: annual
    :ptype period: str
    :param hourly: if True, load hourly data
    :ptype hourly: bool

    :return: dataset of given parameter
    """

    param_str = ''
    parameter = opts.parameter
    if hourly:
        # use correct parameter for hourly data
        if opts.parameter == 'Tx':
            parameter = 'T'

    if opts.dataset == 'SPARTACUS' and not opts.precip:
        param_str = f'{parameter}'
    elif opts.dataset == 'SPARTACUS' and opts.precip:
        param_str = 'RR'

    filenames = get_input_filenames(period=period, start=start, end=end,
                                    inpath=opts.input_data_path,
                                    param_str=param_str, hourly=hourly, ds_name=opts.dataset,
                                    hourly_path=opts.hourly_data_path)

    # load relevant years
    if filenames == []:
        raise FileNotFoundError(f'No input files found for {param_str} in {opts.input_data_path}. Please check '
                                f'input_data_path and parameter settings.')

    logger.info(f'Loading data from {filenames}...')
    try:
        ds = xr.open_mfdataset(filenames, combine='by_coords', data_vars='all')
    except ValueError as e:
        logger.warning(f'Error loading data: {e} Trying again with combine="nested"')
        ds = xr.open_mfdataset(filenames, combine='nested', data_vars='all')

    # select variable
    if opts.dataset == 'SPARTACUS' and parameter == 'P24h_7to7':
        ds = ds.rename({'RR': parameter})
    
    # get temporal resolution of data
    time_diff = (ds.time[1] - ds.time[0]).values
    if time_diff < np.timedelta64(1, 'D'):
        subdaily = True
    else:
        subdaily = False

    if opts.dataset == 'EOBS':
        if opts.parameter == 'Tx':
            ds = ds.rename({'tx': parameter})
        elif opts.parameter == 'Tn':
            ds = ds.rename({'tn': parameter})
    try:
        data = ds[parameter]
    except KeyError as e:
        if subdaily:
            if opts.parameter == 'Tn' or opts.parameter == 'Tx':
                data = ds['T']
            else:
                raise e
        else:
            raise e

    # get only values from selected period
    data = extract_period(ds=data, period=period, start_year=start, end_year=end)

    # resample to daily data if hourly data is loaded but daily data is wanted
    if subdaily and not hourly:
        if opts.aggregation_method is not None:
            agg_method = opts.aggregation_method
            resampled = data.resample(time='1D')
            data = getattr(resampled, agg_method)()
        elif opts.parameter == 'Tx':
            data = data.resample(time='1D').max()
        elif opts.parameter == 'Tn':
            data = data.resample(time='1D').min()
        elif opts.precip:
            data = data.resample(time='1D').sum()
        else:
            data = data.resample(time='1D').mean()

    if opts.dataset == 'SPARTACUS':
        data = data.drop('lambert_conformal_conic')

    if not opts.use_dask:
        # load data into memory
        data.load()

    return data


def get_csv_data(opts):
    """
    load station data
    Args:
        opts: Config parameters as defined in CFG-PARAMS-doc.md and TEA_CFG_DEFAULT.yaml

    Returns:
        data: interpolated station data

    """

    if opts.parameter == 'Tx':
        pstr = 'Tmax'
        rename_dict = {'tmax': opts.parameter}
    else:
        pstr = 'RR'
        rename_dict = {'nied': opts.parameter}

    # read csv file of station data and set time as index of df
    filenames = f'{opts.input_data_path}{pstr}_{opts.station}*18770101*.csv'
    file = glob.glob(filenames)
    if len(file) == 0:
        filenames = f'{opts.input_data_path}{pstr}_{opts.station}*.csv'
        file = glob.glob(filenames)
    data = pd.read_csv(file[0])
    data['time'] = pd.to_datetime(data['time'])
    data = data.set_index('time')

    # remove timezone information
    data = data.tz_localize(None)

    # rename columns
    data = data.rename(columns=rename_dict)

    # interpolate missing data
    data = interpolate_gaps(opts=opts, data=data)

    # convert to xarray DataArray
    data = data.to_xarray()[opts.parameter]

    # extract only timestamps of interest
    data = extract_period(ds=data, period=opts.period, start_year=opts.start, end_year=opts.end)

    return data


def interpolate_gaps(opts, data):
    """
    interpolates data gaps with average of missing day from other years
    Args:
        opts: CLI parameter
        data: station data

    Returns:
        data: interpolated data
    """

    non_nan = data.loc[data[opts.parameter].notnull(), :]
    start_yr = non_nan.index[0]

    gaps = data[data[opts.parameter].isnull()]
    for igap in gaps.index:
        if igap < start_yr:
            continue
        # select all values from that day of year
        day_data = data[data.index.month == igap.month]
        day_data = day_data[day_data.index.day == igap.day]
        # calculate mean
        fill_val = day_data[opts.parameter].mean(skipna=True)
        # fill gap with fill value
        data.at[igap, opts.parameter] = fill_val

    return data


def calc_percentiles(opts, threshold_min=None, data=None):
    """
    calculate percentile of reference period for each grid point
    Args:
        opts: CLI parameter
        threshold_min: minimum data value for threshold calculation (e.g. 0.99 for precip); optional
        data: data to calculate percentiles for; if not provided, data will be loaded

    Returns:
        thresh: threshold (percentile) grid

    """

    # load data if not provided
    if data is None:
        data = get_gridded_data(start=opts.perc_period_yrs[0], end=opts.perc_period_yrs[1],
                                opts=opts,
                                period=opts.period)
    else:
        data = extract_period(ds=data, period=opts.perc_period, start_year=opts.perc_period_yrs[0],
                              end_year=opts.perc_period_yrs[1])

    if threshold_min is not None:
        data = data.where(data > threshold_min)

    # calc the chosen percentile for each grid point as threshold
    percent = data.chunk(dict(time=-1)).quantile(q=opts.threshold / 100, dim='time')

    # smooth SPARTACUS precip percentiles (for each grid point calculate the average of all grid
    # points within the given radius)
    if 'smoothing' in opts:
        radius = opts.smoothing_radius
    else:
        radius = 0

    if radius > 0:
        percent_smooth = smooth_data(percent, radius)
        percent = percent_smooth

    percent = percent.drop_vars('quantile')
    percent.load()

    return percent


def smooth_data(data, radius):
    y_size, x_size = data.shape
    percent_smooth_arr = np.full_like(data.values, np.nan)
    percent_tmp = np.zeros((y_size + 2 * radius, x_size + 2 * radius),
                           dtype='float32') * np.nan
    percent_tmp[radius:radius + y_size, radius:radius + x_size] = data
    rad_circ = radius + 0.5
    x_vec = np.arange(0, x_size + 2 * radius)
    y_vec = np.arange(0, y_size + 2 * radius)
    iy_new = 0
    for iy in range(radius, y_size):
        ix_new = 0
        for ix in range(radius, x_size):
            circ_mask = (x_vec[np.newaxis, :] - ix) ** 2 + (y_vec[:, np.newaxis] - iy) ** 2 \
                        < rad_circ ** 2
            percent_smooth_arr[iy_new, ix_new] = np.nanmean(percent_tmp[circ_mask])
            ix_new += 1
        iy_new += 1
    percent_smooth = xr.full_like(data, np.nan)
    percent_smooth[:, :] = percent_smooth_arr
    return percent_smooth


def create_threshold_grid(opts, data=None):
    """
    create threshold grid for the given parameter and reference period
    Args:
        opts: options as defined in CFG-PARAMS-doc.md and TEA_CFG_DEFAULT.yaml
        data: data to calculate threshold grid for; if not provided, data will be loaded

    Returns:
        thr_grid: threshold grid
    """
    if opts.precip:
        threshold_min = 0.99
    else:
        threshold_min = None
    thr_grid = calc_percentiles(opts=opts, threshold_min=threshold_min, data=data)
    thr_grid = thr_grid.rename('threshold')

    if opts.precip:
        ref_str = 'WetDays > 1 mm Ref'
    else:
        ref_str = 'Ref'
    vname = f'{opts.parameter}-p{opts.threshold}{opts.period} {ref_str}{opts.ref_period[0]}-{opts.ref_period[1]}'
    thr_grid.attrs = {'units': opts.unit, 'methods_variable_name': vname,
                      'percentile': f'{opts.threshold}'}
    return thr_grid
