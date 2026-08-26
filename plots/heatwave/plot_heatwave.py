#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot heatwave data
"""
import argparse
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import xarray as xr

from teametrics.common.config import load_opts

logger = logging.getLogger(__name__)

plot_data = True
show_plots = False
save_data = True


def _prepare_output_path(output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        backup_dir = output_path.parent / "bak"
        backup_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup_path = backup_dir / f"{output_path.stem}_{timestamp}{output_path.suffix}"
        counter = 1
        while backup_path.exists():
            backup_path = backup_dir / (
                f"{output_path.stem}_{timestamp}_{counter}{output_path.suffix}"
            )
            counter += 1
        logger.debug("Backing up existing output to %s", backup_path)
        shutil.copy2(output_path, backup_path)

    logger.info("Writing output to %s", output_path)
    return output_path


def _get_parameter_name(opts):
    threshold = f"{opts.threshold:g}"
    suffix = "p" if opts.threshold_type == "perc" else ""
    return f"{opts.parameter}{threshold}{suffix}"


def get_data(data_var="Tx30", data_path=None, time_interval=None, region="AUT"):
    year = int(time_interval[0][0:4])
    if 1961 <= year <= 1970:
        daily_file_names = f"DBV_{data_var}.0degC_{region}_annual_SPARTACUS_1961to1970.nc"
    elif 1971 <= year <= 1980:
        daily_file_names = f"DBV_{data_var}.0degC_{region}_annual_SPARTACUS_1971to1980.nc"
    elif 1981 <= year <= 1990:
        daily_file_names = f"DBV_{data_var}.0degC_{region}_annual_SPARTACUS_1981to1990.nc"
    elif 1991 <= year <= 2000:
        daily_file_names = f"DBV_{data_var}.0degC_{region}_annual_SPARTACUS_1991to2000.nc"
    elif 2001 <= year <= 2010:
        daily_file_names = f"DBV_{data_var}.0degC_{region}_annual_SPARTACUS_2001to2010.nc"
    elif 2011 <= year <= 2020:
        daily_file_names = f"DBV_{data_var}.0degC_{region}_annual_SPARTACUS_2011to2020.nc"
    elif 2021 <= year <= 2026:
        daily_file_names = f"DBV_{data_var}.0degC_{region}_annual_SPARTACUS_2021to2026.nc"
    else:
        logger.warning("No data available for the specified time interval: %s", time_interval)
        return None
    try:
        logger.info("Loading data from %s for variable %s and time interval %s",
                    data_path + daily_file_names, data_var, time_interval)
        data = xr.open_dataset(data_path + daily_file_names)
    except FileNotFoundError:
        logger.warning("File %s not found in %s", daily_file_names, data_path)
        return None
    return data


def _get_gr_area_size(data):
    if data is None or 'area_grid' not in data:
        return None
    return float(data['area_grid'].sum(skipna=True).values)


def _save_heatwave_csv(data, output_path, gr_area_size=None):
    dataframe = data.to_dataframe()
    if gr_area_size is not None:
        dataframe['GR_area_size'] = gr_area_size
    dataframe.to_csv(_prepare_output_path(output_path))


def _select_dtem(data, heat_map_date, source_name):
    if data is None or 'DTEM' not in data:
        return None
    try:
        return data['DTEM'].sel(time=heat_map_date)
    except KeyError as error:
        raise ValueError(f"Date {heat_map_date} is not available in {source_name} DTEM data") from error


def plot_dtem_heatmap(data, heat_map_date, output_dir, parameter, region,
                      detrended_data=None, detrend_ctp=None, separate=False):
    real_world = _select_dtem(data, heat_map_date, 'real-world')
    detrended = _select_dtem(detrended_data, heat_map_date, 'detrended')
    if real_world is None and detrended is None:
        raise ValueError('DTEM is not available in either real-world or detrended data')

    fields = [field for field in (real_world, detrended) if field is not None]
    vmax = max(float(field.max(skipna=True).values) for field in fields)
    if vmax <= 0:
        vmax = 1
    date_label = np.datetime_as_string(np.asarray(fields[0].time.values), unit='D')
    parameter_suffix = f'{parameter}_' if parameter else ''

    map_fields = []
    if real_world is not None:
        map_fields.append((real_world, 'Real-world', ''))
    if detrended is not None:
        map_fields.append((detrended, 'Detrended', f'_DETREND_{detrend_ctp}'))

    for field, title_suffix, filename_suffix in map_fields:
        if separate:
            figure, axes = plt.subplots(figsize=(10, 7), constrained_layout=True)
            axes = [axes]
        else:
            figure, axes = plt.subplots(1, len(map_fields), figsize=(7 * len(map_fields), 6),
                                        squeeze=False, constrained_layout=True)
            axes = axes[0]

        for axis, (map_field, panel_title, _) in zip(axes, map_fields if not separate else [(field, title_suffix, '')]):
            plotted = map_field.where(map_field > 0).plot.pcolormesh(
                ax=axis, cmap='YlOrRd', vmin=0, vmax=vmax, add_colorbar=False,
                shading='auto')
            axis.set_title(panel_title)
            axis.set_xlabel('x (m)')
            axis.set_ylabel('y (m)')
            axis.set_aspect('equal')

        figure.colorbar(plotted, ax=axes, label='DTEM (K)', shrink=0.85)
        figure.suptitle(f'Gridded DTEM on {date_label} | {parameter} | {region}')
        filename = f'heatwave_map_DTEM_{parameter_suffix}{date_label}{filename_suffix}_{region}.png'
        figure.savefig(_prepare_output_path(output_dir / filename), dpi=180)
        plt.close(figure)


def plot_daily_heatwave(heatwave_data, heatwave_period, data_var="DTEMA_GR",
                        detrended_heatwave_data=None,
                        detrend_ctp=None,
                        output_dir=None,
                        region="AUT",
                        parameter=None,
                        logarithmic=False,
                        ):
    
    plt.figure(figsize=(10, 6))
    myplot = heatwave_data.plot(marker='o')[0]
    if detrended_heatwave_data is not None:
        detrended_heatwave_data.plot(marker='o')
        plt.legend(['Original', 'Detrended'], loc='upper left')
    plt.title(f'Daily {data_var} during Heatwave Period: {heatwave_period[0]} to {heatwave_period[1]}')
    plt.xlabel('Date')
    # set x-axis ticks to show every day in the heatwave period
    
    # locator = mdates.AutoDateLocator()
    locator = mdates.DayLocator(interval=1)
    formatter = mdates.ConciseDateFormatter(locator)
    
    myplot.axes.xaxis.set_major_locator(locator)
    myplot.axes.xaxis.set_major_formatter(formatter)
    
    if logarithmic:
        myplot.axes.set_yscale('log')
    # tight layout to prevent overlap
    plt.tight_layout()
    plt.ylabel('areal degC')
    plt.grid(True)
    
    log_suffix = '_LOG' if logarithmic else ''
    parameter_suffix = f'{parameter}_' if parameter else ''
    if detrended_heatwave_data is not None:
        filename = (
            f'heatwave_daily{log_suffix}_{parameter_suffix}DETREND_{detrend_ctp}_{data_var}_'
            f'{region}_{heatwave_period[0]}_{heatwave_period[1]}.png'
        )
    else:
        filename = (
            f'heatwave_daily{log_suffix}_{parameter_suffix}{data_var}_'
            f'{region}_{heatwave_period[0]}_{heatwave_period[1]}.png'
        )
    plt.savefig(_prepare_output_path(output_dir / filename))
    if show_plots:
        plt.show()


def plot_cumulative_heatwave(heatwave_cumulative, heatwave_period, data_var="DTEMA_GR",
                             detrended_data=None, detrend_ctp=None, output_dir=None, region="AUT",
                             parameter=None):
    set_ylim = True
    plt.figure(figsize=(10, 6))
    myplot = heatwave_cumulative.plot()[0]
    if detrended_data is not None:
        detrended_data.plot()
        plt.legend(['Original', 'Detrended'], loc='upper left')
    plt.title(f'Cumulative {data_var} during Heatwave Period: {heatwave_period[0]} to {heatwave_period[1]}')
    plt.xlabel('Date')
    y_max = heatwave_cumulative.max().values
    if set_ylim:
        # round to next 1000
        y_max = np.ceil(y_max / 1000) * 1000
        plt.ylim(0, y_max)
    # set x-axis ticks to show every day in the heatwave period
    
    # locator = mdates.AutoDateLocator()
    locator = mdates.DayLocator(interval=1)
    formatter = mdates.ConciseDateFormatter(locator)
    
    myplot.axes.xaxis.set_major_locator(locator)
    myplot.axes.xaxis.set_major_formatter(formatter)
    # xticks = heatwave_cumulative.time.values
    # plt.xticks(xticks, [str(np.datetime64(t, 'D')) for t in xticks], rotation=45)
    # tight layout to prevent overlap
    plt.tight_layout()
    plt.ylabel('areal degC days')
    plt.grid(True)
    
    parameter_suffix = f'{parameter}_' if parameter else ''
    if detrended_data is not None:
        filename = (
            f'heatwave_cumulative_{parameter_suffix}DETREND_{detrend_ctp}_{data_var}_'
            f'{region}_{heatwave_period[0]}_{heatwave_period[1]}.png'
        )
    else:
        filename = (
            f'heatwave_cumulative_{parameter_suffix}{data_var}_'
            f'{region}_{heatwave_period[0]}_{heatwave_period[1]}.png'
        )
    plt.savefig(_prepare_output_path(output_dir / filename))
    if show_plots:
        plt.show()


def calc_heatwave_metrics(data, heatwave_period, data_var="DTEMA_GR", add_values=None):
    # Select the heatwave period
    heatwave_data = data.sel(time=slice(heatwave_period[0], heatwave_period[1]))[data_var]
    if add_values is not None:
        new_times = [np.datetime64("2026-06-30")]
        new_data = xr.DataArray(add_values, coords=[new_times], dims=["time"])
        heatwave_data = xr.concat([heatwave_data, new_data], dim="time")
    heatwave_total = heatwave_data.sum(dim='time')
    heatwave_cumulative = heatwave_data.cumsum(dim='time')
    mean_heatwave = heatwave_data.mean(dim='time')
    return heatwave_data, heatwave_total, heatwave_cumulative, mean_heatwave


# noinspection PyStringConversionWithoutDunderMethod
def calc_and_plot_heatwave(data, heatwave_period, data_var="DTEMA_GR", detrended_data=None,
                           detrend_ctp=None,
                           output_dir=None,
                           region="AUT",
                           parameter=None,
                           ):
    gr_area_size = _get_gr_area_size(data)
    heatwave_data, heatwave_total, heatwave_cumulative, mean_heatwave = calc_heatwave_metrics(
        data, heatwave_period, data_var=data_var, add_values=None)
    logger.info("Heatwave TEX_GR = S_GR for period %s to %s: %s areal degC days / event, "
                "event_mean MA_GR = %s areal degC, event_max MA_GR = %s areal degC",
                heatwave_period[0], heatwave_period[1], f"{heatwave_total.values:.0f}",
                f"{mean_heatwave.values:.0f}", f"{heatwave_data.max().values:.0f}")
    if save_data:
        # save csv files for heatwave_data, heatwave_total, heatwave_cumulative, mean_heatwave
        parameter_suffix = f'{parameter}_' if parameter else ''
        daily_output = output_dir / (
            f'heatwave_daily_{parameter_suffix}{data_var}_{region}_{heatwave_period[0]}_{heatwave_period[1]}.csv'
        )
        cumulative_output = output_dir / (
            f'heatwave_cumulative_{parameter_suffix}{data_var}_{region}_{heatwave_period[0]}_{heatwave_period[1]}.csv'
        )
        _save_heatwave_csv(heatwave_data, daily_output, gr_area_size)
        _save_heatwave_csv(heatwave_cumulative, cumulative_output, gr_area_size)
    
    if detrended_data is not None:
        detrended_heatwave_data, detrended_heatwave_total, detrended_heatwave_cumulative, detrended_mean_heatwave = (
            calc_heatwave_metrics(
                detrended_data, heatwave_period, data_var=data_var, add_values=None))
        logger.info("Detrended Heatwave TEX_GR = S_GR for period %s to %s: %s areal degC days / event, "
                    "event_mean MA_GR = %s areal degC, event_max MA_GR = %s areal degC",
                    heatwave_period[0], heatwave_period[1], f"{detrended_heatwave_total.values:.0f}",
                    f"{detrended_mean_heatwave.values:.0f}",
                    f"{detrended_heatwave_data.max().values:.0f}")
        if save_data:
            # save csv files for detrended_heatwave_data, detrended_heatwave_total, detrended_heatwave_cumulative,
            #  detrended_mean_heatwave
            detrended_daily_output = output_dir / (
                f'heatwave_daily_{parameter_suffix}DETREND_{detrend_ctp}_{data_var}_'
                f'{region}_{heatwave_period[0]}_{heatwave_period[1]}.csv'
            )
            detrended_cumulative_output = output_dir / (
                f'heatwave_cumulative_{parameter_suffix}DETREND_{detrend_ctp}_{data_var}_'
                f'{region}_{heatwave_period[0]}_{heatwave_period[1]}.csv'
            )
            _save_heatwave_csv(detrended_heatwave_data, detrended_daily_output, gr_area_size)
            _save_heatwave_csv(detrended_heatwave_cumulative, detrended_cumulative_output, gr_area_size)
    else:
        detrended_heatwave_data = None
        detrended_heatwave_cumulative = None
    
    if not plot_data:
        return
    
    # Plotting
    plot_cumulative_heatwave(heatwave_cumulative, heatwave_period, data_var=data_var,
                             detrended_data=detrended_heatwave_cumulative, detrend_ctp=detrend_ctp,
                             output_dir=output_dir, region=region, parameter=parameter)
    
    plot_daily_heatwave(heatwave_data, heatwave_period, data_var=data_var,
                        detrended_heatwave_data=detrended_heatwave_data, detrend_ctp=detrend_ctp,
                        output_dir=output_dir, region=region, parameter=parameter)
    plot_daily_heatwave(heatwave_data, heatwave_period, data_var=data_var,
                        detrended_heatwave_data=detrended_heatwave_data, detrend_ctp=detrend_ctp,
                        output_dir=output_dir,
                        region=region,
                        parameter=parameter,
                        logarithmic=True)


def _getopts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', '-cf',
                        dest='config_file',
                        type=str,
                        default='../TEA_CFG.yaml',
                        help='TEA configuration file (default: TEA_CFG.yaml)')
    parser.add_argument('--region',
                        type=str,
                        default=None,
                        help='GeoRegion to plot (default: region from the configuration file)')
    parser.add_argument('--heat-map-date',
                        type=str,
                        default=None,
                        help='Plot gridded DTEM for one date (YYYY-MM-DD)')
    parser.add_argument('--heat-map-separate',
                        action='store_true',
                        help='Save real-world and detrended DTEM maps as separate files')
    opts = parser.parse_args()
    if opts.heat_map_separate and opts.heat_map_date is None:
        parser.error('--heat-map-separate requires --heat-map-date')
    return opts


def run_main(opts, detrend_ctp="JJA", heat_map_date=None, heat_map_separate=False):
    parameter = _get_parameter_name(opts)
    data_var = parameter
    configured_output_path = Path(opts.outpath)
    heatwave_output_dir = configured_output_path / "heatwave_data"
    heatwave_output_dir.mkdir(parents=True, exist_ok=True)
    daily_data_path_real_world = str(configured_output_path / "daily_basis_variables") + "/"
    daily_data_path_detrended = str(
        configured_output_path / f"SPARTACUS_detrended/{detrend_ctp}" / "daily_basis_variables"
    ) + "/"
    
    # heatwave_period = ["2026-06-17", "2026-07-01"]
    # worker(daily_data_path_detrended, daily_data_path_real_world, data_var, heatwave_period,
    #        detrend_ctp=detrend_ctp, output_dir=heatwave_output_dir)
    #
    heatwave_period = ["2026-07-25", "2026-08-12"]
    data, detrended_data = worker(
        daily_data_path_detrended, daily_data_path_real_world, data_var, heatwave_period,
        region=opts.region, detrend_ctp=detrend_ctp, output_dir=heatwave_output_dir,
        parameter=parameter)
    if heat_map_date is not None:
        plot_dtem_heatmap(data, heat_map_date, heatwave_output_dir, parameter, opts.region,
                          detrended_data=detrended_data, detrend_ctp=detrend_ctp,
                          separate=heat_map_separate)

    # heatwave_period = ["2013-07-16", "2013-08-09"]
    # worker(daily_data_path_detrended, daily_data_path_real_world, data_var, heatwave_period, detrend_ctp=detrend_ctp)
    
    # heatwave_period = ["1983-07-16", "1983-08-02"]
    # worker(daily_data_path_detrended, daily_data_path_real_world, data_var, heatwave_period, detrend_ctp=detrend_ctp)


def worker(daily_data_path_detrended: str, daily_data_path_real_world: str, data_var: str, heatwave_period: list[str],
           region: str = "AUT", detrend_ctp="JJA", output_dir=None, parameter=None):
    data = get_data(data_var, daily_data_path_real_world, heatwave_period, region=region)
    detrended_data = get_data(data_var, daily_data_path_detrended, heatwave_period, region=region)
    for output_var in ('DTEA_GR', 'DTEM_GR', 'DTEMA_GR'):
        if output_var not in data:
            logger.warning("Variable %s not found in real-world data; skipping", output_var)
            continue
        if detrended_data is not None and output_var not in detrended_data:
            logger.warning("Variable %s not found in detrended data; plotting real-world data only", output_var)
            detrended_output = None
        else:
            detrended_output = detrended_data
        calc_and_plot_heatwave(data, heatwave_period, data_var=output_var,
                               detrended_data=detrended_output, detrend_ctp=detrend_ctp,
                               output_dir=output_dir, region=region, parameter=parameter)
    return data, detrended_data


if __name__ == "__main__":
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.addFilter(lambda record: record.levelno == logging.INFO)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        handlers=[stdout_handler, stderr_handler],
    )
    cmd_opts = _getopts()
    opts = load_opts(fname=__file__, config_file=cmd_opts.config_file)
    if cmd_opts.region is not None:
        opts.region = cmd_opts.region
    run_main(opts, 'JJA', heat_map_date=cmd_opts.heat_map_date,
             heat_map_separate=cmd_opts.heat_map_separate)
    # run_main(opts, 'June')
