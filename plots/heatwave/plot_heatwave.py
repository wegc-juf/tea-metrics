#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot heatwave data
"""
import argparse
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import xarray as xr

from teametrics.common.config import load_opts

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
        shutil.copy2(output_path, backup_path)

    return output_path


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
        print(f"No data available for the specified time interval: {time_interval}")
        return None
    try:
        print(f"Loading data from {data_path + daily_file_names} for variable {data_var} and time interval"
              f" {time_interval}")
        data = xr.open_dataset(data_path + daily_file_names)
    except FileNotFoundError:
        print(f"File {daily_file_names} not found in {data_path}. Please check the path and file name.")
        return None
    return data


def plot_daily_heatwave(heatwave_data, heatwave_period, data_var="DTEMA_GR",
                        detrended_heatwave_data=None,
                        detrend_ctp=None,
                        output_dir=None,
                        region="AUT",
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
    if detrended_heatwave_data is not None:
        filename = (
            f'heatwave_daily{log_suffix}_DETREND_{detrend_ctp}_{data_var}_'
            f'{region}_{heatwave_period[0]}_{heatwave_period[1]}.png'
        )
    else:
        filename = (
            f'heatwave_daily{log_suffix}_{data_var}_'
            f'{region}_{heatwave_period[0]}_{heatwave_period[1]}.png'
        )
    plt.savefig(_prepare_output_path(output_dir / filename))
    if show_plots:
        plt.show()


def plot_cumulative_heatwave(heatwave_cumulative, heatwave_period, data_var="DTEMA_GR",
                             detrended_data=None, detrend_ctp=None, output_dir=None, region="AUT"):
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
    
    if detrended_data is not None:
        filename = (
            f'heatwave_cumulative_DETREND_{detrend_ctp}_{data_var}_'
            f'{region}_{heatwave_period[0]}_{heatwave_period[1]}.png'
        )
    else:
        filename = (
            f'heatwave_cumulative_{data_var}_'
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
                           ):
    heatwave_data, heatwave_total, heatwave_cumulative, mean_heatwave = calc_heatwave_metrics(
        data, heatwave_period, data_var=data_var, add_values=None)
    print(f"Heatwave TEX_GR = S_GR for period {heatwave_period[0]} to {heatwave_period[1]}:"
          f" {heatwave_total.values:.0f} areal degC days / event, event_mean MA_GR = {mean_heatwave.values:.0f} "
          f"areal degC, event_max MA_GR = {heatwave_data.max().values:.0f} areal degC")
    if save_data:
        # save csv files for heatwave_data, heatwave_total, heatwave_cumulative, mean_heatwave
        daily_output = output_dir / (
            f'heatwave_daily_{data_var}_{region}_{heatwave_period[0]}_{heatwave_period[1]}.csv'
        )
        cumulative_output = output_dir / (
            f'heatwave_cumulative_{data_var}_{region}_{heatwave_period[0]}_{heatwave_period[1]}.csv'
        )
        heatwave_data.to_dataframe().to_csv(_prepare_output_path(daily_output))
        heatwave_cumulative.to_dataframe().to_csv(_prepare_output_path(cumulative_output))
    
    if detrended_data is not None:
        detrended_heatwave_data, detrended_heatwave_total, detrended_heatwave_cumulative, detrended_mean_heatwave = (
            calc_heatwave_metrics(
                detrended_data, heatwave_period, data_var=data_var, add_values=None))
        print(f"Detrended Heatwave TEX_GR = S_GR for period {heatwave_period[0]} to {heatwave_period[1]}:"
              f" {detrended_heatwave_total.values:.0f} areal degC days / event, event_mean MA_GR ="
              f" {detrended_mean_heatwave.values:.0f} "
              f"areal degC, event_max MA_GR = {detrended_heatwave_data.max().values:.0f} areal degC")
        if save_data:
            # save csv files for detrended_heatwave_data, detrended_heatwave_total, detrended_heatwave_cumulative,
            #  detrended_mean_heatwave
            detrended_daily_output = output_dir / (
                f'heatwave_daily_DETREND_{detrend_ctp}_{data_var}_'
                f'{region}_{heatwave_period[0]}_{heatwave_period[1]}.csv'
            )
            detrended_cumulative_output = output_dir / (
                f'heatwave_cumulative_DETREND_{detrend_ctp}_{data_var}_'
                f'{region}_{heatwave_period[0]}_{heatwave_period[1]}.csv'
            )
            detrended_heatwave_data.to_dataframe().to_csv(_prepare_output_path(detrended_daily_output))
            detrended_heatwave_cumulative.to_dataframe().to_csv(
                _prepare_output_path(detrended_cumulative_output)
            )
    else:
        detrended_heatwave_data = None
        detrended_heatwave_cumulative = None
    
    if not plot_data:
        return
    
    # Plotting
    plot_cumulative_heatwave(heatwave_cumulative, heatwave_period, data_var=data_var,
                             detrended_data=detrended_heatwave_cumulative, detrend_ctp=detrend_ctp,
                             output_dir=output_dir, region=region)
    
    plot_daily_heatwave(heatwave_data, heatwave_period, data_var=data_var,
                        detrended_heatwave_data=detrended_heatwave_data, detrend_ctp=detrend_ctp,
                        output_dir=output_dir, region=region)
    plot_daily_heatwave(heatwave_data, heatwave_period, data_var=data_var,
                        detrended_heatwave_data=detrended_heatwave_data, detrend_ctp=detrend_ctp,
                        output_dir=output_dir,
                        region=region,
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
    return parser.parse_args()


def run_main(opts, detrend_ctp="JJA"):
    data_var = "Tx30"
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
    heatwave_period = ["2026-07-25", "2026-08-08"]
    worker(daily_data_path_detrended, daily_data_path_real_world, data_var, heatwave_period,
           region=opts.region, detrend_ctp=detrend_ctp, output_dir=heatwave_output_dir)

    # heatwave_period = ["2013-07-16", "2013-08-09"]
    # worker(daily_data_path_detrended, daily_data_path_real_world, data_var, heatwave_period, detrend_ctp=detrend_ctp)
    
    # heatwave_period = ["1983-07-16", "1983-08-02"]
    # worker(daily_data_path_detrended, daily_data_path_real_world, data_var, heatwave_period, detrend_ctp=detrend_ctp)


def worker(daily_data_path_detrended: str, daily_data_path_real_world: str, data_var: str, heatwave_period: list[str],
           region: str = "AUT", detrend_ctp="JJA", output_dir=None):
    data = get_data(data_var, daily_data_path_real_world, heatwave_period, region=region)
    detrended_data = get_data(data_var, daily_data_path_detrended, heatwave_period, region=region)
    calc_and_plot_heatwave(data, heatwave_period, detrended_data=detrended_data, detrend_ctp=detrend_ctp,
                           output_dir=output_dir, region=region)


if __name__ == "__main__":
    cmd_opts = _getopts()
    opts = load_opts(fname=__file__, config_file=cmd_opts.config_file)
    if cmd_opts.region is not None:
        opts.region = cmd_opts.region
    run_main(opts, 'JJA')
    # run_main(opts, 'June')
