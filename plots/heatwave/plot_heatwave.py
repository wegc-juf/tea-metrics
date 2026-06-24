#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot heatwave data
"""
import xarray as xr
import numpy as np


def get_data(data_var="Tx30", data_path=None, time_interval=None):
    year = int(time_interval[0][0:4])
    if 1961 <= year <= 1970:
        daily_file_names = f"DBV_{data_var}.0degC_AUT_annual_SPARTACUS_1961to1970.nc"
    elif 1971 <= year <= 1980:
        daily_file_names = f"DBV_{data_var}.0degC_AUT_annual_SPARTACUS_1971to1980.nc"
    elif 1981 <= year <= 1990:
        daily_file_names = f"DBV_{data_var}.0degC_AUT_annual_SPARTACUS_1981to1990.nc"
    elif 1991 <= year <= 2000:
        daily_file_names = f"DBV_{data_var}.0degC_AUT_annual_SPARTACUS_1991to2000.nc"
    elif 2001 <= year <= 2010:
        daily_file_names = f"DBV_{data_var}.0degC_AUT_annual_SPARTACUS_2001to2010.nc"
    elif 2011 <= year <= 2020:
        daily_file_names = f"DBV_{data_var}.0degC_AUT_annual_SPARTACUS_2011to2020.nc"
    elif 2021 <= year <= 2026:
        daily_file_names = f"DBV_{data_var}.0degC_AUT_annual_SPARTACUS_2021to2026.nc"
    else:
        print(f"No data available for the specified time interval: {time_interval}")
        return None
    try:
        data = xr.open_dataset(data_path + daily_file_names)
    except FileNotFoundError:
        print(f"File {daily_file_names} not found in {data_path}. Please check the path and file name.")
        return None
    return data


def plot_heatwave(data, heatwave_period, data_var="DTEMA_GR", detrended_data=None):
    set_ylim = False
    
    import matplotlib.pyplot as plt
    import numpy as np

    def calc_heatwave_metrics(data, heatwave_period, data_var="DTEMA_GR", add_values=None):
        # Select the heatwave period
        heatwave_data = data.sel(time=slice(heatwave_period[0], heatwave_period[1]))[data_var]
        if add_values is not None:
            new_times = [np.datetime64("2026-06-29"), np.datetime64("2026-06-30")]
            new_data = xr.DataArray(add_values, coords=[new_times], dims=["time"])
            heatwave_data = xr.concat([heatwave_data, new_data], dim="time")
        heatwave_total = heatwave_data.sum(dim='time')
        heatwave_cumulative = heatwave_data.cumsum(dim='time')
        mean_heatwave = heatwave_data.mean(dim='time')
        return heatwave_data, heatwave_total, heatwave_cumulative, mean_heatwave
    
    heatwave_data, heatwave_total, heatwave_cumulative, mean_heatwave = calc_heatwave_metrics(
        data, heatwave_period, data_var=data_var, add_values=[900, 300] if heatwave_period[0] == "2026-06-17" else None)
    y_max = heatwave_cumulative.max().values

    if detrended_data is not None:
        detrended_heatwave_data, detrended_heatwave_total, detrended_heatwave_cumulative, detrended_mean_heatwave = (
            calc_heatwave_metrics(
                detrended_data, heatwave_period, data_var=data_var, add_values=[150, 50] if
            heatwave_period[0] == "2026-06-17" else None))

    # Plotting
    plt.figure(figsize=(10, 6))
    myplot = heatwave_cumulative.plot()[0]
    if detrended_data is not None:
        detrended_heatwave_cumulative.plot()
        plt.legend(['Original', 'Detrended'], loc='upper left')
    plt.title(f'Cumulative {data_var} during Heatwave Period: {heatwave_period[0]} to {heatwave_period[1]}')
    plt.xlabel('Date')
    if set_ylim:
        # round to next 1000
        y_max = np.ceil(y_max / 1000) * 1000
        plt.ylim(0, y_max)
    # set x-axis ticks to show every day in the heatwave period
    xticks = heatwave_data.time.values
    import matplotlib.dates as mdates
    
    locator = mdates.AutoDateLocator()
    locator = mdates.DayLocator(interval=1)
    formatter = mdates.ConciseDateFormatter(locator)
    
    myplot.axes.xaxis.set_major_locator(locator)
    myplot.axes.xaxis.set_major_formatter(formatter)
    # plt.xticks(xticks, [str(np.datetime64(t, 'D')) for t in xticks], rotation=45)
    # tight layout to prevent overlap
    plt.tight_layout()
    plt.ylabel('areal degC days')
    plt.grid(True)
    if detrended_data is not None:
        plt.savefig(f'heatwave_cumulative_DETREND_{data_var}_{heatwave_period[0]}_{heatwave_period[1]}.png')
    else:
        plt.savefig(f'heatwave_cumulative_{data_var}_{heatwave_period[0]}_{heatwave_period[1]}.png')
    plt.show()


def run_main():
    data_var = "Tx30"
    daily_data_path_real_world = "/home/wegnet/results/TEA_indicators_test/daily_basis_variables/"
    
    daily_data_path_detrended = "/home/wegnet/results/SPARTACUS_DETRENDED_JJA/daily_basis_variables/"
    # Add plotting code here
    heatwave_period = ["2026-06-17", "2026-06-30"]
    data = get_data(data_var, daily_data_path_real_world, heatwave_period)
    
    # expand the data to include the new values
    new_times = [np.datetime64("2026-06-29"), np.datetime64("2026-06-30")]
    detrended_data = get_data(data_var, daily_data_path_detrended, heatwave_period)
    plot_heatwave(data, heatwave_period, detrended_data=detrended_data)
    
    heatwave_period = ["2013-07-16", "2013-08-09"]
    data = get_data(data_var, daily_data_path_real_world, heatwave_period)
    detrended_data = get_data(data_var, daily_data_path_detrended, heatwave_period)
    plot_heatwave(data, heatwave_period, detrended_data=detrended_data)
    #
    heatwave_period = ["1983-07-16", "1983-08-02"]
    data = get_data(data_var, daily_data_path_real_world, heatwave_period)
    detrended_data = get_data(data_var, daily_data_path_detrended, heatwave_period)
    plot_heatwave(data, heatwave_period, detrended_data=detrended_data)


if __name__ == "__main__":
    run_main()
