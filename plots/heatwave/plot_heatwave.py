#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot heatwave data
"""
import xarray as xr


def get_data(data_var="Tx30", data_path=None):
    daily_file_names = f"DBV_{data_var}.0degC_AUT_annual_SPARTACUS_2021to2026.nc"
    data = xr.open_dataset(data_path + daily_file_names)
    return data


def plot_heatwave(data, heatwave_period, data_var="DTEMA_GR", detrended_data=None):
    import matplotlib.pyplot as plt
    import numpy as np

    def calc_heatwave_metrics(data, heatwave_period, data_var="DTEMA_GR"):
        # Select the heatwave period
        heatwave_data = data.sel(time=slice(heatwave_period[0], heatwave_period[1]))[data_var]
        heatwave_total = heatwave_data.sum(dim='time')
        heatwave_cumulative = heatwave_data.cumsum(dim='time')
        mean_heatwave = heatwave_data.mean(dim='time')
        return heatwave_data, heatwave_total, heatwave_cumulative, mean_heatwave
    heatwave_data, heatwave_total, heatwave_cumulative, mean_heatwave = calc_heatwave_metrics(data, heatwave_period, data_var=data_var)
    if detrended_data is not None:
        detrended_heatwave_data, detrended_heatwave_total, detrended_heatwave_cumulative, detrended_mean_heatwave = calc_heatwave_metrics(detrended_data, heatwave_period, data_var=data_var)

    # Plotting
    plt.figure(figsize=(10, 6))
    heatwave_cumulative.plot()
    if detrended_data is not None:
        detrended_heatwave_cumulative.plot()
        plt.legend(['Original', 'Detrended'], loc='upper left')
    plt.title(f'Cumulative {data_var} during Heatwave Period: {heatwave_period[0]} to {heatwave_period[1]}')
    plt.xlabel('Date')
    y_max = 14000
    plt.ylim(0, y_max)
    # set x-axis ticks to show every day in the heatwave period
    xticks = heatwave_data.time.values
    plt.xticks(xticks, [str(np.datetime64(t, 'D')) for t in xticks], rotation=45)
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
    daily_data_path = "/home/wegnet/results/TEA_indicators_test/daily_basis_variables/"
    data = get_data(data_var, daily_data_path)
    
    daily_data_path = "/home/wegnet/results/SPARTACUS_DETRENDED_June/daily_basis_variables/"
    detrended_data = get_data(data_var, daily_data_path)
    # Add plotting code here
    heatwave_period = ["2026-06-17", "2026-06-30"]
    plot_heatwave(data, heatwave_period, detrended_data=detrended_data)


if __name__ == "__main__":
    run_main()
