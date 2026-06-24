#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot heatwave data
"""
import xarray as xr

def get_data(data_var="Tx30"):
    daily_data_path = "/home/wegnet/results/TEA_indicators_test/daily_basis_variables/"
    daily_file_names = f"DBV_{data_var}.0degC_AUT_annual_SPARTACUS_2021to2026.nc"
    data = xr.open_dataset(daily_data_path + daily_file_names)
    return data


def plot_heatwave(data, heatwave_period, data_var="DTEMA_GR"):
    import matplotlib.pyplot as plt
    import numpy as np

    # Select the heatwave period
    heatwave_data = data.sel(time=slice(heatwave_period[0], heatwave_period[1]))[data_var]
    heatwave_total = heatwave_data.sum(dim='time')
    heatwave_cumulative = heatwave_data.cumsum(dim='time')
    mean_heatwave = heatwave_data.mean(dim='time')

    # Plotting
    plt.figure(figsize=(10, 6))
    heatwave_cumulative.plot()
    plt.title(f'Cumulative {data_var} during Heatwave Period: {heatwave_period[0]} to {heatwave_period[1]}')
    plt.xlabel('Date')
    # set x-axis ticks to show every day in the heatwave period
    xticks = heatwave_data.time.values
    plt.xticks(xticks, [str(np.datetime64(t, 'D')) for t in xticks], rotation=45)
    # tight layout to prevent overlap
    plt.tight_layout()
    plt.ylabel('areal degC days')
    plt.grid(True)
    plt.savefig(f'heatwave_cumulative_{data_var}_{heatwave_period[0]}_{heatwave_period[1]}.png')
    plt.show()


def run_main():
    data_var = "Tx30"
    data = get_data(data_var)
    # Add plotting code here
    heatwave_period = ["2026-06-17", "2026-06-30"]
    plot_heatwave(data, heatwave_period)

if __name__ == "__main__":
    run_main()