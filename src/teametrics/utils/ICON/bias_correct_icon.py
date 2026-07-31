#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bias correction of ICON forecasts using SPARTACUS Tmax data.
"""

import logging
import xarray as xr
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from get_icon_data import ICON_PATH

SPARTACUS_PATH = "/data/reloclim/backup/ZAMG_SPARTACUS/data/current/"
logger = logging.getLogger(__name__)


def extend_spartacus_with_icon_single(
    spartacus_tmax,
    icon_yesterday,
    icon_today,
    temp_var="t2m",
):
    """
    Parameters
    ----------
    spartacus_tmax : xr.DataArray
        Daily Tmax from SPARTACUS.
        It contains yesterday.

    icon_yesterday : xr.Dataset
        Yesterday's ICON run (e.g. yesterday Z00).
        Must contain hourly temperatures and valid_time.

    icon_today : xr.Dataset
        Today's ICON run (e.g. today Z06).

    temp_var : str
        Temperature variable name.

    Returns
    -------
    forecast_hourly : xr.DataArray
        Bias-adjusted hourly forecast.

    forecast_tmax : xr.DataArray
        Daily Tmax forecast.
    """

    # --------------------------
    # 1. Find last observed day
    # --------------------------

    last_day = spartacus_tmax.time.max()

    obs_tmax = spartacus_tmax.sel(time=last_day)

    # --------------------------
    # 2. Compute ICON Tmax
    #    for same day
    # --------------------------

    day_start = last_day.values
    day_end = day_start + np.timedelta64(1, "D")

    icon_hist = icon_yesterday[temp_var].sel(
        valid_time=slice(day_start, day_end)
    )

    icon_tmax = icon_hist.max("valid_time")

    # --------------------------
    # 3. Bias field
    # --------------------------

    bias = obs_tmax - icon_tmax

    # dimensions:
    # lat x lon

    # --------------------------
    # 4. Correct current forecast
    # --------------------------

    corrected = bias_correct_forecast(icon_today[temp_var], bias)

    # --------------------------
    # 5. Daily Tmax forecast
    # --------------------------

    daily_tmax = (
        corrected
        .groupby("valid_time.date")
        .max()
        .rename("tmax")
    )

    return corrected, daily_tmax


def compute_bias_field_rolling(
    spartacus_tmax: xr.DataArray,
    icon_hist_tmax: xr.DataArray,
    window: int = 7,
):
    """
    Rolling mean bias field.

    bias = observed - forecast

    Returns bias for the most recent day.
    """

    common_days = np.intersect1d(
        spartacus_tmax.time.values,
        icon_hist_tmax.time.values,
    )

    obs = spartacus_tmax.sel(time=common_days)
    fc = icon_hist_tmax.sel(time=common_days)
    # convert K to °C
    fc = fc - 273.15

    daily_error = obs - fc
    
    window = min(window, common_days.size)
    weights = xr.DataArray(
        [1, 2, 3, 4, 5, 6, 7][:window],
        dims=["time"]
    )
    
    bias = (
        daily_error
        .isel(time=slice(-window, None))
        .weighted(weights)
        .mean("time")
    )

    return bias


def bias_correct_forecast(
    icon_forecast: xr.DataArray,
    bias_field: xr.DataArray,
):
    """
    Apply additive Tmax bias correction.
    """

    if np.isnan(bias_field.values).all():
        print("Bias field all NaNs. Skipping bias correction.")
        return icon_forecast
    
    return icon_forecast + bias_field


def forecast_daily_tmax(
    corrected_hourly: xr.DataArray,
        var_name
):
    return (
        corrected_hourly
        .resample(time="1D")
        .max()
        .rename(var_name)
    )


def expand_forecast(
    corrected_tmax: xr.DataArray,
    method="constant",
    source_days=None,
    offsets=None,
):
    """Append forecast days to a daily Tmax time series.

    Parameters
    ----------
    corrected_tmax : xr.DataArray
        Daily Tmax time series with a ``time`` coordinate.
    method : str, optional
        ``"constant"`` duplicates the final day and applies ``offsets``.
        ``"days"`` duplicates the gridded fields selected by ``source_days``.
    source_days : sequence, optional
        Source dates used in order when ``method="days"``. Each date adds one
        forecast day.
    offsets : scalar or sequence, optional
        Additive offsets for the appended days. In ``"days"`` mode, omitted
        offsets leave the copied fields unchanged, a scalar is applied to
        every copied field, and a sequence is applied in source-day order.
        Constant mode uses ``(-5.0, -5.0)`` when omitted.
    """
    if method not in {"constant", "days"}:
        raise ValueError("method must be either 'constant' or 'days'")

    if method == "days":
        if source_days is None or len(source_days) == 0:
            raise ValueError("source_days must contain at least one date when method='days'")
        number_of_days = len(source_days)
        if offsets is None:
            day_offsets = [0.0] * number_of_days
        elif np.isscalar(offsets):
            day_offsets = [offsets] * number_of_days
        else:
            day_offsets = list(offsets)
            if len(day_offsets) != number_of_days:
                raise ValueError("offsets must match the number of source_days when method='days'")
        try:
            expand_data = [
                corrected_tmax.sel(time=day, drop=True) + offset
                for day, offset in zip(source_days, day_offsets)
            ]
        except KeyError as error:
            raise ValueError(f"Source day does not exist in corrected_tmax: {error}") from error
    else:
        if offsets is None:
            offsets = (-5.0, -5.0)
        elif np.isscalar(offsets):
            offsets = (offsets, offsets)
        else:
            offsets = list(offsets)
        if len(offsets) == 0:
            raise ValueError("offsets must contain at least one value when method='constant'")
        number_of_days = len(offsets)
        last_day = corrected_tmax.time.max()
        last_day_index = corrected_tmax.time.get_index("time").get_loc(last_day.values)
        expand_data = [corrected_tmax.isel(time=last_day_index) + offset for offset in offsets]

    last_day = corrected_tmax.time.max()
    expanded_days = []
    for index, day_data in enumerate(expand_data, start=1):
        next_day = last_day + np.timedelta64(index, "D")
        expanded_days.append(day_data.expand_dims(time=[next_day.values]))

    return xr.concat([corrected_tmax, *expanded_days], dim="time")


def first_day_tmax(ds, temp_var="t2m"):
    """
    Extract Tmax for the first forecast day of an ICON run.

    Parameters
    ----------
    ds : xr.Dataset
        ICON dataset with valid_time coordinate.

    temp_var : str
        Temperature variable name.

    Returns
    -------
    xr.DataArray
        Tmax field for forecast day 1.
    """

    first_time = ds.time.min().values
    
    day_start = first_time.astype("datetime64[D]")
    day_end = day_start + np.timedelta64(1, "D")

    return (
        ds[temp_var]
        .sel(time=slice(day_start, day_end))
        .max("time")
    )


def first_day_tmax_stack(runs, temp_var="t2m"):
    """
    Extract first-day Tmax from a list of ICON runs.

    Returns
    -------
    xr.DataArray

    Dimensions:
        run
        latitude
        longitude
    """

    tmax_list = []
    
    for run_file in runs:
        ds = xr.open_dataset(run_file)

        tmax = first_day_tmax(ds, temp_var)

        run_time = ds.time.values[0]

        tmax = tmax.expand_dims(
            time=[run_time]
        )

        tmax_list.append(tmax)

    return xr.concat(tmax_list, dim="time")


def run_main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    spartacus_data = SPARTACUS_PATH + "SPARTACUS2-DAILY_TX_2026.nc"
    logger.info("Loading SPARTACUS Tmax data from %s", spartacus_data)
    spartacus_tmax = xr.open_dataarray(spartacus_data)
    logger.info("Loaded SPARTACUS data covering %s to %s", spartacus_tmax.time.min().values,
                spartacus_tmax.time.max().values)
    
    for icon_dir in [ICON_PATH + "icon_eu_t2m_regridded"]:
        logger.info("Starting ICON bias correction for %s", icon_dir)
        if not Path(icon_dir).exists():
            logger.warning("Directory %s does not exist. Please run regrid_icon_to_spcs.py first.", icon_dir)
            continue
        icon_runs = sorted(Path(icon_dir).glob("*.nc"))
        logger.info("Extracting first-day Tmax from %d ICON runs", len(icon_runs))
        icon_hist_tmax = first_day_tmax_stack(
            icon_runs
        )
        logger.info("Computing rolling seven-day ICON-SPARTACUS bias field")
        bias = compute_bias_field_rolling(spartacus_tmax,
            icon_hist_tmax,
            window=7,
        )
        logger.info("Rolling bias field computed")
        
        todays_run = icon_runs[-1]
        logger.info("Loading latest ICON run from %s", todays_run)
        icon_today = xr.open_dataset(todays_run)
        logger.info("Converting ICON temperatures from Kelvin to Celsius")
        icon_today = icon_today - 273.15  # convert K to °C

        logger.info("Applying bias correction to the latest ICON forecast")
        corrected_hourly = bias_correct_forecast(
            icon_today["t2m"],
            bias,
        )

        logger.info("Calculating daily Tmax from the corrected hourly forecast")
        corrected_tmax = forecast_daily_tmax(
            corrected_hourly[:-1],
            spartacus_tmax.name
        )
        logger.info("Expanding corrected Tmax forecast with configured source days and offsets")
        
        corrected_tmax = expand_forecast(
            corrected_tmax, method="days",
            source_days=['2026-08-01', '2026-08-03', '2026-08-03', '2026-08-01'],
            offsets=[0, -2, 0, 0],
        )
        logger.info("Corrected Tmax forecast now covers %s to %s", corrected_tmax.time.min().values,
                    corrected_tmax.time.max().values)
        
        filename = f"{icon_dir}_bias_corr/bias_corrected_tmax_{todays_run.stem}.nc"
        if not Path(f"{icon_dir}_bias_corr").exists():
            Path(f"{icon_dir}_bias_corr").mkdir(parents=True, exist_ok=True)
        logger.info("Saving bias-corrected Tmax to %s", filename)
        corrected_tmax.to_netcdf(filename)
        logger.info("Bias-corrected Tmax saved")
        
        # expand spartacus_tmax to include the new forecast days
        logger.info("Appending corrected forecast days to SPARTACUS data")
        spartacus_tmax = xr.concat([spartacus_tmax, corrected_tmax], dim="time")
        spartacus_basename = Path(spartacus_data).stem
        outfile = SPARTACUS_PATH + f"forecast/{spartacus_basename}_extended.nc"
        logger.info("Saving extended SPARTACUS forecast to %s", outfile)
        spartacus_tmax.to_netcdf(outfile)
        logger.info("Extended SPARTACUS forecast saved")


if __name__ == "__main__":
    run_main()
