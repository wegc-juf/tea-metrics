import xarray as xr
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


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
        Contains yesterday.

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
        spartacus_tmax.time.values + hack_time_shift,
        icon_hist_tmax.time.values,
    )

    obs = spartacus_tmax.sel(time=common_days - hack_time_shift)
    # shift time of observations to match ICON forecast time (valid_time)
    if hack_time_shift != 0:
        obs = obs.assign_coords(time=common_days)
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
    spartacus_data = "/data/reloclim/backup/ZAMG_SPARTACUS/data/current/SPARTACUS2-DAILY_TX_2026.nc"
    spartacus_tmax = xr.open_dataarray(spartacus_data)
    
    for icon_dir in ["icon_eu_t2m_regridded", "icon_d2_t2m_regridded", "icon_t2m_regridded"]:
        if not Path(icon_dir).exists():
            print(f"Directory {icon_dir} does not exist. Please run regrid_icon_to_spcs.py first.")
            continue
        icon_hist_tmax = first_day_tmax_stack(
            sorted(Path(icon_dir).glob("*.nc"))
        )
        bias = compute_bias_field_rolling(spartacus_tmax,
            icon_hist_tmax,
            window=7,
        )
        
        todays_run = sorted(Path(icon_dir).glob("*.nc"))[-1]
        icon_today = xr.open_dataset(todays_run)
        icon_today = icon_today - 273.15  # convert K to °C

        corrected_hourly = bias_correct_forecast(
            icon_today["t2m"],
            bias,
        )

        corrected_tmax = forecast_daily_tmax(
            corrected_hourly[:-1],
            spartacus_tmax.name
        )
        print(corrected_tmax)
        filename = f"{icon_dir}_bias_corr/bias_corrected_tmax_{todays_run.stem}.nc"
        if not Path(f"{icon_dir}_bias_corr").exists():
            Path(f"{icon_dir}_bias_corr").mkdir(parents=True, exist_ok=True)
        corrected_tmax.to_netcdf(filename)

if __name__ == "__main__":
    run_main()