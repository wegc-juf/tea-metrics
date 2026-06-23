import xarray as xr


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
        spartacus_tmax.time.values,
        icon_hist_tmax.time.values,
    )

    obs = spartacus_tmax.sel(time=common_days)
    fc = icon_hist_tmax.sel(time=common_days)

    daily_error = obs - fc
    
    weights = xr.DataArray(
        [1, 2, 3, 4, 5, 6, 7],
        dims=["time"]
    )
    
    bias = (
        daily_error
        .isel(time=slice(-7, None))
        .weighted(weights)
        .mean("time")
    )

    return bias.isel(time=-1)


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
):
    return (
        corrected_hourly
        .groupby("valid_time.date")
        .max()
        .rename("tmax")
    )


def run_main():
    bias = compute_bias_field_rolling( spartacus_tmax,
        icon_hist_tmax,
        window=7,
    )

    corrected_hourly = bias_correct_forecast(
        icon_today["t2m"],
        bias,
    )

    corrected_tmax = forecast_daily_tmax(
        corrected_hourly
    )

if __name__ == "__main__":
    run_main()