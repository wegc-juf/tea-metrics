import warnings

import numpy as np
import pandas as pd
import xarray as xr

from teametrics.utils.SPARTACUS.aggregate_monthly_RR import aggregate_monthly


def test_aggregate_monthly_uses_explicit_join_without_future_warning(tmp_path):
    """Two daily files combine and aggregate without xarray's join warning."""
    times = pd.date_range("2026-01-01", periods=31, freq="D")
    daily = xr.Dataset(
        {"RR": (("time", "y", "x"), np.ones((31, 1, 1)))},
        coords={"time": times, "x": [0.0], "y": [0.0]},
    )
    daily.isel(time=slice(None, 16)).to_netcdf(tmp_path / "SPARTACUS2-DAILY_RR_a.nc")
    daily.isel(time=slice(16, None)).to_netcdf(tmp_path / "SPARTACUS2-DAILY_RR_b.nc")
    output = tmp_path / "monthly.nc"

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        aggregate_monthly(tmp_path, output)

    with xr.open_dataset(output) as result:
        assert result.sizes["time"] == 1
        assert result["RR"].values.item() == 31.0
