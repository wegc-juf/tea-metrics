import warnings

import numpy as np
import pandas as pd
import xarray as xr

from teametrics.utils.SPARTACUS.aggregate_monthly_RR import aggregate_monthly


def test_aggregate_monthly_uses_explicit_join_without_future_warning(tmp_path):
    """Two daily files combine and aggregate without xarray's join warning."""
    for year in (2025, 2026):
        times = pd.date_range(f"{year}-01-01", periods=31, freq="D")
        daily = xr.Dataset(
            {"RR": (("time", "y", "x"), np.ones((31, 1, 1)))},
            coords={"time": times, "x": [0.0], "y": [0.0]},
        )
        daily.to_netcdf(tmp_path / f"SPARTACUS2-DAILY_RR_{year}.nc")
    daily.to_netcdf(tmp_path / "SPARTACUS2-DAILY_RR_2025.bak.nc")
    output = tmp_path / "monthly.nc"

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        aggregate_monthly(tmp_path, output)

    with xr.open_dataset(output) as result:
        assert result.sizes["time"] == 13
        valid = result["RR"].values[~np.isnan(result["RR"].values)]
        np.testing.assert_allclose(valid, 31.0)
