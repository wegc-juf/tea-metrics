import numpy as np
import pandas as pd
import xarray as xr

from teametrics.TEA import TEAIndicators


EXPECTED_CRS = "EPSG:4326"


def _make_input_grid():
    y = np.array([47.0, 48.0])
    x = np.array([15.0, 16.0])
    values = np.ones((1, len(y), len(x)))
    data = xr.DataArray(
        values,
        coords={"time": [pd.Timestamp("1980-01-01")], "lat": y, "lon": x},
        dims=("time", "lat", "lon"),
        name="input",
    )
    data.attrs["coordinate_sys"] = EXPECTED_CRS

    threshold = xr.DataArray(
        np.full((len(y), len(x)), 0.5),
        coords={"lat": y, "lon": x},
        dims=("lat", "lon"),
        name="threshold",
    )
    threshold.attrs["coordinate_sys"] = EXPECTED_CRS
    return data, threshold


def _make_daily_input():
    times = pd.date_range("1980-01-01", "1994-12-31", freq="D")
    y = np.array([47.0, 48.0])
    x = np.array([15.0, 16.0])

    data = xr.DataArray(
        np.full((len(times), len(y), len(x)), 10.0),
        coords={"time": times, "lat": y, "lon": x},
        dims=("time", "lat", "lon"),
        name="input",
    )
    data.attrs["coordinate_sys"] = EXPECTED_CRS
    return data


def test_crs_propagates_through_ctp_decadal_and_amplification(tmp_path):
    input_data = _make_daily_input()
    _, threshold = _make_input_grid()
    tea = TEAIndicators(input_data=input_data, threshold=threshold, unit="K")
    tea.ref_period = (1980, 1989)

    tea.calc_daily_basis_vars(grid=True, gr=True)
    tea.calc_annual_ctp_indicators(ctp="annual", drop_daily_results=False)
    tea.calc_decadal_indicators(decadal_window=(10, 5, 4), min_duration=0, drop_annual_results=False)
    tea.calc_amplification_factors(
        ref_period=(1980, 1989),
        cc_period=(1985, 1994),
        min_duration=0,
    )

    assert tea._read_crs(tea.ctp_results) == EXPECTED_CRS
    assert tea.ctp_results.EF_GR.attrs["coordinate_sys"] == EXPECTED_CRS
    assert tea._read_crs(tea.decadal_results) == EXPECTED_CRS
    assert tea.decadal_results.EF_GR.attrs["coordinate_sys"] == EXPECTED_CRS
    assert tea._read_crs(tea.amplification_factors) == EXPECTED_CRS
    assert tea.amplification_factors.EF_GR_AF.attrs["coordinate_sys"] == EXPECTED_CRS

    ctp_path = tmp_path / "ctp.nc"
    decadal_path = tmp_path / "decadal.nc"
    amplification_path = tmp_path / "amplification.nc"

    tea.save_ctp_results(ctp_path)
    tea.save_decadal_results(decadal_path)
    tea.save_amplification_factors(amplification_path)

    interval_vars = {
        "TEX_max_interval_start_GR",
        "TEX_max_interval_end_GR",
        "TEX_HW_max_interval_start_GR",
        "TEX_HW_max_interval_end_GR",
    }
    with xr.open_dataset(ctp_path) as saved_ctp:
        assert interval_vars <= set(saved_ctp.data_vars)
        for variable in interval_vars:
            assert np.issubdtype(saved_ctp[variable].dtype, np.datetime64)
    assert not ctp_path.with_suffix(".intervals.csv").exists()

    loaded = TEAIndicators()
    loaded.load_ctp_results(ctp_path, use_dask=False)
    assert interval_vars <= set(loaded.ctp_results.data_vars)
    assert loaded._crs == EXPECTED_CRS
    assert loaded._read_crs(loaded.ctp_results) == EXPECTED_CRS

    loaded.load_decadal_results(decadal_path)
    assert loaded._crs == EXPECTED_CRS
    assert loaded._read_crs(loaded.decadal_results) == EXPECTED_CRS

    loaded.load_amplification_factors(amplification_path)
    assert loaded._crs == EXPECTED_CRS
    assert loaded._read_crs(loaded.amplification_factors) == EXPECTED_CRS
