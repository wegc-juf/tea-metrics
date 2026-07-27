import numpy as np
import pandas as pd
import xarray as xr
import pytest

from teametrics.TEA import TEAIndicators


EXPECTED_CRS = "EPSG:4326"


@pytest.fixture
def lat():
    return np.array([47.0, 48.0])


@pytest.fixture
def lon():
    return np.array([15.0, 16.0])


@pytest.fixture
def daily_times():
    return pd.date_range("1980-01-01", "1994-12-31", freq="D")


@pytest.fixture
def daily_data(daily_times, lat, lon):
    doy = np.array(daily_times.dayofyear)
    seasonal = 22 + 14 * np.cos(2 * np.pi * (doy - 200) / 365.25)
    spatial = np.array([[1.0, 3.0], [-2.0, 0.0]])
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 2, size=(len(daily_times), 2, 2))
    field = seasonal[:, np.newaxis, np.newaxis] + spatial[np.newaxis, :, :] + noise
    data = xr.DataArray(
        field,
        coords={"time": daily_times, "lat": lat, "lon": lon},
        dims=("time", "lat", "lon"),
        name="input",
    )
    data.attrs["coordinate_sys"] = EXPECTED_CRS
    return data


@pytest.fixture
def threshold(lat, lon):
    t = xr.DataArray(
        np.full((2, 2), 30.0),
        coords={"lat": lat, "lon": lon},
        dims=("lat", "lon"),
        name="threshold",
    )
    t.attrs["coordinate_sys"] = EXPECTED_CRS
    return t


@pytest.fixture
def population_grid(lat, lon):
    p = xr.DataArray(
        np.array([[50.0, 120.0], [5.0, 15.0]]),
        coords={"lat": lat, "lon": lon},
        dims=("lat", "lon"),
        name="population",
    )
    p.attrs["coordinate_sys"] = EXPECTED_CRS
    return p


@pytest.fixture
def constant_data(daily_times, lat, lon):
    data = xr.DataArray(
        np.full((len(daily_times), 2, 2), 10.0),
        coords={"time": daily_times, "lat": lat, "lon": lon},
        dims=("time", "lat", "lon"),
        name="input",
    )
    data.attrs["coordinate_sys"] = EXPECTED_CRS
    return data


@pytest.fixture
def low_threshold(lat, lon):
    t = xr.DataArray(
        np.full((2, 2), 0.5),
        coords={"lat": lat, "lon": lon},
        dims=("lat", "lon"),
        name="threshold",
    )
    t.attrs["coordinate_sys"] = EXPECTED_CRS
    return t


@pytest.fixture
def tea(daily_data, threshold, population_grid):
    return TEAIndicators(
        input_data=daily_data,
        threshold=threshold,
        population_grid=population_grid,
        unit="K",
        ref_period=(1980, 1989),
    )


@pytest.fixture
def tea_no_pop(daily_data, threshold):
    return TEAIndicators(
        input_data=daily_data,
        threshold=threshold,
        unit="K",
        ref_period=(1980, 1989),
    )


@pytest.fixture
def tea_constant(constant_data, low_threshold, population_grid):
    return TEAIndicators(
        input_data=constant_data,
        threshold=low_threshold,
        population_grid=population_grid,
        unit="K",
        ref_period=(1980, 1989),
    )


@pytest.fixture
def tea_constant_no_pop(constant_data, low_threshold):
    return TEAIndicators(
        input_data=constant_data,
        threshold=low_threshold,
        unit="K",
        ref_period=(1980, 1989),
    )
