import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import box

from teametrics.utils.downscale_population import (
    _load_destination_grid, _reproject_dataset, allocation_factors,
    create_joint_series, downscale_year,
)


def test_downscale_conserves_each_regional_total():
    population = xr.DataArray(
        [[1.0, 3.0], [2.0, 4.0]],
        dims=("northing", "easting"),
        coords={"northing": [1500, 500], "easting": [500, 1500]},
    )
    cells = gpd.GeoDataFrame(
        {"cell": [0, 1, 2, 3], "weight": [1.0, 3.0, 2.0, 4.0]},
        geometry=[box(0, 1000, 1000, 2000), box(1000, 1000, 2000, 2000),
                  box(0, 0, 1000, 1000), box(1000, 0, 2000, 1000)],
        crs="EPSG:3035",
    )
    regions = gpd.GeoDataFrame(
        {"region": ["west", "east"]},
        geometry=[box(0, 0, 1000, 2000), box(1000, 0, 2000, 2000)],
        crs="EPSG:3035",
    )

    factors = allocation_factors(cells, 1_000_000, regions, "region")
    result = downscale_year(
        factors, pd.Series({"west": 30, "east": 70}), "region", population, 2019
    )

    np.testing.assert_allclose(result.values, [[10, 30], [20, 40]])
    assert result.sum().item() == 100


def test_allocation_factors_split_boundary_cells_by_area():
    cells = gpd.GeoDataFrame(
        {"cell": [0], "weight": [10.0]}, geometry=[box(0, 0, 1000, 1000)], crs="EPSG:3035"
    )
    regions = gpd.GeoDataFrame(
        {"region": ["left", "right"]},
        geometry=[box(0, 0, 500, 1000), box(500, 0, 1000, 1000)], crs="EPSG:3035",
    )

    factors = allocation_factors(cells, 1_000_000, regions, "region")

    np.testing.assert_allclose(factors.fraction.to_numpy(), [1.0, 1.0])


def test_joint_series_records_the_source_transition():
    population = xr.DataArray([[1.0]], dims=("northing", "easting"))
    factors = pd.DataFrame({"region": ["AT1"], "cell": [0], "fraction": [1.0]})
    municipality_factors = {2002: factors.rename(columns={"region": "g_id"})}
    nuts_factors = factors.rename(columns={"region": "NUTS_ID"})

    result = create_joint_series(
        nuts_factors,
        {1960: pd.Series({"AT1": 10}), 2002: pd.Series({"AT1": 20})},
        municipality_factors,
        {2002: pd.Series({"AT1": 30})},
        population,
    )

    assert result.population.sel(year=1960).item() == 10
    assert result.population.sel(year=2002).item() == 30
    assert result.source_region_level.values.tolist() == ["NUTS3", "municipality"]
    assert "Do not use" in result.attrs["trend_use_warning"]


@pytest.mark.filterwarnings(
    "ignore:Use .* matmul instead of .* mul operator for matrix multiplication:PendingDeprecationWarning"
)
def test_destination_grid_defines_exact_output_grid_and_conserves_counts(tmp_path):
    population = xr.Dataset(
        {
            "population": (
                ("year", "northing", "easting"),
                [[[1.0, 2.0], [3.0, 4.0]]],
            )
        },
        coords={
            "year": [2020],
            "northing": [2_500_500.0, 2_499_500.0],
            "easting": [4_500_500.0, 4_499_500.0],
        },
    )

    destination_path = tmp_path / "destination.nc"
    xr.Dataset(
        {"grid": (("y", "x"), np.zeros((3, 3)))},
        coords={
            "x": [288117.02154377, 289117.02154377, 290117.02154377],
            "y": [5052377.66497606, 5051377.66497606, 5050377.66497606],
        },
        attrs={"coordinate_sys": "EPSG:32633"},
    ).to_netcdf(destination_path)
    destination_grid = _load_destination_grid(destination_path)

    result = _reproject_dataset(population, destination_grid, destination_path)

    assert result.attrs["grid_crs"] == "EPSG:32633"
    assert result.attrs["destination_grid"] == str(destination_path)
    np.testing.assert_array_equal(result.x, destination_grid.x)
    np.testing.assert_array_equal(result.y, destination_grid.y)
    assert np.nansum(result.population.values) == 10.0


def test_destination_grid_prefers_mask_variable_with_auxiliary_variables(tmp_path):
    destination_path = tmp_path / "mask.nc"
    xr.Dataset(
        {
            "mask": (("y", "x"), np.ones((2, 2))),
            "area_grid": (("y", "x"), np.full((2, 2), 100.0)),
            "area": ((), 400.0),
        },
        coords={"x": [4_500_500.0, 4_501_500.0], "y": [2_500_500.0, 2_499_500.0]},
        attrs={"coordinate_sys": "EPSG:3035"},
    ).to_netcdf(destination_path)

    destination_grid = _load_destination_grid(destination_path)

    assert destination_grid.name == "mask"
    assert destination_grid.dims == ("y", "x")


def test_destination_grid_rejects_geographic_crs(tmp_path):
    destination_path = tmp_path / "geographic.nc"
    xr.Dataset(
        {"grid": (("lat", "lon"), np.zeros((2, 2)))},
        coords={"lon": [15.0, 15.01], "lat": [47.0, 47.01]},
        attrs={"coordinate_sys": "EPSG:4326"},
    ).to_netcdf(destination_path)

    with pytest.raises(ValueError, match="projected CRS"):
        _load_destination_grid(destination_path)
