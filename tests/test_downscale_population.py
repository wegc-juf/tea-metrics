import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import box

from teametrics.utils.downscale_population import allocation_factors, downscale_year


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
