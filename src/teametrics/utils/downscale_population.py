#!/usr/bin/env python

"""Downscale Austrian population stocks with a 1 km allocation-weight grid."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray  # noqa: F401 - imported to enable the rio accessor
import xarray as xr
from pyproj import CRS
from rasterio.enums import Resampling
from shapely import box as shapely_box


GRID_CRS = "EPSG:3035"


def _cell_size(coordinates):
    differences = np.abs(np.diff(np.asarray(coordinates, dtype=float)))
    if len(differences) == 0 or not np.allclose(differences, differences[0]):
        raise ValueError("The population allocation grid must have regular coordinates.")
    return differences[0]


def _validate_projected_crs(destination_crs):
    """Validate and normalize a projected destination CRS."""
    try:
        crs = CRS.from_user_input(destination_crs)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid destination CRS: {destination_crs}") from error
    if not crs.is_projected:
        raise ValueError("destination-crs must be a projected CRS with linear units.")
    if len(crs.axis_info) < 2 or any(axis.unit_conversion_factor is None for axis in crs.axis_info[:2]):
        raise ValueError("destination-crs must define linear units for both spatial axes.")
    return crs


def _load_destination_grid(path):
    """Load and validate a destination grid template."""
    dataset = xr.open_dataset(path)
    if "population" in dataset:
        grid = dataset.population
    elif len(dataset.data_vars) == 1:
        grid = dataset[next(iter(dataset.data_vars))]
    else:
        raise ValueError("Destination grid must contain 'population' or exactly one data variable.")
    if grid.ndim != 2:
        raise ValueError("The destination grid must have exactly two dimensions.")

    ydim, xdim = grid.dims
    grid = grid.rio.set_spatial_dims(x_dim=xdim, y_dim=ydim, inplace=False)
    destination_crs = grid.rio.crs
    if destination_crs is None:
        for attrs in (grid.attrs, dataset.attrs):
            for name in ("coordinate_sys", "grid_crs", "crs"):
                if attrs.get(name):
                    destination_crs = attrs[name]
                    break
            if destination_crs is not None:
                break
    if destination_crs is None:
        raise ValueError("Destination grid must define a CRS.")
    destination_crs = _validate_projected_crs(destination_crs)
    return grid.rio.write_crs(destination_crs, inplace=False)


def _reproject_dataset(dataset, destination_grid, destination_grid_path):
    """Reproject population counts onto an exact destination grid template."""
    destination_crs = _validate_projected_crs(destination_grid.rio.crs)
    population = dataset.population
    ydim, xdim = population.dims[-2:]
    target = xr.zeros_like(destination_grid, dtype=float)
    target = target.rio.set_spatial_dims(
        x_dim=destination_grid.rio.x_dim,
        y_dim=destination_grid.rio.y_dim,
        inplace=False,
    )
    target = target.rio.write_crs(destination_crs, inplace=False)
    target = target.rio.write_nodata(np.nan, inplace=False)

    arrays = []
    for year in population.year.values:
        array = population.sel(year=year, drop=True)
        array = array.rio.set_spatial_dims(x_dim=xdim, y_dim=ydim, inplace=False)
        array = array.rio.write_crs(GRID_CRS, inplace=False)
        array = array.rio.write_nodata(np.nan, inplace=False)
        array = array.rio.reproject_match(target, resampling=Resampling.sum)
        arrays.append(array.expand_dims(year=[year]))

    result = xr.concat(arrays, dim="year").to_dataset(name="population")
    result.attrs = dict(dataset.attrs)
    result.attrs.update({
        "source_grid_crs": GRID_CRS,
        "grid_crs": destination_crs.to_string(),
        "destination_grid": str(destination_grid_path),
        "reprojection_resampling": "sum",
    })
    result.population.attrs.update({
        "coordinate_sys": destination_crs.to_string(),
    })
    return result


def load_allocation_grid(path):
    """Load a count grid and convert its valid cells to allocation weights."""
    dataset = xr.open_dataset(path)
    if "population" in dataset:
        population = dataset.population
    elif len(dataset.data_vars) == 1:
        population = dataset[next(iter(dataset.data_vars))]
    else:
        raise ValueError("Population grid must contain 'population' or exactly one data variable.")
    if population.ndim != 2:
        raise ValueError("The population allocation grid must have exactly two dimensions.")

    ydim, xdim = population.dims
    values = np.asarray(population.values, dtype=float)
    valid = np.isfinite(values) & (values >= 0)
    if not valid.any():
        raise ValueError("Population allocation grid has no non-negative valid cells.")
    xvals = np.asarray(population[xdim].values)
    yvals = np.asarray(population[ydim].values)
    xsize, ysize = _cell_size(xvals), _cell_size(yvals)
    xgrid, ygrid = np.meshgrid(xvals, yvals)
    row, col = np.indices(values.shape)
    cells = gpd.GeoDataFrame(
        {"cell": (row * values.shape[1] + col)[valid], "weight": values[valid]},
        geometry=shapely_box(
            (xgrid - xsize / 2)[valid], (ygrid - ysize / 2)[valid],
            (xgrid + xsize / 2)[valid], (ygrid + ysize / 2)[valid],
        ),
        crs=GRID_CRS,
    )
    return population, cells, xsize * ysize


def allocation_factors(cells, cell_area, regions, region_id):
    """Return population-weighted allocation fractions for each region and cell."""
    if region_id not in regions:
        raise ValueError(f"Boundary data has no '{region_id}' column.")
    regions = regions[[region_id, "geometry"]].dropna(subset=[region_id, "geometry"]).copy()
    regions[region_id] = regions[region_id].astype(str)
    if regions.crs is None:
        raise ValueError("Boundary data must define a CRS.")
    regions = regions.to_crs(GRID_CRS)
    if regions[region_id].duplicated().any():
        regions = regions.dissolve(by=region_id, as_index=False)

    overlap = gpd.overlay(cells, regions, how="intersection", keep_geom_type=False)
    overlap = overlap[overlap.geometry.area > 0]
    if overlap.empty:
        raise ValueError("No allocation-grid cells overlap the supplied boundaries.")
    overlap["weight"] *= overlap.geometry.area / cell_area
    factors = overlap.groupby([region_id, "cell"], as_index=False)["weight"].sum()
    normalizer = factors.groupby(region_id)["weight"].transform("sum")
    zero_weight = factors.loc[normalizer <= 0, region_id].unique()
    if len(zero_weight):
        raise ValueError(f"Regions have no positive allocation weight: {', '.join(zero_weight)}")
    factors["fraction"] = factors["weight"] / normalizer
    return factors[[region_id, "cell", "fraction"]]


def downscale_year(factors, totals, region_id, population, year):
    """Allocate one year's regional totals to grid cells while conserving every total."""
    totals = totals.copy()
    totals.index = totals.index.astype(str)
    expected = pd.Index(factors[region_id].unique())
    missing = expected.difference(totals.index)
    if len(missing):
        raise ValueError(f"Missing {year} totals for regions: {', '.join(missing)}")
    totals = totals.reindex(expected)
    if totals.isna().any() or (totals < 0).any():
        raise ValueError(f"Invalid {year} regional population totals.")
    allocated = factors.copy()
    allocated["population"] = allocated["fraction"] * allocated[region_id].map(totals)
    flat = np.full(population.size, np.nan, dtype=float)
    by_cell = allocated.groupby("cell")["population"].sum()
    flat[by_cell.index.to_numpy(dtype=int)] = by_cell.to_numpy()
    result = xr.DataArray(
        flat.reshape(population.shape), coords=population.coords, dims=population.dims, name="population"
    )
    result.attrs = {
        "long_name": "Downscaled population count per 1 km grid cell",
        "units": "1",
        "population_representation": "count_per_grid_cell",
        "allocation_method": "regional total multiplied by normalized reference-grid weight",
    }
    return result


def downscale_series(factors_by_year, totals_by_year, region_id, population, source_name):
    """Create a year-indexed downscaled population dataset."""
    arrays = []
    for year in sorted(totals_by_year):
        array = downscale_year(factors_by_year[year], totals_by_year[year], region_id, population, year)
        arrays.append(array.expand_dims(year=[year]))
    result = xr.concat(arrays, dim="year").to_dataset()
    result.attrs = {
        "title": f"Austrian population downscaled from {source_name}",
        "allocation_grid_role": "Normalized within-region allocation weights only",
        "grid_crs": GRID_CRS,
        "conservation": "Regional source totals are conserved except for floating-point precision",
        "source_population_product": source_name,
    }
    return result


def create_joint_series(nuts_factors, ardeco_totals, municipality_factors, statat_totals, population):
    """Use ARDECO before municipal stocks become available, then StatAT stocks."""
    arrays = []
    sources = []
    levels = []
    for year in sorted(ardeco_totals):
        if year in statat_totals:
            array = downscale_year(
                municipality_factors[year], statat_totals[year], "g_id", population, year
            )
            sources.append("Statistics Austria municipality population stock")
            levels.append("municipality")
        else:
            array = downscale_year(nuts_factors, ardeco_totals[year], "NUTS_ID", population, year)
            sources.append("ARDECO NUTS-3 population stock")
            levels.append("NUTS3")
        arrays.append(array.expand_dims(year=[year]))
    result = xr.concat(arrays, dim="year").to_dataset()
    result["source_population_product"] = xr.DataArray(
        sources, dims="year", coords={"year": result.year}
    )
    result["source_region_level"] = xr.DataArray(
        levels, dims="year", coords={"year": result.year}
    )
    result["source_population_product"].attrs["long_name"] = "Population source used for each year"
    result["source_region_level"].attrs["long_name"] = "Administrative level constraining each year"
    result.attrs = {
        "title": "Austrian joint best-guess downscaled population",
        "allocation_grid_role": "Normalized within-region allocation weights only",
        "grid_crs": GRID_CRS,
        "conservation": "Regional source totals are conserved except for floating-point precision",
        "source_hierarchy": "ARDECO NUTS-3 before municipal data; Statistics Austria municipality stocks thereafter",
        "trend_use_warning": "Do not use this mixed-resolution product for grid-cell or small-region trend studies "
                             "across 2001-2002. The allocation regime changes from NUTS-3 to municipality in 2002.",
    }
    return result


def add_provenance(dataset, allocation_grid, population_source, boundaries):
    """Attach enough provenance to reproduce a generated population product."""
    dataset.attrs.update({
        "date_created": datetime.now(timezone.utc).isoformat(),
        "allocation_grid": str(allocation_grid),
        "population_source": str(population_source),
        "regional_boundaries": str(boundaries),
    })
    return dataset


def load_ardeco_totals(path, start_year, end_year):
    """Load Austrian NUTS-3 total population stocks from the ARDECO CSV."""
    data = pd.read_csv(path)
    data = data[
        (data["TERRITORY_ID"].astype(str).str.startswith("AT"))
        & (data["YEAR"].between(start_year, end_year))
        & (data["SEX"] == "TOTAL") & (data["AGE"] == "TOTAL") & (data["UNIT"] == "NR")
    ].copy()
    if data.empty:
        raise ValueError("No ARDECO Austrian NUTS-3 totals matched the requested years.")
    data["VALUE"] = pd.to_numeric(data["VALUE"], errors="raise")
    if data.duplicated(["YEAR", "TERRITORY_ID"]).any():
        raise ValueError("ARDECO input has duplicate total-population records.")
    return {int(year): group.set_index("TERRITORY_ID")["VALUE"] for year, group in data.groupby("YEAR")}


def load_statat_totals(directory, start_year, end_year):
    """Load annual Statistics Austria municipality population stocks."""
    totals = {}
    for year in range(start_year, end_year + 1):
        path = Path(directory) / f"OGD_bevstandjbab2002_BevStand_{year}.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        data = pd.read_csv(path, sep=";", usecols=["C-GRGEMAKT-0", "F-ISIS-1"])
        codes = data["C-GRGEMAKT-0"].astype(str).str.replace("GRGEMAKT-", "", regex=False)
        totals[year] = data.assign(code=codes).groupby("code")["F-ISIS-1"].sum()
    return totals


def _write(dataset, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(path)


def _read_boundaries(path):
    """Read supplied shapefiles whose DBF attributes commonly use Latin-1."""
    return gpd.read_file(path, encoding="latin1")


def run(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", required=True, help="2019 1 km population allocation grid NetCDF")
    parser.add_argument("--ardeco", required=True, help="ARDECO NUTS-3 population CSV")
    parser.add_argument("--statat-directory", required=True, help="Directory containing annual StatAT CSV files")
    parser.add_argument("--nuts3-boundaries", required=True, help="NUTS-3 boundary file")
    parser.add_argument("--municipality-boundary-template", required=True,
                        help="Boundary path or URL; use {year} only with same-vintage totals")
    parser.add_argument("--output-directory", required=True)
    parser.add_argument(
        "--destination-grid",
        help="Optional NetCDF grid defining the exact projected output grid.",
    )
    parser.add_argument("--start-year", type=int, default=2002,
                        help="First Statistics Austria municipality and joint-product year")
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--nuts3-start-year", type=int, default=1960,
                        help="First ARDECO NUTS-3 and joint-product year")
    parsed = parser.parse_args(args)
    if parsed.end_year < parsed.start_year:
        parser.error("end-year must not precede start-year")

    if parsed.nuts3_start_year > parsed.start_year:
        parser.error("nuts3-start-year must not follow start-year")

    destination_grid = None
    if parsed.destination_grid is not None:
        destination_grid = _load_destination_grid(parsed.destination_grid)

    population, cells, cell_area = load_allocation_grid(parsed.grid)
    municipality_years = range(parsed.start_year, parsed.end_year + 1)
    ardeco = load_ardeco_totals(parsed.ardeco, parsed.nuts3_start_year, parsed.end_year)
    statat = load_statat_totals(parsed.statat_directory, parsed.start_year, parsed.end_year)
    missing_years = set(range(parsed.nuts3_start_year, parsed.end_year + 1)).difference(ardeco)
    if missing_years:
        raise ValueError(f"ARDECO is missing years: {sorted(missing_years)}")

    nuts = _read_boundaries(parsed.nuts3_boundaries)
    nuts = nuts[nuts["NUTS_ID"].astype(str).str.startswith("AT")]
    nuts_factors = allocation_factors(cells, cell_area, nuts, "NUTS_ID")
    nuts_series = downscale_series(
        {year: nuts_factors for year in ardeco}, ardeco, "NUTS_ID", population, "ARDECO NUTS-3 stocks"
    )

    municipality_factors = {}
    boundary_template = parsed.municipality_boundary_template
    if "{year}" in boundary_template:
        for year in municipality_years:
            boundaries = _read_boundaries(boundary_template.format(year=year))
            municipality_factors[year] = allocation_factors(cells, cell_area, boundaries, "g_id")
    else:
        boundaries = _read_boundaries(boundary_template)
        factors = allocation_factors(cells, cell_area, boundaries, "g_id")
        municipality_factors = {year: factors for year in municipality_years}
    municipality_series = downscale_series(
        municipality_factors, statat, "g_id", population, "Statistics Austria municipality stocks"
    )
    joint_series = create_joint_series(
        nuts_factors, ardeco, municipality_factors, statat, population
    )

    if destination_grid is not None:
        nuts_series = _reproject_dataset(nuts_series, destination_grid, parsed.destination_grid)
        municipality_series = _reproject_dataset(
            municipality_series, destination_grid, parsed.destination_grid
        )
        joint_series = _reproject_dataset(joint_series, destination_grid, parsed.destination_grid)

    add_provenance(nuts_series, parsed.grid, parsed.ardeco, parsed.nuts3_boundaries)
    add_provenance(municipality_series, parsed.grid, parsed.statat_directory, boundary_template)
    add_provenance(joint_series, parsed.grid, f"{parsed.ardeco}; {parsed.statat_directory}",
                   f"{parsed.nuts3_boundaries}; {boundary_template}")

    output = Path(parsed.output_directory)
    _write(nuts_series, output / "nuts3_ardeco" /
           f"population_1km_{min(ardeco)}-{max(ardeco)}_ardeco_nuts3.nc")
    _write(municipality_series, output / "municipality_statat" /
           f"population_1km_{min(statat)}-{max(statat)}_statat_municipality.nc")
    _write(joint_series, output / "joint_best_guess" /
           f"population_1km_{min(ardeco)}-{max(ardeco)}_joint_best_guess.nc")


if __name__ == "__main__":
    run()
