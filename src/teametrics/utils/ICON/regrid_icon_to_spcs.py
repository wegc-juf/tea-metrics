#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regrid ICON data to SPCS grid
"""

import xarray as xr
import numpy as np
from pathlib import Path
from scipy.interpolate import griddata

from get_icon_data import ICON_PATH

SPCS_GRID_FILE = "/data/reloclim/backup/ZAMG_SPARTACUS/data/v2024_v2.1/SPARTACUS2-DAILY_TX_2026.nc"


def crop_icon_to_spcs(files):
    """
    Crop ICON data to SPCS domain and regrid to SPCS grid
    Returns:

    """
    datasets = []

    for f in files:

        ds = xr.open_dataset(
            f,
            engine="cfgrib",
            backend_kwargs={"indexpath": ""}
        )

        datasets.append(ds)

    ds = xr.concat(datasets, dim="step")
    austria = ds.sel(
        latitude=slice(46.1, 49.1),
        longitude=slice(9.3, 17.4)
    )
    
    return austria


def convert_forecast_lead_time_to_datetime(ds):
    """
    Convert forecast lead time to datetime
    Args:
        ds: xarray dataset

    Returns:
        ds: xarray dataset with datetime coordinate
    """
    ds = ds.assign_coords(
        time=ds.time + ds.step
    )
    
    ds = ds.swap_dims({"step": "time"})
    return ds


def interpolate_icon_to_spcs(ds):
    """
    Interpolate ICON data to SPCS grid
    Args:
        ds: xarray dataset

    Returns:
        ds: xarray dataset with interpolated data
    """
    spcs_grid = xr.open_dataset(SPCS_GRID_FILE)
    ds_interp = ds.interp(
        latitude=spcs_grid.lat,
        longitude=spcs_grid.lon,
        method="linear"
    )
    
    return ds_interp
    
    
def regrid(files):
    ds = crop_icon_to_spcs(files)
    ds = convert_forecast_lead_time_to_datetime(ds)
    ds = interpolate_icon_to_spcs(ds)
    return ds


def icon_global_to_spartacus(
    ds_icon,
    spartacus,
    temp_var="t2m",
    lat_name="latitude",
    lon_name="longitude",
):
    """
    Regrid ICON Global icosahedral data directly onto
    the SPARTACUS grid.

    Parameters
    ----------
    ds_icon : xr.Dataset
        ICON Global dataset.

    spartacus : xr.Dataset or xr.DataArray
        Object containing SPARTACUS latitude/longitude grids.

    temp_var : str
        Temperature variable.

    Returns
    -------
    xr.Dataset
        Same spatial grid as SPARTACUS.
    """

    # ICON coordinates (radians -> degrees)
    src_lon = np.rad2deg(ds_icon.clon.values)
    src_lat = np.rad2deg(ds_icon.clat.values)

    # SPARTACUS target grid
    tgt_lat = spartacus[lat_name].values
    tgt_lon = spartacus[lon_name].values

    # dimensions of target grid
    target_shape = tgt_lat.shape

    target_points = (
        tgt_lon.ravel(),
        tgt_lat.ravel(),
    )

    fields = []

    for t in range(ds_icon.sizes["valid_time"]):

        values = ds_icon[temp_var].isel(
            valid_time=t
        ).values

        interp = griddata(
            (src_lon, src_lat),
            values,
            target_points,
            method="linear",
        )

        fields.append(
            interp.reshape(target_shape)
        )

    fields = np.stack(fields)

    return xr.Dataset(
        {
            temp_var: (
                ("valid_time",) + spartacus[lat_name].dims,
                fields,
            )
        },
        coords={
            "valid_time": ds_icon.valid_time,
            lat_name: spartacus[lat_name],
            lon_name: spartacus[lon_name],
        },
    )

def icon_eu_t2m_regrid():
    files = sorted(
        Path(ICON_PATH + "/icon_eu_t2m").glob("*.grib2")
    )
    dates = [f.stem.split("_")[-4] for f in files]
    unique_dates = sorted(set(dates))
    for date in unique_dates:
        date_files = [f for f in files if date in f.stem]
        ds = regrid(date_files)
        if not Path(ICON_PATH + "./icon_eu_t2m_regridded").exists():
            Path(ICON_PATH + "./icon_eu_t2m_regridded").mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(ICON_PATH + f"./icon_eu_t2m_regridded/icon_eu_t2m_spcs_{date}.nc")
        print(f"icon_eu_t2m files for {date} regridded to SPCS grid and saved to " + ICON_PATH +
              f"icon_eu_t2m_regridded/icon_eu_t2m_spcs_{date}.nc")


def icon_d2_t2m_regrid():
    files = sorted(
        Path(ICON_PATH + "icon_d2_t2m").glob("*.grib2")
    )
    dates = [f.stem.split("_")[-5] for f in files]
    unique_dates = sorted(set(dates))
    for date in unique_dates:
        date_files = [f for f in files if date in f.stem]
        ds = regrid(date_files)
        if not Path(ICON_PATH + "./icon_d2_t2m_regridded").exists():
            Path(ICON_PATH + "./icon_d2_t2m_regridded").mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(ICON_PATH + f"./icon_d2_t2m_regridded/icon_d2_t2m_spcs_{date}.nc")
        print(f"icon_d2_t2m files for {date} regridded to SPCS grid and saved to " +
              ICON_PATH + f"/icon_d2_t2m_regridded/icon_d2_t2m_spcs_{date}.nc")


def icon_global_t2m_regrid():
    files = sorted(
        Path(ICON_PATH + "icon_global_t2m").glob("*.grib2")
    )
    dates = [f.stem.split("_")[-5] for f in files]
    unique_dates = sorted(set(dates))
    spartacus = xr.open_dataset(SPCS_GRID_FILE)
    for date in unique_dates:
        date_files = [f for f in files if date in f.stem]
        one_file = xr.open_dataset(date_files[0], engine="cfgrib", backend_kwargs={"indexpath": ""})
        print(one_file)
        import cfgrib
        
        ds = cfgrib.open_datasets(
            date_files[0],
            backend_kwargs={"indexpath": ""}
        )
        
        for d in ds:
            print(d)
        icon_data = xr.open_mfdataset(date_files, combine="nested", concat_dim="valid_time", engine="cfgrib", backend_kwargs={"indexpath": ""})
        ds_spcs = icon_global_to_spartacus(icon_data, spartacus)
        if not Path(ICON_PATH + "./icon_t2m_regridded").exists():
            Path(ICON_PATH + "./icon_t2m_regridded").mkdir(parents=True, exist_ok=True)
        ds_spcs.to_netcdf(ICON_PATH + f"./icon_t2m_regridded/icon_global_t2m_spcs_{date}.nc")
        print(f"icon_global_t2m files for {date} regridded to SPCS grid and saved to " +
              ICON_PATH + f"/icon_t2m_regridded/icon_global_t2m_spcs_{date}.nc")
        
        
def run_main():
    
    run = "00"
    
    # icon eu t2m files
    icon_eu_t2m_regrid()
    
    # icon d2 t2m files
    icon_d2_t2m_regrid()
    
    # icon global t2m files
    # icon_global_t2m_regrid()


if __name__ == "__main__":
    run_main()