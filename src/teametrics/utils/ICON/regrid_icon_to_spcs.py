import xarray as xr
from pathlib import Path

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


def run_main():
    
    run = "00"
    
    # icon eu t2m files
    files = sorted(
        Path("icon_eu_t2m").glob("*.grib2")
    )
    dates = [f.stem.split("_")[-4] for f in files]
    unique_dates = sorted(set(dates))
    for date in unique_dates:
        date_files = [f for f in files if date in f.stem]
        ds = regrid(date_files)
        if not Path("./icon_eu_t2m_regridded").exists():
            Path("./icon_eu_t2m_regridded").mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(f"./icon_eu_t2m_regridded/icon_eu_t2m_spcs_{date}.nc")
        print(f"icon_eu_t2m files for {date} regridded to SPCS grid and saved to "
              f"icon_eu_t2m_regridded/icon_eu_t2m_spcs_{date}.nc")
    
    # icon d2 t2m files
    files = sorted(
        Path("icon_d2_t2m").glob("*.grib2")
    )
    dates = [f.stem.split("_")[-5] for f in files]
    unique_dates = sorted(set(dates))
    for date in unique_dates:
        date_files = [f for f in files if date in f.stem]
        ds = regrid(date_files)
        if not Path("./icon_d2_t2m_regridded").exists():
            Path("./icon_d2_t2m_regridded").mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(f"./icon_d2_t2m_regridded/icon_d2_t2m_spcs_{date}.nc")
        print(f"icon_d2_t2m files for {date} regridded to SPCS grid and saved to "
              f"icon_d2_t2m_regridded/icon_d2_t2m_spcs_{date}.nc")
    
    
if __name__ == "__main__":
    run_main()