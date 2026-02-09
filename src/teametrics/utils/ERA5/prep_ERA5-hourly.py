import glob
import xarray as xr


def run():
    file_path = '/data/arsclisys/normal/clim-hydro/TEA-Indicators/ERA5/EUR/hourly/single_levels/'
    files = sorted(glob.glob(f'{file_path}*nc'))

    for file in files:
        fname = file.split('/')[-1]
        print(fname)
        ds = xr.open_dataset(file)
        ds = ds.rename({'t2m': 'T', 'latitude': 'lat', 'longitude': 'lon'})
        t2m = ds['T']
        t2m.to_netcdf(f'{file_path}/prepped4TEA/{fname}')
        ds.close()
        t2m.close()

def compress_2023to2025_data():
    files = sorted(glob.glob(
        '/data/arsclisys/normal/clim-hydro/TEA-Indicators/ERA5/EUR/hourly/single_levels/prepped4TEA/*nc'))
    files = files[-3:]

    for file in files:
        fname = file[:-3]
        ds = xr.open_dataset(file)
        ds = ds.rename({'valid_time': 'time'})
        ds.to_netcdf(f'{fname}_NEW.nc',
                     encoding={'T': {'dtype': 'float32', 'zlib': True, 'complevel': 5}})

if __name__ == '__main__':
    # run()
    compress_2023to2025_data()