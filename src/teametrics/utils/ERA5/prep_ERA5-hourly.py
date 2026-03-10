import glob
import numpy as np
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

def convert_to_degc():
    inpath = '/data/arsclisys/normal/clim-hydro/TEA-Indicators/ERA5/EUR/hourly/single_levels/prepped4TEA/'
    files = sorted(glob.glob(f'{inpath}*nc'))

    for file in files:
        fname = file.split('/')[-1]
        print(fname)
        da = xr.open_dataarray(file)

        # Convert Kelvin to Celsius
        celsius_data = da - 273.15

        # Find min/max for scaling (e.g., ignoring fill values)
        valid_data = celsius_data.values[np.isfinite(celsius_data.values)]
        cmin, cmax = float(valid_data.min()), float(valid_data.max())

        # Reserve -32767 as _FillValue
        fill_value = -32767

        # Calculate scale_factor and add_offset for best precision
        int16_max = np.iinfo(np.int16).max
        int16_min = np.iinfo(np.int16).min + 1  # +1 for reserved fill_value

        # Choose add_offset and scale_factor
        add_offset = (cmax + cmin) / 2
        scale_factor = (cmax - cmin) / (int16_max - int16_min)

        # Prepare encoding
        encoding = {'T': {'dtype': 'int16',
                          '_FillValue': fill_value,
                          'scale_factor': scale_factor,
                          'add_offset': add_offset,
                          'zlib': True,
                          'complevel': 5,
                          'contiguous': False}}

        # Attach attributes
        celsius_data.attrs['units'] = 'degC'

        # Write packed data out
        celsius_data.to_netcdf(f'{inpath}degC/{fname}', encoding=encoding, format='NETCDF4', engine='netcdf4')
        pass

if __name__ == '__main__':
    # run()
    # compress_2023to2025_data()
    convert_to_degc()