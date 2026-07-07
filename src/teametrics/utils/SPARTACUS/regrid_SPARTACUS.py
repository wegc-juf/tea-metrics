#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: hst

"""

import numpy as np
import os
from pathlib import Path
import pyproj
from tqdm import trange
import xarray as xr
import argparse

from teametrics.common.general_functions import create_history_from_cfg
from teametrics.common.config import load_opts


def get_target_grid_definition(opts):
    """
    Return target grid metadata.

    Args:
        opts: CLI/config parameters

    Returns:
        dict with target grid definition
    """
    target_grid = getattr(opts, 'target_grid', 'wegn').lower()
    if target_grid == 'wegn':
        return {'name': 'wegn', 'epsg': '32633', 'grid_mapping': 'UTM33N'}
    if target_grid == 'statat':
        return {'name': 'statat', 'epsg': '3035', 'grid_mapping': 'ETRS89_LAEA'}
    raise ValueError(f'Unknown target_grid "{target_grid}". Use "wegn" or "statat".')


def define_wegn_grid_1000x1000(opts):
    """
    create 1 km resolution grid over SPARTACUS region with identical points of
    WegenerNet grid within the FBR region
    Args:
        opts: CLI parameter

    Returns:
        grid: dummy da with new grid coordinates

    """

    # Load sample SPARTACUS data
    original_grid = xr.open_dataset(Path(opts.raw_data_path) / f'SPARTACUS2-DAILY_{opts.parameter.upper()}_2000.nc')

    # Open WEGN sample data
    wegnet = xr.open_dataset(opts.wegn_file)
    # Select 2020-08-15 (just a random date)
    wegnet_first = wegnet.isel(time=14)

    # New coordinates in 1 km x 1 km instead of 200 m x 200 m
    fbr_x = wegnet_first.X.values[2:-2:5]
    fbr_y = wegnet_first.Y.values[2:-2:5]

    # Change SPARTACUS projection to UTM
    x_spa, y_spa = epsg3416_to_utm_grid(original_grid.x, original_grid.y)

    # Round corners to 1000
    xmin, xmax = np.round(np.min(x_spa) / 1000) * 1000, np.round(np.max(x_spa) / 1000) * 1000
    ymin, ymax = np.round(np.min(y_spa) / 1000) * 1000, np.round(np.max(y_spa) / 1000) * 1000

    # FBR corners
    xmin_fbr, xmax_fbr = np.min(fbr_x), np.max(fbr_x)
    ymin_fbr, ymax_fbr = np.min(fbr_y), np.max(fbr_y)

    left = np.arange(xmin_fbr, xmin, -1000)[::-1]
    right = np.arange(xmax_fbr, xmax, 1000)
    bottom = np.arange(ymin_fbr, ymin, -1000)[::-1]
    top = np.arange(ymax_fbr, ymax, 1000)

    x_new = np.concatenate((left[:-1], fbr_x, right[1:]), axis=0)
    y_new = np.concatenate((bottom[:-1], fbr_y, top[1:]), axis=0)

    # Create DataArray with new grid and dummy values
    dummy_data = np.zeros((len(y_new), len(x_new)))

    grid = xr.Dataset(data_vars=dict(data=(["y", "x"], dummy_data), ),
                      coords=dict(x=(["x"], x_new), y=(["y"], y_new), ), )

    return grid


def define_statat_grid_1000x1000(opts):
    """
    create 1 km resolution Statistik Austria-like grid in EPSG:3035
    over the SPARTACUS domain
    Args:
        opts: CLI parameter

    Returns:
        grid: dummy ds with new grid coordinates

    """
    # Load sample SPARTACUS data
    original_grid = xr.open_dataset(Path(opts.raw_data_path) / f'SPARTACUS2-DAILY_{opts.parameter.upper()}_2000.nc')

    # Change SPARTACUS projection to EPSG:3035
    x_spa, y_spa = epsg3416_to_epsg3035_grid(original_grid.x, original_grid.y)

    # Round corners to 1000 m and build regular 1 km target grid
    xmin, xmax = np.round(np.min(x_spa) / 1000) * 1000, np.round(np.max(x_spa) / 1000) * 1000
    ymin, ymax = np.round(np.min(y_spa) / 1000) * 1000, np.round(np.max(y_spa) / 1000) * 1000
    x_new = np.arange(xmin, xmax + 1000, 1000)
    y_new = np.arange(ymin, ymax + 1000, 1000)

    # Create DataArray with new grid and dummy values
    dummy_data = np.zeros((len(y_new), len(x_new)))

    grid = xr.Dataset(data_vars=dict(data=(["y", "x"], dummy_data), ),
                      coords=dict(x=(["x"], x_new), y=(["y"], y_new), ), )

    return grid


def utm_to_epsg3416_grid(x, y):
    """
    transform UTM33N coords to EPSG:3416 grid
    Args:
        x: x-coordinates in UTM33N
        y: y-coordinates in UTM33N

    Returns:
        newx: x-coordinates in EPSG:3416
        newy: y-coordinates in EPSG:3416

     """
    transformer = pyproj.Transformer.from_crs('EPSG:32633', 'EPSG:3416')
    ny, nx = len(y), len(x)
    x, y = np.meshgrid(x, y)
    newy, newx = transformer.transform(x.flatten(), y.flatten())
    newx = np.asarray(newx).reshape((ny, nx))
    newy = np.asarray(newy).reshape((ny, nx))
    return newx, newy


def epsg3416_to_utm_grid(x, y):
    """
    transform EPSG:3416 coords to UTM33N grid
    Args:
        x: x-coordinates in EPSG:3416
        y: y-coordinates in EPSG:341

    Returns:
        newx: x-coordinates in UTM33N
        newy: y-coordinates in UTM33N

    """
    transformer = pyproj.Transformer.from_crs('EPSG:3416', 'EPSG:32633')
    ny, nx = len(y), len(x)
    y, x = np.meshgrid(y, x)
    newx, newy = transformer.transform(y.flatten(), x.flatten())
    newy = np.sort(newy)
    newx = np.asarray(newx).reshape((nx, ny))
    newx = newx.T
    newy = np.asarray(newy).reshape((ny, nx))

    return newx, newy


def epsg3035_to_epsg3416_grid(x, y):
    """
    transform EPSG:3035 coords to EPSG:3416 grid
    Args:
        x: x-coordinates in EPSG:3035
        y: y-coordinates in EPSG:3035

    Returns:
        newx: x-coordinates in EPSG:3416
        newy: y-coordinates in EPSG:3416

     """
    transformer = pyproj.Transformer.from_crs('EPSG:3035', 'EPSG:3416')
    ny, nx = len(y), len(x)
    x, y = np.meshgrid(x, y)
    newx, newy = transformer.transform(x.flatten(), y.flatten())
    newx = np.asarray(newx).reshape((ny, nx))
    newy = np.asarray(newy).reshape((ny, nx))
    return newx, newy


def epsg3416_to_epsg3035_grid(x, y):
    """
    transform EPSG:3416 coords to EPSG:3035 grid
    Args:
        x: x-coordinates in EPSG:3416
        y: y-coordinates in EPSG:3416

    Returns:
        newx: x-coordinates in EPSG:3035
        newy: y-coordinates in EPSG:3035

    """
    transformer = pyproj.Transformer.from_crs('EPSG:3416', 'EPSG:3035')
    ny, nx = len(y), len(x)
    y, x = np.meshgrid(y, x)
    newx, newy = transformer.transform(y.flatten(), x.flatten())
    newy = np.sort(newy)
    newx = np.asarray(newx).reshape((nx, ny))
    newx = newx.T
    newy = np.asarray(newy).reshape((ny, nx))

    return newx, newy


def regrid_spartacus(opts, ds_in, method="linear"):
    """
    regrid SPARTACUS data (projection EPSG3416) to the extended FBR region (projection UTM33N)
    Args:
        opts: CLI parameter
        ds_in: ds for regridding
        method: interpolation method (default: linear)

    Returns:
        ds_regridded: regridded ds
    """

    target_grid = get_target_grid_definition(opts)
    if target_grid['name'] == 'wegn':
        grid = define_wegn_grid_1000x1000(opts=opts)
        x_new = grid.x.values
        y_new = grid.y.values
        # UTM to EPSG because input dataset is SPARTACUS, which is in EPSG3416
        x_new_epsg3416, y_new_epsg3416 = utm_to_epsg3416_grid(x_new, y_new)
    else:
        grid = define_statat_grid_1000x1000(opts=opts)
        x_new = grid.x.values
        y_new = grid.y.values
        # EPSG:3035 to EPSG:3416 for interpolation from SPARTACUS source grid
        x_new_epsg3416, y_new_epsg3416 = epsg3035_to_epsg3416_grid(x_new, y_new)

    # Get remapping arrays in SPARTACUS CRS, but keep target grid coords
    x = xr.DataArray(x_new_epsg3416, dims=["y", "x"], coords={"x": x_new, "y": y_new})
    y = xr.DataArray(y_new_epsg3416, dims=["y", "x"], coords={"x": x_new, "y": y_new})

    # Interpolation
    ds_regridded = ds_in.interp(x=x, y=y, method=method)

    return ds_regridded


def regrid_orog(opts):
    """
    regrid orography to new grid
        Args:
        opts: CLI parameter

    Returns:

    """

    orog_file = xr.open_dataset(opts.orog_file)
    target_grid = get_target_grid_definition(opts)
    oro_new = regrid_spartacus(opts=opts, ds_in=orog_file.orog, method='linear')
    oro_new = oro_new.assign_attrs(grid_mapping=target_grid['grid_mapping'])
    create_history_from_cfg(cfg_params=opts, ds=oro_new)
    oro_new.attrs['crs'] = f"EPSG:{target_grid['epsg']}"
    oro_new = oro_new.drop_vars(['lat', 'lon'])

    path = Path(opts.regridded_data_path)
    path.mkdir(parents=True, exist_ok=True)
    oro_new.to_netcdf(path / 'SPARTACUSreg_orography.nc')


def _getopts():
    """
    get command line arguments

    Returns:
        opts: command line parameters
    """
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config-file', '-cf',
                        dest='config_file',
                        type=str,
                        default='../TEA_CFG.yaml',
                        help='TEA configuration file (default: TEA_CFG.yaml)')
    parser.add_argument('--year', '-y',
                        dest='year',
                        type=int,
                        help='Year for processing')
    parser.add_argument('--target-grid', '-tg',
                        dest='target_grid',
                        type=str,
                        choices=['wegn', 'statat'],
                        help='Target grid: "wegn" (EPSG:32633) or "statat" (EPSG:3035). '
                             'Overrides config value if set.')

    myopts = parser.parse_args()
    
    return myopts


def run():
    # get command line parameters
    cmd_opts = _getopts()
    
    # load CFG parameters
    opts = load_opts(fname=__file__, config_file=cmd_opts.config_file)
    if cmd_opts.target_grid is not None:
        opts.target_grid = cmd_opts.target_grid
    target_grid = get_target_grid_definition(opts)

    if opts.orography:
        regrid_orog(opts=opts)
    else:
        input_path = Path(opts.input_data_path)
        if cmd_opts.year is not None:
            input_files = sorted(input_path.glob(f'*{opts.parameter.upper()}*{cmd_opts.year}.nc'))
        else:
            input_files = sorted(input_path.glob(f'*{opts.parameter.upper()}*.nc'))
        if len(input_files) == 0:
            raise FileNotFoundError(f'No input files found in {input_path}/*{opts.parameter}*.nc')
        for ifile in trange(len(input_files), desc='Regridding files'):
            filename = input_files[ifile].name

            # Open SPARTACUS file
            ds = xr.open_dataset(input_files[ifile], engine='netcdf4')

            # Regrid to extended WEGN grid
            ds_new = regrid_spartacus(opts=opts, ds_in=ds, method='linear')
            ds_new = ds_new.assign_attrs(grid_mapping=target_grid['grid_mapping'])

            # Add history to attributes and set output CRS to target grid CRS
            create_history_from_cfg(cfg_params=opts, ds=ds_new)
            ds_new.attrs['crs'] = f"EPSG:{target_grid['epsg']}"

            # Rename variables if necessary
            if opts.parameter in ['TX', 'Tx']:
                ds_new = ds_new.rename({'TX': 'Tx'})
                opts.parameter = 'Tx'
            elif opts.parameter in ['TN', 'Tn']:
                ds_new = ds_new.rename({'TN': 'Tn'})
                opts.parameter = 'Tn'

            # drop unnecessary coords
            ds_new = ds_new.drop_vars(['lat', 'lon'])

            # Save output
            encoding = {opts.parameter: {'dtype': 'int16', 'scale_factor': 0.1,
                                         '_FillValue': -9999}}

            path = Path(opts.regridded_data_path)
            path.mkdir(parents=True, exist_ok=True)
            filename_parts = filename.split(opts.parameter.upper())
            filename_out = f'{filename_parts[0]}{opts.parameter}{filename_parts[1]}'
            ds_new.to_netcdf(path / filename_out, encoding=encoding, engine='netcdf4')
            print(f"Saved regridded file to {path / filename_out}")


if __name__ == '__main__':
    run()
