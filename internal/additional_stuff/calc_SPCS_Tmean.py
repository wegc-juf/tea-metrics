#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
calculate SPARTACUS Tmean from Tmin and Tmax files
"""
import os
from pathlib import Path
import xarray as xr
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate SPARTACUS Tmean from Tmin and Tmax files"
    )
    parser.add_argument(
        "--raw_data_path",
        type=str,
        required=True,
        help="Path to the raw SPARTACUS data directory",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=False,
        help="Path to the output directory for Tmean files, default is raw_data_path",
    )
    parser.add_argument(
        "--year",
        type=int,
        required=False,
        help="Year to process, default is all years",
    )
    args = parser.parse_args()
    if not args.output_path:
        args.output_path = args.raw_data_path

    raw_data_path = Path(args.raw_data_path)
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if args.year:
        file_mask = f'SPARTACUS2-DAILY_Tn*{args.year}.nc'
    else:
        file_mask = 'SPARTACUS2-DAILY_Tn*.nc'
    
    all_files = sorted(raw_data_path.glob(file_mask))
    for tmin_file in all_files:
        print(f'Processing file: {tmin_file.name}')
        
        tmax_file = Path(raw_data_path, tmin_file.name.replace('Tn', 'Tx'))
    
        # Load Tmin and Tmax datasets
        ds_tmin = xr.open_dataset(tmin_file)
        ds_tmax = xr.open_dataset(tmax_file)

        # Calculate Tmean
        ds_tmean = (ds_tmin.Tn + ds_tmax.Tx) / 2
        ds_tmean = ds_tmean.to_dataset(name='Tm')

        # Save Tmean dataset
        tmean_file = Path(output_path, tmin_file.name.replace('Tn', 'Tm'))
        ds_tmean.to_netcdf(tmean_file)

        print(f'Saved Tmean file to {tmean_file}')