#!/usr/bin/env python3

import sys; print('Python %s on %s' % (sys.version, sys.platform))
import xarray as xr
from pathlib import Path

sys.path.extend(['/home/wegnet/TEA-indicators', '/home/wegnet/tmp', '/home/wegnet/TEA-indicators/src', '/home/wegnet/TEA-indicators/src/teametrics'])
input_path = Path('/data/arsclisys/normal/clim-hydro/TEA-Indicators/results/heatwave_paper/ctp_indicator_variables')
data = xr.open_mfdataset(input_path.glob('*.nc'), data_vars='minimal', combine='by_coords', coords='minimal',
                         compat="override", join="exact", chunks="auto")


def save_sorted_extremity_with_interval(ds, value_var, interval_start_var, interval_end_var, output_file):
    available = {value_var, interval_start_var, interval_end_var}
    missing = available - set(ds.data_vars)
    if missing:
        raise KeyError(f"Missing required variables for export {output_file}: {sorted(missing)}")

    df = ds[[value_var, interval_start_var, interval_end_var]].to_dataframe().reset_index()
    df = df.sort_values(by=value_var, na_position="last")
    df.to_csv(output_file, index=False)


save_sorted_extremity_with_interval(
    data,
    "TEX_HW_max_GR",
    "TEX_HW_max_interval_start_GR",
    "TEX_HW_max_interval_end_GR",
    "TEXHWx_GR.csv",
)
save_sorted_extremity_with_interval(
    data,
    "TEX_max_GR",
    "TEX_max_interval_start_GR",
    "TEX_max_interval_end_GR",
    "TEX_max_GR.csv",
)
