#!/usr/bin/env python3

import sys; print('Python %s on %s' % (sys.version, sys.platform))
import xarray as xr
from pathlib import Path

sys.path.extend(['/home/wegnet/TEA-indicators', '/home/wegnet/tmp', '/home/wegnet/TEA-indicators/src', '/home/wegnet/TEA-indicators/src/teametrics'])
input_path = Path('/data/arsclisys/normal/clim-hydro/TEA-Indicators/results/heatwave_paper/ctp_indicator_variables')
data = xr.open_mfdataset(input_path.glob('*.nc'), data_vars='minimal', combine='by_coords', coords='minimal',
                         compat="override", join="exact", chunks="auto")
sorted = data.TEX_HW_max_GR.sortby(data.TEX_HW_max_GR)
sorted.to_series().to_csv("TEXHWx_GR.csv")
sorted_tex = data.TEX_max_GR.sortby(data.TEX_max_GR)
sorted_tex.to_series().to_csv("TEX_max_GR.csv")
