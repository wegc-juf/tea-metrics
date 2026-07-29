#!/bin/bash

cd ~/TEA-indicators/src
# download and regrid ICON data
./teametrics/utils/ICON/get_icon_data.py
./teametrics/utils/ICON/regrid_icon_to_spcs.py
./teametrics/utils/ICON/bias_correct_icon.py

# regrid SPCS data
./teametrics/utils/SPARTACUS/regrid_SPARTACUS.py --config-file /home/juf/TEA-indicators/internal/config/WEGC/heatwave_paper_Tx30_update.yaml --year 2026

# recalc daily and ctp data
python -m teametrics.calc_TEA --config-file /home/juf/TEA-indicators/internal/config/WEGC/heatwave_paper_Tx30_update.yaml --loglevel INFO

# detrend data
# ~/wegenerNet/misc_analyses/climate_trends/detrend.py --folder-spartacus /data/arsclisys/normal/clim-hydro/TEA-Indicators/SPARTACUS/reproj_StatAT/v2.1  --end 2025 --data-var Tx --output-folder /data/arsclisys/normal/SPARTACUS/reproj_StatAT/detrended/ --year 2026 --cache

# plot heatwave data
../plots/heatwave/plot_heatwave.py -cf ~/TEA-indicators/internal/config/WEGC/heatwave_paper_Tx30_update.yaml