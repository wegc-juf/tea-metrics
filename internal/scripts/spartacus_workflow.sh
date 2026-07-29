#!/bin/bash

cd ~/TEA-indicators/src
# regrid SPCS data
./teametrics/utils/SPARTACUS/regrid_SPARTACUS.py --config-file /home/juf/TEA-indicators/internal/config/WEGC/heatwave_paper_Tx30.yaml --year 2026

# recalc daily and ctp data
python -m teametrics.calc_TEA -
-config-file /home/juf/TEA-indicators/internal/config/WEGC/heatwave_paper_Tx30_update
.yaml --loglevel INFO