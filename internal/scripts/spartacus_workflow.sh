#!/usr/bin/env bash

HEATWAVE_PLOT_DIR=/data/arsclisys/normal/clim-hydro/TEA-Indicators/results/heatwaves/
PLOT_SYNC_DIR="/nas/share/ccr/wegnet/projects/TEA/heatwaves/202607"
PLOT_SYNC_DIR_UNICLOUD="/mnt/unicloud/juergen.fuchsberger/TEA-indicators/heatwaves/202607"

source /home/juf/TEA-indicators/.venv/bin/activate


export PYTHONPATH=$HOME/wegenerNet/WPS/QCS:$HOME/wegenerNet/WPS/DPG:$HOME/wegenerNet/WPS:$HOME/wegenerNet:$HOME/wegenerNet/misc_analyses:~/cdr_DPS/scripts

set -Eeuo pipefail

LOG_DIR="$HOME/.local/state/tea-indicators"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/spartacus_workflow_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

trap 'status=$?; if (( status == 0 )); then printf "[%s] Workflow completed successfully.\n" "$(date --iso-8601=seconds)"; else printf "[%s] Workflow failed with exit status %s.\n" "$(date --iso-8601=seconds)" "$status" >&2; fi' EXIT
trap 'status=$?; printf "[%s] Command failed with exit status %s: %s\n" "$(date --iso-8601=seconds)" "$status" "$BASH_COMMAND" >&2' ERR

run_step() {
    printf '\n[%s] Running:' "$(date --iso-8601=seconds)"
    printf ' %q' "$@"
    printf '\n'
    "$@"
}

printf '[%s] Starting SPARTACUS workflow. Log: %s\n' "$(date --iso-8601=seconds)" "$LOG_FILE"
cd "$HOME/TEA-indicators/src"

# download and regrid ICON data
run_step ./teametrics/utils/ICON/get_icon_data.py
run_step ./teametrics/utils/ICON/regrid_icon_to_spcs.py
run_step ./teametrics/utils/ICON/bias_correct_icon.py

# regrid SPCS data
run_step ./teametrics/utils/SPARTACUS/regrid_SPARTACUS.py --config-file /home/juf/TEA-indicators/internal/config/WEGC/heatwave_paper_Tx30_update.yaml --year 2026

# recalc daily and ctp data
run_step python -m teametrics.calc_TEA --config-file /home/juf/TEA-indicators/internal/config/WEGC/heatwave_paper_Tx30_update.yaml --loglevel INFO
run_step python -m teametrics.calc_TEA --config-file /home/juf/TEA-indicators/internal/config/WEGC/heatwave_paper_Tx35_update.yaml --loglevel INFO
run_step python -m teametrics.calc_TEA --config-file /home/juf/TEA-indicators/internal/config/WEGC/heatwave_paper_Tx40_update.yaml --loglevel INFO

# detrend data
run_step ~/wegenerNet/misc_analyses/climate_trends/detrend.py --folder-spartacus /data/arsclisys/normal/clim-hydro/TEA-Indicators/SPARTACUS/reproj_StatAT/v2.1  --end 2025 --data-var Tx --output-folder /data/arsclisys/normal/SPARTACUS/reproj_StatAT/detrended/ --year 2026 --cache

# recalc daily and ctp data for detrend
run_step python -m teametrics.calc_TEA --config-file /home/juf/TEA-indicators/internal/config/WEGC/CW2_daily.yaml --loglevel INFO

# plot heatwave data
run_step ../plots/heatwave/plot_heatwave.py -cf "$HOME/TEA-indicators/internal/config/WEGC/heatwave_paper_Tx30_update.yaml"
run_step ../plots/heatwave/plot_heatwave.py -cf "$HOME/TEA-indicators/internal/config/WEGC/heatwave_paper_Tx35_update.yaml"
run_step ../plots/heatwave/plot_heatwave.py -cf "$HOME/TEA-indicators/internal/config/WEGC/heatwave_paper_Tx40_update.yaml"

run_step rsync -av  $HEATWAVE_PLOT_DIR $PLOT_SYNC_DIR

# calculate decadal data
run_step python -m teametrics.calc_TEA --config-file /home/juf/TEA-indicators/internal/config/WEGC/heatwave_paper_Tx30_decadal.yaml --loglevel INFO
run_step python -m teametrics.calc_TEA --config-file /home/juf/TEA-indicators/internal/config/WEGC/heatwave_paper_Tx30_decadal_detrend.yaml --loglevel INFO

# plot data
run_step /home/juf/TEA-indicators/plots/paper-figs/plot_Fig5.py --output-dir "$HEATWAVE_PLOT_DIR" --csv --no-show
run_step /home/juf/TEA-indicators/plots/paper-figs/plot_Fig5.py --output-dir "$HEATWAVE_PLOT_DIR" --csv --no-show --run-name current_detrended

# now run for Styria
run_step python -m teametrics.calc_TEA --config-file /home/juf/TEA-indicators/internal/config/WEGC/Tx30_Styria_StatATGrid.yaml --loglevel INFO
run_step python -m teametrics.calc_TEA --config-file /home/juf/TEA-indicators/internal/config/WEGC/CW2_daily_Styria.yaml --loglevel INFO

# plot heatwave data
run_step ../plots/heatwave/plot_heatwave.py -cf "$HOME/TEA-indicators/internal/config/WEGC/Tx30_Styria_StatATGrid.yaml"

# calculate decadal data
run_step python -m teametrics.calc_TEA --config-file /home/juf/TEA-indicators/internal/config/WEGC/Tx30_Styria_decadal.yaml --loglevel INFO
run_step python -m teametrics.calc_TEA --config-file /home/juf/TEA-indicators/internal/config/WEGC/CW2_decadal_Styria.yaml --loglevel INFO

# plot data
run_step /home/juf/TEA-indicators/plots/paper-figs/plot_Fig5.py --output-dir "$HEATWAVE_PLOT_DIR" --region Steiermark --csv --no-show
run_step /home/juf/TEA-indicators/plots/paper-figs/plot_Fig5.py --output-dir "$HEATWAVE_PLOT_DIR" --region Steiermark --csv --no-show --run-name current_detrended

# sync
run_step rsync -av  $HEATWAVE_PLOT_DIR $PLOT_SYNC_DIR
run_step rsync -av  $HEATWAVE_PLOT_DIR $PLOT_SYNC_DIR_UNICLOUD
