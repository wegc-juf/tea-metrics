#!/usr/bin/env bash

set -Eeuo pipefail

# ============================================================================
# BASH BOOLEAN WARNING: unlike Python, Bash treats exit status 0 as TRUE.
# A function returns 0 for SUCCESS and a non-zero status for FAILURE.
# ============================================================================

usage() {
    cat <<'EOF'
Usage: spartacus_workflow.sh [OPTIONS]

Run the configured default stages when no options are supplied, or select one
or more workflow stages to run:

  --icon                 Download, regrid, and bias-correct ICON data
  --regrid-spartacus     Regrid SPARTACUS data
  --calculate             Recalculate daily and CTP data
  --detrend               Detrend data
  --calculate-detrended   Recalculate daily and CTP data for detrended data
  --plot-heatwave         Plot heatwave data
  --decadal               Calculate decadal data
  --plot                  Plot figures
  --styria                Run all Styria-specific stages
  --sync                  Synchronize generated plots
  -h, --help              Show this help

Selected stages are always executed in workflow order, regardless of the
order in which their options are provided.
EOF
}

requested_stages=()
# Configure the stages enabled when the script is run without options.
default_stages=(
    --icon
    --regrid-spartacus
    --calculate
    --detrend
    --calculate-detrended
    --plot-heatwave
    --sync
    --decadal
    --plot
    --styria
)

while (($# > 0)); do
    case "$1" in
        --icon|--regrid-spartacus|--calculate|--detrend|--calculate-detrended|\
        --plot-heatwave|--decadal|--plot|--styria|--sync)
            requested_stages+=("$1")
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'Unknown option: %s\n\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

HEATWAVE_PLOT_DIR=/data/arsclisys/normal/clim-hydro/TEA-Indicators/results/heatwaves/
PLOT_SYNC_DIR="/nas/share/ccr/wegnet/projects/TEA/heatwaves/202607"
PLOT_SYNC_DIR_UNICLOUD="/mnt/unicloud/juergen.fuchsberger/TEA-indicators/heatwaves/202607"

source /home/juf/TEA-indicators/.venv/bin/activate


export PYTHONPATH=$HOME/wegenerNet/WPS/QCS:$HOME/wegenerNet/WPS/DPG:$HOME/wegenerNet/WPS:$HOME/wegenerNet:$HOME/wegenerNet/misc_analyses:~/cdr_DPS/scripts

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

stage_enabled() {
    local stage=$1
    local requested

    if ((${#requested_stages[@]} == 0)); then
        for requested in "${default_stages[@]}"; do
            [[ "$requested" == "$stage" ]] && return 0  # 0 = TRUE: stage enabled
        done
        return 1  # non-zero = FALSE: stage disabled by default
    fi
    for requested in "${requested_stages[@]}"; do
        [[ "$requested" == "$stage" ]] && return 0  # 0 = TRUE: stage enabled
    done
    return 1  # non-zero = FALSE: stage not selected
}

printf '[%s] Starting SPARTACUS workflow. Log: %s\n' "$(date --iso-8601=seconds)" "$LOG_FILE"
cd "$HOME/TEA-indicators/src"

# ============================================================================
# BASH BOOLEAN WARNING: these `if` conditions enter their blocks when the
# function returns 0. That is Bash SUCCESS, and Bash interprets it as TRUE.
# ============================================================================

if stage_enabled --icon; then
    # download and regrid ICON data
    run_step ./teametrics/utils/ICON/get_icon_data.py
    run_step ./teametrics/utils/ICON/regrid_icon_to_spcs.py
    run_step ./teametrics/utils/ICON/bias_correct_icon.py
fi

if stage_enabled --regrid-spartacus; then
    # regrid SPCS data
    run_step ./teametrics/utils/SPARTACUS/regrid_SPARTACUS.py --config-file /home/juf/TEA-indicators/internal/config/WEGC/heatwave_paper_Tx30_update.yaml --year 2026
fi

if stage_enabled --calculate; then
    # recalc daily and ctp data
    run_step python -m teametrics.calc_TEA --config-file /home/juf/TEA-indicators/internal/config/WEGC/heatwave_paper_Tx30_update.yaml --loglevel INFO
    run_step python -m teametrics.calc_TEA --config-file /home/juf/TEA-indicators/internal/config/WEGC/heatwave_paper_Tx35_update.yaml --loglevel INFO
    run_step python -m teametrics.calc_TEA --config-file /home/juf/TEA-indicators/internal/config/WEGC/heatwave_paper_Tx40_update.yaml --loglevel INFO
fi

if stage_enabled --detrend; then
    # detrend data
    run_step ~/wegenerNet/misc_analyses/climate_trends/detrend.py --folder-spartacus /data/arsclisys/normal/clim-hydro/TEA-Indicators/SPARTACUS/reproj_StatAT/v2.1  --end 2025 --data-var Tx --output-folder /data/arsclisys/normal/SPARTACUS/reproj_StatAT/detrended/ --year 2026 --cache
fi

if stage_enabled --calculate-detrended; then
    # recalc daily and ctp data for detrend
    run_step python -m teametrics.calc_TEA --config-file /home/juf/TEA-indicators/internal/config/WEGC/CW2_daily.yaml --loglevel INFO
fi

if stage_enabled --plot-heatwave; then
    # plot heatwave data
    run_step ../plots/heatwave/plot_heatwave.py -cf "$HOME/TEA-indicators/internal/config/WEGC/heatwave_paper_Tx30_update.yaml" --period 2026-08-19 2026-09-15 --outpath /data/arsclisys/normal/clim-hydro/TEA-Indicators/results/heatwaves/current
    run_step ../plots/heatwave/plot_heatwave.py -cf "$HOME/TEA-indicators/internal/config/WEGC/heatwave_paper_Tx35_update.yaml" --period 2026-08-19 2026-09-15 --outpath /data/arsclisys/normal/clim-hydro/TEA-Indicators/results/heatwaves/current
    run_step ../plots/heatwave/plot_heatwave.py -cf "$HOME/TEA-indicators/internal/config/WEGC/heatwave_paper_Tx40_update.yaml" --period 2026-08-19 2026-09-15 --outpath /data/arsclisys/normal/clim-hydro/TEA-Indicators/results/heatwaves/current
fi

if stage_enabled --sync; then
    run_step rsync -av "$HEATWAVE_PLOT_DIR" "$PLOT_SYNC_DIR"
fi

if stage_enabled --decadal; then
    # calculate decadal data
    run_step python -m teametrics.calc_TEA --config-file /home/juf/TEA-indicators/internal/config/WEGC/heatwave_paper_Tx30_decadal.yaml --loglevel INFO
    run_step python -m teametrics.calc_TEA --config-file /home/juf/TEA-indicators/internal/config/WEGC/heatwave_paper_Tx30_decadal_detrend.yaml --loglevel INFO
fi

if stage_enabled --plot; then
    # plot data
    run_step /home/juf/TEA-indicators/plots/paper-figs/plot_Fig5.py --output-dir "$HEATWAVE_PLOT_DIR" --csv --no-show
    run_step /home/juf/TEA-indicators/plots/paper-figs/plot_Fig5.py --output-dir "$HEATWAVE_PLOT_DIR" --csv --no-show --run-name current_detrended
fi

if stage_enabled --styria; then
    # now run for Styria
    run_step python -m teametrics.calc_TEA --config-file /home/juf/TEA-indicators/internal/config/WEGC/Tx30_Styria_StatATGrid.yaml --loglevel INFO
    run_step python -m teametrics.calc_TEA --config-file /home/juf/TEA-indicators/internal/config/WEGC/CW2_daily_Styria.yaml --loglevel INFO

    # plot heatwave data
    run_step ../plots/heatwave/plot_heatwave.py -cf "$HOME/TEA-indicators/internal/config/WEGC/Tx30_Styria_StatATGrid.yaml" --period 2026-08-19 2026-09-15 --outpath /data/arsclisys/normal/clim-hydro/TEA-Indicators/results/heatwaves/current

    # calculate decadal data
    run_step python -m teametrics.calc_TEA --config-file /home/juf/TEA-indicators/internal/config/WEGC/Tx30_Styria_decadal.yaml --loglevel INFO
    run_step python -m teametrics.calc_TEA --config-file /home/juf/TEA-indicators/internal/config/WEGC/CW2_decadal_Styria.yaml --loglevel INFO

    # plot data
    run_step /home/juf/TEA-indicators/plots/paper-figs/plot_Fig5.py --output-dir "$HEATWAVE_PLOT_DIR" --region Steiermark --csv --no-show
    run_step /home/juf/TEA-indicators/plots/paper-figs/plot_Fig5.py --output-dir "$HEATWAVE_PLOT_DIR" --region Steiermark --csv --no-show --run-name current_detrended
fi

if stage_enabled --sync; then
    # sync
    run_step rsync -av "$HEATWAVE_PLOT_DIR" "$PLOT_SYNC_DIR"
    run_step rsync -av "$HEATWAVE_PLOT_DIR" "$PLOT_SYNC_DIR_UNICLOUD"
fi
