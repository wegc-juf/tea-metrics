#!/usr/bin/env bash

set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=${PYTHON:-"$ROOT/.venv/bin/python"}
BASE_CONFIG=${BASE_CONFIG:-"$ROOT/internal/config/WEGC/heatwave_paper_Tx30_update.yaml"}
RUN_ID=${BENCHMARK_RUN_ID:-"$(hostname -s)_$(date +%Y%m%d_%H%M%S)"}
RESULT_ROOT=${RESULT_ROOT:-"/tmp/tea-processing-benchmark/$RUN_ID"}
CASES=${CASES:-"eager auto dask4 dask8 dask16"}
REPEATS=${REPEATS:-1}
PROFILE_CASE=${PROFILE_CASE:-""}
KEEP_OUTPUTS=${KEEP_OUTPUTS:-false}

if [[ ! -x "$PYTHON" ]]; then
    printf 'Python executable not found or not executable: %s\n' "$PYTHON" >&2
    exit 1
fi
if [[ ! -f "$BASE_CONFIG" ]]; then
    printf 'Base config not found: %s\n' "$BASE_CONFIG" >&2
    exit 1
fi
if ! [[ "$REPEATS" =~ ^[1-9][0-9]*$ ]]; then
    printf 'REPEATS must be a positive integer: %s\n' "$REPEATS" >&2
    exit 1
fi

mkdir -p "$RESULT_ROOT/configs" "$RESULT_ROOT/logs" "$RESULT_ROOT/metrics" \
         "$RESULT_ROOT/profiles" "$RESULT_ROOT/outputs"
printf 'trial\tcase\telapsed_seconds\tmax_rss_kb\tuser_cpu_seconds\tsystem_cpu_seconds\n' \
    > "$RESULT_ROOT/results.tsv"

case_settings() {
    case "$1" in
        eager) printf 'false\t' ;;
        auto) printf 'auto\t' ;;
        dask4) printf 'true\t4' ;;
        dask8) printf 'true\t8' ;;
        dask16) printf 'true\t16' ;;
        *) printf 'Unknown case: %s\n' "$1" >&2; return 1 ;;
    esac
}

make_config() {
    local mode=$1
    local workers=$2
    local output_path=$3
    local config_path=$4

    sed \
        -e '/^  zlib_compression:/d' \
        -e '/^  dask_workers:/d' \
        -e "s|^  outpath: .*|  outpath: $output_path/|" \
        -e "s|^  use_dask: .*|  use_dask: $mode|" \
        -e 's|^  parallel_workers: .*|  parallel_workers: 1|' \
        -e '/^  compression_level:/a\  zlib_compression: false' \
        "$BASE_CONFIG" > "$config_path"
    if [[ -n "$workers" ]]; then
        sed -i "/^  use_dask:/a\\  dask_workers: $workers" "$config_path"
    fi
}

for ((trial = 1; trial <= REPEATS; trial++)); do
    for case_name in $CASES; do
        IFS=$'\t' read -r mode workers <<< "$(case_settings "$case_name")"
        run_name="${case_name}-trial${trial}"
        output_path="$RESULT_ROOT/outputs/$run_name"
        config_path="$RESULT_ROOT/configs/$run_name.yaml"
        log_path="$RESULT_ROOT/logs/$run_name.log"
        metrics_path="$RESULT_ROOT/metrics/$run_name.json"
        profile_args=()
        mkdir -p "$output_path"
        make_config "$mode" "$workers" "$output_path" "$config_path"
        if [[ "$PROFILE_CASE" == "$case_name" && "$trial" == 1 ]]; then
            profile_args=(--profile-file "$RESULT_ROOT/profiles/$run_name.prof")
        fi

        printf 'Running %s (mode=%s workers=%s)\n' "$run_name" "$mode" "${workers:-auto}"
        "$PYTHON" "$ROOT/internal/scripts/profile_calc_tea.py" \
            --config-file "$config_path" --metrics-file "$metrics_path" \
            "${profile_args[@]}" > "$log_path" 2>&1

        "$PYTHON" - "$trial" "$case_name" "$metrics_path" "$RESULT_ROOT/results.tsv" <<'PY'
import json
import sys

trial, case, metrics_path, results_path = sys.argv[1:]
with open(metrics_path) as stream:
    metrics = json.load(stream)
with open(results_path, 'a') as stream:
    stream.write(
        f"{trial}\t{case}\t{metrics['elapsed_seconds']:.3f}\t{metrics['max_rss_kb']}\t"
        f"{metrics['user_cpu_seconds']:.3f}\t{metrics['system_cpu_seconds']:.3f}\n"
    )
PY
        if [[ "$KEEP_OUTPUTS" != true ]]; then
            rm -rf "$output_path"
        fi
    done
done

printf '\nResults saved to %s\n' "$RESULT_ROOT/results.tsv"
column -t -s $'\t' "$RESULT_ROOT/results.tsv"
