#!/usr/bin/env bash

set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=${PYTHON:-"$ROOT/.venv/bin/python"}
BASE_CONFIG=${BASE_CONFIG:-"$ROOT/internal/config/WEGC/heatwave_paper_Tx30_update.yaml"}
RUN_ID=${BENCHMARK_RUN_ID:-"$(hostname -s)_$(date +%Y%m%d_%H%M%S)"}
DATA_ROOT=${DATA_ROOT:-"/data/arsclisys/normal/clim-hydro/TEA-Indicators/results/heatwave_paper/tea-compression-benchmark/$RUN_ID"}
TMP_ROOT=${TMP_ROOT:-"/tmp/tea-compression-benchmark/$RUN_ID"}
CONFIG_ROOT=${CONFIG_ROOT:-"/tmp/tea-compression-benchmark-configs/$RUN_ID"}

if [[ ! -x "$PYTHON" ]]; then
    printf 'Python executable not found or not executable: %s\n' "$PYTHON" >&2
    exit 1
fi
if [[ ! -f "$BASE_CONFIG" ]]; then
    printf 'Base config not found: %s\n' "$BASE_CONFIG" >&2
    exit 1
fi

mkdir -p "$DATA_ROOT/zlib-on" "$DATA_ROOT/zlib-off" \
         "$TMP_ROOT/zlib-on" "$TMP_ROOT/zlib-off" "$CONFIG_ROOT"

make_config() {
    local output_path=$1
    local zlib=$2
    local config_path=$3

    sed \
        -e '/^  zlib_compression:/d' \
        -e "s|^  outpath: .*|  outpath: $output_path/|" \
        -e "s|^     significant_digits: .*|     significant_digits: 3|" \
        -e "/^  compression_level:/a\\  zlib_compression: $zlib" \
        "$BASE_CONFIG" > "$config_path"
}

run_case() {
    local name=$1
    local output_path=$2
    local zlib=$3
    local config_path="$CONFIG_ROOT/$name.yaml"
    local log_path="$CONFIG_ROOT/$name.log"
    local metrics_path="$CONFIG_ROOT/$name.metrics"

    make_config "$output_path" "$zlib" "$config_path"
    printf 'Running %s: output=%s zlib=%s\n' "$name" "$output_path" "$zlib"
    "$PYTHON" -c '
import resource
import subprocess
import sys
import time

started = time.perf_counter()
result = subprocess.run(sys.argv[1:])
usage = resource.getrusage(resource.RUSAGE_CHILDREN)
print(f"elapsed_seconds={time.perf_counter() - started:.3f}", file=sys.stderr)
print(f"child_max_rss_kb={usage.ru_maxrss}", file=sys.stderr)
raise SystemExit(result.returncode)
' "$PYTHON" -m teametrics.calc_TEA \
        --config-file "$config_path" --loglevel INFO \
        > "$log_path" 2> "$metrics_path"
}

run_case data-zlib-on "$DATA_ROOT/zlib-on" true
run_case data-zlib-off "$DATA_ROOT/zlib-off" false
run_case tmp-zlib-on "$TMP_ROOT/zlib-on" true
run_case tmp-zlib-off "$TMP_ROOT/zlib-off" false

printf '\nResults\n'
printf '%-16s %-12s %-12s %-12s\n' 'case' 'logical' 'allocated' 'metrics'
for case_path in \
    "data-zlib-on:$DATA_ROOT/zlib-on" \
    "data-zlib-off:$DATA_ROOT/zlib-off" \
    "tmp-zlib-on:$TMP_ROOT/zlib-on" \
    "tmp-zlib-off:$TMP_ROOT/zlib-off"; do
    case_name=${case_path%%:*}
    output_path=${case_path#*:}
    logical=$(du -sh --apparent-size "$output_path" | cut -f1)
    allocated=$(du -sh "$output_path" | cut -f1)
    printf '%-16s %-12s %-12s %s\n' "$case_name" "$logical" "$allocated" \
        "$(tr '\n' ' ' < "$CONFIG_ROOT/$case_name.metrics")"
done

printf '\nArtifacts:\nconfigs/logs: %s\ndata outputs: %s\ntmp outputs: %s\n' \
    "$CONFIG_ROOT" "$DATA_ROOT" "$TMP_ROOT"
