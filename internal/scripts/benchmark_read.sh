#!/usr/bin/env bash

set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=${PYTHON:-"$ROOT/.venv/bin/python"}
READ_TRIALS=${READ_TRIALS:-3}
RESULT_ROOT=${RESULT_ROOT:-"/tmp/tea-read-benchmark-$(hostname -s)_$(date +%Y%m%d_%H%M%S)"}
CACHE_MODE=${CACHE_MODE:-warm}
BENCHMARK_RUN_ID=${BENCHMARK_RUN_ID:-}

DATA_ZLIB_ON=${DATA_ZLIB_ON:-"/data/arsclisys/normal/clim-hydro/TEA-Indicators/results/heatwave_paper/tea-compression-benchmark/data-zlib-on"}
DATA_ZLIB_OFF=${DATA_ZLIB_OFF:-"/data/arsclisys/normal/clim-hydro/TEA-Indicators/results/heatwave_paper/tea-compression-benchmark/data-zlib-off"}
TMP_ZLIB_ON=${TMP_ZLIB_ON:-"/tmp/tea-compression-benchmark/tmp-zlib-on"}
TMP_ZLIB_OFF=${TMP_ZLIB_OFF:-"/tmp/tea-compression-benchmark/tmp-zlib-off"}
ZFS_LZ4_BASE=${ZFS_LZ4_BASE:-"/tank/local/TEA/test"}
ZFS_ZSTD_BASE=${ZFS_ZSTD_BASE:-"/tank/data/wegnet/products/test"}

resolve_zfs_root() {
    local base=$1
    local latest=
    local candidate

    if [[ -n "$BENCHMARK_RUN_ID" ]]; then
        printf '%s/%s\n' "$base" "$BENCHMARK_RUN_ID"
        return
    fi

    for candidate in "$base"/*; do
        if [[ -d "$candidate/zlib-on" && -d "$candidate/zlib-off" \
              && ( -z "$latest" || "$candidate" > "$latest" ) ]]; then
            latest=$candidate
        fi
    done
    [[ -n "$latest" ]] || {
        printf 'No benchmark run found below: %s\n' "$base" >&2
        exit 1
    }
    printf '%s\n' "$latest"
}

ZFS_LZ4_ROOT=${ZFS_LZ4_ROOT:-"$(resolve_zfs_root "$ZFS_LZ4_BASE")"}
ZFS_ZSTD_ROOT=${ZFS_ZSTD_ROOT:-"$(resolve_zfs_root "$ZFS_ZSTD_BASE")"}

if [[ ! -x "$PYTHON" ]]; then
    printf 'Python executable not found or not executable: %s\n' "$PYTHON" >&2
    exit 1
fi
if ! [[ "$READ_TRIALS" =~ ^[1-9][0-9]*$ ]]; then
    printf 'READ_TRIALS must be a positive integer: %s\n' "$READ_TRIALS" >&2
    exit 1
fi
if [[ "$CACHE_MODE" != warm ]]; then
    printf 'Only CACHE_MODE=warm is currently supported; no cache invalidation is performed: %s\n' \
        "$CACHE_MODE" >&2
    exit 1
fi

declare -A CASE_PATHS=(
    [data-zlib-on]="$DATA_ZLIB_ON"
    [data-zlib-off]="$DATA_ZLIB_OFF"
    [tmp-zlib-on]="$TMP_ZLIB_ON"
    [tmp-zlib-off]="$TMP_ZLIB_OFF"
    [zfs-lz4-zlib-on]="$ZFS_LZ4_ROOT/zlib-on"
    [zfs-lz4-zlib-off]="$ZFS_LZ4_ROOT/zlib-off"
    [zfs-zstd-zlib-on]="$ZFS_ZSTD_ROOT/zlib-on"
    [zfs-zstd-zlib-off]="$ZFS_ZSTD_ROOT/zlib-off"
)

for case_name in "${!CASE_PATHS[@]}"; do
    if [[ ! -d "${CASE_PATHS[$case_name]}" ]]; then
        printf 'Benchmark directory not found for %s: %s\n' "$case_name" "${CASE_PATHS[$case_name]}" >&2
        exit 1
    fi
done

mkdir -p "$RESULT_ROOT"
RESULTS="$RESULT_ROOT/results.tsv"
printf 'trial\tcase\tcache_mode\tfile\topen_seconds\tload_seconds\ttotal_seconds\tmax_rss_kb\tlogical_bytes\tallocated_kb\n' > "$RESULTS"

read_file() {
    local trial=$1
    local case_name=$2
    local file_type=$3
    local file_path=$4
    local metrics
    local open_seconds load_seconds total_seconds max_rss_kb

    metrics=$(
        "$PYTHON" - "$file_path" <<'PY'
import resource
import sys
import time

import xarray as xr

path = sys.argv[1]
started = time.perf_counter()
dataset = xr.open_dataset(path)
opened = time.perf_counter()
dataset.load()
loaded = time.perf_counter()
dataset.close()
usage = resource.getrusage(resource.RUSAGE_SELF)
print(f"{opened - started:.3f}\t{loaded - opened:.3f}\t{loaded - started:.3f}\t{usage.ru_maxrss}")
PY
    )
    IFS=$'\t' read -r open_seconds load_seconds total_seconds max_rss_kb <<< "$metrics"
    logical_bytes=$(stat -c '%s' "$file_path")
    allocated_kb=$(du -sk "$file_path" | cut -f1)
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$trial" "$case_name" "$CACHE_MODE" "$file_type" "$open_seconds" "$load_seconds" \
        "$total_seconds" "$max_rss_kb" "$logical_bytes" "$allocated_kb" >> "$RESULTS"
    printf '%s trial=%s cache=%s file=%s open=%ss load=%ss total=%ss rss=%sKB logical=%sB allocated=%sKB\n' \
        "$case_name" "$trial" "$CACHE_MODE" "$file_type" "$open_seconds" "$load_seconds" \
        "$total_seconds" "$max_rss_kb" "$logical_bytes" "$allocated_kb"
}

run_case() {
    local trial=$1
    local case_name=$2
    local root=${CASE_PATHS[$case_name]}
    local daily_file
    local ctp_file

    daily_file=$(printf '%s\n' "$root"/daily_basis_variables/*.nc)
    ctp_file=$(printf '%s\n' "$root"/ctp_indicator_variables/*.nc)
    [[ -f "$daily_file" ]] || { printf 'Daily NetCDF file not found: %s\n' "$daily_file" >&2; exit 1; }
    [[ -f "$ctp_file" ]] || { printf 'CTP NetCDF file not found: %s\n' "$ctp_file" >&2; exit 1; }

    read_file "$trial" "$case_name" daily "$daily_file"
    read_file "$trial" "$case_name" ctp "$ctp_file"
}

printf 'Running %s read trial(s), cache mode: %s (no cache invalidation)\n' \
    "$READ_TRIALS" "$CACHE_MODE"
for ((trial = 1; trial <= READ_TRIALS; trial++)); do
    if ((trial % 2 == 1)); then
        order=(data-zlib-on data-zlib-off tmp-zlib-on tmp-zlib-off \
               zfs-lz4-zlib-on zfs-lz4-zlib-off zfs-zstd-zlib-on zfs-zstd-zlib-off)
    else
        order=(zfs-zstd-zlib-off zfs-zstd-zlib-on zfs-lz4-zlib-off zfs-lz4-zlib-on \
               tmp-zlib-off tmp-zlib-on data-zlib-off data-zlib-on)
    fi
    for case_name in "${order[@]}"; do
        run_case "$trial" "$case_name"
    done
done

printf '\nResults saved to: %s\n' "$RESULTS"
