#!/usr/bin/env bash

set -Eeuo pipefail

export PATH='/usr/local/bin:/usr/bin:/bin'

SOURCE_ROOT='https://public.hub.geosphere.at/datahub/resources/spartacus-v2-1d-1km/filelisting'
DEST_DIR='/data/reloclim/backup/ZAMG_SPARTACUS/data/current'
LOCK_FILE='/tmp/update_spartacus_current.lock'
JANUARY_OVERLAP_DAYS=7

mkdir -p "$DEST_DIR"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    printf '%s\n' 'Another SPARTACUS update is already running; exiting.'
    exit 0
fi

current_year=$(date +%Y)
current_month=$(date +%m)
current_day=$(date +%d)
years=("$current_year")
if [[ "$current_month" == '01' && 10#$current_day -le "$JANUARY_OVERLAP_DAYS" ]]; then
    years+=("$((current_year - 1))")
fi

update_file() {
    local parameter=$1
    local year=$2
    local filename="SPARTACUS2-DAILY_${parameter}_${year}.nc"
    local url="${SOURCE_ROOT}/${parameter}/${filename}"
    local destination="${DEST_DIR}/${filename}"
    local headers temporary remote_modified remote_epoch local_epoch

    headers=$(mktemp)
    temporary=''
    trap 'rm -f -- "$headers" "$temporary"' RETURN

    printf 'Checking %s\n' "$url"
    if ! curl --fail --silent --show-error --location --head \
        --retry 3 --retry-delay 5 --connect-timeout 30 --max-time 120 \
        --dump-header "$headers" --output /dev/null "$url"; then
        printf 'Unable to check %s\n' "$url" >&2
        return 1
    fi

    remote_modified=$(awk '
        tolower($0) ~ /^last-modified:/ {
            value = $0
            sub(/^[^:]*:[[:space:]]*/, "", value)
            sub(/[[:space:]]*$/, "", value)
        }
        END { print value }' "$headers")

    if [[ -z "$remote_modified" ]]; then
        if [[ -e "$destination" ]]; then
            printf 'No Last-Modified header for %s; leaving existing file unchanged.\n' "$filename" >&2
            return 1
        fi
        printf 'No Last-Modified header for %s; downloading because the local file is missing.\n' "$filename"
    elif [[ -e "$destination" ]]; then
        remote_epoch=$(date --date="$remote_modified" +%s)
        local_epoch=$(stat --format='%Y' "$destination")
        if (( remote_epoch <= local_epoch )); then
            printf '%s is current (remote %s, local %s).\n' \
                "$filename" "$remote_modified" "$(date --date="@$local_epoch" --iso-8601=seconds)"
            return 0
        fi
        printf '%s is newer remotely (%s); downloading.\n' "$filename" "$remote_modified"
    else
        printf '%s is missing locally; downloading.\n' "$filename"
    fi

    temporary=$(mktemp "${DEST_DIR}/.${filename}.tmp.XXXXXX")
    if ! curl --fail --silent --show-error --location \
        --retry 3 --retry-delay 5 --connect-timeout 30 --max-time 0 \
        --output "$temporary" "$url"; then
        printf 'Download failed for %s\n' "$filename" >&2
        return 1
    fi
    if [[ ! -s "$temporary" ]]; then
        printf 'Downloaded file is empty: %s\n' "$filename" >&2
        return 1
    fi
    if [[ -n "$remote_modified" ]]; then
        touch --date="$remote_modified" "$temporary"
    fi
    mv --force -- "$temporary" "$destination"
    temporary=''
    chgrp reloclim "$destination"
    chmod 664 "$destination"
    printf 'Updated %s\n' "$destination"
}

failures=0
for year in "${years[@]}"; do
    for parameter in TN TX; do
        if ! update_file "$parameter" "$year"; then
            failures=$((failures + 1))
        fi
    done
done

if (( failures > 0 )); then
    printf '%d SPARTACUS file update(s) failed.\n' "$failures" >&2
    exit 1
fi

printf 'SPARTACUS update check completed successfully.\n'
