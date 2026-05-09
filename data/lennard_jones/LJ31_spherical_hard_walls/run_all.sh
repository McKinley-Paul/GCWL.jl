#!/bin/bash
# Run all LJ31 spherical hard-wall simulations one at a time.
#
# Usage:              bash run_all.sh
# Keep Mac awake:     caffeinate -s bash run_all.sh
# Background + awake: caffeinate -s bash run_all.sh &

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATUS_LOG="$SCRIPT_DIR/run_all_status.txt"

format_duration() {
    local total=$1
    local h=$((total / 3600))
    local m=$(( (total % 3600) / 60 ))
    local s=$((total % 60))
    if   [ "$h" -gt 0 ]; then printf "%dh %02dm %02ds" "$h" "$m" "$s"
    elif [ "$m" -gt 0 ]; then printf "%dm %02ds" "$m" "$s"
    else                      printf "%ds" "$s"
    fi
}

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg"
    echo "$msg" >> "$STATUS_LOG"
}

run_job() {
    local run_dir="$1"
    local label
    label="$(basename "$run_dir")"
    local start_epoch
    start_epoch=$(date +%s)
    local start_fmt
    start_fmt=$(date '+%Y-%m-%d %H:%M:%S')

    log "STARTED   $label"

    cd "$run_dir" || { log "ERROR: cannot cd to $run_dir"; return 1; }

    echo "Started:  $start_fmt" > bashoutput.txt
    julia ./main.jl >> julia_out.txt 2>&1

    local end_epoch
    end_epoch=$(date +%s)
    local end_fmt
    end_fmt=$(date '+%Y-%m-%d %H:%M:%S')
    local duration=$(( end_epoch - start_epoch ))

    {
        echo "Finished: $end_fmt"
        echo "Runtime:  $(format_duration "$duration")"
    } >> bashoutput.txt

    log "FINISHED  $label — $(format_duration "$duration")"
}

TEMPS=(0.010 0.015 0.020 0.025 0.027 0.030 0.035 0.040 0.045 0.050)

run_dirs=()
for T in "${TEMPS[@]}"; do
    d="$SCRIPT_DIR/T_${T}"
    [ -f "$d/main.jl" ] && run_dirs+=("$d")
done

total=${#run_dirs[@]}

> "$STATUS_LOG"
log "=== Starting $total jobs (1 at a time) ==="
log "=== Status log: $STATUS_LOG ==="

for run_dir in "${run_dirs[@]}"; do
    run_job "$run_dir"
done

log "=== All $total jobs complete ==="
