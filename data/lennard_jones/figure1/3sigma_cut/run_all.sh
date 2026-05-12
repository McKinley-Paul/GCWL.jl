#!/bin/bash
# Run all 3sigma_cut simulations with up to 4 jobs running in parallel at all times.
# As soon as one finishes the next queued job starts.
#
# Usage:              bash run_all.sh
# Keep Mac awake:     caffeinate -s bash run_all.sh
# Background + awake: caffeinate -s bash run_all.sh &
# Resume:             re-run the same command; jobs with "Finished:" in
#                     bashoutput.txt are skipped automatically.

MAX_JOBS=4
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
    label="$(basename "$(dirname "$run_dir")")/$(basename "$run_dir")"
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

# Collect run directories in ascending temperature order
TEMPS=(93.64 99.49 105.35 111.20 117.05 122.90 128.76 134.61 140.46)
RUNS=(1 2 3 4)

run_dirs=()
for T in "${TEMPS[@]}"; do
    for R in "${RUNS[@]}"; do
        d="$SCRIPT_DIR/$T/run$R"
        [ -f "$d/main.jl" ] && run_dirs+=("$d")
    done
done

total=${#run_dirs[@]}
skipped=0
queued=()

for d in "${run_dirs[@]}"; do
    if grep -q "^Finished:" "$d/bashoutput.txt" 2>/dev/null; then
        label="$(basename "$(dirname "$d")")/$(basename "$d")"
        log "SKIPPED   $label (already finished)"
        (( skipped++ ))
    else
        queued+=("$d")
    fi
done

log "=== Starting/resuming: $total total, $skipped skipped, ${#queued[@]} to run (max $MAX_JOBS concurrent) ==="
log "=== Status log: $STATUS_LOG ==="

active_pids=()

for run_dir in "${queued[@]}"; do
    # Wait until a slot is free
    while [ "${#active_pids[@]}" -ge "$MAX_JOBS" ]; do
        remaining=()
        for pid in "${active_pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                remaining+=("$pid")
            fi
        done
        active_pids=("${remaining[@]}")
        if [ "${#active_pids[@]}" -ge "$MAX_JOBS" ]; then
            sleep 15
        fi
    done

    run_job "$run_dir" &
    active_pids+=($!)
done

wait
log "=== All jobs complete ==="
