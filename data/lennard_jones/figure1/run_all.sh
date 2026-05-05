#!/bin/bash
# Run all figure1 simulations 3 at a time.
#
# Usage:              bash run_all.sh
# Keep Mac awake:     caffeinate -s bash run_all.sh
# Background + awake: caffeinate -s bash run_all.sh &

MAX_JOBS=3
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

# Collect all run directories, sorted by temperature then run number
run_dirs=()
while IFS= read -r d; do
    run_dirs+=("$d")
done < <(find "$SCRIPT_DIR" -name "main.jl" | sed 's|/main\.jl$||' | sort)

total=${#run_dirs[@]}

# Clear / start the status log
> "$STATUS_LOG"
log "=== Starting $total jobs (max $MAX_JOBS concurrent) ==="
log "=== Status log: $STATUS_LOG ==="

active_pids=()

for run_dir in "${run_dirs[@]}"; do
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

# Wait for all remaining jobs
wait
log "=== All $total jobs complete ==="
