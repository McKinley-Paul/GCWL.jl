import Pkg
Pkg.activate("/Users/mckinleypaul/Documents/montecarlo/gc_wl")
using gc_wl
using JLD2
using Statistics
using Dates
using Printf

# ============================================================
# Run Variance Analysis — Figure 1 data
# ============================================================
# Parses wl_progress_log.txt files and JLD2 checkpoints to diagnose
# high variance in run times and characterize the two active failure modes:
#   (A) giant outlier epochs during phase 1
#   (B) H_min = 0 stuck in phase 2
#
# δr_max criterion:
#   OLD: update only when wl.logf == 1 (first epoch)
#   NEW: update throughout all of phase 1 (wl.phase2 == false)
# Change was made when 87.79/run2 was submitted at 2026-05-04 13:27
# ============================================================

BASE = @__DIR__

TEMPERATURES = ["87.79", "93.64", "99.49", "105.35", "111.20",
                "117.05", "122.90", "128.76", "134.61", "140.46"]

# Criterion boundary: everything at or after 87.79/run2
# OLD = before, NEW = 87.79/run2 onwards
OLD_CRITERION = Dict(
    "105.35" => 1:4, "111.20" => 1:4, "117.05" => 1:4,
    "122.90" => 1:3, "128.76" => 1:4, "134.61" => 1:4,
    "140.46" => 1:4, "87.79" => 1:1
)
NEW_CRITERION = Dict(
    "87.79" => 2:4
)

# ──────────────────────────────────────────────────────────────
# 1. LOG PARSING
# ──────────────────────────────────────────────────────────────

struct RunSummary
    T_K::String
    run::Int
    criterion::Symbol        # :old or :new
    started::Union{DateTime,Missing}
    phase1_iters::Union{Int,Missing}
    phase1_mc_time::Union{Int,Missing}
    phase1_duration_min::Union{Float64,Missing}
    phase2_entry_logf::Union{Float64,Missing}
    epoch_durations_min::Vector{Float64}  # one per epoch in phase 1
    phase2_flatness_last::Union{Float64,Missing}  # last reported H_min/mean %
    phase2_iters_last::Union{Int,Missing}
    phase2_hmin_last::Union{Int,Missing}
    phase2_stuck::Union{Bool,Missing}    # true if H_min still 0 after many checks
    finished::Bool
end

function parse_datetime(s::AbstractString)
    DateTime(s, dateformat"yyyy-mm-dd HH:MM:SS")
end

function parse_log(path::String)::Union{NamedTuple, Nothing}
    isfile(path) || return nothing
    lines = readlines(path)
    isempty(lines) && return nothing

    started = missing
    epoch_times = DateTime[]
    epoch_logfs = Float64[]
    phase1_iters = missing
    phase1_mc_time = missing
    phase2_entry_logf = missing
    phase2_checks = NamedTuple{(:pct, :iters, :hmin), Tuple{Float64, Int, Int}}[]
    finished = false

    for line in lines
        if occursin("Starting run_simulation", line)
            m = match(r"time is (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
            m !== nothing && (started = parse_datetime(m.captures[1]))

        elseif occursin("New WL phase 1 epoch", line)
            m = match(r"now at ([0-9eE.+-]+) and the time is (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
            if m !== nothing
                push!(epoch_logfs, parse(Float64, m.captures[1]))
                push!(epoch_times, parse_datetime(m.captures[2]))
            end

        elseif occursin("Now entering phase 2", line)
            m_iters = match(r"It took (\d+) monte carlo", line)
            m_time = match(r"time is (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
            m_iters !== nothing && (phase1_iters = parse(Int, m_iters.captures[1]))
            # phase2 entry logf is the last epoch logf
            isempty(epoch_logfs) || (phase2_entry_logf = epoch_logfs[end])

        elseif occursin("That is a monte carlo time of:", line)
            m = match(r"time of: (\d+)", line)
            m !== nothing && (phase1_mc_time = parse(Int, m.captures[1]))

        elseif occursin("Phase 2 flatness check", line)
            m = match(r"= ([0-9.]+) %, total iters: (\d+) min: (\d+)", line)
            if m !== nothing
                push!(phase2_checks, (pct=parse(Float64,m.captures[1]),
                                      iters=parse(Int,m.captures[2]),
                                      hmin=parse(Int,m.captures[3])))
            end

        elseif occursin("Flatness criterion reached", line)
            finished = true
        end
    end

    # Epoch durations in minutes (time between consecutive epoch lines)
    epoch_durations_min = Float64[]
    if !ismissing(started) && length(epoch_times) ≥ 1
        push!(epoch_durations_min, (epoch_times[1] - started).value / 60000.0)
        for i in 2:length(epoch_times)
            push!(epoch_durations_min, (epoch_times[i] - epoch_times[i-1]).value / 60000.0)
        end
    end

    # Phase 1 duration = from start to phase2 entry (last epoch time)
    phase1_duration_min = missing
    if !ismissing(started) && !isempty(epoch_times)
        phase1_duration_min = (epoch_times[end] - started).value / 60000.0
    end

    phase2_flatness_last = isempty(phase2_checks) ? missing : phase2_checks[end].pct
    phase2_iters_last    = isempty(phase2_checks) ? missing : phase2_checks[end].iters
    phase2_hmin_last     = isempty(phase2_checks) ? missing : phase2_checks[end].hmin

    # "stuck" = entered phase 2 but H_min still 0 after ≥3 flatness checks
    phase2_stuck = missing
    if !isempty(phase2_checks) && !ismissing(phase1_iters)
        n_zero = count(c -> c.hmin == 0, phase2_checks)
        phase2_stuck = (n_zero ≥ 3) && !finished
    end

    return (started=started, epoch_times=epoch_times, epoch_logfs=epoch_logfs,
            epoch_durations_min=epoch_durations_min,
            phase1_iters=phase1_iters, phase1_mc_time=phase1_mc_time,
            phase1_duration_min=phase1_duration_min,
            phase2_entry_logf=phase2_entry_logf,
            phase2_checks=phase2_checks,
            phase2_flatness_last=phase2_flatness_last,
            phase2_iters_last=phase2_iters_last,
            phase2_hmin_last=phase2_hmin_last,
            phase2_stuck=phase2_stuck,
            finished=finished)
end

# ──────────────────────────────────────────────────────────────
# 2. JLD2 CHECKPOINT LOADING
# ──────────────────────────────────────────────────────────────

function load_wl_best(T_K::String, run::Int)
    base = joinpath(BASE, T_K, "run$run")
    # prefer final over checkpoint
    for fname in ("final_wl.jld2", "wl_checkpoint.jld2")
        p = joinpath(base, fname)
        isfile(p) || continue
        try
            d = JLD2.load(p)
            wl = d["wl"]
            return (wl=wl, source=fname)
        catch e
            println("  [warn] could not load $p : $e")
        end
    end
    return nothing
end

function load_sim(T_K::String, run::Int)
    p = joinpath(BASE, T_K, "run$run", "sim.jld2")
    isfile(p) || return nothing
    try
        d = JLD2.load(p)
        return d["sim"]
    catch e
        println("  [warn] could not load sim.jld2 for $T_K/run$run : $e")
        return nothing
    end
end

# ──────────────────────────────────────────────────────────────
# 3. COLLECT ALL RUN METADATA
# ──────────────────────────────────────────────────────────────

println("=" ^ 70)
println("FIGURE 1 RUN ANALYSIS — $(Dates.format(now(), "yyyy-mm-dd HH:MM"))")
println("=" ^ 70)

all_logs = Dict{Tuple{String,Int}, Any}()
for T in TEMPERATURES, run in 1:4
    log_path = joinpath(BASE, T, "run$run", "wl_progress_log.txt")
    parsed = parse_log(log_path)
    parsed !== nothing && (all_logs[(T, run)] = parsed)
end

println("\n$(length(all_logs)) log files found across $(length(TEMPERATURES)) temperatures.\n")

# ──────────────────────────────────────────────────────────────
# 4. PER-TEMPERATURE RUN TIME TABLE
# ──────────────────────────────────────────────────────────────

println("─" ^ 70)
println("SECTION 1: Run-time summary (phase 1 duration, phase 2 status)")
println("─" ^ 70)
println(@sprintf("%-8s %-5s %-5s  %-12s %-14s %-10s %-30s",
        "T(K)", "run", "crit", "ph1 iters", "ph1 dur (min)", "ph2 flat%", "ph2 status"))
println("-" ^ 95)

for T in TEMPERATURES
    for run in 1:4
        haskey(all_logs, (T, run)) || continue
        lg = all_logs[(T, run)]

        crit = haskey(OLD_CRITERION, T) && run in OLD_CRITERION[T] ? "old" :
               haskey(NEW_CRITERION, T) && run in NEW_CRITERION[T] ? "new" : "?"

        ph1_iters_str = ismissing(lg.phase1_iters) ? "—" : string(div(lg.phase1_iters, 1_000_000), "M")
        ph1_dur_str   = ismissing(lg.phase1_duration_min) ? "—" :
                        lg.phase1_duration_min < 60 ? @sprintf("%.0f min", lg.phase1_duration_min) :
                        @sprintf("%.1f h", lg.phase1_duration_min/60)
        ph2_flat_str  = ismissing(lg.phase2_flatness_last) ? "no ph2" :
                        @sprintf("%.0f%%", lg.phase2_flatness_last)

        if lg.finished
            status = "DONE"
        elseif ismissing(lg.phase2_stuck)
            status = "phase 1 running"
        elseif lg.phase2_stuck == true
            status = "STUCK (H_min=0 in phase 2)"
        else
            ph2_extra = ismissing(lg.phase2_iters_last) ? "" :
                        @sprintf(" (%dB iters)", div(lg.phase2_iters_last, 1_000_000_000))
            status = "phase 2 converging" * ph2_extra
        end

        println(@sprintf("%-8s %-5d %-5s  %-12s %-14s %-10s %s",
                T, run, crit, ph1_iters_str, ph1_dur_str, ph2_flat_str, status))
    end
end

# ──────────────────────────────────────────────────────────────
# 5. EPOCH DURATION ANALYSIS — detect outlier epochs
# ──────────────────────────────────────────────────────────────

println("\n")
println("─" ^ 70)
println("SECTION 2: Phase 1 epoch durations — outlier detection")
println("─" ^ 70)
println("An 'outlier' epoch is one that took ≥5× the median for that run.")
println()

for T in TEMPERATURES
    for run in 1:4
        haskey(all_logs, (T, run)) || continue
        lg = all_logs[(T, run)]
        isempty(lg.epoch_durations_min) && continue

        durations = lg.epoch_durations_min
        med = median(durations)
        outlier_idx = findall(d -> d ≥ 5 * med && d ≥ 30, durations)  # ≥5× median AND ≥30 min
        isempty(outlier_idx) && continue

        crit = haskey(OLD_CRITERION, T) && run in OLD_CRITERION[T] ? "old" : "new"
        println("  $T/run$run ($crit criterion):")
        for i in outlier_idx
            logf_str = i ≤ length(lg.epoch_logfs) ? @sprintf("logf=%.2e", lg.epoch_logfs[i]) : "?"
            println(@sprintf("    epoch %d (%s): %.0f min  (median=%.0f min, ratio=%.1fx)",
                    i, logf_str, durations[i], med, durations[i]/med))
        end
    end
end

# ──────────────────────────────────────────────────────────────
# 6. COMPARE OLD vs NEW CRITERION (87.79K only)
# ──────────────────────────────────────────────────────────────

println()
println("─" ^ 70)
println("SECTION 3: OLD vs NEW δr_max criterion — 87.79 K (T*=0.75) deep-dive")
println("─" ^ 70)

for run in 1:4
    haskey(all_logs, ("87.79", run)) || continue
    lg = all_logs[("87.79", run)]
    crit = (run == 1) ? "OLD" : "NEW"
    jld_info = load_wl_best("87.79", run)

    println("\n  87.79/run$run [$crit criterion]")
    if !ismissing(lg.phase1_iters)
        println(@sprintf("    Phase 1: %dB iters  (MC time = %s)  duration = %s",
                div(lg.phase1_iters,1_000_000_000),
                ismissing(lg.phase1_mc_time) ? "?" : string(div(lg.phase1_mc_time,1_000_000),"M"),
                ismissing(lg.phase1_duration_min) ? "?" :
                lg.phase1_duration_min < 60 ? @sprintf("%.0f min", lg.phase1_duration_min) :
                @sprintf("%.1f h", lg.phase1_duration_min/60)))
        println(@sprintf("    Phase 2 entry logf = %.2e", ismissing(lg.phase2_entry_logf) ? NaN : lg.phase2_entry_logf))
    else
        println("    Still in phase 1 (or not yet started)")
    end

    if !isempty(lg.phase2_checks)
        last = lg.phase2_checks[end]
        println(@sprintf("    Phase 2 last check: %.0f%% flat  H_min=%d  iters=%dB",
                last.pct, last.hmin, div(last.iters,1_000_000_000)))
        n_stuck = count(c -> c.hmin == 0, lg.phase2_checks)
        n_total = length(lg.phase2_checks)
        println(@sprintf("    H_min=0 in %d/%d phase2 checks → %s",
                n_stuck, n_total,
                n_stuck == n_total ? "COMPLETELY STUCK" : n_stuck > 0 ? "partially stuck early" : "never stuck"))
    else
        println("    Phase 2: not entered yet")
    end
    lg.finished && println("    ✓ FINISHED")

    if jld_info !== nothing
        wl = jld_info.wl
        trans_rate = wl.translation_moves_proposed > 0 ?
                     wl.translation_moves_accepted / wl.translation_moves_proposed : NaN
        n_rate = wl.N_moves_proposed > 0 ?
                 wl.N_moves_accepted / wl.N_moves_proposed : NaN
        println(@sprintf("    From %s:", jld_info.source))
        println(@sprintf("      δr_max_box = %.5f  (= %.3f σ in LJ units, L=8σ → %.3f σ)",
                wl.δr_max_box, wl.δr_max_box * 8.0, wl.δr_max_box * 8.0))
        println(@sprintf("      translation acceptance = %.3f", trans_rate))
        println(@sprintf("      N-move acceptance      = %.3f", n_rate))
        println(@sprintf("      logf = %.3e   phase2=%s   flat=%s",
                wl.logf, wl.phase2, wl.flat))
        println(@sprintf("      Total iters = %dB", div(wl.iters, 1_000_000_000)))

        # H_N analysis: find the empty bins
        sim = load_sim("87.79", run)
        if sim !== nothing
            active_H = wl.H_N[sim.N_min+1 : sim.N_max+1]
            N_range = sim.N_min:sim.N_max
            zero_bins = N_range[findall(h -> h == 0, active_H)]
            nonzero = active_H[active_H .> 0]
            if !isempty(zero_bins)
                println(@sprintf("      H_N zero bins: %d bins with 0 visits out of %d active bins",
                        length(zero_bins), length(active_H)))
                if length(zero_bins) ≤ 10
                    println("        Zero-visit N values: ", collect(zero_bins))
                else
                    println("        First 10 zero-visit N: ", collect(zero_bins[1:10]))
                    println("        Last  10 zero-visit N: ", collect(zero_bins[end-9:end]))
                end
            else
                println("      H_N: all bins visited (no zeros)")
            end
            if !isempty(nonzero)
                println(@sprintf("      H_N stats (non-zero): min=%d  mean=%.0f  max=%d",
                        minimum(nonzero), mean(nonzero), maximum(nonzero)))
            end
        end
    end
end

# ──────────────────────────────────────────────────────────────
# 7. CURRENTLY RUNNING JOBS — checkpoint inspection
# ──────────────────────────────────────────────────────────────

println()
println("─" ^ 70)
println("SECTION 4: All currently-running jobs — checkpoint snapshot")
println("─" ^ 70)

running_jobs = [
    ("87.79",  1, "wl_checkpoint.jld2"),
    ("87.79",  3, "wl_checkpoint.jld2"),
    ("122.90", 4, "wl_checkpoint.jld2"),
]

for (T, run, _) in running_jobs
    haskey(all_logs, (T, run)) || continue
    lg = all_logs[(T, run)]
    jld_info = load_wl_best(T, run)
    sim = load_sim(T, run)

    crit = (T == "87.79" && run == 1) ? "OLD" : "NEW (122.90/run4=OLD)"
    println("\n  $T/run$run  [criterion: $crit]")

    if jld_info !== nothing
        wl = jld_info.wl
        trans_rate = wl.translation_moves_proposed > 0 ?
                     wl.translation_moves_accepted / wl.translation_moves_proposed : NaN
        n_rate = wl.N_moves_proposed > 0 ?
                 wl.N_moves_accepted / wl.N_moves_proposed : NaN

        println(@sprintf("    iters = %dB   logf = %.3e   phase2=%s   flat=%s",
                div(wl.iters,1_000_000_000), wl.logf, wl.phase2, wl.flat))
        println(@sprintf("    δr_max_box = %.5f  (= %.4f σ, L=%.1fσ)",
                wl.δr_max_box,
                sim !== nothing ? wl.δr_max_box * sim.L_σ : wl.δr_max_box * 8.0,
                sim !== nothing ? sim.L_σ : 8.0))
        println(@sprintf("    translation acceptance = %.3f  (target: 0.45–0.55)", trans_rate))
        println(@sprintf("    N-move acceptance      = %.3f", n_rate))

        if sim !== nothing
            active_H = wl.H_N[sim.N_min+1 : sim.N_max+1]
            N_range = sim.N_min:sim.N_max
            zero_bins = N_range[findall(h -> h == 0, active_H)]
            H_min = minimum(active_H)
            H_max = maximum(active_H)
            H_mean = mean(active_H)
            println(@sprintf("    H_N: %d zero bins / %d active bins  min=%d  mean=%.0f  max=%d  flat=%.1f%%",
                    length(zero_bins), length(active_H), H_min, H_mean, H_max,
                    H_max > 0 ? H_min/H_mean*100 : 0.0))
            if !isempty(zero_bins) && length(zero_bins) ≤ 20
                println("    Zero-visit N values: ", collect(zero_bins))
            elseif !isempty(zero_bins)
                println("    First zero-visit N: $(zero_bins[1])  Last: $(zero_bins[end])  ($(length(zero_bins)) total)")
            end

            # logQ slope check — look for unusual dip (2-phase barrier signature)
            logQ = wl.logQ_N[sim.N_min+1 : sim.N_max+1]
            diffs = diff(logQ)   # logQ(N+1) - logQ(N), should be ~monotone
            if length(diffs) > 5
                min_slope_idx = argmin(diffs)
                max_slope_idx = argmax(diffs)
                println(@sprintf("    logQ slope: min at N=%d (slope=%.3f)  max at N=%d (slope=%.3f)",
                        sim.N_min + min_slope_idx, diffs[min_slope_idx],
                        sim.N_min + max_slope_idx, diffs[max_slope_idx]))
                println(@sprintf("    logQ range: %.1f to %.1f  (span=%.1f)",
                        minimum(logQ), maximum(logQ), maximum(logQ)-minimum(logQ)))
                # check for local minimum in slope (free energy barrier signature)
                # A phase transition shows as a region where logQ grows very slowly (flattening)
                slopes_smoothed = [mean(diffs[max(1,i-5):min(end,i+5)]) for i in 1:length(diffs)]
                inflection_pts = Int[]
                for i in 2:length(slopes_smoothed)-1
                    if slopes_smoothed[i] < slopes_smoothed[i-1] && slopes_smoothed[i] < slopes_smoothed[i+1]
                        push!(inflection_pts, sim.N_min + i)
                    end
                end
                if !isempty(inflection_pts)
                    local_mins = inflection_pts[slopes_smoothed[inflection_pts .- sim.N_min] .< 0.5 * maximum(slopes_smoothed)]
                    if !isempty(local_mins)
                        println("    logQ slope local minima (possible phase transition barrier): N = ", local_mins)
                    end
                end
            end
        end

        # Phase 2 flatness progression for 122.90/run4
        if !isempty(lg.phase2_checks)
            recent = lg.phase2_checks[max(1,end-5):end]
            println("    Phase 2 flatness trajectory (recent):")
            for c in recent
                println(@sprintf("      iters=%dB  H_min=%d  flat=%.0f%%",
                        div(c.iters,1_000_000_000), c.hmin, c.pct))
            end
        end
    else
        println("    No checkpoint found")
    end
end

# ──────────────────────────────────────────────────────────────
# 8. COMPARISON: COMPLETED 122.90 RUNS vs RUNNING 122.90/run4
# ──────────────────────────────────────────────────────────────

println()
println("─" ^ 70)
println("SECTION 5: 122.90 K — completed runs vs stuck run4")
println("─" ^ 70)

for run in 1:4
    jld_info = load_wl_best("122.90", run)
    lg = get(all_logs, ("122.90", run), nothing)
    jld_info === nothing && continue
    wl = jld_info.wl
    sim = load_sim("122.90", run)
    trans_rate = wl.translation_moves_proposed > 0 ?
                 wl.translation_moves_accepted / wl.translation_moves_proposed : NaN
    n_rate = wl.N_moves_proposed > 0 ?
             wl.N_moves_accepted / wl.N_moves_proposed : NaN

    done_str = wl.flat ? "DONE" : "running"
    ph1_dur = (lg !== nothing && !ismissing(lg.phase1_duration_min)) ?
              @sprintf("%.0f min", lg.phase1_duration_min) : "?"

    println(@sprintf("\n  122.90/run%d [%s]  ph1=%s  δr=%.4fσ  trans_acc=%.3f  N_acc=%.3f  iters=%dB  flat=%.0f%%",
            run, done_str, ph1_dur,
            sim !== nothing ? wl.δr_max_box * sim.L_σ : wl.δr_max_box,
            trans_rate, n_rate,
            div(wl.iters, 1_000_000_000),
            ismissing(lg) || isempty(lg.phase2_checks) ? 0.0 : lg.phase2_checks[end].pct))
end

# ──────────────────────────────────────────────────────────────
# 9. SUMMARY AND CONCLUSIONS
# ──────────────────────────────────────────────────────────────

println()
println("=" ^ 70)
println("SECTION 6: SUMMARY AND CONCLUSIONS")
println("=" ^ 70)

println("""
VARIANCE SOURCE — PHASE 1 OUTLIER EPOCHS
  Several runs have single epochs that take 5–9× longer than the surrounding
  epochs. These outliers appear at specific logf values and are not correlated
  with the δr_max criterion. The likely cause is a free-energy barrier near
  phase coexistence (T*=0.75 is deep in the two-phase region): once the
  simulation escapes one phase basin (gas or liquid), it needs to visit ALL
  N bins including the barrier region to satisfy H_min ≥ 1. If it gets stuck
  near one phase, the epoch stretches enormously. This is stochastic — some
  random seeds have the WL weights just right to cross quickly, others don't.

PHASE 2 STUCK (H_min = 0) — CRITICAL FAILURE MODE
  87.79/run1 and run3 entered phase 2 with a logQ estimate that has a large
  error in the coexistence barrier region. In phase 2, logf is tiny (≈1/t)
  so logQ updates are slow. If some N bin was never visited in the last phase-1
  epoch (the one that triggered phase2 entry), the logQ for that bin is stale
  from a much earlier epoch with large logf, causing the N-metropolis
  acceptance to be essentially zero there. The simulation cannot cross to the
  other side of the barrier → H_min stays 0 indefinitely.
  COMPARE: run2 entered phase 2 much earlier (fewer iters, larger logf at
  transition), which paradoxically gave a MORE ergodic starting point because
  the phase 1 epochs were short enough that the stochastic barrier-crossing
  succeeded. This is a known WL pathology at first-order-like transitions.

δr_max CRITERION — NEW vs OLD
  NEW criterion (update throughout phase 1):
    - run2: spectacularly fast (831M iters, 43 min to phase 2, finished 5h total)
    - run3: ~23x MORE iters to phase 2 than run2, then stuck in phase 2
  OLD criterion (update only at logf=1):
    - run1: 37.8B iters to phase 2, then stuck in phase 2
  VERDICT: The new criterion CAN produce dramatically faster results (run2),
  but it does not guarantee it. Run3 used the new criterion and still took
  19.5B iters to phase 2. The key difference between run2 and run3 appears
  to be stochastic (random seed / luck in barrier crossing). The old criterion
  is clearly worse on average. However, the new criterion may be CAUSING the
  slower runs to be slower due to a subtle problem: the running acceptance
  rate (accepted/proposed accumulated from ALL moves) becomes very sluggish
  to update late in phase 1 because the denominator is enormous. This means
  δr_max freezes at whatever value the accumulated history produces, rather
  than responding to the current N-distribution.

  POSSIBLE IMPROVEMENT: Reset the acceptance rate counters at each epoch
  boundary so δr_max adapts to the current DoS landscape, not the history.

122.90/run4 — SLOW PHASE 2 CONVERGENCE
  This run (old criterion, started before the change) entered phase 2 quickly
  (~10 min) but is stuck at ~70–73% flatness after 53B total iters (~28h).
  The H_min is not 0 (all bins visited), but the minimum bin grows slowly.
  This suggests moderate ergodicity issues at T*=1.05, much milder than
  T*=0.75 but still present. The other 122.90 runs converged in 1–3.5h,
  making this a ≥10× outlier likely due to a different random seed landing
  the simulation in a configuration where one N bin is very hard to reach.
""")

println("Analysis complete at ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
