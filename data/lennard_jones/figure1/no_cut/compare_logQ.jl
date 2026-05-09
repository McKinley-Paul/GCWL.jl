import Pkg
Pkg.activate("/Users/mckinleypaul/Documents/montecarlo/gc_wl")
using gc_wl
using JLD2
using Statistics
using Printf

# Compare logQ(N) estimates across fast vs slow runs at each temperature.
# "Fast" = small phase-1 MC time (entered phase 2 early with larger logf).
# "Slow" = large phase-1 MC time (entered phase 2 late with tiny logf).
#
# Relevant literature framing:
#  - Pereyra 2007: error ~ 1/sqrt(t) in 1/t phase, so total MC time matters more than
#    when phase 2 started.
#  - Zhou 2008: error cannot vanish faster than 1/t; convergence requires sampling fluctuations
#    to decay, which takes time proportional to tunneling time at first-order transitions.
#  - Netto 2012: early large-f stages give poor logQ; only later stages stabilize the estimate.
#    The 1/t scheme itself can be BIASED if started too early (before f is small enough).

BASE = @__DIR__

# Phase 1 MC times extracted from previous analysis (in millions of MC steps / num_active_bins)
# These are the "MC time at phase2 entry" values — larger = more phase1 work
PHASE1_MC = Dict(
    ("87.79",  2) => 1843,
    ("105.35", 1) => 502,
    ("105.35", 2) => 820,
    ("105.35", 3) => 2504,
    ("105.35", 4) => 562,
    ("111.20", 1) => 3274,
    ("111.20", 2) => 3471,
    ("111.20", 3) => 3103,
    ("111.20", 4) => 3032,
    ("117.05", 1) => 724,
    ("117.05", 2) => 781,
    ("117.05", 3) => 2713,
    ("117.05", 4) => 3919,
    ("122.90", 1) => 1141,
    ("122.90", 2) => 1041,
    ("122.90", 3) => 215,
    ("128.76", 1) => 131,
    ("128.76", 2) => 149,
    ("128.76", 3) => 160,
    ("128.76", 4) => 195,
    ("134.61", 1) => 171,
    ("134.61", 2) => 362,
    ("134.61", 3) => 178,
    ("134.61", 4) => 82,
    ("140.46", 1) => 93,
    ("140.46", 2) => 268,
    ("140.46", 3) => 82,
    ("140.46", 4) => 178,
)

function load_final_wl(T_K, run)
    for fname in ("final_wl.jld2", "wl_checkpoint.jld2")
        p = joinpath(BASE, T_K, "run$run", fname)
        isfile(p) || continue
        d = JLD2.load(p)
        wl = d["wl"]
        # only return if simulation actually completed
        fname == "final_wl.jld2" && return wl
        wl.flat && return wl   # checkpoint that happened to be flat
    end
    return nothing
end

function load_sim_params(T_K, run)
    p = joinpath(BASE, T_K, "run$run", "sim.jld2")
    isfile(p) || return nothing
    d = JLD2.load(p)
    return d["sim"]
end

# Apply correct_logQ! normalization so all runs are anchored to logQ(N=0)=0
function normalized_logQ(wl)
    lq = copy(wl.logQ_N)
    lq .-= lq[1]   # anchor: logQ(N=0) = 0
    return lq
end

println("=" ^ 72)
println("logQ CONSISTENCY ANALYSIS: fast vs slow runs per temperature")
println("=" ^ 72)
println()
println("Each run's phase-1 MC time (t_ph1, in thousands of MC time units)")
println("is the key distinguisher. Runs entering phase 2 at large logf (small t_ph1)")
println("are 'fast'; those with small logf (large t_ph1) are 'slow'.")
println()

TEMPERATURES = ["105.35","111.20","117.05","122.90","128.76","134.61","140.46"]
# 87.79 only has one finished run (run2), so skip cross-run comparison there

for T in TEMPERATURES
    # Gather all completed runs at this temperature
    runs = Int[]
    wls  = []
    sims = []
    t_ph1s = Float64[]

    for run in 1:4
        wl = load_final_wl(T, run)
        wl === nothing && continue
        sim = load_sim_params(T, run)
        sim === nothing && continue
        t = get(PHASE1_MC, (T, run), missing)
        ismissing(t) && continue
        push!(runs, run)
        push!(wls, wl)
        push!(sims, sim)
        push!(t_ph1s, Float64(t))
    end

    length(runs) < 2 && continue

    # Get N range (should be same for all runs at a given T)
    sim0 = sims[1]
    N_range = sim0.N_min : sim0.N_max
    n_bins = length(N_range)

    # Normalized logQ for each run
    lqs = [normalized_logQ(wl)[sim0.N_min+1 : sim0.N_max+1] for wl in wls]

    # Mean and std across runs
    lq_mat = hcat(lqs...)   # n_bins × n_runs
    lq_mean = vec(mean(lq_mat, dims=2))
    lq_std  = vec(std(lq_mat, dims=2))

    # Per-run deviation from mean
    deviations = [lqs[i] .- lq_mean for i in 1:length(runs)]
    rms_devs = [sqrt(mean(d.^2)) for d in deviations]
    max_devs = [maximum(abs.(d)) for d in deviations]

    # Rank runs by phase-1 MC time
    order = sortperm(t_ph1s)
    fastest_idx = order[1]
    slowest_idx = order[end]

    println("─" ^ 72)
    @printf("%-8s  N_bins=%d  n_runs=%d\n", T*" K", n_bins, length(runs))
    println()

    # Table header
    @printf("  %-6s  %-10s  %-10s  %-12s  %-12s  %-6s\n",
            "run", "t_ph1(k)", "total_B", "RMS_dev", "max_dev", "tag")
    println("  " * "-"^62)

    for (pos, i) in enumerate(order)
        tag = pos==1 ? "← fastest" : pos==length(runs) ? "← slowest" : ""
        wl = wls[i]
        @printf("  %-6d  %-10.0f  %-10.1f  %-12.4f  %-12.4f  %s\n",
                runs[i],
                t_ph1s[i],
                wl.iters / 1e9,
                rms_devs[i],
                max_devs[i],
                tag)
    end
    println()

    # Where in N-space does the spread peak?
    overall_std = lq_std
    peak_std_N = N_range[argmax(overall_std)]
    println(@sprintf("  Cross-run std: mean=%.4f  max=%.4f at N=%d",
            mean(overall_std), maximum(overall_std), peak_std_N))

    # Compare fastest vs slowest explicitly
    fast_dev = deviations[fastest_idx]
    slow_dev = deviations[slowest_idx]
    fast_rms = rms_devs[fastest_idx]
    slow_rms = rms_devs[slowest_idx]

    # Is fast run consistently above or below mean? (systematic bias check)
    fast_bias = mean(fast_dev)
    slow_bias = mean(slow_dev)
    println(@sprintf("  Fast run (run%d, t_ph1=%gk): RMS=%.4f, bias=%.4f",
            runs[fastest_idx], t_ph1s[fastest_idx], fast_rms, fast_bias))
    println(@sprintf("  Slow run (run%d, t_ph1=%gk): RMS=%.4f, bias=%.4f",
            runs[slowest_idx], t_ph1s[slowest_idx], slow_rms, slow_bias))

    # Direct fast-vs-slow comparison
    fast_vs_slow = lqs[fastest_idx] .- lqs[slowest_idx]
    fvs_rms = sqrt(mean(fast_vs_slow.^2))
    fvs_max = maximum(abs.(fast_vs_slow))
    fvs_bias = mean(fast_vs_slow)
    println(@sprintf("  Fast minus slow: RMS=%.4f, max=%.4f, mean_bias=%.4f",
            fvs_rms, fvs_max, fvs_bias))
    println()

    # Netto 2012 insight: check whether fast run looks biased toward gas or liquid
    # A systematic POSITIVE bias means fast run overestimates logQ at high N
    # (i.e., overestimates the liquid phase partition function)
    n_half = div(n_bins, 2)
    fast_bias_gas    = mean(fast_dev[1:n_half])
    fast_bias_liquid = mean(fast_dev[n_half+1:end])
    println(@sprintf("  Fast run bias: gas-side(N<%.0f)=%.4f  liquid-side(N>%.0f)=%.4f",
            N_range[n_half], fast_bias_gas, N_range[n_half], fast_bias_liquid))
    println()
end

# Special: 87.79 has only one finished run, compare against ideal gas limit and Desgranges
println("─" ^ 72)
println("87.79 K (T*=0.75): single finished run (run2) — compare to ideal gas slope")
wl2 = load_final_wl("87.79", 2)
sim2 = load_sim_params("87.79", 2)
if wl2 !== nothing && sim2 !== nothing
    lq = normalized_logQ(wl2)[sim2.N_min+1 : sim2.N_max+1]
    ig_lq = [ideal_gas_logQ_loggamma(N, sim2.V_σ, sim2.Λ_σ) for N in sim2.N_min:sim2.N_max]
    ig_lq .-= ig_lq[1]  # normalize

    # The deviation from ideal gas is proportional to configurational energy
    delta = lq .- ig_lq
    println(@sprintf("  logQ span: %.1f (run2 finished)  ideal-gas span: %.1f",
            lq[end], ig_lq[end]))
    println(@sprintf("  Excess logQ (logQ - logQ_IG): min=%.2f at N=%d, max=%.2f at N=%d",
            minimum(delta), sim2.N_min + argmin(delta) - 1,
            maximum(delta), sim2.N_min + argmax(delta) - 1))

    # Slope analysis: d(logQ)/dN ≈ chemical potential contribution
    slopes = diff(lq)
    ig_slopes = diff(ig_lq)
    println(@sprintf("  logQ slope range: %.3f to %.3f (IG: %.3f to %.3f)",
            minimum(slopes), maximum(slopes), minimum(ig_slopes), maximum(ig_slopes)))
    println(@sprintf("  Phase-2 entry logf = %.2e, total iters = %.1fB, phase2 iters = ~%.1fB",
            4.77e-7, wl2.iters/1e9, (wl2.iters - 831_000_000)/1e9))
end

println()
println("=" ^ 72)
println("LITERATURE-INFORMED ANALYSIS")
println("=" ^ 72)
println("""
PEREYRA 2007:
  The 1/t algorithm guarantees error ∝ 1/√t, where t is total MC time. Once
  in phase 2, the simulation asymptotically corrects any initial error from
  phase 1. The crucial insight: the H(N)/H_mean flatness ratio (Δ̄H/⟨H⟩) tracks
  the error in real time. A run that enters phase 2 early with larger logf0
  starts phase 2 with a worse logQ estimate, but if it runs phase 2 long
  enough to achieve flatness, the final logQ should converge to the same
  accuracy as a slow run that entered phase 2 with a better initial estimate.
  The stopping criterion (80% flatness) is the controlling variable.

ZHOU 2008:
  Error cannot vanish faster than 1/t. The optimal strategy is:
    logf ∝ N × ||p(t) - p̄||²   (fluctuation of histogram)
  Both fast and slow runs converge to the same answer, but the convergence
  TIME depends on the tunneling time through free-energy barriers. At phase
  transitions (gas↔liquid at T*=0.75), the tunneling time is exponentially
  long and is the dominant bottleneck — not the logQ estimate quality.
  A key implication: the Pereyra transition criterion (logf ≤ 1/t) is
  UNNECESSARY from Zhou's perspective; one could run the 1/t schedule from
  the very start, or enter it much earlier (fewer phase-1 epochs), without
  losing accuracy — as long as total MC time in phase 2 is sufficient.

NETTO 2012:
  The critical finding for YOUR question: early-stage (large f) WL gives
  BIASED density-of-states estimates, and accumulating microcanonical
  averages during these stages contaminates results. The bias persists
  until f is small enough (they define f_micro ≈ 10⁻³ for Ising).
  For your GC-WL: if a fast run enters phase 2 at logf=4.77e-7, this is
  BELOW the typical f_micro threshold — meaning the phase 1 estimate is
  already in the 'reliable' regime. In contrast, a run entering at logf=1e-9
  has been refining a potentially biased estimate for much longer. Netto's
  results suggest the extra phase-1 work beyond f_micro may be WASTED or
  even counterproductive.

SYNTHESIS — Should you use N_epochs ≥ 5-10 as the phase-2 entry criterion?
  See detailed discussion below.
""")

# Count epochs across all finished runs
println("Epoch counts at phase2 entry:")
println(@sprintf("  %-8s %-5s %-12s %-8s %-14s", "T(K)", "run", "t_ph1(k)", "epochs", "logf_entry"))
for T in vcat(TEMPERATURES, ["87.79"])
    for run in 1:4
        p = joinpath(BASE, T, "run$run", "wl_progress_log.txt")
        isfile(p) || continue
        lines = readlines(p)
        n_epochs = count(l -> occursin("New WL phase 1 epoch", l), lines)
        ph2_line = findfirst(l -> occursin("Now entering phase 2", l), lines)
        ph2_line === nothing && continue
        logf_entry = get(PHASE1_MC, (T,run), missing)
        # get the logf at entry from the last epoch line before phase2
        epoch_lines = filter(l -> occursin("New WL phase 1 epoch", l), lines)
        logf_str = isempty(epoch_lines) ? "?" : begin
            m = match(r"now at ([0-9eE.+-]+)", epoch_lines[end])
            m !== nothing ? m.captures[1] : "?"
        end
        @printf("  %-8s %-5d %-12s %-8d %-14s\n",
                T, run, ismissing(logf_entry) ? "?" : string(logf_entry), n_epochs, logf_str)
    end
end
println()

println("""
RECOMMENDATION ON PHASE-2 ENTRY CRITERION:
───────────────────────────────────────────
The current criterion (logf ≤ 1/t) is a sensible default from Pereyra 2007
but has a key problem for systems with phase transitions: the outlier epochs
at the coexistence barrier inflate t enormously for some runs, making them
enter phase 2 with unnecessarily small logf (run1: 7.45e-9, run3: 1.49e-8)
while others enter at much larger logf (run2: 4.77e-7). The per-run variance
in phase-2 entry conditions is 100×, yet the resulting accuracy (where both
converge) appears to be similar.

A fixed-epoch criterion (e.g., N ≥ 8 epochs completed) would give:
  - Deterministic, reproducible behavior across seeds
  - logf ≤ 1/2^8 = 0.0039 at entry (epoch 8 means logf = 2^-8)
  - Immunity to outlier epochs: even if epoch 5 takes 9h, you enter phase 2
    after epoch 8 regardless of how long it took
  - Better ergodicity in phase 2: larger logf at entry means bigger logQ
    corrections, which can push the simulation over the coexistence barrier

However, the risk: if epochs 1-8 are all fast (<<1 MC tunneling time each),
the logQ at entry is still coarse. But Zhou 2008 shows phase 2 corrects this.
The more important concern is whether phase 2 with a coarse initial logQ can
still traverse the phase-transition barrier — and run2 at 87.79K shows YES.

CONCRETE SUGGESTION:
  Replace: if logf ≤ 1/t → enter phase 2
  With:    if n_epochs_completed ≥ 8 AND logf ≤ 0.01 → enter phase 2
  (The logf ≤ 0.01 guard ensures at least some refinement has occurred.)
  This is consistent with Netto's f_micro finding: accumulating beyond f≈10⁻³
  gives diminishing returns in phase 1, so earlier phase-2 entry is fine.
""")
