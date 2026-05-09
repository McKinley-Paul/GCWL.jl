import Pkg
Pkg.activate("/Users/mckinleypaul/Documents/montecarlo/gc_wl")
using gc_wl
using Random
using Statistics
using Printf
using JLD2
using Dates

const T_σ     = 0.010
const Λ_σ     = 1.0
const L_σ     = 10.0    # Rc = 5σ, sphere inscribed in box with L = 2Rc
const N_max   = 31
const N_min   = 0
const n_runs  = 4
const maxiter = 10^14

const output_dir = @__DIR__
const log_path   = joinpath(output_dir, "wl_progress_log.txt")
const n_bins     = N_max + 1   # entries for N = 0..N_max

logQ_store = zeros(Float64, n_bins, n_runs)

t_start = time()
for ii in 1:n_runs
    sim = SimulationParams(
        N_max = N_max, N_min = N_min,
        T_σ = T_σ, Λ_σ = Λ_σ, L_σ = L_σ,
        r_cut_σ = L_σ,           # sphere diameter — no truncation within sphere
        save_directory_path = output_dir,
        maxiter = maxiter,
        rng = MersenneTwister(ii),
    )
    μstate = init_microstate(sim)   # N=0 vacuum start (required for hard-wall sphere)
    wl     = init_WangLandauVars(sim)
    cache  = init_cache(sim, μstate)

    if ii == 1
        initialization_check(sim, μstate, wl)
    end

    hws_run_simulation!(sim, μstate, wl, cache)
    correct_logQ!(wl)

    logQ_store[:, ii] = wl.logQ_N

    rm(log_path; force = true)

    elapsed = time() - t_start
    println("Run $ii / $n_runs — elapsed $(round(elapsed/60, digits=1)) min  ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
    flush(stdout)
end

total_runtime = time() - t_start
println("All $n_runs runs completed in $(round(total_runtime/60, digits=1)) min")
flush(stdout)

logQ_mean = vec(mean(logQ_store; dims=2))
logQ_std  = vec(std(logQ_store;  dims=2))
logQ_sem  = logQ_std ./ sqrt(n_runs)

JLD2.jldsave(joinpath(output_dir, "results_$(n_runs)runs.jld2");
    T_σ             = T_σ,
    n_runs          = n_runs,
    runtime_seconds = total_runtime,
    logQ_store      = logQ_store,
    logQ_mean       = logQ_mean,
    logQ_std        = logQ_std,
    logQ_sem        = logQ_sem,
)

println("Saved results to $(joinpath(output_dir, "results_$(n_runs)runs.jld2"))")

println()
println("═"^60)
@printf("SUMMARY — T* = %.3f, %d runs, %.1f min\n", T_σ, n_runs, total_runtime/60)
println("═"^60)
println("\n── logQ mean ± std ──")
@printf("  %3s │ %14s │ %14s\n", "N", "logQ mean", "logQ std")
println("  ────┼────────────────┼────────────────")
for N in 0:N_max
    @printf("  %3d │ %14.6f │ %14.6f\n", N, logQ_mean[N+1], logQ_std[N+1])
end
println("\nDone.")
