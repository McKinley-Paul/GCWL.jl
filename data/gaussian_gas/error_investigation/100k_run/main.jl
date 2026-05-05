import Pkg
Pkg.activate("/Users/mckinleypaul/Documents/montecarlo/gc_wl")
using gc_wl
using Random
using Statistics
using LinearAlgebra
using SpecialFunctions
using Printf
using JLD2
using Dates 


# ═══════════════════════════════════════════════════════════════════════════
#  100,000-run campaign: per-run statistics + correlation analysis + Jensen
#  gap analysis (bn/Bn from mean logQ vs mean of per-run bn/Bn).
# ═══════════════════════════════════════════════════════════════════════════

# ─────────────────── Potential definition ───────────────────
struct GaussianGas <: PairPotential end
function gc_wl.pair_energy(p::GaussianGas, r2_σ::Float64)::Float64
    # βU(r) = -ln(1 - exp(-(r/σ)²))  with σ=1, temperature-independent at T*=1
    return -log(1.0 - exp(-r2_σ))
end

# ─────────────────── Fixed sim parameters ───────────────────
const T_σ     = 1.0
const Λ_σ     = 1.0
const L_σ     = 20.0
const N_max   = 13
const maxiter = 10^14       # ~0.14s/run at this scale → 100k runs ≈ 4 hours

const n_runs  = 100_000
const n_bins  = N_max + 1  # logQ_N has entries for N = 0..N_max
const n_bns   = N_max      # bns[1..N_max]
const n_Bns   = N_max      # Bns[1..N_max], with Bns[1]=0 placeholder

const output_dir       = "/Users/mckinleypaul/Documents/montecarlo/gc_wl/data/gaussian_gas/error_investigation/100k_run"
mkpath(output_dir)
const log_path         = joinpath(output_dir, "wl_progress_log.txt")
const results_path     = joinpath(output_dir, "correlation_campaign_$(n_runs)runs.jld2")
const checkpoint_every = 5000

# Helper: recompute logZ from a given logQ vector (same formula as compute_logZ)
function logZ_from_logQ(logQ::AbstractVector, sim::SimulationParams)
    Ns = 0:sim.N_max
    return logQ .- Ns .* log(sim.V_σ) .+
           SpecialFunctions.loggamma.(Ns .+ 1) .+
           3 .* Ns .* log(sim.Λ_σ)
end

# Convert covariance matrix → correlation matrix; NaN rows/cols where σ=0
function corr_from_cov(C::AbstractMatrix)
    σ = sqrt.(diag(C))
    σ_safe = ifelse.(σ .< eps(eltype(C)), one(eltype(C)), σ)
    R = C ./ (σ_safe * σ_safe')
    for i in eachindex(σ)
        if σ[i] < eps(eltype(C))
            R[i, :] .= NaN
            R[:, i] .= NaN
        end
    end
    return R
end

# ─────────────────── Storage: every run as a column ───────────────────
# Matrix layout (rows = N or n index, columns = run id) allows cov(M; dims=2).
logQ_store      = zeros(Float64, n_bins, n_runs)
bns_store       = zeros(Float64, n_bns,  n_runs)
tilde_bns_store = zeros(Float64, n_bns,  n_runs)
Bns_store       = zeros(Float64, n_Bns,  n_runs)
bn_Rsum_store   = zeros(Float64, n_bns,  n_runs)
bn_Rsub_store   = zeros(Float64, n_bns,  n_runs)

# ─────────────────── Main loop ───────────────────
t_start = time()
for ii in 1:n_runs
    sim = SimulationParams(
        potential = GaussianGas(),
        N_max = N_max, N_min = 0,
        T_σ = T_σ, Λ_σ = Λ_σ, L_σ = L_σ,
        r_cut_σ = ceil(sqrt(3) * L_σ),
        save_directory_path = output_dir,
        maxiter = maxiter,
        rng = MersenneTwister(ii),
    )
    μstate = init_microstate(sim)
    wl     = init_WangLandauVars(sim)
    cache  = init_cache(sim, μstate)

    run_simulation!(sim, μstate, wl, cache)
    _ = correct_logQ!(wl)

    logZ_big                  = BigFloat.(compute_logZ(wl, sim))
    bns_big, bn_Rsum, bn_Rsub = compute_bn_from_logZ(logZ_big, sim)
    tilde_bns_big, _, _       = compute_bns_rescaled(logZ_big, sim)
    Bns_big, _, _             = compute_Bn_from_bn(bns_big, sim)

    logQ_store[:,      ii] = wl.logQ_N
    bns_store[:,       ii] = Float64.(bns_big)
    tilde_bns_store[:, ii] = Float64.(tilde_bns_big)
    Bns_store[:,       ii] = Float64.(Bns_big)
    bn_Rsum_store[:,   ii] = Float64.(bn_Rsum)
    bn_Rsub_store[:,   ii] = Float64.(bn_Rsub)

    rm(log_path; force = true)  # keep progress log file from growing

    if ii % 1000 == 0
        elapsed = time() - t_start
        eta = elapsed * (n_runs - ii) / ii
        println("Run $ii / $n_runs — elapsed $(round(elapsed/60, digits=1)) min, ETA $(round(eta/60, digits=1)) min ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS")  )
        flush(stdout)
    end

    if ii % checkpoint_every == 0
        JLD2.jldsave(joinpath(output_dir, "checkpoint.jld2");
            logQ_store      = logQ_store[:,      1:ii],
            bns_store       = bns_store[:,       1:ii],
            tilde_bns_store = tilde_bns_store[:, 1:ii],
            Bns_store       = Bns_store[:,       1:ii],
            bn_Rsum_store   = bn_Rsum_store[:,   1:ii],
            bn_Rsub_store   = bn_Rsub_store[:,   1:ii],
            n_completed     = ii,
        )
    end
end

total_runtime = time() - t_start
println("All $n_runs runs completed in $(round(total_runtime/60, digits=1)) min")
flush(stdout)
# ═══════════════════ Analysis ═══════════════════

# ─────── Basic means, stds, SEMs ───────
logQ_mean      = vec(mean(logQ_store;      dims=2))
logQ_std       = vec(std(logQ_store;       dims=2))
logQ_sem       = logQ_std ./ sqrt(n_runs)

bns_mean       = vec(mean(bns_store;       dims=2))
bns_std        = vec(std(bns_store;        dims=2))
bns_sem        = bns_std ./ sqrt(n_runs)

tilde_bns_mean = vec(mean(tilde_bns_store; dims=2))
tilde_bns_std  = vec(std(tilde_bns_store;  dims=2))

Bns_mean       = vec(mean(Bns_store;       dims=2))
Bns_std        = vec(std(Bns_store;        dims=2))
Bns_sem        = Bns_std ./ sqrt(n_runs)

bn_Rsum_mean   = vec(mean(bn_Rsum_store;   dims=2))
bn_Rsub_mean   = vec(mean(bn_Rsub_store;   dims=2))

# ─────── Covariance & correlation matrices ───────
logQ_cov  = cov(logQ_store; dims=2)
bns_cov   = cov(bns_store;  dims=2)
Bns_cov   = cov(Bns_store;  dims=2)

logQ_corr = corr_from_cov(logQ_cov)
bns_corr  = corr_from_cov(bns_cov)
Bns_corr  = corr_from_cov(Bns_cov)

# ─────── Operation-order comparison (Jensen gap) ───────
# Estimator 1 (already computed):  mean(bns_per_run), mean(Bns_per_run)
# Estimator 2:                     bns, Bns computed from mean(logQ) directly
#
# The two differ because bn is a nonlinear function of logQ (involves exp()).
# For a nonlinear f, E[f(X)] ≠ f(E[X]). The gap between them is the "Jensen gap";
# for small fluctuations it's ~ (1/2) f''(E[X]) Var(X).
#
# If the gap is >> SEM, it's a real systematic bias and we should probably prefer
# the bn-from-mean-logQ estimator (the underlying thermodynamic bn is a function
# of the true logQ, not of noisy samples).
sim_for_analysis = SimulationParams(
    potential = GaussianGas(),
    N_max = N_max, N_min = 0,
    T_σ = T_σ, Λ_σ = Λ_σ, L_σ = L_σ,
    r_cut_σ = ceil(sqrt(3) * L_σ),
    save_directory_path = output_dir,
    maxiter = 1,
    rng = MersenneTwister(0),
)

logZ_of_mean_logQ           = BigFloat.(logZ_from_logQ(logQ_mean, sim_for_analysis))
bns_from_meanlogQ_big, _, _ = compute_bn_from_logZ(logZ_of_mean_logQ, sim_for_analysis)
Bns_from_meanlogQ_big, _, _ = compute_Bn_from_bn(bns_from_meanlogQ_big, sim_for_analysis)
bns_from_meanlogQ           = Float64.(bns_from_meanlogQ_big)
Bns_from_meanlogQ           = Float64.(Bns_from_meanlogQ_big)

bns_jensen_gap      = bns_mean .- bns_from_meanlogQ
Bns_jensen_gap      = Bns_mean .- Bns_from_meanlogQ
bns_jensen_over_sem = bns_jensen_gap ./ max.(bns_sem, eps(Float64))
Bns_jensen_over_sem = Bns_jensen_gap ./ max.(Bns_sem, eps(Float64))

# ─────── SEM scaling snapshots ───────
snapshot_ks = [10, 30, 100, 300, 1000, 3000, 10_000, 30_000, 100_000]
snapshot_sem_logQ = Dict{Int, Vector{Float64}}()
snapshot_sem_bns  = Dict{Int, Vector{Float64}}()
snapshot_sem_Bns  = Dict{Int, Vector{Float64}}()
for k in snapshot_ks
    k > n_runs && continue
    snapshot_sem_logQ[k] = vec(std(@view(logQ_store[:, 1:k]); dims=2)) ./ sqrt(k)
    snapshot_sem_bns[k]  = vec(std(@view(bns_store[:,  1:k]); dims=2)) ./ sqrt(k)
    snapshot_sem_Bns[k]  = vec(std(@view(Bns_store[:,  1:k]); dims=2)) ./ sqrt(k)
end

# ─────── Save everything ───────
JLD2.jldsave(results_path;
    n_runs              = n_runs,
    runtime_seconds     = total_runtime,

    # raw per-run arrays (columns = runs)
    logQ_store          = logQ_store,
    bns_store           = bns_store,
    tilde_bns_store     = tilde_bns_store,
    Bns_store           = Bns_store,
    bn_Rsum_store       = bn_Rsum_store,
    bn_Rsub_store       = bn_Rsub_store,

    # summary stats
    logQ_mean           = logQ_mean,
    logQ_std            = logQ_std,
    logQ_sem            = logQ_sem,
    bns_mean            = bns_mean,
    bns_std             = bns_std,
    bns_sem             = bns_sem,
    tilde_bns_mean      = tilde_bns_mean,
    tilde_bns_std       = tilde_bns_std,
    Bns_mean            = Bns_mean,
    Bns_std             = Bns_std,
    Bns_sem             = Bns_sem,
    bn_Rsum_mean        = bn_Rsum_mean,
    bn_Rsub_mean        = bn_Rsub_mean,

    # covariance / correlation
    logQ_cov            = logQ_cov,
    logQ_corr           = logQ_corr,
    bns_cov             = bns_cov,
    bns_corr            = bns_corr,
    Bns_cov             = Bns_cov,
    Bns_corr            = Bns_corr,

    # Jensen gap analysis
    bns_from_meanlogQ   = bns_from_meanlogQ,
    Bns_from_meanlogQ   = Bns_from_meanlogQ,
    bns_jensen_gap      = bns_jensen_gap,
    Bns_jensen_gap      = Bns_jensen_gap,
    bns_jensen_over_sem = bns_jensen_over_sem,
    Bns_jensen_over_sem = Bns_jensen_over_sem,

    # SEM scaling diagnostic
    snapshot_ks         = snapshot_ks,
    snapshot_sem_logQ   = snapshot_sem_logQ,
    snapshot_sem_bns    = snapshot_sem_bns,
    snapshot_sem_Bns    = snapshot_sem_Bns,
)

@info "Saved results to $results_path"

# ─────── Console summary ───────
println()
println("═"^82)
println("SUMMARY — $n_runs runs, $(round(total_runtime/60, digits=1)) min")
println("═"^82)

println("\n── Operation-order comparison for Bns ──")
println("   (mean-of-per-run) vs (from-mean-logQ); |gap/SEM| ≫ 3 means statistically significant bias")
println()
@printf("  %2s │ %18s │ %18s │ %12s │ %9s\n",
        "n", "Bns mean-of-runs", "Bns from-mean-logQ", "Jensen gap", "gap/SEM")
println("  ───┼────────────────────┼────────────────────┼──────────────┼───────────")
for n in 2:N_max
    @printf("  %2d │ %18.10g │ %18.10g │ %12.4g │ %9.3g\n",
            n, Bns_mean[n], Bns_from_meanlogQ[n], Bns_jensen_gap[n], Bns_jensen_over_sem[n])
end

println("\n── Operation-order comparison for bns ──")
@printf("  %2s │ %18s │ %18s │ %12s │ %9s\n",
        "n", "bns mean-of-runs", "bns from-mean-logQ", "Jensen gap", "gap/SEM")
println("  ───┼────────────────────┼────────────────────┼──────────────┼───────────")
for n in 2:N_max
    @printf("  %2d │ %18.10g │ %18.10g │ %12.4g │ %9.3g\n",
            n, bns_mean[n], bns_from_meanlogQ[n], bns_jensen_gap[n], bns_jensen_over_sem[n])
end

println("\n── logQ correlation matrix (full) ──")
print("       ")
for j in 0:N_max; @printf(" N=%2d   ", j); end
println()
for i in 1:n_bins
    @printf(" N=%2d ", i - 1)
    for j in 1:n_bins
        @printf(" %6.3f ", logQ_corr[i, j])
    end
    println()
end

println("\n── bns correlation matrix (indices 2..$N_max) ──")
print("       ")
for j in 2:n_bns; @printf(" n=%2d   ", j); end
println()
for i in 2:n_bns
    @printf(" n=%2d ", i)
    for j in 2:n_bns
        @printf(" %6.3f ", bns_corr[i, j])
    end
    println()
end

println("\n── Bns correlation matrix (indices 2..$N_max) ──")
print("       ")
for j in 2:n_Bns; @printf(" n=%2d   ", j); end
println()
for i in 2:n_Bns
    @printf(" n=%2d ", i)
    for j in 2:n_Bns
        @printf(" %6.3f ", Bns_corr[i, j])
    end
    println()
end

println("\n── Off-diagonal correlation summary ──")
off_diag_logQ = [logQ_corr[i, j] for i in 2:n_bins, j in 2:n_bins if i != j]
off_diag_bns  = [bns_corr[i, j]  for i in 2:n_bns,  j in 2:n_bns  if i != j]
off_diag_Bns  = [Bns_corr[i, j]  for i in 2:n_Bns,  j in 2:n_Bns  if i != j]
@printf("  logQ off-diag (N≥1): min = %6.3f, max = %6.3f, mean = %6.3f\n",
        minimum(off_diag_logQ), maximum(off_diag_logQ), mean(off_diag_logQ))
@printf("  bns  off-diag (n≥2): min = %6.3f, max = %6.3f, mean = %6.3f\n",
        minimum(off_diag_bns),  maximum(off_diag_bns),  mean(off_diag_bns))
@printf("  Bns  off-diag (n≥2): min = %6.3f, max = %6.3f, mean = %6.3f\n",
        minimum(off_diag_Bns),  maximum(off_diag_Bns),  mean(off_diag_Bns))

println("\nDone.")