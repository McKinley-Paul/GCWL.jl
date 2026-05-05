import Pkg
Pkg.activate("/Users/mckinleypaul/Documents/montecarlo/gc_wl")
using gc_wl
using Random
using Statistics
using JLD2

# Gaussian gas potential, named for mayer function:
# Mayer function: f(r) = -exp(-(r/σ)²) implying pairwise potential: U(r) = -(1/β) ln(1 - exp(-(r/σ)²))
# so β in pairwise potential cancels with the β appearing in the boltzmann factor exp(-βU(x)), thus 
# βU = -ln(1 - exp(-r²)) is temperature-independent and so Q(N,V,T) = Q(N,V) 
# Running at T_σ=1 with pair_energy = βU gives exact Boltzmann factor exp(-βU) at any T.

struct GaussianGas <: PairPotential end
function gc_wl.pair_energy(p::GaussianGas, r2_σ::Float64)::Float64
    # βU(r) = -ln(1 - exp(-r²/σ²)), with σ=1
    return -log(1.0 - exp(-r2_σ))
end

# ─────────────────── Fixed sim parameters ───────────────────
T_σ = 1.0
Λ_σ = 1.0
L_σ = 20.0
N_max = 13
maxiter = 10^14   # reduce this from 10^14 for practical runtime over 9000 runs

n_runs = 9000
n_bins = N_max + 1                        # logQ_N has entries for N = 0..N_max
n_bns  = N_max                            # your compute_bn_from_logZ returns bns for n = 1..N_max
n_Bns  = N_max                            # Bns indexed 1..N_max, with Bns[1] = 0 placeholder
log_path = "/Users/mckinleypaul/Documents/montecarlo/gc_wl/data/gaussian_gas/error_investigation/9000run/wl_progress_log.txt"

# ─────────────────── Welford accumulators ───────────────────
# For each quantity, track the running mean (M), sum of squared deviations (S), 
# and snapshots at predetermined sample counts for the 1/√k scaling check.

mutable struct WelfordVector
    n::Int                      # number of samples seen so far
    M::Vector{Float64}          # running mean
    S::Vector{Float64}          # running sum of squared deviations
end
WelfordVector(length::Int) = WelfordVector(0, zeros(length), zeros(length))

function update!(w::WelfordVector, x::AbstractVector{<:Real})
    w.n += 1
    for i in eachindex(x)
        δ = x[i] - w.M[i]
        w.M[i] += δ / w.n
        δ2 = x[i] - w.M[i]
        w.S[i] += δ * δ2
    end
    return w
end

mean_of(w::WelfordVector) = copy(w.M)
var_of(w::WelfordVector)  = w.n > 1 ? w.S ./ (w.n - 1) : fill(NaN, length(w.S))
std_of(w::WelfordVector)  = sqrt.(var_of(w))
sem_of(w::WelfordVector)  = std_of(w) ./ sqrt(w.n)

# ─────────────────── Accumulators ───────────────────
w_logQ      = WelfordVector(n_bins)
w_bns       = WelfordVector(n_bns)
w_tilde_bns = WelfordVector(n_bns)
w_Bns       = WelfordVector(n_Bns)
w_bn_Rsum   = WelfordVector(n_bns)
w_bn_Rsub   = WelfordVector(n_bns)

# Snapshots for scaling check: record SEM at these sample counts
snapshot_ks = [10, 30, 100, 300, 1000, 3000, 9000]
snapshot_sem_logQ = Dict{Int, Vector{Float64}}()
snapshot_sem_bns  = Dict{Int, Vector{Float64}}()
snapshot_sem_Bns  = Dict{Int, Vector{Float64}}()

# ─────────────────── Main loop ───────────────────
for ii in 1:n_runs
    sim = SimulationParams(
        potential = GaussianGas(),
        N_max = N_max, N_min = 0,
        T_σ = T_σ, Λ_σ = Λ_σ, L_σ = L_σ,
        r_cut_σ = ceil(sqrt(3) * L_σ), # No cutoff
        save_directory_path = @__DIR__,
        maxiter = maxiter,
        rng = MersenneTwister(ii),  # independent, reproducible seed per run
    )
    μstate = init_microstate(sim)
    wl     = init_WangLandauVars(sim)
    cache  = init_cache(sim, μstate)

    run_simulation!(sim, μstate, wl, cache)
    _ = correct_logQ!(wl)

    # logQ and downstream
    logZ_big = BigFloat.(compute_logZ(wl, sim))
    bns_big, bn_Rsum, bn_Rsub      = compute_bn_from_logZ(logZ_big, sim)
    tilde_bns_big, _, _            = compute_bns_rescaled(logZ_big, sim)
    Bns_big, _, _                  = compute_Bn_from_bn(bns_big, sim)

    # Convert to Float64 for statistics (BigFloat Welford would work but is slow
    # and unnecessary given your diagnostic showed Float64 is already at machine precision)
    update!(w_logQ,      Float64.(wl.logQ_N))
    update!(w_bns,       Float64.(bns_big))
    update!(w_tilde_bns, Float64.(tilde_bns_big))
    update!(w_Bns,       Float64.(Bns_big))
    update!(w_bn_Rsum,   Float64.(bn_Rsum))
    update!(w_bn_Rsub,   Float64.(bn_Rsub))

    # Snapshot for scaling analysis
    if ii in snapshot_ks
        snapshot_sem_logQ[ii] = sem_of(w_logQ)
        snapshot_sem_bns[ii]  = sem_of(w_bns)
        snapshot_sem_Bns[ii]  = sem_of(w_Bns)
    end

    # Progress
    if ii % 100 == 0
        println("Completed run $ii / $n_runs , ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS") )
        flush(stdout)
    end
    rm(log_path; force = true)
end

# ─────────────────── Results ───────────────────
results = (
    n_runs          = n_runs,
    logQ_mean       = mean_of(w_logQ),
    logQ_std        = std_of(w_logQ),
    logQ_sem        = sem_of(w_logQ),
    bns_mean        = mean_of(w_bns),
    bns_std         = std_of(w_bns),
    bns_sem         = sem_of(w_bns),
    tilde_bns_mean  = mean_of(w_tilde_bns),
    tilde_bns_std   = std_of(w_tilde_bns),
    Bns_mean        = mean_of(w_Bns),
    Bns_std         = std_of(w_Bns),
    Bns_sem         = sem_of(w_Bns),
    bn_Rsum_mean    = mean_of(w_bn_Rsum),
    bn_Rsub_mean    = mean_of(w_bn_Rsub),
    snapshot_ks     = snapshot_ks,
    snapshot_sem_logQ = snapshot_sem_logQ,
    snapshot_sem_bns  = snapshot_sem_bns,
    snapshot_sem_Bns  = snapshot_sem_Bns,
)

JLD2.save(joinpath(@__DIR__, "error_stats_$(n_runs)runs.jld2"), "results", results)

# ─────────────────── 1/√k scaling check ───────────────────
using Plots
ks = sort(collect(keys(snapshot_sem_Bns)))
p = plot(xscale = :log10, yscale = :log10,
         xlabel = "number of runs k", ylabel = "SEM",
         title = "SEM vs k — should scale as k^(-1/2)",
         legend = :outerright)
for n in 2:N_max
    sems = [snapshot_sem_Bns[k][n] for k in ks]
    plot!(p, ks, sems, marker = :circle, label = "B_$n")
end
# reference line: 1/√k normalized to first point
k_ref = ks
plot!(p, k_ref, snapshot_sem_Bns[ks[1]][2] .* sqrt.(ks[1] ./ k_ref),
      linestyle = :dash, color = :black, label = "1/√k reference")
savefig(p, joinpath(@__DIR__, "sem_scaling_Bns.png"))
