# Example: ln Q(N, V, T) for N = 0…40 of the Lennard-Jones fluid with argon parameters at T* = 1.2 in a box of
# V = 64σ³ (L = 4σ), with the 1/t Wang–Landau schedule. Run from the repository root:
#     julia --project=. examples/lj_argon_small.jl [output directory]
using GCWL, Random, Printf

outdir = length(ARGS) ≥ 1 ? ARGS[1] : joinpath(@__DIR__, "output")
mkpath(outdir)

T_σ = 1.2
sim = SimulationParams(N_max = 40, N_min = 0, T_σ = T_σ, Λ_σ = argon_deBroglie(T_σ),
                       L_σ = 4.0, r_cut_σ = 2.0,
                       save_directory_path = outdir, maxiter = 10^13, rng = MersenneTwister(1))

μ     = init_microstate(sim)                       # start from the vacuum, N = 0
wl    = init_WangLandauVars(sim)
cache = init_cache(sim, μ)
initialization_check(sim, μ, wl)

run_simulation!(sim, μ, wl, cache)                 # progress: outdir/wl_progress_log.txt
post_run(sim, μ, wl)                               # normalizes wl.logQ_N so that Q(0) = 1

logQ = wl.logQ_N                                   # logQ[N+1] = ln Q(N, V, T)
open(joinpath(outdir, "logQ.csv"), "w") do io
    println(io, "N,logQ")
    for N in 0:sim.N_max
        @printf(io, "%d,%.17g\n", N, logQ[N+1])
    end
end
@printf("ln Q(1) = %.4f, exact ln(V/Λ³) = %.4f\n", logQ[2], log(sim.V_σ / sim.Λ_σ^3))
println("wrote ", joinpath(outdir, "logQ.csv"))
