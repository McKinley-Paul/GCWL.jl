import Pkg
Pkg.activate("/Users/mckinleypaul/Documents/montecarlo/segc_wl")
using segc_wl
import Random

input_path = "/Users/mckinleypaul/Documents/montecarlo/segc_wl/initial_configs/N350_L8_random_deletion.inp"
T_σ = 1.2
Λ_σ = argon_deBroglie(T_σ)
println("Λ_σ = $Λ_σ")
sim = SimulationParams(
    N_max=350, N_min=0, T_σ=T_σ, Λ_σ=Λ_σ,
    λ_max=99, r_cut_σ=3., input_filename=input_path,
    save_directory_path=@__DIR__,
    rng=Random.MersenneTwister(1),
    maxiter=100_000_000_000_000)

μstate = init_microstate(sim=sim, filename=input_path)
wl = init_WangLandauVars(sim)
cache = init_cache(sim, μstate)
initialization_check(sim, μstate, wl)
run_simulation!(sim, μstate, wl, cache)
post_run(sim, μstate, wl)
logQ = correct_logQ(wl)
println("logQ*(350) = $(logQ[350+1])")
println("Full logQ vector:")
println(logQ)
