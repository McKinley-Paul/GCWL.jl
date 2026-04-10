import Pkg
Pkg.activate("/Users/mckinleypaul/Documents/montecarlo/segc_wl")
using segc_wl
import Random

T_σ = 1.2
Λ_σ = argon_deBroglie(T_σ)
println("Λ_σ = $Λ_σ")
sim = SimulationParams(
    N_max=350,
    N_min=0,
    T_σ=T_σ, Λ_σ=Λ_σ,
    λ_max=2, #M3 !!! 1/t to help fix error saturation and a bit of an alchemical nudge to help high N insertion
    r_cut_σ=3.,
    L_σ = 8.0,
    save_directory_path=@__DIR__,
    maxiter=100_000_000_000_000)

μstate = init_microstate(sim)

wl = init_WangLandauVars(sim)

cache = init_cache(sim, μstate)

initialization_check(sim, μstate, wl)

run_simulation!(sim, μstate, wl, cache)

post_run(sim, μstate, wl)

logQ = correct_logQ(wl)
println("logQ*(350) = $(logQ[350+1])")
println("Full logQ vector:")
println(logQ)
