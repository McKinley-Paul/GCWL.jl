import Pkg
Pkg.activate("/Users/mckinleypaul/Documents/montecarlo/gc_wl")
using gc_wl
import Random


T_σ = 1.0
Λ_σ = argon_deBroglie(T_σ)
L_σ = 8.0 # V=512
println("Λ_σ = $Λ_σ")
sim = SimulationParams(
    N_max=565,
    N_min=0,
    T_σ=T_σ, Λ_σ=Λ_σ,
    L_σ = L_σ,    
    r_cut_σ= ceil(sqrt(3)*L_σ) , # Using effectively no cutoff here because maximum distance in cube is L_σ*√3 (less for pbc technically) so just erring on safe side here 
    save_directory_path=@__DIR__,
    maxiter=100_000_000_000_000)


μstate = init_microstate(sim)

wl = init_WangLandauVars(sim)

cache = init_cache(sim, μstate)

initialization_check(sim, μstate, wl)

run_simulation!(sim, μstate, wl, cache)

post_run(sim, μstate, wl)

constant = correct_logQ!(wl)

println("Value of wang landau logQ multiplicative constant before correction: ", constant )
println("logQ(N_max) = $(wl.logQ_N[sim.N_max+1])")
println("Full logQ vector:")
println(wl.logQ_N)
