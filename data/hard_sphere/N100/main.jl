import Pkg
Pkg.activate("/Users/mckinleypaul/Documents/montecarlo/gc_wl")
using gc_wl
import Random

η = 0.575
Nmax=100
v_sphere = π/6 # σ=1
V = v_sphere*Nmax/η
L_σ = V^(1/3)
println("Volume: ", V)
println("Length: ", L_σ)
println("η: ",compute_packing_frac(Nmax,V,1.))

# hard sphere
struct HardSphere <: PairPotential end
function gc_wl.pair_energy(p::HardSphere,r2_σ::Float64)::Float64
    # MUST to be used with r_cut_σ==1 so if the potential function ever gets called, it means they are overlapping
        #   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    return floatmax(Float64)
end

T_σ = 1.0 # MUST RUN THIS MODEL AT T_σ == 1 or include β in the potential definition above because temp needs to cancel out in boltzmann factor
Λ_σ = 1.0 # doesn't matter for computing of virials because they just depend on configuration integral
println("Λ_σ = $Λ_σ")
sim = SimulationParams(
    potential = HardSphere(),
    N_max=Nmax,# Nmax and L_σ must be carefully chosen for HS 
    N_min=0,
    T_σ=T_σ, Λ_σ=Λ_σ,
    L_σ = L_σ,  
    r_cut_σ= 1.0, # !!! important !!! 
    save_directory_path=@__DIR__,
    maxiter=100_000_000_000_000)

μstate = init_microstate(sim)

wl = init_WangLandauVars(sim)

cache = init_cache(sim, μstate)

initialization_check(sim, μstate, wl)

run_simulation!(sim, μstate, wl, cache)

post_run(sim, μstate, wl)


println("logQ(N_max) = $(wl.logQ_N[sim.N_max+1])")
println("Full logQ vector:")
println(wl.logQ_N)
