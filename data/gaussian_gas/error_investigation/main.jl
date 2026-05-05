import Pkg
Pkg.activate("/Users/mckinleypaul/Documents/montecarlo/gc_wl")
using gc_wl
import Random

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

T_σ = 1.0 # MUST RUN THIS MODEL AT T_σ == 1 or include β in the potential definition above because temp needs to cancel out in boltzmann factor
Λ_σ = 1.0 # doesn't matter for computing of virials because they just depend on configuration integral
L_σ = 20.0
println("Λ_σ = $Λ_σ")
sim = SimulationParams(
    potential = GaussianGas(),
    N_max=13,
    N_min=0,
    T_σ=T_σ, Λ_σ=Λ_σ,
    # dilute: ρ_max = 13/8000 ≈ 0.001625
    L_σ = L_σ ,  
    # Using effectively no cutoff here because maximum distance in cube is L_σ*√3 (less for pbc technically) so just erring on safe side here 
    r_cut_σ= ceil(sqrt(3)*L_σ), 
    # r_cut_σ=5.4,   # u(5.4) = 2.1671553440685403e-13, this is approaching precision level of the machine so following what Wheatley 2013 () did for small values of f(r) approaching machine precision and use a cutoff such that u(r) ≈ 10^-14  is set to zero to avoid numerical errors in floating point arithmetic
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
