import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "..", ".."))
using GCWL
import Random

include(joinpath(@__DIR__, "..", "..", "..", "square_well_potential.jl"))

# ── Square well, λ = 1.75, T* = 1.55, replica 4 ──
# Reduced units: σ = ε = 1, Λ_σ = 1.
# r_cut_σ = λ is the FULL range of the potential, not a truncation, so no
# long-range correction applies. Do not call apply_lrc_to_logQ on this data.
λ_sw  = 1.75
T_σ   = 1.55
L_σ   = 9.0        # V = 729.0 σ³
Λ_σ   = 1.0

println("Square well: λ = $λ_sw, T* = $T_σ, V = $(L_σ^3) σ³, replica 4")

sim = SimulationParams(
    potential = SquareWell(λ_sw),
    N_max     = 530,
    N_min     = 0,
    T_σ       = T_σ,
    Λ_σ       = Λ_σ,
    L_σ       = L_σ,
    r_cut_σ   = λ_sw,
    save_directory_path = @__DIR__,
    maxiter   = 100_000_000_000_000,
    rng       = Random.MersenneTwister(175015504))

μstate = init_microstate(sim)
wl     = init_WangLandauVars(sim)
cache  = init_cache(sim, μstate)

initialization_check(sim, μstate, wl)

run_simulation!(sim, μstate, wl, cache)

post_run(sim, μstate, wl)

constant = correct_logQ!(wl)

println("Value of wang landau logQ multiplicative constant before correction: ", constant)
println("logQ(N_max) = $(wl.logQ_N[sim.N_max+1])")
println("logQ(1) = $(wl.logQ_N[2])  exact ln(V/Λ³) = $(log(sim.V_σ / sim.Λ_σ^3))")
println("Full logQ vector:")
println(wl.logQ_N)
