import Pkg
Pkg.activate("/Users/mckinleypaul/Documents/montecarlo/gc_wl")
using gc_wl
import Random

println("restarting run ")

sim = load_sim_jld2("/Users/mckinleypaul/Documents/montecarlo/gc_wl/data/lennard_jones/figure1/3sigma_cut/87.79/run1/sim.jld2")

wl = load_wanglandau_jld2("/Users/mckinleypaul/Documents/montecarlo/gc_wl/data/lennard_jones/figure1/3sigma_cut/87.79/run1/wl_checkpoint.jld2")

μstate = load_microstate_jld2("/Users/mckinleypaul/Documents/montecarlo/gc_wl/data/lennard_jones/figure1/3sigma_cut/87.79/run1/microstate_checkpoint.jld2")

cache = init_cache(sim, μstate)


run_simulation!(sim, μstate, wl, cache)

post_run(sim, μstate, wl)

constant = correct_logQ!(wl)

println("Value of wang landau logQ multiplicative constant before correction: ", constant)
println("logQ(N_max) = $(wl.logQ_N[sim.N_max+1])")
println("Full logQ vector:")
println(wl.logQ_N)
