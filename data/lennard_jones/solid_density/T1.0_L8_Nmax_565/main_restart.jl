import Pkg
Pkg.activate("/Users/mckinleypaul/Documents/montecarlo/gc_wl")
using gc_wl
import Random

#=      HAD TO RESTART BECAUSE CHANGED THE δr_max UPDATE SCHEME, AT THE START OF THIS RUN, HAD IT AS ONLY TUNING δr_max 
        DURING wl.logf == 1 WHICH WORKED FOR OLD SCHEME BUT IS NOT GOOD FOR NEW SCHEME, MANY RUNS IN /figure1/ TOOK WAY LONGER 
        THAN THEY SHOULD HAVE DUE TO LOW TRANSLATION MOVE ACCEPTANCE RATES. THEREFORE SEVERAL DAYS INTO THIS RUN, FIXED THIS ISSUE
        AND SET the tuning of δr_max to occur during all of phase 1, with an if wl.phase2 == false 
=# 
println(" HAD TO RESTART BECAUSE CHANGED THE δr_max UPDATE SCHEME several days into this run ")

sim = load_sim_jld2("/Users/mckinleypaul/Documents/montecarlo/gc_wl/data/lennard_jones/solid_density/T1.0_L8_Nmax_565/sim.jld2")

μstate = load_microstate_jld2("/Users/mckinleypaul/Documents/montecarlo/gc_wl/data/lennard_jones/solid_density/T1.0_L8_Nmax_565/microstate_checkpoint.jld2")

wl = load_wanglandau_jld2("/Users/mckinleypaul/Documents/montecarlo/gc_wl/data/lennard_jones/solid_density/T1.0_L8_Nmax_565/wl_checkpoint.jld2")

cache = init_cache(sim, μstate)

run_simulation!(sim, μstate, wl, cache)

post_run(sim, μstate, wl)

constant = correct_logQ!(wl)

println("Value of wang landau logQ multiplicative constant before correction: ", constant )
println("logQ(N_max) = $(wl.logQ_N[sim.N_max+1])")
println("Full logQ vector:")
println(wl.logQ_N)
