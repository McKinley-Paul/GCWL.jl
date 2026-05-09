using Pkg 
Pkg.activate("/Users/mckinleypaul/Documents/montecarlo/gc_wl/src/")
Pkg.activate("/Users/mckinleypaul/Documents/montecarlo/gc_wl")
using gc_wl 

using StaticArrays
using Random
using JLD2 



wl = load_wanglandau_jld2("/Users/mckinleypaul/Documents/montecarlo/gc_wl/data/lennard_jones/LJ31_spherical_hard_walls/T_0.010/wl_checkpoint.jld2")

println(wl.iters)