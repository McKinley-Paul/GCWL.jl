using Plots
using Statistics
using StaticArrays
using Random
import Pkg

Pkg.activate("/Users/mckinleypaul/Documents/montecarlo/gc_wl/src/")
Pkg.activate("/Users/mckinleypaul/Documents/montecarlo/gc_wl")
using gc_wl 

struct GaussianGas <: PairPotential end
function gc_wl.pair_energy(p::GaussianGas, r2_σ::Float64)::Float64
    # βU(r) = -ln(1 - exp(-r²/σ²)), with σ=1
    return -log(1.0 - exp(-r2_σ))
end

wl = load_wanglandau_jld2("/Users/mckinleypaul/Documents/montecarlo/gc_wl/data/gaussian_gas/error_investigation/final_wl.jld2")

sim = load_sim_jld2("/Users/mckinleypaul/Documents/montecarlo/gc_wl/data/gaussian_gas/error_investigation/sim.jld2")

logZ = compute_logZ(wl,sim)

bns, bn_Rsum, bn_Rsub, = compute_bn_from_logZ(logZ,sim)
tilde_bns, tilde_bn_Rsum,tilde_bn_Rsub = compute_bns_rescaled(logZ,sim)

println(bns)
println(bn_Rsum)
println(bn_Rsub)
println(tilde_bns)
println(tilde_bn_Rsum)
println(tilde_bn_Rsub)

println("now running diagnostic test")
logZ_f64 = compute_logZ(wl, sim)
logZ_big = BigFloat.(logZ_f64)

bns_f64, _, _ = compute_bn_from_logZ(logZ_f64, sim)
bns_big, _, _ = compute_bn_from_logZ(logZ_big, sim)

# Compare: how different are they?
for n in 1:sim.N_max
    rel_err = abs(bns_big[n] - bns_f64[n]) / abs(bns_big[n])
    println("n=$n: Float64 b = $(bns_f64[n]), BigFloat b = $(Float64(bns_big[n])), rel err = $rel_err")
end

println("Bns test now")
Bns_f64_from_f64, _, _ = compute_Bn_from_bn(bns_f64, sim)
Bns_big_from_big, _, _ = compute_Bn_from_bn(bns_big, sim)

for n in 2:sim.N_max
    rel_err = abs(Bns_big_from_big[n] - Bns_f64_from_f64[n]) / abs(Bns_big_from_big[n])
    println("n=$n: B_Float64 = $(Bns_f64_from_f64[n]), B_BigFloat = $(Float64(Bns_big_from_big[n])), rel err = $rel_err")
end