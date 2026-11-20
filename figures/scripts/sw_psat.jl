# Saturation pressure of the SW fluid for Fig. 4(c).
# Coexistence = equal weight of the two phases (same as NIST's equal phase pressures), and the phase pressure is
# βPV = ln Ξ_gas(z_c) = ln Ξ_liq(z_c) = ln Ξ(z_c) − ln 2. P* = T* ln Ξ_gas / V (σ = ε = Λ = 1).
# Mean from the four-run-mean ln Q; sd over the four runs, each at its own equal-weight z_c.
# Writes sw/lam*/T*/Psat, Psat_sd and the reference arrays into figures/data/figure_data.jld2.
import Pkg; Pkg.activate(joinpath(@__DIR__, "..", ".."); io=devnull)
using GCWL, JLD2, Printf, Statistics
const ROOT = joinpath(@__DIR__, "..", "..")
const SWDIR = joinpath(ROOT, "data", "square_well"); const V = 729.0
const EXPORT = joinpath(ROOT, "figures", "data", "figure_data.jld2")
lse(x) = (m = maximum(x); m + log(sum(exp.(x .- m))))
function valley(lnQ, N, u)
    g = lnQ .+ N .* u; i1 = argmax(g)
    far = [i for i in eachindex(g) if abs(i - i1) > 50]; i2 = far[argmax(g[far])]
    a, b = minmax(i1, i2); return a - 1 + argmin(g[a:b])          # index of the valley
end
function eqweight(lnQ, N, ib; lo = -30.0, hi = 5.0)
    f(u) = lse(lnQ[ib+1:end] .+ N[ib+1:end] .* u) - lse(lnQ[1:ib] .+ N[1:ib] .* u)
    for _ in 1:200; m = (lo + hi) / 2; f(m) > 0 ? (hi = m) : (lo = m); end
    return (lo + hi) / 2
end
psat(lnQ, N, u, ib, T) = T * lse(lnQ[1:ib] .+ N[1:ib] .* u) / V
FD = jldopen(EXPORT, "r"); keys_sw = [(lam, k) for lam in ("1.50", "1.75") for k in keys(FD["sw/lam$lam"])]
lnzc_fd = Dict((lam, k) => FD["sw/lam$lam/$k/lnzc"] for (lam, k) in keys_sw); close(FD)
res = Dict{Tuple{String,String},Tuple{Float64,Float64}}()
println(" λ     T*     ln z_c (mean)   check vs fig data     P*_sat        sd(4 runs)")
for (lam, k) in sort(keys_sw)
    T = parse(Float64, k[2:end])
    dir = joinpath(SWDIR, "lambda_$lam", @sprintf("T_%.3f", T))
    runs = Vector{Float64}[]
    for r in 1:4
        p = joinpath(dir, "run$r", "final_wl.jld2"); isfile(p) || continue
        lq = collect(Float64, load_wanglandau_jld2(p).logQ_N); lq .-= lq[1]; push!(runs, lq)
    end
    lnQ = mean(runs); N = collect(0.0:length(lnQ)-1)
    ib = valley(lnQ, N, lnzc_fd[(lam, k)]); u = eqweight(lnQ, N, ib)
    P = psat(lnQ, N, u, ib, T)
    Pr = [psat(lq, N, eqweight(lq, N, ib), ib, T) for lq in runs]
    res[(lam, k)] = (P, std(Pr))
    @printf(" %s  %.3f   %.6f      %+.1e           %.5e   %.1e\n", lam, T, u, u - lnzc_fd[(lam, k)], P, std(Pr))
end
# references. NIST SAT-TMMC λ = 1.5, V = 729σ³ (same box), equal phase pressures; 95% limits.
nist_T = [0.95, 1.00, 1.05, 1.10, 1.15]; nist_P = [1.727e-2, 2.507e-2, 3.527e-2, 4.837e-2, 6.500e-2]
# Singh, Kofke & Errington, JCP 119, 3405 (2003), Table IV, λ = 1.75 (GC-TMMC), rows within our range.
singh_T = [1.50, 1.55, 1.57, 1.60, 1.62, 1.65, 1.68]; singh_P = [0.04236, 0.0519, 0.0561, 0.06299, 0.06786, 0.0757, 0.08419]
jldopen(EXPORT, "a+") do f
    for ((lam, k), (P, sd)) in res
        for (nm, v) in (("Psat", P), ("Psat_sd", sd))
            key = "sw/lam$lam/$k/$nm"; haskey(f, key) && delete!(f, key); f[key] = v
        end
    end
    for (key, v) in (("sw/nist_P", nist_P), ("sw/nist_P_T", nist_T), ("sw/singh175_T", singh_T), ("sw/singh175_P", singh_P))
        haskey(f, key) && delete!(f, key); f[key] = v
    end
end
println("\nNIST λ=1.5 comparison:")
for (T, Pn) in zip(nist_T, nist_P)
    P, sd = res[("1.50", @sprintf("T%.3f", T))]; @printf("  T*=%.2f  ours %.5f  NIST %.5f  rel diff %+.2e\n", T, P, Pn, P/Pn - 1)
end
println("Singh λ=1.75 comparison (matching T only):")
for (T, Ps) in zip(singh_T, singh_P)
    k = @sprintf("T%.3f", T); haskey(res, ("1.75", k)) || continue
    P, sd = res[("1.75", k)]; @printf("  T*=%.2f  ours %.5f  Singh %.5f  rel diff %+.2e\n", T, P, Ps, P/Ps - 1)
end
