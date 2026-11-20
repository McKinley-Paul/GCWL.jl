# Lee's expansion at the square-well state points, and the error of each order on the ring zeros of the four-run mean
# ln Q. Writes output/lee_expansion_sw.jld2, which figures/make_figures.ipynb reads for Fig. 8 and the numbers quoted
# with Figs. 7 and 8. Recomputing takes about 10 min (one 1024-bit root solve per state point).
#
# The output file is tracked, so by default the script only loads it and prints the summary. To recompute it from the
# runs, overwriting it, run from the repository root:
#     julia data/square_well/lee_expansion_sw.jl --force
import Pkg; Pkg.activate(joinpath(@__DIR__, "..", ".."); io=devnull)
using GCWL, JLD2, Printf, Statistics

const FORCE = "--force" in ARGS   # true: recompute from the runs and overwrite the output file
const BASE = @__DIR__
const V = 729.0
lse(v) = (m = maximum(v); m + log(sum(exp.(v .- m))))

points = [(λ=parse(Float64, split(l, "_")[2]), T=parse(Float64, split(t, "_")[2]), dir=joinpath(BASE, l, t))
          for l in sort(filter(startswith("lambda_"), readdir(BASE))) for t in sort(filter(startswith("T_"), readdir(joinpath(BASE, l))))]
runs_of(p) = [(w = load_wanglandau_jld2(joinpath(p.dir, "run$r", "final_wl.jld2")); x = collect(Float64, w.logQ_N); x .- x[1]) for r in 1:4]

# Equal-weight coexistence: g(N) = ln Q(N) + N u is split at its valley between the outermost maxima, and u is
# bisected until the gas and liquid parts of Ξ have equal weight.
maxima(lnQ, u) = (d = diff(lnQ); [i+1 for i in 1:length(d)-1 if d[i]+u > 0 && d[i+1]+u <= 0])
valley(lnQ, Nv, u) = (mx = maxima(lnQ, u); length(mx) < 2 ? div(length(lnQ), 2) :
                      mx[1]-1+argmin(@view (lnQ .+ Nv .* u)[mx[1]:mx[end]]))
function coex(lnQ, Nv, u0)
    f(u) = (j = valley(lnQ, Nv, u); g = lnQ .+ Nv .* u; lse(g[j+1:end]) - lse(g[1:j]))
    lo, hi = u0, u0
    for w in (0.02, 0.05, 0.1, 0.2, 0.6, 1.5)
        lo, hi = u0-w, u0+w; f(lo)*f(hi) < 0 && break
    end
    for _ in 1:100; m = (lo+hi)/2; f(m) > 0 ? (hi = m) : (lo = m); end
    (lo+hi)/2
end
# Cumulants of N in the gas and liquid parts at u, their jumps Δ per volume, and the coefficients b_n of Lee's series
# t_k = Σ_n b_n w_k^n for the kth zero, t = ln(z/z_c), w_k = (2πi(k+½) − ln(Ξ_liq/Ξ_gas))/V.
function lee(lnQ, Nv, u; ib = valley(lnQ, Nv, u))
    g = lnQ .+ Nv .* u
    cum(r) = begin
        p = exp.(g[r] .- lse(g[r])); N = Nv[r]; m = sum(p .* N); d = N .- m
        μ2, μ3, μ4 = sum(p .* d.^2), sum(p .* d.^3), sum(p .* d.^4)
        [m, μ2, μ3, μ4 - 3μ2^2]
    end
    kg, kl = cum(1:ib), cum(ib+1:length(Nv))
    Δ = (kl .- kg) ./ V; a = Δ ./ factorial.(1:4)
    b = [1/a[1], -a[2]/a[1]^3, (2a[2]^2 - a[1]*a[3])/a[1]^5, (5a[1]*a[2]*a[3] - a[1]^2*a[4] - 5a[2]^3)/a[1]^7]
    (u=u, ib=ib, Nb=Nv[ib], kg=kg, kl=kl, Δ=Δ, b=b, lnratio=lse(g[ib+1:end]) - lse(g[1:ib]))
end
wk(L, k) = (2π*im*(k+0.5) - L.lnratio)/V
leepred(L, k, n) = sum(L.b[j] * wk(L, k)^j for j in 1:n)
spacing(L) = 2π/(V*L.Δ[1])
# The ring zeros are the upper-half-plane zeros with |ln(|z|/z_c)| < RING_BAND; the kth is the kth by increasing
# argument, k = 0, 1, ...
const RING_BAND = 0.03
function ring_zeros(lnQ, Nv, L, K)
    z = GCWL.yang_lee_zeros(lnQ, Nv[1]:Nv[end]; prec_bits=1024, tol=1e-10).roots
    ring = sort([x for x in z if imag(x) > 0 && abs(log(abs(x)) - L.u) < RING_BAND], by = angle)
    length(ring) < K && error("only $(length(ring)) ring zeros in the upper half-plane, K = $K")
    ComplexF64.(ring[1:K])
end
# err[k+1, n]: distance of the order-n prediction from the kth ring zero, in units of the zero spacing 2π/ΔN
function lee_on(lnQ, Nv, u0, K)
    L = lee(lnQ, Nv, coex(lnQ, Nv, u0)); zs = ring_zeros(lnQ, Nv, L, K); t = log.(zs) .- L.u
    (L=L, err=[abs(leepred(L, k, n) - t[k+1])/spacing(L) for k in 0:K-1, n in 1:4])
end

const OUT = joinpath(BASE, "output", "lee_expansion_sw.jld2")
const K = 40
rows = isfile(OUT) && !FORCE ? jldopen(f -> f["rows"], OUT, "r") : begin
    rows = Any[]
    for p in points
        m = mean(runs_of(p)); Nv = collect(0:length(m)-1)
        i = length(m) ÷ 2; u = coex(m, Nv, -(m[i+1] - m[i-1])/2)   # equal-weight ln z_c, from the slope at mid-range
        push!(rows, (λ=p.λ, T=p.T, L=lee(m, Nv, u), err=lee_on(m, Nv, u, K).err))
        @printf("done λ=%.2f T=%.3f\n", p.λ, p.T); flush(stdout)
    end
    jldopen(f -> (f["rows"] = rows), OUT, "w"); rows
end

firstworse(e) = (i = findfirst(k -> e[k, 3] > e[k, 2], 1:size(e, 1)); i === nothing ? "-" : string(i - 1))
println(" λ     T*      error / spacing at k = 10, orders 1 / 2 / 3     first k at which order 3 is further than order 2")
for r in rows
    @printf(" %.2f  %.3f   %.2g / %.2g / %.2g                      %s\n", r.λ, r.T, r.err[11, 1:3]..., firstworse(r.err))
end
