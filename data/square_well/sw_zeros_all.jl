"""
Yang–Lee zeros of the four-run mean ln Q at each square-well state point in data/square_well.

For each state point (lambda_<λ>/T_<T*>):
  1. Load the four runs' ln Q(N) from final_wl.jld2 (no long-range correction: r_c = λ is the full range of the
     potential). The stored ln Q already has ln Q(0) = 0; it is subtracted again, which changes nothing.
  2. Average ln Q(N) across runs.
  3. Compute the zeros of Ξ(z) = Σ_N Q(N,V,T) z^N by the Aberth–Ehrlich iteration at 1024 bits (`yang_lee_zeros`)
     and check their residuals (`verify_yang_lee_zeros`).
  4. Save them to output/sw_zeros_lam<λ>_T<T*>_avg.jld2, which figures/make_figures.ipynb reads (it also discovers
     the state points from these file names).

Run from the repository root:
    julia data/square_well/sw_zeros_all.jl              # overwrites the tracked output/sw_zeros_*_avg.jld2
    julia data/square_well/sw_zeros_all.jl <outdir>     # writes to <outdir> instead
It takes about 10 min (one 1024-bit root solve per state point, degree 470–590).
"""

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."); io=devnull)

using GCWL
using JLD2
using Statistics
using Printf

const BASE_DIR   = @__DIR__
const OUTPUT_DIR = isempty(ARGS) ? joinpath(BASE_DIR, "output") : abspath(ARGS[1])
mkpath(OUTPUT_DIR)

const N_RUNS    = 4
const PREC_BITS = 1024
const MAX_ITER  = 400
const TOL       = 1e-10

points = [(λ = parse(Float64, split(l, "_")[2]), T = parse(Float64, split(t, "_")[2]), dir = joinpath(BASE_DIR, l, t))
          for l in sort(filter(startswith("lambda_"), readdir(BASE_DIR)))
          for t in sort(filter(startswith("T_"), readdir(joinpath(BASE_DIR, l))))]

function load_runs(p)
    runs = Vector{Float64}[]
    for r in 1:N_RUNS
        path = joinpath(p.dir, "run$r", "final_wl.jld2")
        isfile(path) || continue
        lnQ = collect(Float64, load_wanglandau_jld2(path).logQ_N)
        lnQ .-= lnQ[1]                      # Q(N=0) = 1; already true of the stored values
        push!(runs, lnQ)
    end
    isempty(runs) && error("no completed runs in $(p.dir)")
    return runs
end

for p in points
    runs  = load_runs(p)
    lnQ   = mean(runs)
    N_max = length(lnQ) - 1
    @printf("λ = %.2f  T* = %.3f  (%d runs, N_max = %d)\n", p.λ, p.T, length(runs), N_max)

    res = yang_lee_zeros(lnQ, 0:N_max; prec_bits = PREC_BITS, max_iter = MAX_ITER, tol = TOL)
    ver = verify_yang_lee_zeros(res)
    @printf("  %3d roots  converged = %-5s  max rel. residual = %.1e  conjugate violations = %d\n",
            length(res.roots), res.converged, ver.max_relative_residual, ver.conjugate_pair_violations)

    out = joinpath(OUTPUT_DIR, @sprintf("sw_zeros_lam%.2f_T%.3f_avg.jld2", p.λ, p.T))
    jldsave(out; roots = res.roots, converged = res.converged, z_ref = res.z_ref,
                 max_res = ver.max_relative_residual, conj_viol = ver.conjugate_pair_violations)
    println("  saved $out")
end
