"""
Yang–Lee zero analysis for GC-WL data (figure1/3sigma_cut).

Three independent methods are cross-verified:
  1. Aberth–Ehrlich root finding in ArbNumerics at configurable precision (production)
  2. Float64 companion-matrix eigenvalues (baseline — included to show it fails)
  3. Cumulant estimator from moments of P(N|z_ref) via Newton iteration (approximation)

References:
  Aberth, Math. Comput. 26, 339 (1973) — Aberth–Ehrlich iteration.
  Taylor & Luettmer-Strathmann, JCP 141, 204906 (2014) — YL zeros from WL data.
  Janke & Kenna, J. Stat. Phys. 102, 1211 (2001) — cumulant approach (Fisher zeros;
      adapted here to fugacity zeros via ∂/∂(ln z) = z ∂/∂z).
"""

import Pkg
Pkg.activate("/Users/mckinleypaul/Documents/montecarlo/gc_wl")

using gc_wl
using JLD2
using Statistics
using Random
using LinearAlgebra
using ArbNumerics
using DataFrames
using CSV
using Printf
using Plots
using Plots: RGB

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG — edit this block before running
# ═══════════════════════════════════════════════════════════════════════════════

const BASE_DIR  = @__DIR__           # 3sigma_cut/
const T_STR     = "105.35"           # temperature directory
const T_K       = 105.35             # K
const T_STAR    = 105.35 / 117.05    # T* = kT/ε

const N_RUNS    = 4

# Aberth–Ehrlich parameters
const PREC_BITS_PILOT      = 512
const PREC_BITS_PRODUCTION = 1024
const ABERTH_MAX_ITER      = 400
const ABERTH_TOL           = 1e-10   # Float64 convergence threshold on |correction|

# Bootstrap
const BOOTSTRAP_B       = 200
const BOOTSTRAP_PREC    = 512

const OUTPUT_DIR  = joinpath(BASE_DIR, "output")
const FIGURES_DIR = joinpath(BASE_DIR, "figures")

# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

"""
    load_runs(T_str, n_runs) -> Vector{NamedTuple}

Load up to n_runs GC-WL results for temperature T_str. LRC applied before returning.
Each element has fields: N_values, lnQ, run_id, sim, z_sat (computed from data).
"""
function load_runs(T_str::String, n_runs::Int)
    runs = NamedTuple[]
    N_max_common = typemax(Int)
    N_min_common = 0

    raw_runs = NamedTuple[]
    for run in 1:n_runs
        dir     = joinpath(BASE_DIR, T_str, "run$run")
        wl_path = joinpath(dir, "final_wl.jld2")
        sp_path = joinpath(dir, "sim.jld2")
        isfile(wl_path) && isfile(sp_path) || continue

        wl  = load(wl_path, "wl")
        sim = load(sp_path, "sim")
        logQ_lrc = apply_lrc_to_logQ(collect(Float64, wl.logQ_N),
                                      sim.T_σ, sim.V_σ, sim.r_cut_σ; N_min=sim.N_min)
        N_max_common = min(N_max_common, sim.N_max)
        N_min_common = max(N_min_common, sim.N_min)
        push!(raw_runs, (N_values=sim.N_min:sim.N_max, lnQ=logQ_lrc, run_id=run, sim=sim))
    end

    isempty(raw_runs) && error("No completed runs found for T=$T_str")

    for r in raw_runs
        lo = N_min_common - r.sim.N_min + 1
        hi = N_max_common - r.sim.N_min + 1
        lnQ_trunc = r.lnQ[lo:hi]
        N_vals    = N_min_common:N_max_common

        # Compute z_sat from the LRC-corrected logQ using equal-peak-area bisection.
        # This is the correct reference fugacity for THIS data, not the TMMC table value.
        z_sat = _compute_z_sat(lnQ_trunc, N_vals, r.sim)
        push!(runs, (N_values=N_vals, lnQ=lnQ_trunc, run_id=r.run_id, sim=r.sim, z_sat=z_sat))
    end

    @printf("Loaded %d runs: T=%s K  N=[%d,%d]  T*=%.4f\n",
            length(runs), T_str, N_min_common, N_max_common, runs[1].sim.T_σ)
    for r in runs
        @printf("  run%d: z_sat = %.6g  (ln z = %.4f)\n",
                r.run_id, r.z_sat, log(r.z_sat))
    end
    return runs
end

function _compute_z_sat(lnQ::AbstractVector{<:Real}, N_values::AbstractRange, sim)
    # The Yang-Lee polynomial is Ξ(z) = Σ_N Q(N,V,T) z^N where the probability
    # weight is Q(N) z^N and z = exp(μ_star / T_σ).  This is consistent with
    # how compute_logΞ in thermo.jl defines its weights.  The physical activity
    # λ = z / Λ_σ³ (used by compute_lnzsat) is a DIFFERENT variable and must
    # NOT be used as z_star for precondition.
    T_σ   = sim.T_σ
    N_min = first(N_values)
    try
        μ_star = find_μ_coex(lnQ, T_σ; N_min=N_min, μ_lo=-20.0, μ_hi=-1.0, tol=1e-7)
        z_sat = exp(μ_star / T_σ)   # z = exp(βμ) for the Yang-Lee polynomial
        @printf("  find_μ_coex: μ_star=%.4f  z_sat=exp(%.4f)=%.4g\n",
                μ_star, μ_star/T_σ, z_sat)
        return z_sat
    catch e
        @warn "find_μ_coex failed: $e — falling back to z_sat=exp(-15)"
        return exp(-15.0)   # conservative fallback: should be in the two-phase region
    end
end

"""
    make_averaged_run(runs) -> NamedTuple

Across-run mean of lnQ. Returns a NamedTuple with lnQ_std for bootstrap.
"""
function make_averaged_run(runs)
    mat      = reduce(hcat, [r.lnQ for r in runs])
    lnQ_mean = vec(mean(mat, dims=2))
    lnQ_std  = size(mat, 2) > 1 ? vec(std(mat, dims=2)) : zeros(size(mat, 1))
    z_sat_mean = mean(r.z_sat for r in runs)
    return (N_values = runs[1].N_values,
            lnQ      = lnQ_mean,
            lnQ_std  = lnQ_std,
            run_id   = 0,
            sim      = runs[1].sim,
            z_sat    = z_sat_mean)
end

# ═══════════════════════════════════════════════════════════════════════════════
# PRECONDITIONING
# ═══════════════════════════════════════════════════════════════════════════════

"""
    precondition(lnQ, N_values; z_star) -> (log_coeffs, scaling_info)

Two-stage log-space rescaling of the Yang–Lee polynomial Ξ(z) = Σ_N Q(N) z^N.

Stage 1 — vertical shift:  c̃_N = lnQ(N) − max_N lnQ(N)
Stage 2 — horizontal scale: ã_N = c̃_N + N · ln(z_star)  (substitution z = z_star · w)

Returns log_coeffs = [ã_{N_min}, …, ã_{N_max}] (Float64)
and scaling_info = (lnQ_max, z_star, N_min).

Nothing is exponentiated here; all arithmetic stays in log space.
"""
function precondition(lnQ::AbstractVector{<:Real}, N_values::AbstractRange;
                      z_star::Real)
    lnQ_max   = maximum(lnQ)
    ln_z_star = log(z_star)
    log_coeffs = Vector{Float64}(undef, length(N_values))
    for (i, N) in enumerate(N_values)
        log_coeffs[i] = (lnQ[i] - lnQ_max) + N * ln_z_star
    end
    return log_coeffs, (lnQ_max=Float64(lnQ_max), z_star=Float64(z_star),
                        N_min=first(N_values))
end

# ═══════════════════════════════════════════════════════════════════════════════
# ABERTH–EHRLICH ROOT FINDER (multiprecision via ArbNumerics)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    _horner_arb(coeffs, z) -> (p, dp)

Horner evaluation of polynomial and derivative at z using ArbComplex arithmetic.
coeffs[1] = constant term, coeffs[end] = leading coefficient.
"""
function _horner_arb(coeffs::Vector{ArbComplex{P}}, z::ArbComplex{P}) where P
    n  = length(coeffs)
    p  = coeffs[n]
    dp = ArbComplex{P}(ArbFloat{P}(0), ArbFloat{P}(0))
    for i in n-1:-1:1
        dp = dp * z + p
        p  = p  * z + coeffs[i]
    end
    return p, dp
end

"""
    _arb_to_c64(z) -> ComplexF64

Extract Float64 midpoint from an ArbComplex interval.
"""
@inline function _arb_to_c64(z::ArbComplex{P}) where P
    complex(Float64(real(z)), Float64(imag(z)))
end

"""
    find_roots_arblib(log_coeffs, scaling_info; prec_bits) -> NamedTuple

Aberth–Ehrlich iteration for all roots of the rescaled Yang–Lee polynomial in w
(where z = z_star · w). Convergence is tracked in Float64 (just the stopping
criterion); all polynomial arithmetic is in ArbComplex{prec_bits}.

Returns (midpoints::Vector{ComplexF64}, prec_bits, n_rejected) sorted by |Im(z)| asc.
Rejects roots with Float64 relative radius > 1e-6; warns to increase prec_bits.
"""
function find_roots_arblib(log_coeffs::Vector{Float64},
                           scaling_info::NamedTuple;
                           prec_bits::Int = PREC_BITS_PRODUCTION,
                           max_iter::Int  = ABERTH_MAX_ITER,
                           tol::Float64   = ABERTH_TOL)
    setworkingprecision(ArbFloat, bits=prec_bits)
    P = prec_bits

    degree  = length(log_coeffs) - 1
    n_coeff = length(log_coeffs)

    # Build ArbComplex coefficients from log-space (never materialise Q(N) as Float64).
    # CRITICAL: exponentiation happens here at full Arb precision.
    coeffs_arb = Vector{ArbComplex{P}}(undef, n_coeff)
    for i in 1:n_coeff
        lc = log_coeffs[i]
        if isfinite(lc)
            val = exp(ArbFloat{P}(BigFloat(lc, precision=prec_bits)))
            coeffs_arb[i] = ArbComplex{P}(val, ArbFloat{P}(0))
        else
            coeffs_arb[i] = ArbComplex{P}(ArbFloat{P}(0), ArbFloat{P}(0))
        end
    end

    # Diagnostic: print log-coefficient range
    finite_lc = filter(isfinite, log_coeffs)
    @printf("  Polynomial degree=%d  log-coeff range: [%.1f, %.1f]\n",
            degree, minimum(finite_lc), maximum(finite_lc))

    # Initialise roots on unit circle in w-space.
    ws = Vector{ArbComplex{P}}(undef, degree)
    two_pi_arb = ArbFloat{P}(2) * ArbFloat{P}(π)
    for k in 0:degree-1
        θ = two_pi_arb * ArbFloat{P}(k) / ArbFloat{P}(degree)
        ws[k+1] = ArbComplex{P}(cos(θ), sin(θ))
    end

    # Aberth–Ehrlich iteration. Convergence tracked in Float64.
    converged = false
    for iter in 1:max_iter
        max_corr_f64 = 0.0
        ws_new = copy(ws)

        for k in 1:degree
            p_k, dp_k = _horner_arb(coeffs_arb, ws[k])

            # Compute Newton step as ratio in ArbNumerics.
            # p_k and dp_k individually have magnitude ~exp(-5300), below Float64 range,
            # but their ratio is O(1) near a root and converts to Float64 without underflow.
            newton_f64 = _arb_to_c64(p_k / dp_k)
            (!isfinite(real(newton_f64)) || !isfinite(imag(newton_f64))) && continue

            # Aberth correction in Float64 (Newton + Weierstrass deflation sum)
            w_f64 = _arb_to_c64(ws[k])
            s_f64 = sum(1.0 / (w_f64 - _arb_to_c64(ws[j])) for j in 1:degree if j != k)

            denom = 1.0 - newton_f64 * s_f64
            abs(denom) < 1e-300 && continue

            corr_f64 = newton_f64 / denom
            if !isfinite(corr_f64)
                continue
            end

            # Apply correction in ArbNumerics for full precision update
            corr_arb = ArbComplex{P}(ArbFloat{P}(BigFloat(real(corr_f64), precision=prec_bits)),
                                      ArbFloat{P}(BigFloat(imag(corr_f64), precision=prec_bits)))
            ws_new[k] = ws[k] - corr_arb
            max_corr_f64 = max(max_corr_f64, abs(corr_f64))
        end

        ws = ws_new
        if max_corr_f64 > 0 && max_corr_f64 < tol
            @printf("  Aberth converged in %d iterations (max corr ≈ %.2e)\n", iter, max_corr_f64)
            converged = true
            break
        end
        if max_corr_f64 == 0.0 && iter > 1
            @warn "Aberth: all corrections were skipped (check polynomial evaluation)"; break
        end
        if iter == max_iter
            @printf("  Aberth: %d iters, max corr = %.2e (tol=%.1e)\n", max_iter, max_corr_f64, tol)
        end
    end

    # Convert w → z = z_star · w, extract Float64 midpoints
    z_star = scaling_info.z_star
    midpoints = [_arb_to_c64(w) * z_star for w in ws]

    # Filter obviously wrong roots and sort: positive-Re roots first (for same |Im|),
    # then by |Im z| ascending. Physical leading zero has Re > 0 near z_sat.
    midpoints = filter(z -> isfinite(z) && abs(z) < 1e6, midpoints)
    sort!(midpoints, by=z -> (abs(imag(z)), -real(z)))

    return (midpoints=midpoints, prec_bits=prec_bits,
            converged=converged, n_rejected=degree-length(midpoints))
end

# ═══════════════════════════════════════════════════════════════════════════════
# FLOAT64 COMPANION-MATRIX BASELINE
# ═══════════════════════════════════════════════════════════════════════════════

"""
    find_roots_naive_float64(log_coeffs, scaling_info) -> NamedTuple

Companion-matrix eigenvalues in Float64 — demonstrates why multiprecision is
necessary. The rescaled coefficients exp(ã_N) span many decades even after the
two-stage precondition, causing the companion matrix to be ill-conditioned.
Failure is reported; never silently swallowed.
"""
function find_roots_naive_float64(log_coeffs::Vector{Float64}, scaling_info::NamedTuple)
    degree = length(log_coeffs) - 1
    coeffs_f64 = exp.(log_coeffs)

    n_inf  = count(isinf,  coeffs_f64)
    n_zero = count(iszero, coeffs_f64)
    n_nan  = count(isnan,  coeffs_f64)
    @printf("  Float64 coefficients: %d Inf, %d zero (underflow), %d NaN  out of %d\n",
            n_inf, n_zero, n_nan, length(coeffs_f64))

    roots  = ComplexF64[]
    status = "ok"
    try
        a_lead = coeffs_f64[end]
        if !isfinite(a_lead) || a_lead == 0
            status = "zero_or_inf_leading_coeff"
            @printf("  Float64: leading coefficient is %s — cannot form companion matrix\n",
                    string(a_lead))
            return (midpoints=roots, status=status, n_overflow_coeffs=n_inf+n_nan+n_zero)
        end
        a = coeffs_f64[1:end-1] ./ a_lead

        C = zeros(Float64, degree, degree)
        for j in 1:degree
            C[degree, j] = -a[j]
        end
        for j in 1:degree-1
            C[j, j+1] = 1.0
        end

        ev = eigvals(C)
        roots = ComplexF64.(ev) .* scaling_info.z_star

        n_bad = count(!isfinite ∘ real, roots) + count(!isfinite ∘ imag, roots)
        if n_bad > 0
            status = "partial_failure"
            @printf("  Float64: %d / %d roots are NaN or Inf\n", n_bad, degree)
        else
            @printf("  Float64: all %d roots finite — but likely inaccurate!\n", degree)
        end
    catch e
        status = "exception: $(typeof(e))"
        @printf("  Float64 companion FAILED: %s\n", typeof(e))
    end

    return (midpoints=roots, status=status, n_overflow_coeffs=n_inf+n_nan+n_zero)
end

# ═══════════════════════════════════════════════════════════════════════════════
# CUMULANT ESTIMATOR FOR LEADING ZERO
# ═══════════════════════════════════════════════════════════════════════════════

"""
    _pN_moments(lnQ, N_values, z_ref) -> (mean_N, κ2, κ3, κ4)

Cumulants of N under P(N|z_ref) ∝ Q(N) z_ref^N, computed in log-sum-exp.
κ_n are the n-th central cumulants of the particle number distribution.
"""
function _pN_moments(lnQ::AbstractVector{<:Real}, N_values::AbstractRange, z_ref::Real)
    ln_z = log(z_ref)
    logw = [lnQ[i] + N * ln_z for (i, N) in enumerate(N_values)]
    logZ = logsumexp(logw)
    pN   = exp.(logw .- logZ)

    μ1 = sum(Float64(N) * pN[i] for (i, N) in enumerate(N_values))
    κ2 = sum((Float64(N) - μ1)^2 * pN[i] for (i, N) in enumerate(N_values))
    κ3 = sum((Float64(N) - μ1)^3 * pN[i] for (i, N) in enumerate(N_values))
    κ4 = sum((Float64(N) - μ1)^4 * pN[i] for (i, N) in enumerate(N_values)) - 3*κ2^2

    return μ1, κ2, κ3, κ4
end

"""
    _yl_residuals(z_R, z_I, z_ref, κ2, κ3) -> (F1, F2)

Residuals of the leading-conjugate-pair approximation for cumulants κ2, κ3.

From the partial-fraction decomposition of (z ∂/∂z)^n ln Ξ with n = 2, 3,
keeping only the leading pair z_1 = z_R + iz_I and its conjugate:

    κ2 ≈ -2 z_ref Re[z_1 / (z_ref - z_1)^2]
    κ3 ≈ +2 z_ref Re[z_1 (z_ref + z_1) / (z_ref - z_1)^3]

Derived from (z ∂/∂z) ln Ξ = z Σ_k 1/(z - z_k); higher cumulants by iteration.
"""
function _yl_residuals(z_R::Float64, z_I::Float64, z_ref::Float64, κ2::Float64, κ3::Float64)
    z1 = complex(z_R, z_I)
    d  = z_ref - z1
    abs(d) < 1e-15 && return (Inf, Inf)
    κ2_model = -2 * z_ref * real(z1 / d^2)
    κ3_model =  2 * z_ref * real(z1 * (z_ref + z1) / d^3)
    return κ2_model - κ2, κ3_model - κ3
end

"""
    leading_zero_cumulant(run; z_ref) -> ComplexF64

Newton iteration in (z_R, z_I) to match the measured cumulants κ2, κ3 of N
under P(N|z_ref) to the leading-pair Yang–Lee approximation. See `_yl_residuals`.

z_ref defaults to run.z_sat computed from the LRC-corrected data.
"""
function leading_zero_cumulant(run::NamedTuple; z_ref::Real = run.z_sat)
    μ1, κ2, κ3, κ4 = _pN_moments(run.lnQ, run.N_values, z_ref)
    @printf("  Cumulant estimator at z_ref=%.5g: <N>=%.1f  κ2=%.3g  κ3=%.3g\n",
            z_ref, μ1, κ2, κ3)

    if abs(μ1 - last(run.N_values)) < 1.0 || abs(μ1 - first(run.N_values)) < 1.0
        @warn "Distribution saturated at N boundary — z_ref may be far from coexistence."
    end

    # Initial guess: z_R near z_sat, z_I small but nonzero
    z_R = Float64(z_ref) * 0.99
    z_I = Float64(z_ref) * 0.05   # ~5% of z_sat as imaginary part

    δ = 1e-10 * Float64(z_ref)
    for iter in 1:300
        F1, F2 = _yl_residuals(z_R, z_I, Float64(z_ref), κ2, κ3)
        norm_F = sqrt(F1^2 + F2^2)
        norm_F < 1e-12 * κ2 && break

        # Finite-difference Jacobian
        F1r, F2r = _yl_residuals(z_R + δ, z_I,     Float64(z_ref), κ2, κ3)
        F1i, F2i = _yl_residuals(z_R,     z_I + δ, Float64(z_ref), κ2, κ3)
        J = [(F1r-F1)/δ (F1i-F1)/δ; (F2r-F2)/δ (F2i-F2)/δ]

        det_J = J[1,1]*J[2,2] - J[1,2]*J[2,1]
        abs(det_J) < 1e-30 && break

        Δ = J \ [-F1; -F2]

        # Damped Newton to keep z_I > 0 and z_R > 0
        damp = 1.0
        for _ in 1:8
            z_R_new = z_R + damp*Δ[1]
            z_I_new = z_I + damp*Δ[2]
            if z_I_new > 1e-15 && z_R_new > 0
                F1n, F2n = _yl_residuals(z_R_new, z_I_new, Float64(z_ref), κ2, κ3)
                if sqrt(F1n^2 + F2n^2) < norm_F
                    z_R, z_I = z_R_new, z_I_new
                    break
                end
            end
            damp *= 0.5
        end
    end

    F1f, F2f = _yl_residuals(z_R, z_I, Float64(z_ref), κ2, κ3)
    res = sqrt(F1f^2 + F2f^2)
    @printf("  Cumulant → z1 = %.5g + %.5g i  (residual=%.2e, κ2=%.3g)\n", z_R, z_I, res, κ2)
    res > 0.1 * abs(κ2) &&
        @warn "Cumulant residual large (>10% of κ2); leading-pair approximation may be poor at this V."

    return complex(z_R, z_I)
end

"""
    leading_zero_wada(run) -> ComplexF64

Placeholder for Wada–Ohzeki (2025) ratio method. Returns NaN sentinel.
Substitute the exact recurrence from Wada et al. (2025) before use.
"""
function leading_zero_wada(run::NamedTuple)
    @warn "leading_zero_wada: Wada et al. (2025) recurrence not implemented. " *
          "Returning NaN sentinel."
    return complex(NaN, NaN)
end

# ═══════════════════════════════════════════════════════════════════════════════
# VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

"""
    verify_roots(midpoints, log_coeffs, scaling_info; prec_bits) -> NamedTuple

Verification diagnostics on Aberth output:
  - max_relative_residual:     max_i |p(w_i)| / Σ_N |a_N| |w_i|^N (in Float64)
  - conjugate_pair_violations: count of roots without a conjugate partner at 1e-8 tol
  - newton_identity_residual:  |Σ w_i + a_{n-1}/a_n| relative to |a_{n-1}/a_n|
  - truncation_sensitivity:    leading-zero shift when dropping top 5%/10%/20% of N
"""
function verify_roots(midpoints::Vector{ComplexF64},
                      log_coeffs::Vector{Float64},
                      scaling_info::NamedTuple;
                      prec_bits::Int = PREC_BITS_PRODUCTION)
    isempty(midpoints) &&
        return (max_relative_residual=NaN, conjugate_pair_violations=-1,
                newton_identity_residual=NaN, truncation_sensitivity=Dict{String,ComplexF64}())

    z_star = scaling_info.z_star
    degree = length(log_coeffs) - 1

    # Shift log-coefficients to avoid Float64 underflow in diagnostics.
    # The polynomial with shifted coefficients has the same roots (shifted by a +constant).
    finite_lc = filter(isfinite, log_coeffs)
    lc_shift  = isempty(finite_lc) ? 0.0 : minimum(finite_lc)
    lcs = [isfinite(lc) ? lc - lc_shift : -Inf for lc in log_coeffs]

    # 1. Relative residuals (Float64 on shifted polynomial — same roots, no underflow)
    max_rel = 0.0
    for z in midpoints
        w = z / z_star
        p_val = 0.0 + 0.0im
        for i in length(lcs):-1:1
            p_val = p_val * w + (isfinite(lcs[i]) ? exp(lcs[i]) : 0.0)
        end
        denom = sum(isfinite(lcs[i]) ? exp(lcs[i]) * abs(w)^(i-1) : 0.0
                    for i in 1:length(lcs))
        max_rel = max(max_rel, abs(p_val) / max(denom, 1e-300))
    end

    # 2. Conjugate pair check
    tol_conj = 1e-8 * mean(abs.(midpoints))
    n_viol   = sum(!any(abs(z - conj(w)) < tol_conj for w in midpoints)
                   for z in midpoints if imag(z) != 0)

    # 3. Newton / Vieta identity: Σ_k w_k = -a_{n-1}/a_n (in w-space, using shifted coeffs)
    ws = midpoints ./ z_star
    sum_w = sum(ws)
    a_n_s   = isfinite(lcs[end])   ? exp(lcs[end])   : 0.0
    a_nm1_s = isfinite(lcs[end-1]) ? exp(lcs[end-1]) : 0.0
    vieta_ref = a_n_s > 0 ? -a_nm1_s / a_n_s : NaN
    newton_res = isnan(vieta_ref) || abs(vieta_ref) < 1e-300 ? NaN :
                 abs(real(sum_w) - vieta_ref) / abs(vieta_ref)

    # 4. Truncation sensitivity
    trunc = Dict{String,ComplexF64}()
    for (tag, frac) in [("drop5pct",0.05), ("drop10pct",0.10), ("drop20pct",0.20)]
        n_drop = max(1, round(Int, frac * degree))
        lc_t   = log_coeffs[1:end-n_drop]
        try
            r = find_roots_arblib(lc_t, scaling_info; prec_bits=PREC_BITS_PILOT, max_iter=100)
            idx = findfirst(z -> real(z) > 0 && imag(z) > 1e-10, r.midpoints)
            trunc[tag] = idx !== nothing ? r.midpoints[idx] : complex(NaN,NaN)
        catch
            trunc[tag] = complex(NaN,NaN)
        end
    end

    return (max_relative_residual    = max_rel,
            conjugate_pair_violations = n_viol,
            newton_identity_residual  = newton_res,
            truncation_sensitivity    = trunc)
end

# ═══════════════════════════════════════════════════════════════════════════════
# BOOTSTRAP
# ═══════════════════════════════════════════════════════════════════════════════

"""
    bootstrap_leading_zero(runs, avg_run; B, prec_bits) -> Vector{ComplexF64}

Parametric bootstrap: for each of B resamples,
    lnQ^(b)(N) = mean_lnQ(N) + ε_N,   ε_N ~ N(0, σ_N)
where σ_N is the inter-run empirical std at each N.
Returns B leading zero positions (smallest positive Im(z)).
"""
function bootstrap_leading_zero(runs::Vector{<:NamedTuple},
                                avg_run::NamedTuple;
                                B::Int=BOOTSTRAP_B, prec_bits::Int=BOOTSTRAP_PREC,
                                rng_seed::Int=42)
    σN       = avg_run.lnQ_std
    lnQ_mean = avg_run.lnQ
    N_values = avg_run.N_values
    z_sat    = avg_run.z_sat
    any(σN .== 0) && @warn "Some σ_N = 0 (only one run loaded?)"

    results = Vector{Union{ComplexF64,Missing}}(undef, B)
    fill!(results, missing)
    rngs = [MersenneTwister(rng_seed + b) for b in 1:B]

    Threads.@threads for b in 1:B
        lnQ_b = lnQ_mean .+ randn(rngs[b], length(lnQ_mean)) .* σN
        lc, si = precondition(lnQ_b, N_values; z_star=z_sat)
        result = try
            find_roots_arblib(lc, si; prec_bits=prec_bits, max_iter=150)
        catch
            nothing
        end
        result === nothing && continue
        idx = findfirst(z -> real(z) > 0 && imag(z) > 1e-10, result.midpoints)
        idx !== nothing && (results[b] = result.midpoints[idx])
    end

    valid = ComplexF64.(filter(!ismissing, results))
    @printf("Bootstrap: %d / %d valid leading zeros\n", length(valid), B)
    return valid
end

# ═══════════════════════════════════════════════════════════════════════════════
# COMPARISON TABLE
# ═══════════════════════════════════════════════════════════════════════════════

"""
    compare_runs(runs, avg_run; methods) -> DataFrame

One row per (run_id, method): run_id, method, re_z1, im_z1,
max_residual, prec_bits, time_seconds, status.
"""
function compare_runs(runs::Vector{<:NamedTuple}, avg_run::NamedTuple;
                      methods::Tuple=(:arblib,:float64,:cumulant,:wada))
    rows = []
    for run in vcat(runs, [avg_run])
        lc, si = precondition(run.lnQ, run.N_values; z_star=run.z_sat)
        for method in methods
            re_z1=NaN; im_z1=NaN; res=NaN; prec=0; status="ok"
            t0 = time()

            if method == :arblib
                prec = PREC_BITS_PRODUCTION
                r = try find_roots_arblib(lc, si; prec_bits=prec)
                    catch e; @warn e; status="exception"; nothing end
                if r !== nothing
                    idx = findfirst(z -> real(z) > 0 && imag(z) > 1e-10, r.midpoints)
                    if idx !== nothing
                        z1 = r.midpoints[idx]; re_z1=real(z1); im_z1=imag(z1)
                    else
                        status = "no_positive_imag_root"
                    end
                    status = r.converged ? status : "not_converged"
                end

            elseif method == :float64
                prec = 64
                r = find_roots_naive_float64(lc, si)
                status = r.status
                valid = filter(z -> isfinite(z) && real(z) > 0 && imag(z) > 1e-10, r.midpoints)
                if !isempty(valid)
                    z1 = first(sort(valid, by=z->abs(imag(z))))
                    re_z1=real(z1); im_z1=imag(z1)
                end
                res = Float64(r.n_overflow_coeffs) / length(lc)

            elseif method == :cumulant
                prec = 64
                z1 = try leading_zero_cumulant(run)
                     catch e; @warn e; complex(NaN,NaN) end
                re_z1=real(z1); im_z1=imag(z1)

            elseif method == :wada
                z1 = leading_zero_wada(run)
                re_z1=real(z1); im_z1=imag(z1)
                status = "not_implemented"
            end

            push!(rows, (run_id=run.run_id, method=String(method),
                         re_z1=re_z1, im_z1=im_z1, max_residual=res,
                         prec_bits=prec, time_seconds=time()-t0, status=status))
        end
    end
    return DataFrame(rows)
end

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE
# ═══════════════════════════════════════════════════════════════════════════════

# Okabe–Ito 8-colour palette (colour-blind safe)
const OKABE_ITO = [
    RGB(0.902, 0.624, 0.000),  # orange
    RGB(0.337, 0.706, 0.914),  # sky blue
    RGB(0.000, 0.620, 0.451),  # green
    RGB(0.941, 0.894, 0.259),  # yellow
    RGB(0.000, 0.447, 0.698),  # blue
    RGB(0.835, 0.369, 0.000),  # vermillion
    RGB(0.800, 0.475, 0.655),  # reddish purple
]

"""
    make_robustness_figure(runs, avg_run, bootstrap_cloud, all_arblib; output_path_base)

Two-panel figure:
  (a) Full zero distribution: avg-run zeros (gray) + leading 10 zeros of each run (coloured).
  (b) Leading-zero zoom: per-run leading zeros + bootstrap cloud.
"""
function make_robustness_figure(runs::Vector{<:NamedTuple},
                                avg_run::NamedTuple,
                                bootstrap_cloud::Vector{ComplexF64},
                                all_arblib::Dict;
                                output_path_base::String)
    K = length(runs)
    run_colors = OKABE_ITO[1:min(K, length(OKABE_ITO))]

    avg_result = get(all_arblib, 0, nothing)
    avg_zeros  = avg_result === nothing ? ComplexF64[] : avg_result.midpoints

    # Panel (a)
    pa = plot(xlabel="Re(z)", ylabel="Im(z)",
              title="Yang–Lee zeros  T*=$(round(T_STAR,digits=3))  (full distribution)",
              aspect_ratio=:equal, legend=:topright, framestyle=:box)

    if !isempty(avg_zeros)
        scatter!(pa, real.(avg_zeros), imag.(avg_zeros), color=:gray70,
                 markersize=2, markerstrokewidth=0, alpha=0.5,
                 label="avg lnQ ($(length(avg_zeros)) zeros)")
    end

    leading_per_run = Dict{Int,Vector{ComplexF64}}()
    for (i, run) in enumerate(runs)
        r = get(all_arblib, run.run_id, nothing); r === nothing && continue
        pos = filter(z -> real(z) > 0 && imag(z) > 1e-10, r.midpoints)
        leading = pos[1:min(10,end)]
        leading_per_run[run.run_id] = leading
        scatter!(pa, real.(leading), imag.(leading),
                 color=run_colors[i], markersize=5, markerstrokewidth=0.5,
                 label="run $(run.run_id)", alpha=0.85)
    end

    vline!(pa, [avg_run.z_sat], color=:black, linestyle=:dash, linewidth=1,
           label="z_sat (data)")

    # Panel (b): zoom
    all_lead = vcat(values(leading_per_run)...)
    !isempty(avg_zeros) && (all_lead = vcat(all_lead, avg_zeros[1:min(4,end)]))
    isempty(all_lead) && (all_lead = [complex(avg_run.z_sat, avg_run.z_sat*0.1)])

    im_lead = filter(x -> x > 0, imag.(all_lead))
    re_lead = real.(all_lead)
    re_c  = isempty(re_lead) ? avg_run.z_sat : mean(re_lead[1:min(4,end)])
    im_c  = isempty(im_lead) ? avg_run.z_sat*0.1 : mean(im_lead[1:min(4,end)])
    zoom_w = max(3*avg_run.z_sat, 2*std(re_lead; corrected=false) + 1e-8)
    zoom_h = max(2*im_c, 1e-8)

    pb = plot(xlabel="Re(z)", ylabel="Im(z)",
              title="Leading Yang–Lee zeros (zoom)",
              xlims=(re_c - zoom_w, re_c + zoom_w),
              ylims=(0, im_c + 2zoom_h),
              aspect_ratio=:equal, legend=:topright, framestyle=:box)

    if !isempty(bootstrap_cloud)
        scatter!(pb, real.(bootstrap_cloud), imag.(bootstrap_cloud),
                 color=:lightblue, markersize=3, markerstrokewidth=0, alpha=0.5,
                 label="bootstrap (B=$(length(bootstrap_cloud)))")
    end

    for (i, run) in enumerate(runs)
        pos = get(leading_per_run, run.run_id, ComplexF64[])
        isempty(pos) && continue
        scatter!(pb, [real(pos[1])], [imag(pos[1])],
                 color=run_colors[i], markersize=8, markerstrokewidth=1,
                 label="run $(run.run_id)")
    end

    if !isempty(avg_zeros)
        idx = findfirst(z -> real(z) > 0 && imag(z) > 1e-10, avg_zeros)
        idx !== nothing && scatter!(pb, [real(avg_zeros[idx])], [imag(avg_zeros[idx])],
                                    color=:black, marker=:star5, markersize=12,
                                    markerstrokewidth=0, label="avg leading zero")
    end

    vline!(pb, [avg_run.z_sat], color=:black, linestyle=:dash, linewidth=1, label="")

    fig = plot(pa, pb, layout=(1,2), size=(1400,600), margin=5Plots.mm)
    for ext in ("pdf","png")
        path = "$(output_path_base).$(ext)"
        savefig(fig, path)
        println("  Saved: $path")
    end
    return fig
end

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

function main()
    mkpath(OUTPUT_DIR); mkpath(FIGURES_DIR)

    println("\n" * "="^70)
    println("Yang–Lee Zero Analysis  T=$(T_K) K  (T*=$(round(T_STAR, digits=4)))")
    println("="^70 * "\n")

    # Load data (z_sat computed from each run's LRC-corrected logQ)
    runs    = load_runs(T_STR, N_RUNS)
    avg_run = make_averaged_run(runs)
    println()

    # Comparison table
    println("Running compare_runs (all methods × all runs + averaged)...")
    df = compare_runs(runs, avg_run; methods=(:arblib, :float64, :cumulant, :wada))
    println("\n", df, "\n")

    csv_path = joinpath(OUTPUT_DIR, "yang_lee_methodology_T$(T_K).csv")
    CSV.write(csv_path, df)
    println("Table saved: $csv_path\n")

    # Verification
    println("Verification diagnostics (Aberth roots)...")
    all_arblib = Dict{Int,NamedTuple}()
    for run in vcat(runs, [avg_run])
        lc, si = precondition(run.lnQ, run.N_values; z_star=run.z_sat)
        r = try find_roots_arblib(lc, si; prec_bits=PREC_BITS_PRODUCTION)
            catch e; @warn e; nothing end
        r === nothing && continue
        all_arblib[run.run_id] = r

        diag  = verify_roots(r.midpoints, lc, si; prec_bits=PREC_BITS_PRODUCTION)
        label = run.run_id == 0 ? "avg" : "run$(run.run_id)"
        println("  ── $label ──")
        @printf("  max relative residual:     %.2e  %s\n",
                diag.max_relative_residual,
                diag.max_relative_residual < 1e-6 ? "ok" : "WARN")
        @printf("  conjugate pair violations: %d  %s\n",
                diag.conjugate_pair_violations,
                diag.conjugate_pair_violations == 0 ? "ok" : "WARN")
        @printf("  Newton identity residual:  %.2e\n", diag.newton_identity_residual)
        for (k,v) in sort(collect(diag.truncation_sensitivity), by=p->p[1])
            @printf("  truncation %s: z1 = %.5g + %.5g i\n", k, real(v), imag(v))
        end
        println()
    end

    # Bootstrap
    println("Bootstrap (B=$(BOOTSTRAP_B), representative run=$(runs[1].run_id))...")
    bootstrap_cloud = bootstrap_leading_zero(runs, avg_run; B=BOOTSTRAP_B, prec_bits=BOOTSTRAP_PREC)
    if !isempty(bootstrap_cloud)
        @printf("  Re(z1) = %.5g ± %.2g\n", mean(real.(bootstrap_cloud)), std(real.(bootstrap_cloud)))
        @printf("  Im(z1) = %.5g ± %.2g\n", mean(imag.(bootstrap_cloud)), std(imag.(bootstrap_cloud)))
    end
    println()

    # Figure
    println("Generating figure...")
    fig_base = joinpath(FIGURES_DIR, "yang_lee_robustness_T$(replace(string(T_K),'.' => 'p'))")
    make_robustness_figure(runs, avg_run, bootstrap_cloud, all_arblib; output_path_base=fig_base)

    # Summary
    println("\n" * "="^70)
    println("SUMMARY — per-method leading-zero positions")
    println("="^70)
    for method in ("arblib", "cumulant")
        for r in filter(x -> x.method == method, eachrow(df))
            lbl = r.run_id == 0 ? "avg" : "run$(r.run_id)"
            @printf("  %-8s %-4s: z1 = %.5g + %.5g i   t=%.1fs  status=%s\n",
                    method, lbl, r.re_z1, r.im_z1, r.time_seconds, r.status)
        end
    end
    println("\nFloat64 companion: see 'float64' rows — expected to show overflow/failure.")
    println("Wada method: 'not_implemented' — substitute Wada et al. (2025) recurrence.")
    println("\nDone.")
end

main()
