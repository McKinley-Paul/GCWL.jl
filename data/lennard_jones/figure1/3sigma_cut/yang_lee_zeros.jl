"""
Yang–Lee zero analysis for GC-WL data (figure1/3sigma_cut).

Two methods:
  1. Aberth–Ehrlich root finding in ArbNumerics at configurable precision (production)
  2. Float64 companion-matrix eigenvalues (baseline — included to show it fails)

Verification:
  (a) Taylor-style multi-run overlay of zero maps (run-to-run scatter = uncertainty)
  (b) Thermodynamic observable reconstruction: ρ(μ) and κ_T(μ) from roots vs from ln Q

References:
  Aberth, Math. Comput. 26, 339 (1973) — Aberth–Ehrlich iteration.
  Taylor & Luettmer-Strathmann, JCP 141, 204906 (2014) — YL zeros from WL data.
"""

import Pkg
Pkg.activate("/Users/mckinleypaul/Documents/montecarlo/gc_wl")

using gc_wl
using JLD2
using Statistics
using LinearAlgebra
using ArbNumerics
using DataFrames
using CSV
using Printf
using Plots
using Plots: RGB

# ═══════════════════════════════════════════════════════════════════════════════
# PHYSICAL CONSTANTS (argon / LJ)
# ═══════════════════════════════════════════════════════════════════════════════

const k_B      = 1.380649e-23
const N_A      = 6.02214076e23
const ε_kB_Ar  = 117.05          # K
const M_Ar     = 39.948e-3       # kg/mol
const LJ_TO_KJKG = ε_kB_Ar * k_B * N_A / M_Ar / 1000.0   # kJ/kg per LJ ε unit

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

const BASE_DIR  = @__DIR__
const T_STR     = "105.35"
const T_K       = 105.35
const T_STAR    = 105.35 / 117.05

const N_RUNS    = 4

const PREC_BITS_PRODUCTION = 1024
const ABERTH_MAX_ITER      = 400
const ABERTH_TOL           = 1e-10

const OUTPUT_DIR  = joinpath(BASE_DIR, "output")
const FIGURES_DIR = joinpath(BASE_DIR, "figures")

# ═══════════════════════════════════════════════════════════════════════════════
# DATA-DRIVEN Z_REF SELECTION
# ═══════════════════════════════════════════════════════════════════════════════

"""
    compute_z_ref(lnQ, N_values, sim) -> z_ref

Centered finite difference at N_mid = (N_min + N_max) / 2:
    ln z_ref = -(lnQ[i_mid+1] - lnQ[i_mid-1]) / 2

This places the peak of ln Q(N) + N ln z_ref at N_mid so the polynomial
coefficients are roughly symmetric and roots cluster near |w| = 1.

Prints ln z_ref, preconditioned coefficient range, and an equal-area
consistency check (which will fail gracefully for supercritical T).
"""
function compute_z_ref(lnQ::AbstractVector{<:Real}, N_values::AbstractRange, sim)
    n = length(lnQ)
    n < 3 && error("compute_z_ref: need at least 3 points")

    i_mid = clamp(n ÷ 2 + 1, 2, n-1)
    lnQ_f = Float64.(lnQ)
    ln_z_ref = -(lnQ_f[i_mid+1] - lnQ_f[i_mid-1]) / 2
    z_ref = exp(ln_z_ref)

    lnQ_max = maximum(lnQ)
    a_tilde = [(lnQ[i] - lnQ_max) + N * ln_z_ref for (i, N) in enumerate(N_values)]
    @printf("    ln z_ref = %+.4f   z_ref = %.5g\n", ln_z_ref, z_ref)
    @printf("    ã_N range: [%.1f, %.1f]\n", minimum(a_tilde), maximum(a_tilde))

    try
        μ_ea = find_μ_coex(lnQ, sim.T_σ; N_min=first(N_values),
                            μ_lo=-20.0, μ_hi=-5.0, tol=1e-7)
        ln_z_ea = μ_ea / sim.T_σ
        @printf("    Equal-area check: ln z_sat(EA) = %+.4f   Δln z = %+.4f\n",
                ln_z_ea, ln_z_ref - ln_z_ea)
    catch
        @printf("    Equal-area check: failed (no bimodal p(N) — likely supercritical)\n")
    end

    return z_ref
end

# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

function load_runs(T_str::String, n_runs::Int)
    raw_runs = NamedTuple[]
    N_max_common = typemax(Int)
    N_min_common = 0

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

    runs = NamedTuple[]
    for r in raw_runs
        lo = N_min_common - r.sim.N_min + 1
        hi = N_max_common - r.sim.N_min + 1
        lnQ_t = r.lnQ[lo:hi]
        N_vals = N_min_common:N_max_common
        @printf("  run%d z_ref:\n", r.run_id)
        z_ref = compute_z_ref(lnQ_t, N_vals, r.sim)
        push!(runs, (N_values=N_vals, lnQ=lnQ_t, run_id=r.run_id, sim=r.sim, z_ref=z_ref))
    end

    @printf("Loaded %d runs: T=%s K  N=[%d,%d]  T*=%.4f\n",
            length(runs), T_str, N_min_common, N_max_common, runs[1].sim.T_σ)
    return runs
end

function make_averaged_run(runs)
    mat      = reduce(hcat, [r.lnQ for r in runs])
    lnQ_mean = vec(mean(mat, dims=2))
    lnQ_std  = size(mat, 2) > 1 ? vec(std(mat, dims=2)) : zeros(size(mat, 1))
    println("  avg run z_ref:")
    z_ref = compute_z_ref(lnQ_mean, runs[1].N_values, runs[1].sim)
    return (N_values=runs[1].N_values, lnQ=lnQ_mean, lnQ_std=lnQ_std,
            run_id=0, sim=runs[1].sim, z_ref=z_ref)
end

# ═══════════════════════════════════════════════════════════════════════════════
# PRECONDITIONING
# ═══════════════════════════════════════════════════════════════════════════════

function precondition(lnQ::AbstractVector{<:Real}, N_values::AbstractRange; z_star::Real)
    lnQ_max    = maximum(lnQ)
    ln_z_star  = log(z_star)
    log_coeffs = [(lnQ[i] - lnQ_max) + N * ln_z_star for (i, N) in enumerate(N_values)]
    return log_coeffs, (lnQ_max=Float64(lnQ_max), z_star=Float64(z_star),
                        N_min=first(N_values))
end

# ═══════════════════════════════════════════════════════════════════════════════
# ABERTH–EHRLICH ROOT FINDER (ArbNumerics)
# ═══════════════════════════════════════════════════════════════════════════════

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

@inline _arb_to_c64(z::ArbComplex{P}) where P = complex(Float64(real(z)), Float64(imag(z)))

function find_roots_arblib(log_coeffs::Vector{Float64},
                           scaling_info::NamedTuple;
                           prec_bits::Int = PREC_BITS_PRODUCTION,
                           max_iter::Int  = ABERTH_MAX_ITER,
                           tol::Float64   = ABERTH_TOL)
    setworkingprecision(ArbFloat, bits=prec_bits)
    P       = prec_bits
    degree  = length(log_coeffs) - 1
    n_coeff = length(log_coeffs)

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

    finite_lc = filter(isfinite, log_coeffs)
    @printf("  Polynomial degree=%d  log-coeff range: [%.1f, %.1f]\n",
            degree, minimum(finite_lc), maximum(finite_lc))

    ws = Vector{ArbComplex{P}}(undef, degree)
    two_pi = ArbFloat{P}(2) * ArbFloat{P}(π)
    for k in 0:degree-1
        θ = two_pi * ArbFloat{P}(k) / ArbFloat{P}(degree)
        ws[k+1] = ArbComplex{P}(cos(θ), sin(θ))
    end

    converged = false
    for iter in 1:max_iter
        max_corr = 0.0
        ws_new   = copy(ws)
        for k in 1:degree
            p_k, dp_k = _horner_arb(coeffs_arb, ws[k])
            newton    = _arb_to_c64(p_k / dp_k)
            (!isfinite(real(newton)) || !isfinite(imag(newton))) && continue
            w_f64 = _arb_to_c64(ws[k])
            s     = sum(1.0 / (w_f64 - _arb_to_c64(ws[j])) for j in 1:degree if j != k)
            denom = 1.0 - newton * s
            abs(denom) < 1e-300 && continue
            corr = newton / denom
            !isfinite(corr) && continue
            corr_arb = ArbComplex{P}(ArbFloat{P}(BigFloat(real(corr), precision=prec_bits)),
                                      ArbFloat{P}(BigFloat(imag(corr), precision=prec_bits)))
            ws_new[k] = ws[k] - corr_arb
            max_corr  = max(max_corr, abs(corr))
        end
        ws = ws_new
        if max_corr > 0 && max_corr < tol
            @printf("  Aberth converged in %d iterations (max corr ≈ %.2e)\n", iter, max_corr)
            converged = true; break
        end
        max_corr == 0.0 && iter > 1 && (@warn "Aberth: all corrections skipped"; break)
        iter == max_iter &&
            @printf("  Aberth: %d iters, max corr = %.2e (tol=%.1e)\n", max_iter, max_corr, tol)
    end

    z_star    = scaling_info.z_star
    midpoints = [_arb_to_c64(w) * z_star for w in ws]
    midpoints = filter(z -> isfinite(z) && abs(z) < 1e6, midpoints)
    sort!(midpoints, by=z -> (abs(imag(z)), -real(z)))
    return (midpoints=midpoints, prec_bits=prec_bits,
            converged=converged, n_rejected=degree-length(midpoints))
end

# ═══════════════════════════════════════════════════════════════════════════════
# FLOAT64 COMPANION-MATRIX BASELINE
# ═══════════════════════════════════════════════════════════════════════════════

function find_roots_naive_float64(log_coeffs::Vector{Float64}, scaling_info::NamedTuple)
    degree     = length(log_coeffs) - 1
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
        for j in 1:degree; C[degree, j] = -a[j]; end
        for j in 1:degree-1; C[j, j+1] = 1.0; end
        ev    = eigvals(C)
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
# VERIFICATION — conjugate-pair check + residuals only
# ═══════════════════════════════════════════════════════════════════════════════

function verify_roots(midpoints::Vector{ComplexF64},
                      log_coeffs::Vector{Float64},
                      scaling_info::NamedTuple;
                      prec_bits::Int = PREC_BITS_PRODUCTION)
    isempty(midpoints) &&
        return (max_relative_residual=NaN, conjugate_pair_violations=-1)

    setworkingprecision(ArbFloat, bits=prec_bits)
    P       = prec_bits
    z_star  = scaling_info.z_star
    n_coeff = length(log_coeffs)

    # Rebuild Arb coefficients — avoids Float64 overflow for |w| >> 1.
    # With z_star ~ 5e-6 and |z| up to 1e6, |w| can reach 2e11 and |w|^450 ~ 10^5000,
    # which overflows Float64 (max ~10^308) but is exact in ArbNumerics.
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

    max_rel = 0.0
    for z in midpoints
        w_arb = ArbComplex{P}(ArbFloat{P}(BigFloat(real(z/z_star), precision=prec_bits)),
                               ArbFloat{P}(BigFloat(imag(z/z_star), precision=prec_bits)))
        p_val, _ = _horner_arb(coeffs_arb, w_arb)

        # Denominator Σ |a_i| |w|^(i-1) accumulated iteratively to avoid repeated pow
        abs_w     = abs(w_arb)
        abs_w_pow = ArbFloat{P}(1)
        denom     = ArbFloat{P}(0)
        for i in 1:n_coeff
            denom    += abs(coeffs_arb[i]) * abs_w_pow
            abs_w_pow *= abs_w
        end

        rel = denom > ArbFloat{P}(0) ? Float64(abs(p_val) / denom) : NaN
        isfinite(rel) && (max_rel = max(max_rel, rel))
    end

    tol_conj = 1e-8 * mean(abs.(midpoints))
    n_viol   = sum(!any(abs(z - conj(w)) < tol_conj for w in midpoints)
                   for z in midpoints if imag(z) != 0)

    return (max_relative_residual=max_rel, conjugate_pair_violations=n_viol)
end

# ═══════════════════════════════════════════════════════════════════════════════
# THERMODYNAMIC OBSERVABLE RECONSTRUCTION FROM ROOTS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    reconstruct_rho_from_roots(roots, z, V, N_min) -> ρ

    <N>(z) = N_min + Σ_k  z / (z - z_k)
    ρ = <N> / V

The sum is real for real z because z_k come in conjugate pairs.
"""
function reconstruct_rho_from_roots(roots::Vector{ComplexF64},
                                     z::Float64, V::Float64, N_min::Int)
    mean_N = Float64(N_min) + real(sum(z / (z - zk) for zk in roots))
    return mean_N / V
end

"""
    reconstruct_kT_from_roots(roots, z, V, T_star, N_min) -> κ_T

    Var(N)(z) = -Σ_k  z z_k / (z - z_k)²
    κ_T = V / (T* <N>²) × Var(N)
"""
function reconstruct_kT_from_roots(roots::Vector{ComplexF64},
                                    z::Float64, V::Float64,
                                    T_star::Float64, N_min::Int)
    mean_N = Float64(N_min) + real(sum(z / (z - zk) for zk in roots))
    mean_N < 0.1 && return NaN
    var_N  = real(sum(-z * zk / (z - zk)^2 for zk in roots))
    return V / (T_star * mean_N^2) * var_N
end

function compute_var_N_lnQ(logQ_N::AbstractVector{<:Real}, μ_star::Real, T_σ::Real;
                            N_min::Int=0)
    N_max  = length(logQ_N) - 1
    pN     = compute_pN(logQ_N, μ_star, T_σ; N_min=N_min)
    N_vals = N_min:N_max
    μ1     = sum(Float64(N) * pN[i] for (i, N) in enumerate(N_vals))
    μ2     = sum(Float64(N)^2 * pN[i] for (i, N) in enumerate(N_vals))
    return μ2 - μ1^2
end

# ═══════════════════════════════════════════════════════════════════════════════
# COMPARISON TABLE
# ═══════════════════════════════════════════════════════════════════════════════

function compare_runs(runs::Vector{<:NamedTuple}, avg_run::NamedTuple)
    rows = []
    for run in vcat(runs, [avg_run])
        lc, si = precondition(run.lnQ, run.N_values; z_star=run.z_ref)

        z_sat_ea = NaN
        try
            μ_ea = find_μ_coex(run.lnQ, run.sim.T_σ; N_min=first(run.N_values),
                                μ_lo=-20.0, μ_hi=-5.0, tol=1e-7)
            z_sat_ea = exp(μ_ea / run.sim.T_σ)
        catch end

        for method in (:arblib, :float64)
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
                    !r.converged && (status = "not_converged")
                end

            elseif method == :float64
                prec = 64
                r    = find_roots_naive_float64(lc, si)
                status = r.status
                valid = filter(z -> isfinite(z) && real(z) > 0 && imag(z) > 1e-10, r.midpoints)
                if !isempty(valid)
                    z1 = first(sort(valid, by=z -> abs(imag(z))))
                    re_z1=real(z1); im_z1=imag(z1)
                end
                res = Float64(r.n_overflow_coeffs) / length(lc)
            end

            push!(rows, (run_id=run.run_id, method=String(method),
                         re_z1=re_z1, im_z1=im_z1,
                         z_sat_YL=re_z1, z_sat_EA=z_sat_ea,
                         max_residual=res, prec_bits=prec,
                         time_seconds=time()-t0, status=status))
        end
    end
    return DataFrame(rows)
end

# ═══════════════════════════════════════════════════════════════════════════════
# OBSERVABLE VERIFICATION FIGURE
# ═══════════════════════════════════════════════════════════════════════════════

function make_observables_figure(avg_run::NamedTuple,
                                  avg_roots::Vector{ComplexF64};
                                  output_path_base::String)
    sim   = avg_run.sim
    T_σ   = sim.T_σ
    V_σ   = sim.V_σ
    N_min = first(avg_run.N_values)
    logQ  = avg_run.lnQ

    μ_scan    = range(-18.0, -7.0, length=300)
    μ_markers = range(-18.0, -7.0, length=60)

    # Direct from ln Q
    rho_direct = [compute_mean_N(logQ, μ, T_σ; N_min=N_min) / V_σ for μ in μ_scan]
    kT_direct  = [let vN = compute_var_N_lnQ(logQ, μ, T_σ; N_min=N_min),
                       mN = compute_mean_N(logQ, μ, T_σ; N_min=N_min)
                   mN < 0.1 ? NaN : V_σ / (T_σ * mN^2) * vN
                   end for μ in μ_scan]

    # From Yang–Lee roots (subsample for speed)
    rho_roots = [reconstruct_rho_from_roots(avg_roots, exp(μ/T_σ), V_σ, N_min)
                 for μ in μ_markers]
    kT_roots  = [reconstruct_kT_from_roots(avg_roots, exp(μ/T_σ), V_σ, T_σ, N_min)
                 for μ in μ_markers]

    μ_kJkg_scan    = collect(μ_scan)    .* LJ_TO_KJKG
    μ_kJkg_markers = collect(μ_markers) .* LJ_TO_KJKG

    pl = plot(layout=(1,2), size=(1200,500), margin=5Plots.mm)

    plot!(pl[1], μ_kJkg_scan, rho_direct,
          xlabel="μ (kJ/kg)", ylabel="ρ (σ⁻³)",
          title="Density ρ(μ)  T*=$(round(T_STAR, digits=3))",
          label="from ln Q", color=:black, linewidth=2, framestyle=:box)
    scatter!(pl[1], μ_kJkg_markers, rho_roots,
             label="from roots", color=:crimson, markersize=4, markerstrokewidth=0)

    plot!(pl[2], μ_kJkg_scan, kT_direct,
          xlabel="μ (kJ/kg)", ylabel="κ_T (ε⁻¹σ³)",
          title="Compressibility κ_T(μ)  T*=$(round(T_STAR, digits=3))",
          label="from ln Q", color=:black, linewidth=2, framestyle=:box)
    scatter!(pl[2], μ_kJkg_markers, kT_roots,
             label="from roots", color=:crimson, markersize=4, markerstrokewidth=0)

    for ext in ("pdf", "png")
        path = "$(output_path_base).$(ext)"
        savefig(pl, path)
        println("  Saved: $path")
    end
    return pl
end

# ═══════════════════════════════════════════════════════════════════════════════
# TAYLOR-STYLE MULTI-RUN OVERLAY FIGURE
# ═══════════════════════════════════════════════════════════════════════════════

const OKABE_ITO = [
    RGB(0.902, 0.624, 0.000),  # orange
    RGB(0.337, 0.706, 0.914),  # sky blue
    RGB(0.000, 0.620, 0.451),  # green
    RGB(0.000, 0.447, 0.698),  # blue
    RGB(0.835, 0.369, 0.000),  # vermillion
    RGB(0.800, 0.475, 0.655),  # reddish purple
]

function make_overlay_figure(runs::Vector{<:NamedTuple},
                              avg_run::NamedTuple,
                              all_arblib::Dict;
                              output_path_base::String)
    K          = length(runs)
    run_colors = OKABE_ITO[1:min(K, length(OKABE_ITO))]
    avg_result = get(all_arblib, 0, nothing)
    avg_zeros  = avg_result === nothing ? ComplexF64[] : avg_result.midpoints

    # Vertical reference line: prefer z_sat(EA), fall back to z_ref
    z_line     = avg_run.z_ref
    line_label = "z_ref"
    try
        μ_ea = find_μ_coex(avg_run.lnQ, avg_run.sim.T_σ; N_min=first(avg_run.N_values),
                            μ_lo=-20.0, μ_hi=-5.0, tol=1e-7)
        z_line     = exp(μ_ea / avg_run.sim.T_σ)
        line_label = "z_sat(EA)"
    catch end

    # ── Panel (a): full zero distribution ──────────────────────────────────────
    pa = plot(xlabel="Re(z)", ylabel="Im(z)",
              title="Yang–Lee zeros  T*=$(round(T_STAR,digits=3))  (all runs)",
              aspect_ratio=:equal, legend=:topright, framestyle=:box)

    if !isempty(avg_zeros)
        scatter!(pa, real.(avg_zeros), imag.(avg_zeros),
                 color=:gray70, markersize=3, markerstrokewidth=0.5,
                 markerstrokealpha=0.4, alpha=0.5,
                 label="avg lnQ ($(length(avg_zeros)) zeros)")
    end

    per_run_zeros = Dict{Int,Vector{ComplexF64}}()
    for (i, run) in enumerate(runs)
        r = get(all_arblib, run.run_id, nothing); r === nothing && continue
        per_run_zeros[run.run_id] = r.midpoints
        scatter!(pa, real.(r.midpoints), imag.(r.midpoints),
                 color=run_colors[i], markersize=2, markerstrokewidth=0, alpha=0.7,
                 label="run $(run.run_id)")
    end
    vline!(pa, [z_line], color=:black, linestyle=:dash, linewidth=1, label=line_label)

    # ── Panel (b): zoom on leading 3–5 zeros ───────────────────────────────────
    leading_per_run = Dict{Int,Vector{ComplexF64}}()
    for (run_id, zeros) in per_run_zeros
        pos = filter(z -> real(z) > 0 && imag(z) > 1e-10, zeros)
        leading_per_run[run_id] = pos[1:min(5, end)]
    end
    avg_leading = filter(z -> real(z) > 0 && imag(z) > 1e-10, avg_zeros)

    # Collect all displayed leading zeros to compute tight axis bounds
    re_zoom = Float64[]
    im_zoom = Float64[]
    for run in runs
        pts = get(leading_per_run, run.run_id, ComplexF64[])
        append!(re_zoom, real.(pts))
        append!(im_zoom, filter(>(0.0), imag.(pts)))
    end
    if !isempty(avg_leading)
        n_show_avg = min(5, length(avg_leading))
        append!(re_zoom, real.(avg_leading[1:n_show_avg]))
        append!(im_zoom, filter(>(0.0), imag.(avg_leading[1:n_show_avg])))
    end
    isempty(re_zoom) && push!(re_zoom, z_line)
    isempty(im_zoom) && push!(im_zoom, z_line * 0.01)

    re_lo, re_hi = extrema(re_zoom)
    im_hi        = maximum(im_zoom)
    re_spread    = max(re_hi - re_lo, im_hi * 0.5)   # never narrower than half Im range
    pb = plot(xlabel="Re(z)", ylabel="Im(z)",
              title="Leading Yang–Lee zeros (zoom)",
              xlims=(re_lo - 2*re_spread, re_hi + 2*re_spread),
              ylims=(0, im_hi * 3),
              legend=:topright, framestyle=:box)

    for (i, run) in enumerate(runs)
        pos = get(leading_per_run, run.run_id, ComplexF64[])
        isempty(pos) && continue
        scatter!(pb, real.(pos), imag.(pos),
                 color=run_colors[i], markersize=8, markerstrokewidth=1,
                 label="run $(run.run_id)")
    end
    if !isempty(avg_leading)
        scatter!(pb, real.(avg_leading[1:min(5,end)]), imag.(avg_leading[1:min(5,end)]),
                 color=:black, marker=:star5, markersize=12, markerstrokewidth=0,
                 label="avg leading zeros")
    end
    vline!(pb, [z_line], color=:black, linestyle=:dash, linewidth=1, label="")

    fig = plot(pa, pb, layout=(1,2), size=(1400,600), margin=5Plots.mm)
    for ext in ("pdf", "png")
        path = "$(output_path_base).$(ext)"
        savefig(fig, path)
        println("  Saved: $path")
    end

    println("""
Caption: Yang–Lee zeros in the complex fugacity plane at T* = $(round(T_STAR, digits=3)) \
from K = $K independent GC-WL runs. Individual root positions away from the positive \
real axis vary noticeably between runs, but the leading roots and the boundary structure \
of the distribution are well reproduced. The leading-zero position is the physically \
relevant quantity for the phase transition and is robust to WL noise on ln Q(N).""")

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

    println("Loading runs...")
    runs    = load_runs(T_STR, N_RUNS)
    println("\nAveraged run:")
    avg_run = make_averaged_run(runs)
    println()

    # Compute all Aberth roots (production precision)
    println("Computing Aberth–Ehrlich roots (all runs + averaged)...")
    all_arblib = Dict{Int,NamedTuple}()
    for run in vcat(runs, [avg_run])
        lc, si = precondition(run.lnQ, run.N_values; z_star=run.z_ref)
        r = try find_roots_arblib(lc, si; prec_bits=PREC_BITS_PRODUCTION)
            catch e; @warn e; nothing end
        r === nothing && continue
        all_arblib[run.run_id] = r

        diag  = verify_roots(r.midpoints, lc, si)
        label = run.run_id == 0 ? "avg" : "run$(run.run_id)"
        println("  ── $label ──")
        @printf("  max relative residual:     %.2e  %s\n",
                diag.max_relative_residual,
                diag.max_relative_residual < 1e-6 ? "ok" : "WARN")
        @printf("  conjugate pair violations: %d  %s\n",
                diag.conjugate_pair_violations,
                diag.conjugate_pair_violations == 0 ? "ok" : "WARN")
        println()
    end

    # Save computed zeros so root-finding and plotting can be decoupled
    zeros_path = joinpath(OUTPUT_DIR, "yang_lee_zeros_T$(T_K).jld2")
    zeros_meta = Dict{String,Any}(
        "T_K"       => T_K,
        "T_star"    => T_STAR,
        "N_min"     => first(runs[1].N_values),
        "N_max"     => last(runs[1].N_values),
        "prec_bits" => PREC_BITS_PRODUCTION,
    )
    for run in vcat(runs, [avg_run])
        key = run.run_id == 0 ? "avg" : "run$(run.run_id)"
        r   = get(all_arblib, run.run_id, nothing)
        zeros_meta[key * "_zeros"]     = r === nothing ? ComplexF64[] : r.midpoints
        zeros_meta[key * "_converged"] = r === nothing ? false : r.converged
        zeros_meta[key * "_z_ref"]     = run.z_ref
    end
    save(zeros_path, zeros_meta)
    println("Zeros saved: $zeros_path\n")

    # Comparison table (arblib + float64)
    println("Building comparison table...")
    df = compare_runs(runs, avg_run)
    println("\n", df, "\n")

    csv_path = joinpath(OUTPUT_DIR, "yang_lee_T$(T_K).csv")
    CSV.write(csv_path, df)
    println("Table saved: $csv_path\n")

    # Leading-zero summary
    println("="^65)
    println("Leading-zero summary  (arblib rows only)")
    println("  z_sat(YL) = Re(z_1),   z_sat(EA) = equal-area bisection")
    println("="^65)
    @printf("%-6s  %-12s  %-12s  %-12s  %-12s\n",
            "run", "Re(z_1)", "Im(z_1)", "z_sat(YL)", "z_sat(EA)")
    println("-"^60)
    for r in filter(x -> x.method == "arblib", eachrow(df))
        lbl = r.run_id == 0 ? "avg" : "run$(r.run_id)"
        @printf("%-6s  %-12.5g  %-12.5g  %-12.5g  %-12.5g\n",
                lbl, r.re_z1, r.im_z1, r.z_sat_YL, r.z_sat_EA)
    end
    println()

    # Observable verification figure (from avg run roots)
    avg_r = get(all_arblib, 0, nothing)
    if avg_r !== nothing && !isempty(avg_r.midpoints)
        println("Generating observable verification figure...")
        obs_base = joinpath(FIGURES_DIR,
                            "figure_observables_T$(replace(string(T_K), '.' => 'p'))")
        make_observables_figure(avg_run, avg_r.midpoints; output_path_base=obs_base)
        println()
    end

    # Taylor-style overlay figure
    println("Generating Taylor-style overlay figure...")
    overlay_base = joinpath(FIGURES_DIR,
                            "yang_lee_overlay_T$(replace(string(T_K), '.' => 'p'))")
    make_overlay_figure(runs, avg_run, all_arblib; output_path_base=overlay_base)

    println("\n" * "="^70)
    println("Float64 companion: see 'float64' rows — expected to show overflow/failure.")
    println("Done.")
end

main()
