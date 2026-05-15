"""
Yang–Lee zero computation and finite-size scaling for GC-WL density-of-states data.

The grand-canonical partition function is treated as a polynomial in the fugacity z:

    Ξ(z) = Σ_{N=N_min}^{N_max}  Q(N,V,T) z^N

Roots are found by the Aberth–Ehrlich iteration at configurable ArbNumerics precision.
All averaging, long-range corrections, and Q(N=0)=1 normalization are the caller's
responsibility.

References
  Lee & Yang,   Phys. Rev. 87 (1952) 410 — Yang–Lee theory.
  Aberth,       Math. Comput. 26 (1973) 339 — Aberth–Ehrlich iteration.
  Taylor & Luettmer-Strathmann, JCP 141 (2014) 204906 — YL zeros from WL data.
  Janke & Villanova, Nucl. Phys. B (Proc. Suppl.) 83 (2000) 697 — FSS of YL zeros.
"""

using ArbNumerics
using Printf

# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

function _build_arb_coeffs(log_coeffs::Vector{Float64}, P::Int)
    n      = length(log_coeffs)
    coeffs = Vector{ArbComplex{P}}(undef, n)
    for i in 1:n
        lc = log_coeffs[i]
        if isfinite(lc)
            val       = exp(ArbFloat{P}(BigFloat(lc, precision=P)))
            coeffs[i] = ArbComplex{P}(val, ArbFloat{P}(0))
        else
            coeffs[i] = ArbComplex{P}(ArbFloat{P}(0), ArbFloat{P}(0))
        end
    end
    return coeffs
end

# Horner evaluation returning (p, dp) in one pass.
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

@inline _arb_to_c64(z::ArbComplex{P}) where P =
    complex(Float64(real(z)), Float64(imag(z)))

# Aberth–Ehrlich core iteration.
# The Newton step p/p' and the Weierstrass deflation sum are computed in Float64
# (stable near roots where p/p' is O(1)); the polynomial evaluation that determines
# p/p' uses Arb at full precision to avoid underflow/overflow in p and p' individually.
function _aberth_ehrlich(coeffs_arb::Vector{ArbComplex{P}},
                          degree::Int,
                          max_iter::Int, tol::Float64) where P
    ws     = Vector{ArbComplex{P}}(undef, degree)
    two_pi = ArbFloat{P}(2) * ArbFloat{P}(π)
    for k in 0:degree-1
        θ       = two_pi * ArbFloat{P}(k) / ArbFloat{P}(degree)
        ws[k+1] = ArbComplex{P}(cos(θ), sin(θ))
    end

    converged  = false
    final_corr = NaN
    for iter in 1:max_iter
        max_corr = 0.0
        ws_new   = copy(ws)
        for k in 1:degree
            p_k, dp_k = _horner_arb(coeffs_arb, ws[k])
            newton     = _arb_to_c64(p_k / dp_k)
            (!isfinite(real(newton)) || !isfinite(imag(newton))) && continue
            w_k = _arb_to_c64(ws[k])
            s   = sum(1.0 / (w_k - _arb_to_c64(ws[j])) for j in 1:degree if j != k)
            den = 1.0 - newton * s
            abs(den) < 1e-300 && continue
            corr = newton / den
            !isfinite(corr) && continue
            corr_arb   = ArbComplex{P}(ArbFloat{P}(BigFloat(real(corr), precision=P)),
                                        ArbFloat{P}(BigFloat(imag(corr), precision=P)))
            ws_new[k]  = ws[k] - corr_arb
            max_corr   = max(max_corr, abs(corr))
        end
        ws         = ws_new
        final_corr = max_corr
        max_corr > 0 && max_corr < tol && (converged = true; break)
        max_corr == 0.0 && iter > 1 && break
    end
    return ws, converged, final_corr
end

# ══════════════════════════════════════════════════════════════════════════════
# PRECONDITIONING
# ══════════════════════════════════════════════════════════════════════════════

"""
    compute_z_ref(lnQ, N_values) -> Float64

Select a reference fugacity for preconditioning via a centered finite difference
at the midpoint of the N range:

    ln z_ref = −(lnQ[i_mid+1] − lnQ[i_mid-1]) / 2,    i_mid = length(lnQ) ÷ 2 + 1

This places the peak of ln Q(N) + N ln z_ref near N_mid, so the rescaled
polynomial coefficients ã_N are roughly symmetric and all roots of the
substituted polynomial in w = z / z_ref cluster near |w| = 1.
"""
function compute_z_ref(lnQ::AbstractVector{<:Real}, N_values::AbstractRange)
    n     = length(lnQ)
    n < 3 && error("compute_z_ref: need at least 3 points")
    i_mid = clamp(n ÷ 2 + 1, 2, n-1)
    ln_z  = -(Float64(lnQ[i_mid+1]) - Float64(lnQ[i_mid-1])) / 2
    return exp(ln_z)
end

function _precondition(lnQ::AbstractVector{<:Real}, N_values::AbstractRange; z_ref::Real)
    lnQ_max    = maximum(lnQ)
    ln_z       = log(Float64(z_ref))
    log_coeffs = [(Float64(lnQ[i]) - lnQ_max) + N * ln_z for (i, N) in enumerate(N_values)]
    return log_coeffs, (lnQ_max=Float64(lnQ_max), z_ref=Float64(z_ref),
                        N_min=first(N_values))
end

# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

"""
    yang_lee_zeros(lnQ, N_values; prec_bits, max_iter, tol, verbose) -> NamedTuple

Compute all Yang–Lee zeros of Ξ(z) = Σ_N Q(N,V,T) z^N using the Aberth–Ehrlich
iteration at `prec_bits`-bit ArbNumerics precision.

Arguments
  lnQ       : log density-of-states; lnQ[i] = ln Q(N_min + i − 1). Any overall
              additive constant does not affect the roots.
  N_values  : corresponding N range, e.g. `N_min:N_max`.
  prec_bits : ArbNumerics working precision (default 1024). Increase if
              `verify_yang_lee_zeros` reports max_relative_residual > 1e-10.
  max_iter  : maximum Aberth–Ehrlich iterations (default 400).
  tol       : convergence threshold on max |correction| in Float64 (default 1e-10).
  verbose   : if true, print degree, log-coeff range, and convergence summary.

Returns a NamedTuple with fields
  roots        :: Vector{ComplexF64}  — all roots sorted by |Im(z)| ascending
  z_ref        :: Float64             — preconditioning fugacity (see compute_z_ref)
  N_min, N_max :: Int
  converged    :: Bool
  final_corr   :: Float64             — max |correction| at the last iteration
  prec_bits    :: Int
  log_coeffs   :: Vector{Float64}     — preconditioned log-coefficients (pass to verify)
  scaling_info :: NamedTuple          — (lnQ_max, z_ref, N_min)
"""
function yang_lee_zeros(lnQ::AbstractVector{<:Real},
                         N_values::AbstractRange;
                         prec_bits::Int = 1024,
                         max_iter::Int  = 400,
                         tol::Float64   = 1e-10,
                         verbose::Bool  = false)
    length(lnQ) == length(N_values) ||
        error("yang_lee_zeros: length(lnQ) = $(length(lnQ)) ≠ length(N_values) = $(length(N_values))")
    length(lnQ) < 3 && error("yang_lee_zeros: need at least 3 N values")

    setworkingprecision(ArbFloat, bits=prec_bits)
    P = prec_bits

    z_ref      = compute_z_ref(lnQ, N_values)
    lc, si     = _precondition(lnQ, N_values; z_ref=z_ref)
    degree     = length(lc) - 1
    coeffs_arb = _build_arb_coeffs(lc, P)

    if verbose
        finite_lc = filter(isfinite, lc)
        @printf("  yang_lee_zeros: degree=%d  log-coeff range: [%.1f, %.1f]  prec=%d bits\n",
                degree, minimum(finite_lc), maximum(finite_lc), prec_bits)
    end

    ws, converged, final_corr = _aberth_ehrlich(coeffs_arb, degree, max_iter, tol)

    if verbose
        if converged
            @printf("  Converged (final max |corr| ≈ %.2e)\n", final_corr)
        else
            @printf("  Did NOT converge after %d iters (final max |corr| ≈ %.2e)\n",
                    max_iter, final_corr)
        end
    end

    roots = [_arb_to_c64(w) * z_ref for w in ws]
    filter!(z -> isfinite(z) && abs(z) < 1e6, roots)
    sort!(roots, by=z -> (abs(imag(z)), -real(z)))

    return (roots        = roots,
            z_ref        = z_ref,
            N_min        = first(N_values),
            N_max        = last(N_values),
            converged    = converged,
            final_corr   = final_corr,
            prec_bits    = prec_bits,
            log_coeffs   = lc,
            scaling_info = si)
end

# ══════════════════════════════════════════════════════════════════════════════
# VERIFICATION
# ══════════════════════════════════════════════════════════════════════════════

"""
    verify_yang_lee_zeros(result; prec_bits) -> NamedTuple

Self-consistency checks on a `yang_lee_zeros` result:

  max_relative_residual     — max_k |p(w_k)| / Σ_i |a_i| |w_k|^i
  conjugate_pair_violations — count of roots without a conjugate partner

Both diagnostics are evaluated in ArbNumerics at `prec_bits` to handle far-out
roots where |w| = |z|/z_ref can reach O(10^11) and |w|^degree overflows Float64
(cap ~10^308) but is representable in Arb.

Healthy roots show max_relative_residual ≲ 10^{−12} and zero violations.
"""
function verify_yang_lee_zeros(result::NamedTuple;
                                prec_bits::Int = result.prec_bits)
    roots = result.roots
    isempty(roots) &&
        return (max_relative_residual=NaN, conjugate_pair_violations=-1)

    setworkingprecision(ArbFloat, bits=prec_bits)
    P       = prec_bits
    z_ref   = result.scaling_info.z_ref
    lc      = result.log_coeffs
    n_coeff = length(lc)

    coeffs_arb = _build_arb_coeffs(lc, P)

    max_rel = 0.0
    for z in roots
        w_arb = ArbComplex{P}(ArbFloat{P}(BigFloat(real(z/z_ref), precision=P)),
                               ArbFloat{P}(BigFloat(imag(z/z_ref), precision=P)))
        p_val, _ = _horner_arb(coeffs_arb, w_arb)

        # Denominator Σ_i |a_i| |w|^(i-1), accumulated iteratively
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

    tol_conj = 1e-8 * sum(abs, roots) / length(roots)
    n_viol   = sum(!any(abs(z - conj(w)) < tol_conj for w in roots)
                   for z in roots if imag(z) != 0)

    return (max_relative_residual=max_rel, conjugate_pair_violations=n_viol)
end

# ══════════════════════════════════════════════════════════════════════════════
# LEADING ZERO
# ══════════════════════════════════════════════════════════════════════════════

"""
    leading_zero(roots) -> Union{ComplexF64, Nothing}

Return the leading Yang–Lee zero: the root with the smallest positive imaginary
part and a positive real part.  Returns `nothing` when no such root exists.

For a first-order phase transition Re(z₁) → z_sat and Im(z₁) → 0 as V → ∞.
"""
function leading_zero(roots::Vector{ComplexF64})
    idx = findfirst(z -> real(z) > 0 && imag(z) > 1e-12, roots)
    return idx === nothing ? nothing : roots[idx]
end

# ══════════════════════════════════════════════════════════════════════════════
# THERMODYNAMICS FROM ROOTS
# ══════════════════════════════════════════════════════════════════════════════

"""
    mean_N_from_roots(roots, z, N_min) -> Float64

Grand-canonical mean particle number from the partial-fraction identity

    ⟨N⟩(z) = N_min + Σ_k  z / (z − z_k)

The sum is real for real z > 0 because the z_k come in conjugate pairs.
Sum over all roots, not just the leading pair.
"""
function mean_N_from_roots(roots::Vector{ComplexF64}, z::Float64, N_min::Int)
    Float64(N_min) + real(sum(z / (z - zk) for zk in roots))
end

"""
    var_N_from_roots(roots, z, N_min) -> Float64

Grand-canonical variance of particle number from the partial-fraction identity

    Var(N)(z) = −Σ_k  z z_k / (z − z_k)²

The sum is real for real z > 0 because the z_k come in conjugate pairs.
"""
function var_N_from_roots(roots::Vector{ComplexF64}, z::Float64, N_min::Int)
    real(sum(-z * zk / (z - zk)^2 for zk in roots))
end

"""
    density_from_roots(roots, z, V, N_min) -> Float64

Number density ρ = ⟨N⟩/V reconstructed from Yang–Lee roots.
"""
function density_from_roots(roots::Vector{ComplexF64},
                              z::Float64, V::Float64, N_min::Int)
    mean_N_from_roots(roots, z, N_min) / V
end

"""
    compressibility_from_roots(roots, z, V, T_star, N_min) -> Float64

Isothermal compressibility in LJ reduced units (k_B = 1):

    κ_T(z) = V / (T* ⟨N⟩²) × Var(N)(z)

Returns `NaN` when ⟨N⟩ < 0.1 (near-vacuum, numerically ill-defined).
"""
function compressibility_from_roots(roots::Vector{ComplexF64},
                                     z::Float64, V::Float64,
                                     T_star::Float64, N_min::Int)
    mN = mean_N_from_roots(roots, z, N_min)
    mN < 0.1 && return NaN
    V / (T_star * mN^2) * var_N_from_roots(roots, z, N_min)
end

# ══════════════════════════════════════════════════════════════════════════════
# FINITE-SIZE SCALING
# ══════════════════════════════════════════════════════════════════════════════

"""
    fss_power_law(z1_vec, V_vec) -> NamedTuple

Finite-size scaling of the leading Yang–Lee zero over a set of system volumes.

Fits Im(z₁) ~ B · V^{−κ} by OLS in log–log space, then fits
Re(z₁) = z_sat + A · V^{−κ} with κ fixed.

Returned fields
  z_sat     :: Float64 — thermodynamic-limit coexistence fugacity
  κ         :: Float64 — FSS exponent; first-order → κ ≈ 1;
                         continuous → κ = 1/(νd) where ν is the correlation-length
                         exponent and d is the spatial dimension
  B_im      :: Float64 — Im(z₁) prefactor
  A_re      :: Float64 — Re(z₁) finite-size correction prefactor
  r2_log_im :: Float64 — R² of the log–log Im fit

Requires at least 3 distinct volumes.
"""
function fss_power_law(z1_vec::Vector{ComplexF64}, V_vec::Vector{Float64})
    n = length(z1_vec)
    length(V_vec) == n ||
        error("fss_power_law: z1_vec and V_vec must have the same length")
    n < 3 && error("fss_power_law: need at least 3 volumes")

    re_z1 = real.(z1_vec)
    im_z1 = imag.(z1_vec)
    lV    = log.(V_vec)
    lI    = log.(im_z1)

    # OLS in log-log: log(Im) = log(B) − κ · log(V)
    lV_m = sum(lV) / n
    lI_m = sum(lI) / n
    Sxx  = sum((lV .- lV_m).^2)
    Sxy  = sum((lV .- lV_m) .* (lI .- lI_m))
    κ    = -Sxy / Sxx
    lB   = lI_m + κ * lV_m
    B_im = exp(lB)
    lI_pred   = lB .- κ .* lV
    r2_log_im = 1.0 - sum((lI .- lI_pred).^2) / sum((lI .- lI_m).^2)

    # OLS: Re(z₁) = z_sat + A · V^{−κ}
    x     = V_vec .^ (-κ)
    x_m   = sum(x) / n
    r_m   = sum(re_z1) / n
    A_re  = sum((x .- x_m) .* (re_z1 .- r_m)) / sum((x .- x_m).^2)
    z_sat = r_m - A_re * x_m

    return (z_sat=z_sat, κ=κ, B_im=B_im, A_re=A_re, r2_log_im=r2_log_im)
end

"""
    fss_first_order(z1_vec, V_vec) -> NamedTuple

Finite-size scaling of the leading Yang–Lee zero assuming a first-order transition
(κ = 1, linear in 1/V):

    Re(z₁) = z_sat + A / V
    Im(z₁) = B / V

Both lines are fit by OLS in x = 1/V. Requires at least 2 volumes.

Returned fields: z_sat, A_re, B_im.
"""
function fss_first_order(z1_vec::Vector{ComplexF64}, V_vec::Vector{Float64})
    n = length(z1_vec)
    length(V_vec) == n ||
        error("fss_first_order: z1_vec and V_vec must have the same length")
    n < 2 && error("fss_first_order: need at least 2 volumes")

    re_z1 = real.(z1_vec)
    im_z1 = imag.(z1_vec)
    x     = 1.0 ./ V_vec

    x_m = sum(x) / n
    Sxx = sum((x .- x_m).^2)

    r_m  = sum(re_z1) / n
    A_re = sum((x .- x_m) .* (re_z1 .- r_m)) / Sxx
    z_sat = r_m - A_re * x_m

    i_m  = sum(im_z1) / n
    B_im = sum((x .- x_m) .* (im_z1 .- i_m)) / Sxx

    return (z_sat=z_sat, A_re=A_re, B_im=B_im)
end
