# This file contains functions general to doing thermodynamics or stat mech including some thermodynamic post processing of the GC-WL found configurational integrals 
using ArbNumerics  # arbitrary-precision arithmetic
using SpecialFunctions

function correct_logQ!(wl::WangLandauVars)::Float64  
    #= Wang–Landau gives the canonical partition functions Q(N,V,T) only up to a multiplicative constant.
     This function fixes the constant from the exact value for the empty box:
        Q(N=0,V,T) == 1
     so that
         Q_true(N=0) = 1 = C * Q_wl(N=0)   =>   C = 1/Q_wl(N=0)
     and then
         Q_true(N) = C * Q_wl(N)
     but of course all this is done in log space to get most precision possible.

    returns the constant, and makes the wang landau logQ the true partition functions logQ[1] = logQ(N=0|V,T) and so on
    =#
    logC = -1*wl.logQ_N[1] # natural log of multiplicative constant 
    wl.logQ_N .= wl.logQ_N .+ logC
    # logQtrue = wl.logQ_N .+ logC
    # wl.logQ_N = logQtrue
    return(logC)
end #correct_logQ!

function compute_logZ(wl::WangLandauVars, sim::SimulationParams)::Vector{Float64}
    #= computes the log of the configuration integral (logZ) from the log of the partition function (logQ) as 

         logZ(N,V,T) = logQ(N,V,T) - NlogV + log(N!) + 3N log(Λ)

         derived from Q(N,V,T) = V^N/(N! Λ^3N) Z(N,V,T)

         note that logZ[1] = 0, and in general the vector indexing is shifted by 1 logZ(N,V,T) == logZ[N+1]
    =#
    if sim.N_min != 0
        error("compute_logZ requires sim.N_min == 0")
    end
    logZ = wl.logQ_N .- (0:sim.N_max).*log(sim.V_σ) .+ loggamma.((1: (sim.N_max+1) )) .+ (0:sim.N_max).*log(sim.Λ_σ)*3
    return(logZ)
end # compute_logZ()


function ideal_gas_logQNVT(N::Int,V::Float64,Λ::Float64)::Float64
    #= computes the ideal gas partition function Q(N,V,T):

            Q(N,V,T) = 1/N! (V/Λ^3)^N 

        using Stirling's approximation:

            logQ(N,V,T) = N log (V/Λ^3) + log(1/N!)
                        = N log (V/Λ^3) + log(1) - log(N!)
                        now stirling approx:
                        = N log (V/Λ^3) - N log(N) + N

        V and Λ^3 must be in the same units.

        Inputs: 
        - N = number of particles
        - V = volume
        - Λ = thermal debroglie wavelength: Λ = (h^2/[2 π m k_B T])^1/2
        Outputs:
        - logQ_id = ideal gas partition function
    =# 
    if N == 0
        return(0.0) #  Q(N=0) = (1/0!) (V/Λ^3)^0 = 1*1 thus log(1) = 0
    end
    logQ_id = N*log(V/ (Λ^3) ) - N*log(N) + N 
    return(logQ_id)
end # ideal_gas_logQNVT


function ideal_gas_logQ_loggamma(N::Int, V::Float64, Λ::Float64)::Float64
    #= computes the ideal gas partition function Q(N,V,T) exactly:

            Q(N,V,T) = 1/N! (V/Λ^3)^N

            logQ(N,V,T) = N log (V/Λ^3) - log(N!)
                        = N log (V/Λ^3) - loggamma(N+1)   [exact, no Stirling approximation]

        V and Λ^3 must be in the same units.

        Inputs:
        - N = number of particles
        - V = volume
        - Λ = thermal debroglie wavelength: Λ = (h^2/[2 π m k_B T])^1/2
        Outputs:
        - logQ_id = ideal gas partition function (exact)
    =#
    if N == 0
        return(0.0)
    end
    logQ_id = N*log(V / (Λ^3)) - loggamma(Float64(N+1))
    return(logQ_id)
end # ideal_gas_logQ_loggamma

function compute_packing_frac(N::Int64, V::Float64,σ::Float64)::Float64
    # σ is diameter
    v_sphere =  (π * σ^3)/6
    η = v_sphere* N /V
    return(η)
end

# ─── Grand-canonical thermodynamic analysis (Desgranges & Delhommelle 2012) ───

function logsumexp(v::AbstractVector{<:Real})
    m = maximum(v)
    return m + log(sum(x -> exp(x - m), v))
end

# logΞ(μ,V,T) = log Σ_N Q(N,V,T) exp(βμN), where β=1/(ε T_σ), μ_star=μ/ε
function compute_logΞ(logQ_N::AbstractVector{<:Real}, μ_star::Real, T_σ::Real; N_min::Int=0)
    N_max = length(logQ_N) - 1
    logw = [logQ_N[N+1] + N * μ_star / T_σ for N in N_min:N_max]
    return logsumexp(logw)
end

# p(N) = Q(N,V,T) exp(βμN) / Ξ; returns vector of length (N_max - N_min + 1)
function compute_pN(logQ_N::AbstractVector{<:Real}, μ_star::Real, T_σ::Real; N_min::Int=0)
    N_max = length(logQ_N) - 1
    logw = [logQ_N[N+1] + N * μ_star / T_σ for N in N_min:N_max]
    logΞ = logsumexp(logw)
    return exp.(logw .- logΞ)
end

# <N>(μ,V,T) = Σ_N N p(N)
function compute_mean_N(logQ_N::AbstractVector{<:Real}, μ_star::Real, T_σ::Real; N_min::Int=0)
    N_max = length(logQ_N) - 1
    pN = compute_pN(logQ_N, μ_star, T_σ; N_min=N_min)
    return sum(Float64(N) * pN[i] for (i, N) in enumerate(N_min:N_max))
end

# P = k_B T ln(Ξ) / V in LJ reduced units (P* in units of ε/σ³)
function compute_pressure_σ(logΞ::Real, V_σ::Real, T_σ::Real)
    return T_σ * logΞ / V_σ
end

"""
Find the valley between the two largest local maxima of pN.
Returns a 1-based index into pN. Returns 0 if fewer than 2 peaks found.
"""
function find_Nb_idx(pN::AbstractVector{<:Real}; min_separation::Int=10)
    n = length(pN)
    peaks = Int[]
    for i in 2:n-1
        if pN[i] > pN[i-1] && pN[i] > pN[i+1]
            push!(peaks, i)
        end
    end
    isempty(peaks) && return 0
    sort!(peaks, by = i -> -pN[i])
    p1 = peaks[1]
    p2 = 0
    for cand in peaks[2:end]
        if abs(cand - p1) >= min_separation
            p2 = cand
            break
        end
    end
    p2 == 0 && return 0
    i1, i2 = minmax(p1, p2)
    return i1 + argmin(pN[i1:i2]) - 1
end


# Find μ_coex (in LJ reduced units) via the equal-peak-area criterion (bisection).
# area_diff(μ) = 2 * Σ_{N≤N_b} p(N) - 1; positive = vapor-dominated, negative = liquid-dominated.
function find_μ_coex(logQ_N::AbstractVector{<:Real}, T_σ::Real;
                     N_min::Int=0, μ_lo::Real=-20.0, μ_hi::Real=-5.0,
                     tol::Real=1e-6, max_iter::Int=100)
    function area_diff(μ_star)
        pN = compute_pN(logQ_N, μ_star, T_σ; N_min=N_min)
        idx_b = find_Nb_idx(pN)
        if idx_b == 0
            # No bimodal structure: decide vapor- vs liquid-dominated by where ⟨N⟩ sits.
            # Vapor-dominated (low μ): ⟨N⟩ small → return +1.0 (vapor side too heavy)
            # Liquid-dominated (high μ): ⟨N⟩ large → return -1.0 (liquid side too heavy)
            N_mean = sum((i - 1) * pN[i] for i in eachindex(pN))
            return N_mean < (length(pN) - 1) / 2 ? 1.0 : -1.0
        end
        return 2*sum(pN[1:idx_b]) - 1.0
    end
    f_lo = area_diff(μ_lo)
    f_hi = area_diff(μ_hi)
    f_lo * f_hi > 0 && error("find_μ_coex: no sign change in [$(μ_lo), $(μ_hi)]; f_lo=$(f_lo), f_hi=$(f_hi)")
    for _ in 1:max_iter
        μ_mid = (μ_lo + μ_hi) / 2
        abs(μ_hi - μ_lo) < tol && return μ_mid
        f_mid = area_diff(μ_mid)
        if f_lo * f_mid < 0
            μ_hi = μ_mid; f_hi = f_mid
        else
            μ_lo = μ_mid; f_lo = f_mid
        end
    end
    return (μ_lo + μ_hi) / 2
end

# ══════════════════════════════════════════════════════════════════════════════
# TWO FUGACITY CONVENTIONS — read this before comparing any number to the paper
# ══════════════════════════════════════════════════════════════════════════════
#
#   z = exp(βμ)              "fugacity"  — Λ lives in the COEFFICIENTS Q(N)
#   y = exp(βμ) / Λ³ = z/Λ³  "activity"  — Λ lives in the VARIABLE
#
# Both give the same Ξ, since  Q(N)·z^N = [Z(N)/N!]·y^N.  They differ only in
# where the Λ^(3N) is bookkept, so the two are related by a pure rescaling
#
#       y = z / Λ³        and        y_k = z_k / Λ³
#
# WHICH IS USED WHERE:
#   • z  — everything in this codebase that touches the polynomial: logQ_N is
#          the full canonical Q (including Λ^(-3N)), so the Yang–Lee roots
#          returned by yang_lee_zeros() and every z_c = exp(βμ_c) are z-values.
#          Typical magnitude for argon here: z_c ~ 1e-5.
#   • y  — what Desgranges & Delhommelle (2012) Table I and the TMMC literature
#          report as "ln z_sat".  Typical magnitude: ln y_sat ~ -3 to -6.
#
# At T*=1.2 the two differ by 1/Λ³ ≈ 3.1e3, so mixing them is a three-orders-of-
# magnitude error.  Anything compared against the literature must be in y;
# anything plotted in the complex plane alongside the roots must be in z.
#
# NOTE: observables are invariant under the rescaling — e.g.
#       ⟨N⟩ = N_min + Σ_k z/(z-z_k) = N_min + Σ_k y/(y-y_k)
# — because z/(z-z_k) is unchanged by z→λz, z_k→λz_k.  Only the numerical
# values of z and z_k themselves depend on the convention.

"""
    compute_ln_y_sat(μ_coex_star, T_σ, Λ_σ) -> Float64

Log ACTIVITY at coexistence, `ln y_sat = -3 ln Λ_σ + βμ_coex`, where
`y = exp(βμ)/Λ³`.

This is the quantity Desgranges & Delhommelle (2012) Table I and the TMMC
literature label "ln z_sat" — use it for literature comparison ONLY.

For the fugacity used everywhere else in this package (the one whose roots are
the Yang–Lee zeros), use `ln z = μ_coex_star / T_σ` directly; the two differ by
`-3 ln Λ_σ`.  See the convention block above.
"""
function compute_ln_y_sat(μ_coex_star::Real, T_σ::Real, Λ_σ::Real)
    return -3*log(Λ_σ) + μ_coex_star / T_σ
end

# Liquid and vapor densities at coexistence (Desgranges Eqs. 22–23).
# Returns (ρ_liq_σ, ρ_vap_σ) in LJ reduced units (particles / σ³).
# Nb splits at the argmin of p(N) in the interior (find_Nb_idx).
# Liquid: N > Nb; vapor: N ≤ Nb.
function compute_phase_densities(logQ_N::AbstractVector{<:Real}, μ_star::Real, T_σ::Real, V_σ::Real;
                                  N_min::Int=0)
    pN    = compute_pN(logQ_N, μ_star, T_σ; N_min=N_min)
    idx_b = find_Nb_idx(pN)
    N_vals = N_min : N_min + length(pN) - 1

    A_vap   = sum(pN[1:idx_b])
    ρ_vap_σ = A_vap > 0 ? sum(Float64(N_vals[i]) * pN[i] for i in 1:idx_b) / (A_vap * V_σ) : 0.0

    A_liq   = sum(pN[idx_b+1:end])
    ρ_liq_σ = A_liq > 0 ? sum(Float64(N_vals[i]) * pN[i] for i in idx_b+1:length(pN)) / (A_liq * V_σ) : 0.0

    return ρ_liq_σ, ρ_vap_σ
end

# Convert LJ reduced density (particles / σ³) to g / cm³.
# σ_Å: LJ σ in Ångströms; M_g_per_mol: molar mass in g/mol.
function ljdens_to_gcm3(ρ_σ::Real; σ_Å::Real=3.4, M_g_per_mol::Real=39.948)
    σ_cm = σ_Å * 1e-8
    return ρ_σ * M_g_per_mol / (6.02214076e23 * σ_cm^3)
end

# Saturation pressure in bar from P* = T_σ * logΞ / V_σ (LJ reduced units ε/σ³ → bar).
# ε_kB: LJ ε / k_B in K; σ_Å: σ in Å.
function compute_Psat_bar(logΞ::Real, T_σ::Real, V_σ::Real;
                           ε_kB::Real=117.05, σ_Å::Real=3.4)
    P_star = T_σ * logΞ / V_σ
    k_B_SI = 1.380649e-23
    σ_m    = σ_Å * 1e-10
    ε_SI   = ε_kB * k_B_SI
    return P_star * ε_SI / σ_m^3 / 1e5  # Pa → bar
end

# Apply the standard LJ isotropic long-range correction (LRC) to a logQ_N vector.
# Because U_LRC(N) depends only on N (not on particle positions), it factors out of
# the configurational integral exactly:
#   logQ_LRC(N) = logQ_raw(N) - U_LRC(N) / T_σ
# where (LJ reduced units, ε=σ=1):
#   U_LRC(N) = (8π/3) × (N²/V_σ) × [(1/3)/r_cut_σ⁹ - 1/r_cut_σ³]
# U_LRC < 0 (attractive tail), so logQ_LRC > logQ_raw.
# The correction is zero at N=0, preserving the Q(N=0)=1 anchor.
function apply_lrc_to_logQ(logQ_N::AbstractVector{<:Real}, T_σ::Real, V_σ::Real, r_cut_σ::Real;
                            N_min::Int=0)
    C = (8π/3) * ((1/3) * r_cut_σ^(-9) - r_cut_σ^(-3))   # < 0 for LJ
    logQ_corrected = copy(Float64.(logQ_N))
    for N in N_min : length(logQ_N) - 1
        U_lrc = C * N^2 / V_σ
        logQ_corrected[N + 1] -= U_lrc / T_σ   # subtract negative → increases logQ
    end
    return logQ_corrected
end