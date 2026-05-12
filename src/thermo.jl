# This file contains functions general to doing thermodynamics or stat mech including some thermodynamic post processing of the GC-WL found configurational integrals 
#  ✅  == checked in /test
using ArbNumerics  # Julia's arbitrary precision library — add with `Pkg.add("ArbNumerics")`
using SpecialFunctions

function correct_logQ!(wl::WangLandauVars)::Float64  
    #= after doing WL, we solve only for the simplified expanded canonical partition functions  Q(N|V,T) up to a multiplicative constant
     this function solves for this constant and returns the true Q's by recognizing that we know,
     quite profoundly, that there is only one way to have nothing:
        Q(N=0,l=0,|T,V) == 1 
     as such we solve for the arbitrary constant, C, by doing:
         Q_true(N=0,l=0) = 1 = C * Q_wl(N=0,l=0)
         1/Q_wl(N=0,l=0) = C 
     then we can solve for the true partition functions by doing:
         Qtrue(N,l=0) = C * Q_wl(N,l=0)
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
        error("Tried to use compute_configuration_integral() when sim.N_Min != 0 ")
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

        make sure that V and Λ^3 have the same units!!!

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

        make sure that V and Λ^3 have the same units!!!

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

# Find the 1-based index into pN of the valley between vapor and liquid peaks.
# Searches in the interior (20%–80% of the array) to avoid edge effects.
function find_Nb_idx(pN::AbstractVector{<:Real})
    n = length(pN)
    lo = max(2, n ÷ 5)
    hi = min(n - 1, 4n ÷ 5)
    return lo + argmin(pN[lo:hi]) - 1
end

# Find μ_coex (in LJ reduced units) via the equal-peak-area criterion (bisection).
# area_diff(μ) = 2 * Σ_{N≤N_b} p(N) - 1; positive = vapor-dominated, negative = liquid-dominated.
function find_μ_coex(logQ_N::AbstractVector{<:Real}, T_σ::Real;
                     N_min::Int=0, μ_lo::Real=-20.0, μ_hi::Real=-5.0,
                     tol::Real=1e-6, max_iter::Int=100)
    function area_diff(μ_star)
        pN = compute_pN(logQ_N, μ_star, T_σ; N_min=N_min)
        idx_b = find_Nb_idx(pN)
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

# ln z_sat = -3 ln(Λ_σ) + μ_coex_star / T_σ  (activity at coexistence, Desgranges Eq. after Table I)
function compute_lnzsat(μ_coex_star::Real, T_σ::Real, Λ_σ::Real)
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