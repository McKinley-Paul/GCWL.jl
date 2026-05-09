# This file contains functions general to doing thermodynamics or stat mech including some thermodynamic post processing of the SEGC-WL found configurational integrals 
# also contains some functions to do analysis
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

function compute_bn_from_logZ(logZ::AbstractVector{<:Real},sim::SimulationParams)
    #= computes the activity coefficients/cluster integrals/ bn's via inversion of Ushcats' 2018 (10.1103/PhysRevE.98.032135) eqn 6 recursion relation. Note that he calls the partition function
    Z and the configuration integral Q while we use the opposite notation, our partition function is Q and our configuration integral is z1
        Z_i = V/i \sum_{n=1}^i n * b_n * Z_{i-n}

    with Z_0 = 1. This was first inverted by Kofke 2023 eqn 25 (https://doi.org/10.1021/acs.jpcb.3c00807)

        b_n(V,T) = 1/V Z(n,V,T) - 1/n \sum_{k=1}^{n-1} k * b_k(V,T) * Z(n-k,V,T)

    with b_1(V,T) = 1. Confusingly, although logZ is shifted indexed, the bns[] returned from this is not shifted 
    and bn[1] = b_1 = 1 

    We also compute two conditioning numbers because there are two possible places for catastrophic
        cancellation in our algorithm. The first comes from the recursive sum and is defined as:
            R_n_sum = ( \sum |terms| )/ |sum|
        which when implemented in this context looks like 
            R_n_sum = \sum_{k=1}^{n-1} | k b_k Z_{n-k} | / |\sum_{k=1}^{n-1} k  b_k Z_{n-k} |
        The second place comes from the final subtraction 
            R_n_sub = |1/V Z(n,V,T) | + |1/n \sum_{k=1}^{n-1} k * b_k(V,T) * Z(n-k,V,T)| / | b_n | 

    =#
    if !isapprox(logZ[1], 0.0, atol=0.01)
        error("Trying to use compute_bn_from_logZ(), but logZ does not appear to be indexed like logZ(N,V,T) = logZ[N+1], because exp(logZ[1]) != 1 (got logZ[1]=$(logZ[1]))")
    end

    bns = [1.0]
    Rns_summation = [1.0]
    Rns_subtraction = [1.0]

    for n in (2:sim.N_max)
        sum = 0 
        Rn_sum_numerator = 0
        for k in (1:n-1)
            sum = sum + k*bns[k]*exp(logZ[n-k+1])

            Rn_sum_numerator = Rn_sum_numerator + abs(k*bns[k]*exp(logZ[n-k+1]))
        end
        A = (1/sim.V_σ)*exp(logZ[n+1])
        B = (1/n)*sum

        # bn = A(1-B/A) # this is more numerically stable than bn = A - B if A is huge compared to B
        bn = A - B 

        Rn_sum = Rn_sum_numerator / abs(sum)
 
        Rn_sub = ( abs((1/sim.V_σ)*exp(logZ[n+1])) + (1/n)*abs(sum) ) / abs(bn)

        push!(bns,bn)
        push!(Rns_summation, Rn_sum)
        push!(Rns_subtraction,Rn_sub)
    end
    return(bns,Rns_summation,Rns_subtraction)
end # bns from logZ 

function compute_bns_rescaled(logZ::AbstractVector{<:Real},sim::SimulationParams)
    #= computes a rescaled activity coefficient tilde_b_n = b_n/Z_n such that the recursion relation is 

        tilde_b_n = 1/V - 1/n \sum_{k=1}^{n-1} k * tilde_b_k * (Z_k * Z_{n-k}) / Z_n

        now, the advantage is that our wang landau simulation is a monte carlo estimate and therefore 
        does not give us the true logZ but in fact is:
            
            Y_n = logZ_n + noise 
        
        where Y_n is our monte carlo estimate of the log of config integral and logZ_n is the true value. 
        Thus the trick here is for tilde_b_n, everything is expressed in terms of ratios like:
            Z_k Z_{n-k} / Z_n = exp( Y_k + Y_{n-k} - Y_n)
        thus exponentials cancel out analytically, and all terms become O(1) instead of exponential
        and our sensitivity is to Y_k + Y_n-k - Y_n so errors are subtracted in log space which does 
        not have exponential amplification along the recursive series. Fundamentally we expect this to improve
        the numerical conditioning of the recursion.
        
        We also compute two conditioning numbers because there are two possible places for catastrophic
            cancellation in our algorithm. The first comes from the recursive sum and is defined as:
                R_n_sum = ( \sum |terms| )/ |sum|
            which when implemented in this context looks like 
                R_n_sum = \sum_{k=1}^{n-1} | k tilde_b_k (Z_k*Z_{n-k})/Z_n | / |\sum_{k=1}^{n-1} k \tilde_b_k  (Z_k*Z_{n-k})/Z_n  |
            The second place comes from the final subtraction 
                R_n_sub = |1/V| + |1/n \sum_{k=1}^{n-1} k * b_k(V,T) * (Z_k*Z_{n-k})/Z_n| / | tilde_b_n | 

    =#
    if !isapprox(logZ[1], 0.0, atol=0.01)
        error("Trying to use compute_bns_rescaled(), but logZ does not appear to be indexed like logZ(N,V,T) = logZ[N+1], because exp(logZ[1]) != 1 (got logZ[1]=$(logZ[1]))")
    end

    tilde_bns = [1.0/exp(logZ[2])] 
    Rns_summation = [1.0]
    Rns_subtraction = [1.0]


    for n in (2:sim.N_max)
        sum = 0 
        Rn_sum_numerator = 0

        for k in (1:n-1)
            exponent = (logZ[k+1] + logZ[n-k+1] - logZ[n+1])
            sum = sum + k * tilde_bns[k] * exp(exponent)
            Rn_sum_numerator = Rn_sum_numerator + abs(k * tilde_bns[k] * exp(exponent))
        end


        tilde_bn = (1/sim.V_σ) - (1/n)*sum # again if A is huge compared to B, can do tilde_bn = A(1-B/A) which will help a little with numerical stability

        Rn_sum = Rn_sum_numerator / abs(sum)
 
        Rn_sub = ( abs((1/sim.V_σ)) + (1/n)*abs(sum) ) / abs(tilde_bn)


        push!(tilde_bns,tilde_bn)
        push!(Rns_summation, Rn_sum)
        push!(Rns_subtraction,Rn_sub)
    end
    return(tilde_bns,Rns_summation,Rns_subtraction)
end #rescaled bns from logZ 

###### start of machinery for density virial 

function compute_Bn_from_bn(bns::AbstractVector{<:Real}, sim::SimulationParams)
    #= Computes density virial coefficients B_n from activity/cluster integrals b_n
    using Hellmann & Bich 2011 (J. Chem. Phys. 135, 084117) Eq. (25), converted
    to the Hill/Ushcats convention used by compute_bn_from_logZ (i.e., NO extra j!
    factor, so that b_1 = 1 and beta*p = sum_{n>=1} b_n z^n).

    The formula in this convention is:

        B_n = (n-1)/n! * sum_{m} (-1)^M * (n + M - 2)! * prod_{j>=2} j^{m_j} b_j^{m_j} / m_j!

    where the sum is over integer partitions {m_j}_{j>=2} satisfying
        sum_{j>=2} (j-1) m_j = n - 1,    M = sum_j m_j.

    Returns
    -------
    Bns                 : Vector{Float64} with Bns[n] = B_n, Bns[1] = 0 (placeholder).
                          Length N_max.
    betas               : Vector{Float64} with betas[k] = beta_k = -(k+1)/k * B_{k+1}
                          for k = 1 .. N_max-1. Length N_max-1.
    diag                : NamedTuple of conditioning diagnostics, each a Vector{Float64}
                          of length N_max with entry [n]:
        .abs_Bn_over_bn     - |B_n| / |b_n|. Hellmann-Bich warn this blows up;
                              they found the approach "only works satisfactorily
                              up to the fourth virial coefficient".
        .R_sum              - sum_i |t_i| / |sum_i t_i|, where t_i are the individual
                              partition terms contributing to B_n. Catastrophic
                              cancellation indicator: R_sum ~ 1 is clean, R_sum >>
                              1 means result is a small difference of large numbers.
                              Rough rule of thumb: lose log10(R_sum) digits of precision.
        .max_term_ratio     - max_i |t_i| / |B_n|. How dominant is the largest
                              single term relative to the final answer.
        .n_partitions       - number of {m_j} partitions summed for B_n. Grows
                              combinatorially in n (sanity check, also tells you
                              cost scaling).
    =#

    # Basic sanity: your compute_bn_from_logZ returns b_n in Hill/Ushcats convention
    # with b_1 = 1 and bns is *not* shifted (bns[1] = b_1).
    if !isapprox(bns[1], 1.0, atol=1e-6)
        @warn "bns[1] = $(bns[1]), expected 1.0 (Hill/Ushcats convention: b_1 = 1). Proceeding anyway, but results likely wrong."
    end

    N = length(bns)
    Bns = zeros(Float64, N)
    R_sum = zeros(Float64, N)
    max_term_ratio = zeros(Float64, N)
    abs_Bn_over_bn = zeros(Float64, N)
    n_partitions = zeros(Int64, N)

    # B_1 is not defined by the virial expansion (the rho^1 coefficient is always 1);
    # we leave Bns[1] = 0 as a placeholder.
    Bns[1] = 0.0

    for n in 2:N
        # Enumerate all {m_j : j >= 2} with sum_{j>=2} (j-1) m_j = n - 1
        # i.e., partitions of (n-1) into positive integer parts, where a part of
        # size p corresponds to m_{p+1} being incremented.
        terms = Float64[]
        for mj_dict in partitions_as_mjdict(n - 1)
            M = sum(values(mj_dict))
            sign_factor = iseven(M) ? 1.0 : -1.0
            # (n + M - 2)!  --- use loggamma for overflow safety, then exp back
            # For n, M in the ranges we care about this is fine as Float64 up through
            # about n + M ~ 170, but we stay safe with logs.
            log_weight = SpecialFunctions.loggamma(n + M - 1)  # log((n+M-2)!)
            # product over j of j^{m_j} * b_j^{m_j} / m_j!
            log_combinatoric = 0.0
            sign_of_b_product = 1.0
            mag_of_b_product = 0.0  # in log
            for (j, mj) in mj_dict
                log_combinatoric += mj * log(j) - SpecialFunctions.loggamma(mj + 1)
                bj = bns[j]
                if bj == 0.0
                    # term vanishes
                    mag_of_b_product = -Inf
                    break
                end
                sign_of_b_product *= (bj < 0 ? (iseven(mj) ? 1.0 : -1.0) : 1.0)
                mag_of_b_product += mj * log(abs(bj))
            end
            if mag_of_b_product == -Inf
                continue
            end
            log_abs_term = log_weight + log_combinatoric + mag_of_b_product
            term = sign_factor * sign_of_b_product * exp(log_abs_term)
            push!(terms, term)
        end

        n_partitions[n] = length(terms)

        # Prefactor (n-1)/n!
        # log((n-1)/n!) = log(n-1) - loggamma(n+1)
        log_prefactor = log(n - 1) - SpecialFunctions.loggamma(n + 1)
        prefactor = exp(log_prefactor)

        raw_sum = sum(terms)
        abs_sum = sum(abs, terms)
        Bn = prefactor * raw_sum
        Bns[n] = Bn

        # Diagnostics
        R_sum[n] = abs_sum / max(abs(raw_sum), eps(Float64))
        max_term_ratio[n] = (maximum(abs, terms; init=0.0) * prefactor) / max(abs(Bn), eps(Float64))
        abs_Bn_over_bn[n] = abs(Bn) / max(abs(bns[n]), eps(Float64))
    end

    # beta_k = -(k+1)/k * B_{k+1}  for k = 1 .. N-1
    betas = Float64[-(k + 1) / k * Bns[k + 1] for k in 1:(N-1)]

    diag = (
        abs_Bn_over_bn = abs_Bn_over_bn,
        R_sum = R_sum,
        max_term_ratio = max_term_ratio,
        n_partitions = n_partitions,
    )

    return Bns, betas, diag
end

"""
    partitions_as_mjdict(k)

Iterator over all integer partitions of k, returned as Dict{Int,Int} where
dict[j] = m_j is the multiplicity of part size (j - 1), i.e., a partition of k
into parts {p_1, p_2, ...} becomes {p_i + 1 => count(p_i)}.

A part of size p in the integer partition corresponds to an m_{p+1} in the
cluster formula (since the constraint is sum (j-1) m_j = k, and setting p = j-1
recovers standard integer partitions of k).
"""
function partitions_as_mjdict(k::Int)
    results = Dict{Int,Int}[]
    if k == 0
        push!(results, Dict{Int,Int}())
        return results
    end
    _partition_recurse!(results, Dict{Int,Int}(), k, k)
    return results
end

function _partition_recurse!(results::Vector{Dict{Int,Int}},
                              current::Dict{Int,Int},
                              remaining::Int,
                              max_part::Int)
    if remaining == 0
        push!(results, copy(current))
        return
    end
    for p in min(remaining, max_part):-1:1
        j = p + 1  # convert: part of size p corresponds to m_{p+1}
        current[j] = get(current, j, 0) + 1
        _partition_recurse!(results, current, remaining - p, p)
        current[j] -= 1
        if current[j] == 0
            delete!(current, j)
        end
    end
end
###### end of machinery for density virial 


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