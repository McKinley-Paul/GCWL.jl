# functions used to do cluster expansion analysis, compute activity and density virials and so on
# thermo.jl was getting too big 

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
