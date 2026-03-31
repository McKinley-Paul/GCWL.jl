# This file contains functions general to doing thermodynamics or stat mech including some thermodynamic post processing of the SEGC-WL found configurational integrals 
# also contains some functions to do analysis
#  ✅  == checked in /test
using ArbNumerics  # Julia's arbitrary precision library — add with `Pkg.add("ArbNumerics")`
using SpecialFunctions

function correct_logQ(wl::WangLandauVars)::Array{Float64}   
    #❌ This function does not take into account wl.N_min yet, assumes N_min==0 ❌
    #= after doing WL, we solve only for the simplified expanded canonical partition functions  Q(λ,N|V,T) up to a multiplicative constant
     this function solves for this constant and returns the true Q's by recognizing that we know 
        Q(N=0,l=0,|T,V) == 1 
     as such we solve for the arbitrary constant, C, by doing:
         Q_true(N=0,l=0) = 1 = C * Q_wl(N=0,l=0)
         1/Q_wl(N=0,l=0) = C 
     then we can solve for the true partition functions by doing:
         Qtrue(N,l=0) = C * Q_wl(N,l=0)
     but of course all this is done in log space to get most precision possible.

    returns the corrected,true partition functions as a Vector. logQ[1] = logQ(N=0|V,T) and so on
    =#
    logC = -1*wl.logQ_λN[1,1] # natural log of multiplicative constant 
    logQtrue = wl.logQ_λN[1,:] .+ logC
    return(logQtrue)
end #correct_logQ


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