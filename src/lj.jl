# this module contains system specific stuff to the argon lennard jones system
using Random
#  ✅  == checked in \test 

function argon_deBroglie(T_σ::Float64)::Float64 #  ✅ 
    # computes de broglie wavelength for the LJ model of Argon at a given input reduced temperature T_σ = kB T / ϵ

    # -- physical constants (SI)
    h_Js  = 6.62607015e-34      # Planck constant, J s
    kB_J_K = 1.380649e-23        # Boltzmann constant, J/K
    amu_kg = 1.66053906660e-27  # atomic mass unit, kg

    # -- Argon LJ parameters (typical)
    ϵ_over_kB_K = 117.05              # epsilon / k_B (K)
    σ_Å = 3.4                       # sigma in Angstroms
    mass_amu = 39.948               # atomic mass of Ar in amu

    # -- convert to SI
    T_K = ϵ_over_kB_K * T_σ             # temperature in K
    σ_m = σ_Å * 1e-10                 # sigma in meters
    m_kg= mass_amu * amu_kg             # mass in kg

    # -- de Broglie wavelength (SI)
    Λ_m = h_Js / sqrt(2*pi * m_kg * kB_J_K * T_K)

    # -- dimensionless quantities in LJ reduced units
    Λ_σ = Λ_m / σ_m


    return (Λ_σ)
end # argon debroglie

function E_12_LJ(rij_squared_σ::Float64)::Float64 #  ✅
    #= Computes the interaction energy between two lennard jones particles in LJ units
    rij_squared_\sigma = squared distance between two particles in lennard jones \sigma =1 units
    =#
    E_int = (1/rij_squared_σ)^6 - (1/rij_squared_σ)^3
    E_int = 4*E_int
    return(E_int)
end

struct LennardJones <: PairPotential end # no parameters needed in reduced units

pair_energy(::LennardJones, r2_σ::Float64) = E_12_LJ(r2_σ)
