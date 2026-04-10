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



function potential_1(r_box::Vector{MVector{3,Float64}},ri_box::MVector{3,Float64},i::Int64,N::Int64,L_squared_σ::Float64,r_cut_squared_box::Float64)::Float64 #  ✅ 
    #= Calculates the sum of the interaction potential between 1 special particle labelled i (e.g. the particle involved in a proposal move) in the r list and all others
    Inputs:
        -r_box (Array of Float64s): 3xN position matrix of N  particles in box units
        -ri_box (static array of Float64s): 3x1 vector of the position of ri you want to use for distances between all other particles in box units
        -i (int): index of particle in r list you want to compute interactions between, used for avoiding self interactions.
        - N (int): current number of particles, could be read from _,N = size(r_box) but want to minimize all activity in this loop and we already have to keep track of N outside this function call
        - L_squared_σ (Float64): squared length of box in σ units (LJ this time)
        - r_cut_squared_box  (Float64): cutoff length squared  for the potential in box=1 units
    Returns:
        - E_int_σ (Float64): total interaction energy between particle i at location ri and all other particles including the fractional one in natural units for the potential (i.e. lennard jones units)
    =# 

    E_int_σ = 0

    #  computing interaction with other particles
    @inbounds for j in 1:N
        if j != i # avoid double counting
            rij_squared_box = euclidean_distance_squared_pbc(ri_box,r_box[j])
            if rij_squared_box < r_cut_squared_box # only evaluate potential if pair is inside cutoff distance
                rij_squared_σ = rij_squared_box  * L_squared_σ   # convert to potential natural units (i.e. lennard jones) to compute the potentials see allen and Tildesly one Note note for explanation why
                if rij_squared_σ != 0.0 # no overlap check but still can't be 0 distance because Lj is undefined/+infty there
                    E_int_σ = E_int_σ + E_12_LJ(rij_squared_σ)  
                else
                    return(typemax(Float64))
                end
                # if rij_squared_σ >  0.5 # close to overlap threshold given by allen and tildesly of potential E > 100, ours is E > 224
                #     E_int_σ = E_int_σ + E_12_LJ(rij_squared_σ)
                # else # if particles overlap quit early and return a huge energy
                #     return(typemax(Float64))
                # end 
            end
        end
    end 

    return(E_int_σ)

end # potential_1
