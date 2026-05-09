# This file contains stuff general to the monte carlo or wang landau process
using StaticArrays
using Random
#  ✅  == checked in /test
function euclidean_distance(ri::MVector{3,Float64},rj::MVector{3,Float64})::Float64 #  ✅ 
    # computes the euclidean distance between ri and rj. chatgpt says doing this manually in this way is the fastest b/c it avoids allocations 
    # outputs r = √(Δx^2 +Δy^2 Δz^2 )
    Δx = ri[1]-rj[1]
    Δy = ri[2]-rj[2]
    Δz = ri[3]-rj[3]
    rij  = sqrt(Δx*Δx + Δy*Δy+ Δz*Δz)
    return(rij)
end # euclidean distance

function euclidean_distance_squared(ri::MVector{3,Float64},rj::MVector{3,Float64})::Float64
    # computes the euclidean distance between ri and rj. chatgpt says doing this manually in this way is the fastest b/c it avoids allocations 
    # outputs r = √(Δx^2 +Δy^2 Δz^2 )
    Δx = ri[1]-rj[1]
    Δy = ri[2]-rj[2]
    Δz = ri[3]-rj[3]
    rij_2  = Δx*Δx + Δy*Δy+ Δz*Δz
    return(rij_2)
end # euclidean distance


function min_config_distance(r::Vector{MVector{3,Float64}})::Tuple{Float64,Int64,Int64}
    # checks each pairwise distance in r and returns the minimum distance and the indices of the particles that that distance is between
    # so the user can identify if there is overlap or not. Needs to be called after renormalization to box units
    # and periodic boundary assurances are applied
    # assumes r is 3xN matrix, position of each atom stored as column vector

    N = length(r)
    m = length(r[1])
    @assert m==3 "position matrix doesn't seem to be 3xN"

    min_distance = typemax(Float64) # just initializing to be huge so don't have to add an extra condition to loop to check if it's the first time every time 
    i_min = 0
    j_min = 0
    @inbounds for i in 1:(N-1) # @inbounds tells the compiler to skip some safety checks it makes when doing loops involving arrays, basically we are telling the compiler we are not going to ask for an element not in the array bounds like r[0] or r[N+1] so it doesnt have to check for us, https://docs.julialang.org/en/v1/devdocs/boundscheck/
        for j in (i+1):N # no need to make a copy because we aren't modifying this at all 
            distance = euclidean_distance(r[i],r[j])
            if distance < min_distance
                min_distance = distance
                i_min = i 
                j_min = j 
            end
        end
    end # ij pair loop

    return(min_distance,i_min,j_min)
end #min_config_distance(r)

@inline function euclidean_distance_squared_pbc(ri_box::MVector{3,Float64},rj_box::MVector{3,Float64})::Float64 #  ✅ 
    # computes the euclidean squared distance between ri and rj.  doing this manually in this way may be the fastest b/c it "avoids allocations" but im not sure
    # does so assuming ri and rj are in box units for periodic boundary conditions
    # outputs r2 = Δx^2 +Δy^2 Δz^2
    # avoids unneccesary sqrt for LJ potential which speeds things up
    Δx = ri_box[1]-rj_box[1]
    Δy = ri_box[2]-rj_box[2]
    Δz = ri_box[3]-rj_box[3]

    # now doing PBC in box=1 units
    Δx = Δx - round(Δx)
    Δy = Δy - round(Δy)
    Δz = Δz - round(Δz)

    rij_squared  = Δx*Δx + Δy*Δy+ Δz*Δz
    return(rij_squared)
end # euclidean distance squared


function translate_by_random_vector!(r::MVector{3,Float64},δr_max::Float64,rng::MersenneTwister,c::SimCache)#  ✅ 
    # modifies c.ζ_Mvec to a new vector translated by a random amount less than √3*(δr_max) in a random direction. Does not apply PBC
    # note that δr_max is the max that any x,y,z component can move
    @inbounds for i in 1:3
        r[i] = r[i] + δr_max* (2* rand(rng) - 1.0)
        # rand(rng) generates a random number between 0 and 1, so 2*rand(rng)-1 generates a random number between -1 and 1
        # then we scale it by δr_max to get the desired range for each component
    end
    return nothing
end # translate_by_random_vector

function metropolis(ΔE::Float64,T_σ::Float64,rng::MersenneTwister=MersenneTwister())::Bool #  ✅ 
    # in the case of a translational move, the acceptance criteria reduces to the standard metropolis one. Assumes ΔE and T_σ have LJ units (both have energy units)
    exponent = -1*ΔE/T_σ 

    if exponent > 0 # downhill, accept without evaluating due to min(1,e^exponent) and e^exponent > 1 if exponent > 0
        return(true)
    else
        ζ = rand(rng) # random number between zero and 1
        accept = (exp(exponent) > ζ)   #boolean
        return(accept)
    end

end # metropolis