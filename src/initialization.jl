# contains basic structs and initalization processes used in segc_wl
using StaticArrays
using Printf
using JLD2
using Random


mutable struct microstate # holds the variables of the actual current or proposed state
    N::Int64 # number of particles in the microstate
    r_box::Vector{MVector{3,Float64}}  # position of all N particles (3xN matrix)
end

mutable struct SimCache
    #= to try and reduce heap allocations during simulations we allocate the memory needed for temporary calculations
        such as those by translate_by_random_vector() to a "cache variable" at the start of the simulation so that each time we call these functions, we don't have to reallocate the same conceptual variables as new instances on the heap
        The idea of naming this cache is that hopefully this object is small enough to be saved in the "cache" which is a small, fast access memory storage in between the CPU and and the larger slower main part of the RAM. even if 
        that is not the case, its still good to preallocate things. you can think of this as "temporary allocations" for variables we reinsantiate and use many times during our simulation 

        make sure to update the values of the variables with .= !!!!!! to avoid allocations
    =#
    ζ_Mvec::MVector{3,Float64} # created for use in translate_by_random_vector() to store the new random vector 
    ζ_idx::Int64 # created for use in translation_move! to pick random atom to move 
    ri_proposed_box::MVector{3,Float64} # createed for use in translation_move!() to store new proposed location of fractional or normal particle 
    μ_prop::microstate # used for proposing moves in λ_move!()
end


struct SimulationParams # immutable because structs are by default immutable in julia
    # SimulationParams is kind of like the user inputs to the simulation and things that will remain fixed throughout the simulation
    # could also add other meta params like the convergence threshold to this, how often to do a flatness check (10,000 rn), the minimum visits to be considered flat and so on but just leaving that stuff hard coded for now
    # add a maximum iteration? 

    # ~~~~~~~ Input parameters ~~~~~~~~~
    N_max::Int64 # maximum number of atoms in the simulation
    N_min::Int64 # minimum number of atoms in simulation
    T_σ::Float64 # temperature for Q(N,V,T,λ) in lennard jones units
    Λ_σ::Float64 # DeBroglie Wavelength for system, just passing in as somethign you compute instead of doing inside inner constructor for module loading dependency reasons

    r_cut_σ::Float64 # cutoff to stop evaluating the potential

    input_filename::String
    save_directory_path::String #path to save variables to like binaries of microstate during checkpoints and after runs and so on
    # usually want to use save_directory_path = @__DIR__x
    rng::MersenneTwister # used for seeding tests with random numbers via rng=MersenneTwister(1234) or so on. For production runs, set rng=MersenneTwister()
    maxiter::Int64 # maximum monte carlo iterations, can set to 1 to test single steps

    # ~~~~~~~ Derived parameters ~~~~~~~~~
    L_σ::Float64 # length of box in LJ units, read in from input file 
    V_σ::Float64 # volume
    L_squared_σ::Float64 # used to convert between box units (L_box=1) and LJ units
    r_cut_box::Float64
    r_cut_squared_box::Float64 # used for cutoff to avoid computing sqrts
    dynamic_δr_max_box::Bool # set to true if you want to dynamically adjust δr_max_box during the run, false if you want to use a static one specified from the start

    # Now using an "Inner Constructor" to compute the derived parameters from only the input ones
    function SimulationParams(; N_max,N_min,T_σ,Λ_σ,λ_max,r_cut_σ,input_filename="",L_σ::Float64=0.0,save_directory_path,rng=MersenneTwister(),maxiter=10^9,dynamic_δr_max_box=true)
        # the semicolon above in (; N_max ... ) makes it so all these are keyword arguments, not positional ones so you set them with like 'N_max=100' when initializing a SimulationParams struct
        # Compute derived quantities: L_σ can be supplied directly or read from an input config file
        if input_filename != ""
            _, L_σ, _ = load_configuration(input_filename)
        elseif L_σ == 0.0
            throw(ArgumentError("Must provide either input_filename or L_σ to determine box size"))
        end
        V_σ = L_σ^3
        L_squared_σ = L_σ^2
        r_cut_box = r_cut_σ/L_σ
        r_cut_squared_box = r_cut_box^2

        new(N_max,N_min,T_σ,Λ_σ,λ_max,r_cut_σ,input_filename,save_directory_path,rng,maxiter,
            L_σ,V_σ,L_squared_σ,r_cut_box ,r_cut_squared_box,dynamic_δr_max_box)
    end
end # SimulationParams

mutable struct WangLandauVars
    # WangLandauVars holds variables related to the wang landau monte carlo process that will change throughout the course of the simulation
    logf::Float64 # natural log of f, the modification factor in the Wang Landau scheme which is initiall set to f = the euler constant
    H_N::Vector{Int64} # histogram of number of times each (λ,N) pair has been visited.  λ_max rows, N_max columns.
    logQ_N::Vector{Float64} # "density of states" for Wang Landau method in SEGC is the extended canonical partition functions Q(NVT,λ) because V and T are fixed for given sim., they are left out of name, λ on rows and N on columns
    translation_moves_proposed::Int64
    translation_moves_accepted::Int64
    N_moves_proposed::Int64
    N_moves_accepted::Int64
    iters::Int64 # total number of monte carlo moves thus far
    δr_max_box::Float64
    phase2::Bool # set to true once phase 2 is entered for Pereyra's 1/t algorithm (monte carlo time)
    flat::Bool # set to true when the condition for flatness is met 
    
end

function init_WangLandauVars(sim::SimulationParams,δr_max_box::Float64 = typemax(Float64))::WangLandauVars
    # if you want to dynamically δr_max_box::Float64 = typemax(Float64)
    if sim.dynamic_δr_max_box == true
        δr_max_box = 0.15/sim.L_σ 
    elseif  ( sim.dynamic_δr_max_box == false ) && (δr_max_box == typemax(Float64) )
        throw("you have sim.dynamic_δr_max_box set to false but did not provide a value of delta r_max_box to init_WangLandauVars")
    end

    logf = 1 
    H_N = zeros(Int64,sim.N_max+1)
    logQ_N=zeros(Float64,sim.N_max+1) # has N_max+1 for N, N=0 goes in column 1, N_max goes in column (N_max + 1)
        
    wl = WangLandauVars(logf,H_N,logQ_N,0,0,0,0,0,δr_max_box,false,false)
    return(wl)
end



function copy_microstate!(dest::microstate, src::microstate)
    dest.N = src.N
    for i in 1:src.N
        dest.r_box[i] .= src.r_box[i]  # deep copy each MVector's contents
    end
    # previously this function had:  dest.r_box .= src.r_box  # Copy into existing array (no allocation)
    #The .= operator on a Vector does element-wise assignment. For Vector{MVector{3,Float64}}, dest.r_box[i] = src.r_box[i]
    # makes dest.r_box[i] point to the same MVector object as src.r_box[i]. This was the source of a major bug as This behavior means c.μ_prop.r_box and μ.r_box share the same MVector objects after initialization!
    # thus we must deep copy  element wise  as above
end





function load_configuration(filename::String)::Tuple{Int,Float64,Vector{MVector{3,Float64}}}
    # Loads in a cnf.inp file generated by Allen and Tildesly's initialize.py 
    # returns the first line, which has the number of particles in the configuration as N 
    # returns L_σ, the box length which is from the second line in lennard jones units
    # returns r_σ, the coordinates of the particles as a 3xN array in lennard jones units
#                       # r_N = [ x1  x2  x3 ... x_N
#                                 y1  y2  y3 ... y_N
#                                 z1  z2  z3 ... z_N ]
    # even though in the filename (e.g. cnf.inp) it is stored as an Nx3 array 
    open(filename, "r") do io
        # Read N (int)
        N = parse(Int, strip(readline(io)))

        # Read L (float)
        L_σ = parse(Float64, strip(readline(io)))

        # Read N lines of 3 floats each
        data = Array{Float64}(undef, N, 3)
        for i in 1:N
            data[i, :] = parse.(Float64, split(readline(io)))
        end

        # Transpose to 3×N (your preferred format)
        r_σ = permutedims(data)   # or data' gives 3×N Float64 matrix
        r_σ = matrix_to_svec_positions(r_σ) # convert to Vector of SVector{3,Float64}
        return N, L_σ, r_σ
    end
end # load_configuration(filename)

@inline function matrix_to_svec_positions(r_box::AbstractMatrix{Float64})::Vector{MVector{3,Float64}}
    @assert size(r_box, 1) == 3
    N = size(r_box, 2)

    r = Vector{MVector{3,Float64}}(undef, N)
    @inbounds for j in 1:N
        r[j] = MVector{3,Float64}(
            r_box[1, j],
            r_box[2, j],
            r_box[3, j],
        )
    end
    return r
end


function init_microstate(;sim::SimulationParams,filename::String )::microstate
    # initialize the microstate to feed into run_simulation() in segc_wl from input file
    N,L_σ,r_σ = load_configuration(filename)
    r_σ_box = r_σ ./ L_σ # converting to "box units" by dividing the positions by the length of the box. Now r_i ∈ [-0.5,0.5]^3 because inputs are centered around 0,0
    for i in 1:length(r_σ_box)
        pbc_wrap!(r_σ_box[i])
    end
    # pre-allocate N_max slots so λ-moves can always write to r_box[μ.N] without bounds errors
    r_box = [@MVector zeros(3) for _ in 1:sim.N_max]
    for i in 1:N
        r_box[i] .= r_σ_box[i]
    end

    μstate = microstate(N,r_box)
    return(μstate)
end # init microstate

function init_microstate(sim::SimulationParams)::microstate
    # initialize microstate at N=0  (vacuum state) with r_box pre-allocated to N_max slots
    # use this to start the Wang-Landau walk from the known anchor point Q(N=0)=1
    r_box = [@MVector zeros(3) for _ in 1:sim.N_max]
    return microstate(0, r_box)
end # init_microstate from vacuum

@inline function pbc_wrap!(r::MVector{3,Float64})
    @inbounds for i in 1:3
        r[i] = r[i] - round(r[i])
    end
    return nothing
end    



function check_inputs(s::SimulationParams,μ::microstate,wl::WangLandauVars)
    if μ.N > 1
        min_distance,particle_1,particle_2 = min_config_distance(μ.r_box[1:μ.N])
        println("Minimum distance in box units (L_box=1) between particles in the initial configuration is: ",round(min_distance,digits=3), " between particles ",particle_1, " and ", particle_2)
        println("This is compared to length of σ in box units which is: ", round(1/s.L_σ,digits=3))
    else
        println("Starting from N=$(μ.N), skipping minimum distance check.")
    end

    if s.dynamic_δr_max_box == false
        println("SimulationParams.dynamic_δr_max_box is set to false, therefore wl.δr_max_box will not be updated during the run and
        will be set to ", wl.δr_max_box, " for the entire course of the wang landau simulation")
    end
    if (s.dynamic_δr_max_box == false) && (wl.δr_max_box = (0.15/s.L_σ))
        println("wl.δr_max_box: ", wl.δr_max_box)
        throw("You have SimulationParams.dynamic_δr_max_box set to false yet wl.δr_max_box is the
                 the default value of 0.15/ L_sigma")
    elseif (s.dynamic_δr_max_box == true) && (wl.δr_max_box != (0.15/s.L_σ))
        throw("You have SimulationParams.dynamic_δr_max_box set to true yet wl.δr_max_box is NOT the
                 the default value of 0.15/ L_sigma")
        println("wl.δr_max_box: ", wl.δr_max_box)
    end

    flush(stdout)

end #check_inputs

function init_cache(sim::SimulationParams,initial_μ::microstate)::SimCache
    ζ_Mvec = @MVector zeros(3)
    ζ_idx = Int64(0)
    ri_proposed_box = @MVector zeros(3)
    r_box = [@MVector zeros(3) for _ in 1:sim.N_max]
    μ_prop = microstate(0,r_box)
    copy_microstate!(μ_prop,initial_μ)
    return SimCache(ζ_Mvec,ζ_idx,ri_proposed_box,μ_prop)
end


function print_simulation_params(params::SimulationParams,start::Bool=true)
    println()
    if start
        println("           Starting SEGC Wang Landau Simulation        ")
        println("Due to the occurence of Λ in the metropolis criterion, this simulation cannot be fully carried out for lennard jonesium and is here specific to Argon")
    end
    @printf("T* = %.4f\n", params.T_σ)
    @printf("V = %.4f σ³\n", params.V_σ)
    @printf("Box Length L = %.4f σ\n", params.L_σ)
    @printf("DeBroglie Λ = %.4f σ\n", params.Λ_σ)
    @printf("Max Density (N/V [σ^-3]): ρ = %.4f N\n", params.N_max/params.V_σ)
    
    @printf("Nmax = %d\n", params.N_max)
    @printf("Nmin = %d\n", params.N_min)
    @printf("λmax = %d\n", params.λ_max)

    @printf("r_cut = %.4f σ\n", params.r_cut_σ)
    println("Directory files saved in: ", params.save_directory_path)
end #print_simulation_params

function print_microstate(μ::microstate,print_r::Bool=false)
    println()
    println("Current microstate:")
    @printf("N = %.4f\n", μ.N)
    @printf("λ = %.4f\n", μ.λ)
    if print_r 
        println("Positions of full particles in box units: ")
        display(μ.r_box)
    end
end #print_microstate

function print_wl(wl::WangLandauVars,verbose=true)
    println()
    println("Wangl Landau Variables")
    @printf("logf = %.4f\n", wl.logf)
    @printf("N_moves_proposed = %f\n", wl.N_moves_proposed)
    @printf("N_moves_accepted = %f\n", wl.N_moves_accepted)
    @printf("N acceptance ratio = %.4f\n", ( wl.N_moves_accepted/wl.N_moves_proposed) )

    @printf("δr_max_box = %.4f\n", wl.δr_max_box)
    @printf("translation_moves_proposed = %f\n", wl.translation_moves_proposed)
    @printf("translation_moves_accepted = %f\n", wl.translation_moves_accepted)
    @printf("translation acceptance ratio = %.4f\n", ( wl.translation_moves_accepted/wl.translation_moves_proposed) )
    @printf("Total monte carlo moves so far = %.1f\n", wl.iters )

    if verbose 
        display("LogQ: ")
        display(wl.logQ_N)

        display("Histogram N")
        display(wl.H_N)
    end
end # print_wl

function initialization_check(sim::SimulationParams, μ::microstate,wl::WangLandauVars)
    check_inputs(sim,μ,wl) # tell user the min distance in config and similar things
    print_simulation_params(sim)
    print_microstate(μ,true)
    flush(stdout)
end


function save_wanglandau_jld2(wl::WangLandauVars,sim::SimulationParams, filename::String)
    filepath = joinpath(sim.save_directory_path,filename)
    jldsave(filepath; wl=wl)
end

function load_wanglandau_jld2(filename::String)::WangLandauVars
    return load(filename, "wl")
end
# # Usage
# save_wanglandau_jld2(wl, "checkpoint.jld2")
# wl_loaded = load_wanglandau_jld2("checkpoint.jld2")

function save_microstate_jld2(μ::microstate,sim::SimulationParams,  filename::String)
    filepath = joinpath(sim.save_directory_path,filename)
    jldsave(filepath; μ=μ)
end

function load_microstate_jld2(filename::String)::microstate
    return load(filename, "μ")
end