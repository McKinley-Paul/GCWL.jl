module gc_wl
using StaticArrays
using Dates # mostly for debugging/monitoring long calculations
using Statistics # used for 80% average flatness criterion

include("initialization.jl")
include("utils.jl")
include("lj.jl")
include("thermo.jl")
# ✅  = unit tests for function exist in /tests/
# this module contains the meat of the package including run_simulation and the various High Level Wang Landau Monte Carlo functions. 

# gc_wl.jl functions:
export run_simulation!, translation_move!,N_move!,update_wl!, post_run,  potential_1
# initialization functions exports:
export microstate,SimulationParams,init_microstate,check_inputs, print_simulation_params, print_microstate,print_wl, WangLandauVars,init_WangLandauVars, initialization_check, save_wanglandau_jld2, save_microstate_jld2,load_microstate_jld2, load_wanglandau_jld2, load_configuration, SimCache,init_cache,copy_microstate!
# utils exports:
export euclidean_distance, min_config_distance, euclidean_distance_squared_pbc, translate_by_random_vector!, metropolis
# lj exports:
export argon_deBroglie, E_12_LJ, N_metropolis, PairPotential, LennardJones, pair_energy
# thermo stuff:
export correct_logQ, ideal_gas_logQNVT, ideal_gas_logQ_loggamma


function run_simulation!(sim::SimulationParams, μ::microstate,wl::WangLandauVars,c::SimCache)
    log_path = joinpath(sim.save_directory_path, "wl_progress_log.txt")
    progress_log = open(log_path,"a") # for debugging/monitoring long calculations
    println(progress_log,"Starting run_simulation!(), time is ",  Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))

    num_active_bins = (sim.N_max - sim.N_min + 1)
    
    while (wl.flat == false ) && ( wl.iters < sim.maxiter)
        ζ = rand(sim.rng)
        if ζ < 0.75                         # propose translational moves 75% of the time 
            translation_move!(sim,μ,wl,c)
        else                                # this means  ζ ≥ 0.75 and so we propose λ moves 25% of the time 
            N_move!(sim,μ,wl,c)
        end

        if wl.phase2 == true # change logf every move in phase 2
            wl.logf = num_active_bins/wl.iters # same as 1/t= 1/(wl.iters/num_active_bins)  wl.iters/num_active bins is "monte carlo time" as described in Pereyra's 2007 article introducing the 1/t algorithm
            
            if wl.iters % 1_000_000 == 0 # flatness check and saving every million moves
                H_min = minimum(wl.H_N[(sim.N_min+1) : (sim.N_max+1) ])
                H_avg = mean(wl.H_N[(sim.N_min+1) : (sim.N_max+1) ])

                if H_min/H_avg > 0.8 # flatness check, looks like old school Wang Landau 2001 style but actually inspired by Shchur 2017 section IV.C 10.1103/PhysRevE.96.043307
                    wl.flat = true
                    println(progress_log,"Flatness criterion reached in phase 2 after ", wl.iters, " total iterations ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
                    println(progress_log,"H_λ,N min/mean = ",round(H_min/H_avg*100), " %" )
                    println(progress_log,"H_λ,N max = ", maximum(wl.H_N[(sim.N_min+1) : (sim.N_max+1) ]))
                    flush(progress_log)
                end

                if wl.iters % 500_000_000 == 0   # Progress log every time 500,000,000 iters, approximately every ten minutes 
                    println(progress_log,"Phase 2 flatness check: H_λ,N min/mean = ",round(H_min/H_avg*100), " %, total iters: ",wl.iters, " min: ",H_min," ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS")  )
                    flush(progress_log)
                end
            end

        end

        update_wl!(wl,μ)

        # check when to switch to phase 2:
        if wl.phase2 == false # check if every state has been visited/ tunneling time every move while in phase 1
            if wl.iters % (1_000 * num_active_bins) == 0  # check phase 1 flatness only every 1_000 monte carlo time as suggested in pereyra       
                H_min = minimum(wl.H_N[(sim.N_min+1) : (sim.N_max+1) ]) # the zeroth particle sits in wl.H_λN[:,1] - the 1 indexed column

                if H_min ≥  1 # checking flatness for phase 1 of algorithm in line with Pereyra
                    wl.logf = 0.5*wl.logf
                    println(progress_log,"New WL phase 1 epoch!, now at ",wl.logf," and the time is ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS")) # MOSTLY FOR DEBUGGING AND MONITORING LONG CALCULATIONS
                    flush(progress_log) 

                    monte_carlo_time = wl.iters/num_active_bins 
                    if wl.logf ≤ 1/(monte_carlo_time) # Pererya 2007 phase transition criterion 
                        wl.phase2 = true

                        println(progress_log,"Now entering phase 2! It took ", wl.iters, " monte carlo moves (wl.iters) to get to phase 2! ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
                        println(progress_log, "That is a monte carlo time of: ", Int64(monte_carlo_time))
                        flush(progress_log)
                    end
                    
                    wl.H_N = zeros(Int64,sim.N_max+1)

                end
            end
        end #phase1 -> phase2 check

        if wl.iters % 1_000_000 ==0 # save checkpoint every 1_000,000 moves 
            save_microstate_jld2(μ,sim, "microstate_checkpoint.jld2")
            save_wanglandau_jld2(wl,sim, "wl_checkpoint.jld2") # jld2 is quick save binary, to inspect checkpoint, open up julia ipynb and use wl_loaded = load_wanglandau_jld2("checkpoint.jld2")
        end

    end # while logf ≥ logf_convergence_threshold
    close(progress_log)
end #run_simulation


function potential_1(pot::P,r_box::Vector{MVector{3,Float64}},ri_box::MVector{3,Float64},i::Int64,N::Int64,L_squared_σ::Float64,r_cut_squared_box::Float64)::Float64 where P <: PairPotential #  ✅
    #= Calculates the sum of the interaction potential between 1 special particle labelled i (e.g. the particle involved in a proposal move) in the r list and all others
    Inputs:
        - pot: pair potential (subtype of PairPotential); the compiler generates a specialized version for each concrete type, so there is no runtime dispatch overhead
        -r_box (Array of Float64s): 3xN position matrix of N  particles in box units
        -ri_box (static array of Float64s): 3x1 vector of the position of ri you want to use for distances between all other particles in box units
        -i (int): index of particle in r list you want to compute interactions between, used for avoiding self interactions.
        - N (int): current number of particles, could be read from _,N = size(r_box) but want to minimize all activity in this loop and we already have to keep track of N outside this function call
        - L_squared_σ (Float64): squared length of box in σ units (LJ this time)
        - r_cut_squared_box  (Float64): cutoff length squared  for the potential in box=1 units
    Returns:
        - E_int_σ (Float64): total interaction energy between particle i at location ri and all other particles in natural units for the potential
    =#

    E_int_σ = 0.0

    #  computing interaction with other particles
    @inbounds for j in 1:N
        if j != i # avoid double counting
            rij_squared_box = euclidean_distance_squared_pbc(ri_box,r_box[j])
            if rij_squared_box < r_cut_squared_box # only evaluate potential if pair is inside cutoff distance
                rij_squared_σ = rij_squared_box  * L_squared_σ   # convert to potential natural units (i.e. lennard jones) to compute the potentials see allen and Tildesly one Note note for explanation why
                if rij_squared_σ != 0.0 # no overlap check but still can't be 0 distance because potential is undefined/+infty there
                    E_int_σ = E_int_σ + pair_energy(pot, rij_squared_σ)
                else
                    return(typemax(Float64))
                end
            end
        end
    end

    return(E_int_σ)

end # potential_1

function translation_move!(sim::SimulationParams,μ::microstate,wl::WangLandauVars,c::SimCache) #✅
    # wrote the body of all these functions before refactoring into structs, could rewrite at some point because ugly
    wl.translation_moves_proposed += 1

    if μ.N == 0; return(nothing); end # nothing to translate
    c.ζ_idx = rand(sim.rng,1:μ.N) # randomly pick atom to move

    c.ri_proposed_box .= μ.r_box[c.ζ_idx]
    translate_by_random_vector!(c.ri_proposed_box, wl.δr_max_box, sim.rng,c) # Trial move to new position (in box=1 units), this used to be ri_proposed_box before cache
    pbc_wrap!(c.ri_proposed_box)   # PBC
    E_proposed = potential_1(sim.potential,μ.r_box,c.ri_proposed_box,c.ζ_idx,μ.N,sim.L_squared_σ,sim.r_cut_squared_box)

    if E_proposed == typemax(Float64) # check overlap
        nothing # reject the move and recount this state for histogram and partition function
    else
        E_old = potential_1(sim.potential,μ.r_box,μ.r_box[c.ζ_idx],c.ζ_idx,μ.N,sim.L_squared_σ,sim.r_cut_squared_box)
        ΔE = E_proposed - E_old 
        accept = metropolis(ΔE,sim.T_σ,sim.rng) 
        if accept
            μ.r_box[c.ζ_idx] .= c.ri_proposed_box
            c.μ_prop.r_box[c.ζ_idx] .= c.ri_proposed_box
            wl.translation_moves_accepted += 1
        end
    end


    if wl.logf == 1 # tune δr_max_box during first wang landau epoch
        if sim.dynamic_δr_max_box == true  
            if (wl.translation_moves_accepted/wl.translation_moves_proposed > 0.55) && wl.δr_max_box < 1.0 # tune δr_max_box to get ~50% acceptance, pg 159 Allen Tildesly
                # added the wl.δr_max_box < 1.0 because for dilute systems or ideal gas conditions you accept every move and the δr_max_box grows riducously and unphysically for a periodic system using  box=1 units
                wl.δr_max_box = wl.δr_max_box * 1.05

            elseif wl.translation_moves_accepted/wl.translation_moves_proposed  < 0.45
                wl.δr_max_box = wl.δr_max_box*0.95
            end 
        end
    end
end # translation move

function N_move!(sim::SimulationParams,μ::microstate,wl::WangLandauVars,c::SimCache)
    # currently only implementing ΔN = ±1
    wl.N_moves_proposed += 1

    ΔN = 2*rand(sim.rng,Bool) - 1 # change N by ±1; can go out of our range

    if ΔN == -1 # decrement N 
        c.μ_prop.N = μ.N-1
        if c.μ_prop.N < sim.N_min # return early/reject move if goes below simulation bounds
            # have to do this here otherwise line below idx_deleted will throw error when μ.N == 0
            c.μ_prop.N += 1
            return(nothing)
        end

        idx_deleted = rand(sim.rng,1:μ.N) # pick a random particle to be deleted
        
        c.μ_prop.r_box[idx_deleted] .= μ.r_box[μ.N] # swap last particle into deleted slot

        accept = N_metropolis(μ, c.μ_prop,idx_deleted, wl,sim)
        if accept == true
            wl.N_moves_accepted += 1
            if idx_deleted != μ.N # only do the swap if we actually deleted a particle that wasn't the last one
                μ.r_box[idx_deleted] .= μ.r_box[μ.N] # swap last particle into deleted slot using original μ.N
            end
            μ.N -= 1 # single decrement, after swap

        else # reset the cache state if rejected
            c.μ_prop.N += 1
            c.μ_prop.r_box[idx_deleted] .= μ.r_box[idx_deleted] # put back the deleted particle in the cache
        end

    elseif ΔN == +1 # increment N 
        c.μ_prop.N = μ.N+1
         if c.μ_prop.N  > sim.N_max # return early/reject move if we go out of bounds with too many particles 
            c.μ_prop.N -= 1
            return(nothing)
        end

        @inbounds for i in 1:3 # inserting new particle at random spot
            c.μ_prop.r_box[c.μ_prop.N][i] = rand(sim.rng) - 0.5
        end
        idx_deleted = 0

        accept = N_metropolis(μ, c.μ_prop,idx_deleted, wl,sim)
        if accept == true
            wl.N_moves_accepted += 1
            μ.N += 1
            μ.r_box[μ.N] .= c.μ_prop.r_box[μ.N] # add the new particle to the μ.N position in actual microstate

        else # reset the cache state if rejected
            c.μ_prop.N -= 1
            # we don't reset c.μ_prop.r_box[μ.N+1] because it doesn't matter 
        end
    end # ΔN control flow

end

function  N_metropolis(μ::microstate,
                           μ_prop::microstate, idx_deleted::Int64, # μ_prop = μ_proposed the next proposed microstate
                           wl::WangLandauVars,sim::SimulationParams)::Bool #  ✅ 

        # the purpose of the function is to compute the acceptance criterion for different situations
        # many things change including the form of the metropolis criterion you need to compute, what energies you have to compute to get ΔE, and so on.
        # covered by equations 19 in Ganzmuller and Camp 2007 doi: 10.1063/1.2794042
        # assumes that you have already checked that N_proposed is in bounds  (N_min ≤ N_proposed ≤ N_max)

        # first we compute the multiplicative prefactor term involving Q,V,Λ in eqns 10-12-- because the N! terms can cause overflow, we only compute them once we know something about N old vs new
        logQ_diff = wl.logQ_N[μ.N+1] - wl.logQ_N[μ_prop.N+1]
        partition_ratio = exp(logQ_diff)
        
        V_Λ_prefactor = sim.V_σ^(μ_prop.N - μ.N) * sim.Λ_σ^(3*(μ.N - μ_prop.N))

        # now we compute the exponential part of the criterion having to do with the configurational potential energy

        if μ.N > μ_prop.N # deletion move
            factorial_prefactor = μ.N # N! / (N-1)! = N
            # ΔU = U(N-1) - U(N) = -(energy of deleted particle with all others)
            # so E_old = interaction of deleted particle with all N-1 remaining, E_proposed = 0
            E_old = potential_1(sim.potential,μ.r_box,μ.r_box[idx_deleted],idx_deleted,μ.N,sim.L_squared_σ,sim.r_cut_squared_box)
            E_proposed = 0.0

        else # insertion move
            factorial_prefactor = 1/μ_prop.N  # N! / (N+1)! = 1/(N+1)
            
            #  change in potential energy comes from inserting new particle, all other interactions are the same so ΔE = sum of interactions of new particle with old
            E_old = 0
            i = μ_prop.N 
            E_proposed = potential_1(sim.potential,μ_prop.r_box,μ_prop.r_box[i],i,μ_prop.N,sim.L_squared_σ,sim.r_cut_squared_box)
        end

        ΔE = E_proposed - E_old
        exponent = -1*ΔE/sim.T_σ 
        prob_ratio = partition_ratio*V_Λ_prefactor*factorial_prefactor*exp(exponent)
        if prob_ratio > 1
            return(true)
        else
            ζ = rand(sim.rng)
            accept = (prob_ratio > ζ)   #boolean
            return(accept)
        end
end #N_metropolis

function update_wl!(wl::WangLandauVars,μ::microstate)
    wl.logQ_N[μ.N+1] += wl.logf
    wl.H_N[μ.N+1]    += 1
    wl.iters +=1
end #update_wl!

function post_run(sim::SimulationParams,μ::microstate,wl::WangLandauVars)
    println("Wang Landau converged or reached max iterations, logf has reached ", wl.logf, " and convergence is achieved when logf reaches ", log( exp(10^(-8))) )
    println("Iterations: ", (wl.translation_moves_proposed+wl.N_moves_proposed), " with maxiters: ", sim.maxiter )
    println("Total translation moves proposed: ", wl.translation_moves_proposed, ", translation moves accepted: ", wl.translation_moves_accepted, ", Acceptance ratio: ", wl.translation_moves_accepted/wl.translation_moves_proposed)
    println("Total N moves proposed: ", wl.N_moves_proposed, ", N moves accepted: ", wl.N_moves_accepted, ", Acceptance ratio: ", wl.N_moves_accepted/wl.N_moves_proposed)
    save_wanglandau_jld2(wl,sim,"final_wl.jld2")
    save_microstate_jld2(μ,sim,"final_microstate.jld2")
    println("Final Microstate and Wang Landau variables saved")
end

end