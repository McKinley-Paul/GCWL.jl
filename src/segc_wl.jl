module segc_wl
using StaticArrays
using Dates # mostly for debugging/monitoring long calculations
using Statistics # used for 80% average flatness criterion

include("initialization.jl")
include("utils.jl")
include("lj.jl")
include("thermo.jl")
# ✅  = unit tests for function exist in /tests/
# this module contains the meat of the package including run_simulation and the various High Level Wang Landau Monte Carlo functions. 

# segc_wl.jl functions:
export run_simulation!, translation_move!,λ_move!,update_wl!, post_run
# initialization functions exports:
export microstate,SimulationParams,init_microstate,check_inputs, print_simulation_params, print_microstate,print_wl, WangLandauVars,init_WangLandauVars, initialization_check, save_wanglandau_jld2, save_microstate_jld2,load_microstate_jld2, load_wanglandau_jld2, load_configuration, SimCache,init_cache,copy_microstate!
# utils exports:
export euclidean_distance, min_config_distance, euclidean_distance_squared_pbc, translate_by_random_vector!, metropolis
# lj exports:
export argon_deBroglie, E_12_LJ, E_12_frac_LJ, potential_1_normal, potential_1_frac, λ_metropolis_pm1
# thermo stuff:
export correct_logQ, ideal_gas_logQNVT, ideal_gas_logQ_loggamma


function run_simulation!(sim::SimulationParams, μ::microstate,wl::WangLandauVars,c::SimCache)
    log_path = joinpath(sim.save_directory_path, "wl_progress_log.txt")
    progress_log = open(log_path,"a") # for debugging/monitoring long calculations

    f_convergence_threshold = exp(10^(-8)) # Desgranges paper convergence criterion has a typo, this is from Original 2001 Landau paper which is presumably what Desgranges et. al. meant
    logf_convergence_threshold = log(f_convergence_threshold)
    while (wl.logf ≥ logf_convergence_threshold ) && ( wl.iters < sim.maxiter)
        ζ = rand(sim.rng)
        if ζ < 0.75                         # propose translational moves 75% of the time 
            translation_move!(sim,μ,wl,c)
        else                                # this means  ζ ≥ 0.75 and so we propose λ moves 25% of the time 
            λ_move!(sim,μ,wl,c)
        end

        update_wl!(wl,μ)

        # check flatness and save state:
        if wl.iters % 10_000 ==0 # check flatness only every 1000 cycles
            min = minimum(wl.H_λN[:,(sim.N_min+1) : (sim.N_max+1) ]) # the zeroth particle sits in wl.H_λN[:,1] - the 1 indexed column
                        
            # trying out the traditional flatness criterion as suggested by Wang and Landau in 2001
            # avg = mean(wl.H_λN[:,(sim.N_min+1) : (sim.N_max+1) ])
            # if min ≥ 0.8*avg # flatness here means that for all possible  H(λ,N) is not less than 80% of the average histogram <H(λ,N)>

            if min ≥  1000 # this is our flatness criteria
                save_wanglandau_jld2(wl,sim, "wl_checkpoint_before_rezeroing.jld2") # jld2 is quick save binary, to inspect checkpoint, open up julia ipynb and use wl_loaded = load_wanglandau_jld2("checkpoint.jld2")
                
                wl.logf = 0.5*wl.logf
                println(progress_log,"New WL epoch!, now at ",wl.logf," and the time is ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS")) # MOSTLY FOR DEBUGGING AND MONITORING LONG CALCULATIONS
                flush(progress_log)

                wl.H_λN = zeros(Int64,sim.λ_max+1,sim.N_max+1)
                save_wanglandau_jld2(wl,sim, "wl_checkpoint_after_zeroing.jld2") # jld2 is quick save binary, to inspect checkpoint, open up julia ipynb and use wl_loaded = load_wanglandau_jld2("checkpoint.jld2")
            end
            if wl.iters % 100_000 ==0 # save checkpoint every 100,000 moves 
                #save_wanglandau_jld2(wl,sim, "wl_checkpoint.jld2") # jld2 is quick save binary, to inspect checkpoint, open up julia ipynb and use wl_loaded = load_wanglandau_jld2("checkpoint.jld2")
                save_microstate_jld2(μ,sim, "microstate_checkpoint.jld2")
        end
        end #flatness/printing
    end # while logf ≥ logf_convergence_threshold
end #run_simulation

function translation_move!(sim::SimulationParams,μ::microstate,wl::WangLandauVars,c::SimCache) #✅
    # wrote the body of all these functions before refactoring into structs, could rewrite at some point because ugly
    wl.translation_moves_proposed += 1

    c.ζ_idx = rand(sim.rng,1:(μ.N+1)) # randomly pick atom to move, includes fractional particle
    if c.ζ_idx ≤ μ.N # move normal particle    
        c.ri_proposed_box .= μ.r_box[c.ζ_idx]
        translate_by_random_vector!(c.ri_proposed_box, wl.δr_max_box, sim.rng,c) # Trial move to new position (in box=1 units), this used to be ri_proposed_box before cache
        pbc_wrap!(c.ri_proposed_box)   # PBC
        E_proposed = potential_1_normal(μ.r_box,c.ri_proposed_box,c.ζ_idx,μ.r_frac_box,μ.λ,sim.λ_max,μ.N,sim.L_squared_σ,sim.r_cut_squared_box,  μ.ϵ_ξ,μ.σ_ξ_squared ) 

        if E_proposed == typemax(Float64) # check overlap
            nothing # reject the move and recount this state for histogram and partition function 
        else   
            E_old =potential_1_normal(μ.r_box,μ.r_box[c.ζ_idx],c.ζ_idx,μ.r_frac_box,μ.λ,sim.λ_max,μ.N,sim.L_squared_σ,sim.r_cut_squared_box,  μ.ϵ_ξ,μ.σ_ξ_squared ) 
            ΔE = E_proposed - E_old 
            accept = metropolis(ΔE,sim.T_σ,sim.rng) 
            if accept
                μ.r_box[c.ζ_idx] .= c.ri_proposed_box
                c.μ_prop.r_box[c.ζ_idx] .= c.ri_proposed_box
                wl.translation_moves_accepted += 1
            end
        end

    else # move fractional particle
        c.ri_proposed_box .= μ.r_frac_box
        translate_by_random_vector!(c.ri_proposed_box,wl.δr_max_box,sim.rng,c)#  this used to be r_frac_proposed_box before I introduced the cache
        pbc_wrap!(c.ri_proposed_box)   # PBC
        if μ.λ == 0 # auto accept because if λ =0 , translational move of the fractional particle doesn't change energy
            μ.r_frac_box .= c.ri_proposed_box
            c.μ_prop.r_frac_box .= c.ri_proposed_box
            wl.translation_moves_accepted += 1
        else 
            E_proposed = potential_1_frac(μ.r_box, c.ri_proposed_box  ,μ.λ ,sim.λ_max,μ.N,sim.L_squared_σ,sim.r_cut_squared_box,  μ.ϵ_ξ,μ.σ_ξ_squared )

            E_old =potential_1_frac(μ.r_box,μ.r_frac_box,μ.λ,sim.λ_max,μ.N,sim.L_squared_σ,sim.r_cut_squared_box,  μ.ϵ_ξ,μ.σ_ξ_squared )
            ΔE = E_proposed - E_old 
                accept = metropolis(ΔE,sim.T_σ,sim.rng) 
                if accept
                    μ.r_frac_box .= c.ri_proposed_box
                    c.μ_prop.r_frac_box .= c.ri_proposed_box
                    wl.translation_moves_accepted += 1
                end
        end

    end # if i < N deciding to move normal or translational particle

    if sim.dynamic_δr_max_box == true   
        if wl.logf == 1 # tune δr_max_box during first wang landau epoch
            if (wl.translation_moves_accepted/wl.translation_moves_proposed > 0.55) && wl.δr_max_box < 1.0 # tune δr_max_box to get ~50% acceptance, pg 159 Allen Tildesly
                # added the wl.δr_max_box < 1.0 because for dilute systems or ideal gas conditions you accept every move and the δr_max_box grows riducously and unphysically for a periodic system using  box=1 units
                wl.δr_max_box = wl.δr_max_box * 1.05

            elseif wl.translation_moves_accepted/wl.translation_moves_proposed  < 0.45
                wl.δr_max_box = wl.δr_max_box*0.95
            end 
        end
    end
end# translation move

function λ_move!(sim::SimulationParams,μ::microstate,wl::WangLandauVars,c::SimCache)
    # again wrote the body of this and all subfunctions before introducing Structs in a refactor, could rewrite because as of now it's ugly and verbose but if it works...
    # currently only implementing Δλ = ±1, can do CFMC/translational move style drawing from scaled uniform distribution if this doesnt work well
    wl.λ_moves_proposed += 1

    λ_proposed = μ.λ + 2*rand(sim.rng,Bool) - 1 # change λ by ±1; can go out of our range, and can be -1 or 100 which is our signal to change N 
    
    if -1 < λ_proposed ≤ sim.λ_max # λ_proposed < 99 for λ_max = 99 no change to N or anything besides λ
        c.μ_prop.λ = λ_proposed
        idx_deleted=0

        c.μ_prop.ϵ_ξ = ( c.μ_prop.λ/(sim.λ_max + 1) )^(1/3)
        c.μ_prop.σ_ξ_squared = (c.μ_prop.λ/(sim.λ_max + 1) )^(1/2) 

        accept = λ_metropolis_pm1(μ, c.μ_prop,idx_deleted, wl,sim)
        if accept == true
            wl.λ_moves_accepted += 1
            μ.λ = c.μ_prop.λ
            μ.ϵ_ξ = c.μ_prop.ϵ_ξ
            μ.σ_ξ_squared = c.μ_prop.σ_ξ_squared
        else # reset the cache if state is rejected 
            c.μ_prop.λ = μ.λ # reset proposed λ to current λ if move rejected
            c.μ_prop.ϵ_ξ = μ.ϵ_ξ 
            c.μ_prop.σ_ξ_squared = μ.σ_ξ_squared
        end

    elseif λ_proposed == -1 # decrement N 
        # keep the fractional particle in r_frac_box but delete a random particle from the list, need to pick a random particle I think
        c.μ_prop.λ = sim.λ_max     
        c.μ_prop.N = μ.N-1
        if c.μ_prop.N < sim.N_min # return early/reject move if negative particles appear i.e. if λ_proposed = -1 initially when N=0
            # have to do this here otherwise line below idx_deleted will throw error when μ.N == 0
            c.μ_prop.λ = μ.λ # reset the cache
            c.μ_prop.N += 1
            return(nothing)
        end

        idx_deleted = rand(sim.rng,1:μ.N)
        # swap last particle into deleted slot
        if idx_deleted != μ.N
            c.μ_prop.r_box[idx_deleted] .= μ.r_box[μ.N]
        end
        # deleted particle becomes the new fractional particle: its position must go to r_frac_box.
        # μ.r_box[idx_deleted] is still the deleted position here (the swap above only touched c.μ_prop.r_box)
        c.μ_prop.r_frac_box .= μ.r_box[idx_deleted]

        c.μ_prop.ϵ_ξ = ( c.μ_prop.λ/(sim.λ_max + 1) )^(1/3)
        c.μ_prop.σ_ξ_squared = (c.μ_prop.λ/(sim.λ_max + 1) )^(1/2)

        accept = λ_metropolis_pm1(μ, c.μ_prop,idx_deleted, wl,sim)
        if accept == true
            wl.λ_moves_accepted += 1
            μ.λ = c.μ_prop.λ
            # set r_frac_box to deleted position BEFORE swapping r_box (μ.r_box[idx_deleted] still valid here)
            μ.r_frac_box .= μ.r_box[idx_deleted]
            if idx_deleted != μ.N # only do the swap if we actually deleted a particle that wasn't the last one
                μ.r_box[idx_deleted] .= μ.r_box[μ.N] # swap last particle into deleted slot in actual microstate
            end
            μ.N -= 1

            μ.ϵ_ξ = c.μ_prop.ϵ_ξ
            μ.σ_ξ_squared = c.μ_prop.σ_ξ_squared

        else # reset the cache state if rejected
            c.μ_prop.λ = μ.λ
            c.μ_prop.N += 1
            c.μ_prop.r_box[idx_deleted] .= μ.r_box[idx_deleted] # put back the deleted particle in the cache
            c.μ_prop.r_frac_box .= μ.r_frac_box # restore ghost position in cache
            c.μ_prop.ϵ_ξ = μ.ϵ_ξ
            c.μ_prop.σ_ξ_squared = μ.σ_ξ_squared
        end

    elseif λ_proposed > sim.λ_max # increment N 
        c.μ_prop.λ = 0 
        c.μ_prop.N = μ.N+1
         if c.μ_prop.N  > sim.N_max # return early/reject move if we go out of bounds with too many particles (i.e. λ_prop = 100 when N=N_max)
            c.μ_prop.λ = μ.λ # reset the cache if move goes out of bounds
            c.μ_prop.N -= 1
            return(nothing)
        end
        c.μ_prop.r_box[c.μ_prop.N] .= μ.r_frac_box # add the fractional particle to the μ.N+1 position
        @inbounds for i in 1:3
            c.μ_prop.r_frac_box[i] = rand(sim.rng) - 0.5
        end
        idx_deleted = 0

        c.μ_prop.ϵ_ξ = ( c.μ_prop.λ/(sim.λ_max + 1) )^(1/3)
        c.μ_prop.σ_ξ_squared = (c.μ_prop.λ/(sim.λ_max + 1) )^(1/2) 


        accept = λ_metropolis_pm1(μ, c.μ_prop,idx_deleted, wl,sim)
        if accept == true
            wl.λ_moves_accepted += 1
            μ.λ = 0
            μ.N += 1
            μ.r_frac_box .= c.μ_prop.r_frac_box
            μ.r_box[μ.N] .= c.μ_prop.r_box[μ.N] # add the fractional particle to the μ.N position in actual microstate

            μ.ϵ_ξ = c.μ_prop.ϵ_ξ
            μ.σ_ξ_squared = c.μ_prop.σ_ξ_squared 
        else # reset the cache state if rejected
            c.μ_prop.λ = μ.λ
            c.μ_prop.N -= 1
            c.μ_prop.r_frac_box .= μ.r_frac_box
            # we don't reset c.μ_prop.r_box[μ.N+1] because it doesn't matter 
            c.μ_prop.ϵ_ξ = μ.ϵ_ξ 
            c.μ_prop.σ_ξ_squared = μ.σ_ξ_squared
        end
    else
        throw("Error in λ move ΔN control flow")
    end # ΔN control flow

end

function update_wl!(wl::WangLandauVars,μ::microstate)
    wl.logQ_λN[μ.λ+1,μ.N+1] += wl.logf
    wl.H_λN[μ.λ+1,μ.N+1] += 1
    wl.iters +=1
end #update_wl!

function post_run(sim::SimulationParams,μ::microstate,wl::WangLandauVars)
    println("Wang Landau converged or reached max iterations, logf has reached ", wl.logf, " and convergence is achieved when logf reaches ", log( exp(10^(-8))) )
    println("Iterations: ", (wl.translation_moves_proposed+wl.λ_moves_proposed), " with maxiters: ", sim.maxiter )
    println("Total translation moves proposed: ", wl.translation_moves_proposed, ", translation moves accepted: ", wl.translation_moves_accepted, ", Acceptance ratio: ", wl.translation_moves_accepted/wl.translation_moves_proposed)
    println("Total λ moves proposed: ", wl.λ_moves_proposed, ", λ moves accepted: ", wl.λ_moves_accepted, ", Acceptance ratio: ", wl.λ_moves_accepted/wl.λ_moves_proposed)
    save_wanglandau_jld2(wl,sim,"final_wl.jld2")
    save_microstate_jld2(μ,sim,"final_microstate.jld2")
    println("Final Microstate and Wang Landau variables saved")
end

end