#= HARD WALL BOUNDARY CONDITIONS

Most of the time, we want to run things in PBC with minimum image convention in cubic volumes. 
However, for some systems and benchmarks, we would like to run things in hard walls with non-periodic boundaries.
In particular, for lennard jones cluster benchmarks, spherical boundaries with hard walls will be implemented in this folder.


Everything required to do this will be self contained in this file and will have "hwx_" appended to each function name and so on. if x = s then spherical boundaries if x = c this means cubic boundaries 
spherical boundary simulations must be started from N=0


for the sphere, the approach we take is that the sphere is inscribed inside the box such that L  = 2Rc, ie the diameter of the sphere equals the length of the box. then the sphere is centered at the origin and has radius 0.5 in box units. this is enforced by just checking that any proposal moves can only be accepted if they land inside the box via is_inside_sphere() i.e. x^2 + y^2 + z^2 < R^2 < 0.25

    For matching Roundy , set Rc=2.5σ hence L_σ=5.0 and Vsph=4/3π(2.5)^3≈65.45σ^3
=# 


function hws_run_simulation!(sim::SimulationParams, μ::microstate,wl::WangLandauVars,c::SimCache)
    # !!!!!!!!!!!!!!!!!     !!!!!!!!!!!!!!!!!!!     !!!!!!!!!!!!!!!!!!!!
    # THE ONLY difference between this and the normal run_simulation!() in gc_wl.jl is that this function calls hws_translation_move!() and hws_N_move!() instead of the normal translation_move!() and N_move!()
    # !!!!!!!!!!!!!!!!!     !!!!!!!!!!!!!!!!!!!     !!!!!!!!!!!!!!!!!!!!

    log_path = joinpath(sim.save_directory_path, "wl_progress_log.txt")
    progress_log = open(log_path,"a") # for debugging/monitoring long calculations
    println(progress_log,"Starting run_simulation!(), time is ",  Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
    flush(progress_log)

    num_active_bins = (sim.N_max - sim.N_min + 1)
    
    while (wl.flat == false ) && ( wl.iters < sim.maxiter)
        ζ = rand(sim.rng)
        if ζ < 0.75                         # propose translational moves 75% of the time 
            hws_translation_move!(sim,μ,wl,c)
        else                                # this means  ζ ≥ 0.75 and so we propose λ moves 25% of the time 
            hws_N_move!(sim,μ,wl,c)
        end

        if wl.phase2 == true # change logf every move in phase 2
            wl.logf = num_active_bins/wl.iters # same as 1/t= 1/(wl.iters/num_active_bins)  wl.iters/num_active bins is "monte carlo time" as described in Pereyra's 2007 article introducing the 1/t algorithm
            
            if wl.iters % 1_000_000 == 0 # flatness check and saving every million moves
                H_min = minimum(wl.H_N[(sim.N_min+1) : (sim.N_max+1) ])
                H_avg = mean(wl.H_N[(sim.N_min+1) : (sim.N_max+1) ])

                if H_min/H_avg > 0.8 # flatness check, looks like old school Wang Landau 2001 style but actually inspired by Shchur 2017 section IV.C 10.1103/PhysRevE.96.043307
                    wl.flat = true
                    println(progress_log,"Flatness criterion reached in phase 2 after ", wl.iters, " total iterations ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
                    println(progress_log,"H(N) min/mean = ",round(H_min/H_avg*100), " %" )
                    println(progress_log,"H(N) max = ", maximum(wl.H_N[(sim.N_min+1) : (sim.N_max+1) ]))
                    flush(progress_log)
                end

                if wl.iters % 500_000_000 == 0   # Progress log every time 500,000,000 iters, approximately every ten minutes 
                    println(progress_log,"Phase 2 flatness check: H(N) min/mean = ",round(H_min/H_avg*100), " %, total iters: ",wl.iters, " min: ",H_min," ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS")  )
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
end #hws_run_simulation



function hws_potential_1(pot::P,r_box::Vector{MVector{3,Float64}},ri_box::MVector{3,Float64},i::Int64,N::Int64,L_squared_σ::Float64,r_cut_squared_box::Float64)::Float64 where P <: PairPotential
    # only difference is it uses euclidean_distance_squared() instead of euclidean_distance_squared_pbc() to evaluate distances

    E_int_σ = 0.0

    @inbounds for j in 1:N
        if j != i 
            rij_squared_box = euclidean_distance_squared(ri_box,r_box[j]) # !!!!! major difference no minimum image convention or periodicity as in euclidean_distance_squared_pbc() !!!!!!!!!!!!!!!!!!!!!!!!!!!!
            if rij_squared_box < r_cut_squared_box 
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

end # hws_potential_1

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Key function ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#for the sphere, the approach we take is that the sphere is inscribed inside the box such that L  = 2Rc, ie the diameter of the sphere equals the length of the box. then the sphere is centered at the origin and has radius 0.5 in box units. this is enforced by just checking that any proposal moves can only be accepted if they land inside the box via is_inside_sphere() i.e. x^2 + y^2 + z^2 < R^2 with R^2 = 0.25 in box units
@inline function is_inside_sphere(r::MVector{3,Float64}) 
    @inbounds return r[1]*r[1] + r[2]*r[2] + r[3]*r[3] ≤ 0.25
end
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Key function ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

function hws_translation_move!(sim::SimulationParams, μ::microstate, wl::WangLandauVars, c::SimCache)
    wl.translation_moves_proposed += 1
    if μ.N == 0; return nothing; end
    
    c.ζ_idx = rand(sim.rng, 1:μ.N)
    c.ri_proposed_box .= μ.r_box[c.ζ_idx]
    translate_by_random_vector!(c.ri_proposed_box, wl.δr_max_box, sim.rng, c)
    
    accept = false
    if is_inside_sphere(c.ri_proposed_box)
        E_proposed = hws_potential_1(sim.potential, μ.r_box, c.ri_proposed_box,
                                      c.ζ_idx, μ.N, sim.L_squared_σ, sim.r_cut_squared_box)
        E_old      = hws_potential_1(sim.potential, μ.r_box, μ.r_box[c.ζ_idx],
                                      c.ζ_idx, μ.N, sim.L_squared_σ, sim.r_cut_squared_box)
        ΔE = E_proposed - E_old
        accept = metropolis(ΔE, sim.T_σ, sim.rng)
        if accept
            μ.r_box[c.ζ_idx] .= c.ri_proposed_box
            c.μ_prop.r_box[c.ζ_idx] .= c.ri_proposed_box
            wl.translation_moves_accepted += 1
        end
    end
    # Outside-sphere proposals fall through here as rejections — accept stays false
    
    if sim.dynamic_δr_max_box
        if μ.N ≥ (9 * sim.N_max) ÷ 10
            wl.dense_trans_proposed += 1
            if accept
                wl.dense_trans_accepted += 1
            end
            ratio = wl.dense_trans_accepted / wl.dense_trans_proposed
            if ratio > 0.55 && wl.δr_max_box < 0.5 # no point trying to make moves larger than sphere itself
                wl.δr_max_box *= 1.05
            elseif ratio < 0.45
                wl.δr_max_box *= 0.95
            end
        end
    end
    return nothing
end

@inline function random_point_in_sphere!(rng, r::MVector{3,Float64}) # rejection sampling to get point inside sphere 
    while true
        x = rand(rng) - 0.5
        y = rand(rng) - 0.5
        z = rand(rng) - 0.5

        x*x + y*y + z*z <= 0.25 || continue

        @inbounds begin
            r[1] = x
            r[2] = y
            r[3] = z
        end

        return nothing
    end
end

function hws_N_move!(sim::SimulationParams,μ::microstate,wl::WangLandauVars,c::SimCache)
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

        accept = hw_N_metropolis(μ, c.μ_prop,idx_deleted, wl,sim)
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
        
        random_point_in_sphere!(sim.rng,c.μ_prop.r_box[c.μ_prop.N]) # !!!

        idx_deleted = 0

        accept = hw_N_metropolis(μ, c.μ_prop,idx_deleted, wl,sim)
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


function  hw_N_metropolis(μ::microstate,
                           μ_prop::microstate, idx_deleted::Int64, # μ_prop = μ_proposed the next proposed microstate
                           wl::WangLandauVars,sim::SimulationParams)::Bool #  ✅ 

        # the purpose of the function is to compute the acceptance criterion for different situations
        # many things change including the form of the metropolis criterion you need to compute, what energies you have to compute to get ΔE, and so on.
        # covered by equations 19 in Ganzmuller and Camp 2007 doi: 10.1063/1.2794042
        # assumes that you have already checked that N_proposed is in bounds  (N_min ≤ N_proposed ≤ N_max)

        logQ_diff = wl.logQ_N[μ.N+1] - wl.logQ_N[μ_prop.N+1]
        partition_ratio = exp(logQ_diff)

        V_sphere_σ = (4.0/3.0) * π * (sim.L_σ/2.0)^3  # sphere inscribed in box: R_c = L_σ/2
        V_Λ_prefactor = V_sphere_σ^(μ_prop.N - μ.N) * sim.Λ_σ^(3*(μ.N - μ_prop.N))

        # now we compute the exponential part of the criterion having to do with the configurational potential energy

        if μ.N > μ_prop.N # deletion move
            factorial_prefactor = μ.N # N! / (N-1)! = N
            # ΔU = U(N-1) - U(N) = -(energy of deleted particle with all others)
            # so E_old = interaction of deleted particle with all N-1 remaining, E_proposed = 0
            E_old = hws_potential_1(sim.potential,μ.r_box,μ.r_box[idx_deleted],idx_deleted,μ.N,sim.L_squared_σ,sim.r_cut_squared_box)
            E_proposed = 0.0

        else # insertion move
            factorial_prefactor = 1/μ_prop.N  # N! / (N+1)! = 1/(N+1)
            
            #  change in potential energy comes from inserting new particle, all other interactions are the same so ΔE = sum of interactions of new particle with old
            E_old = 0
            i = μ_prop.N 
            E_proposed = hws_potential_1(sim.potential,μ_prop.r_box,μ_prop.r_box[i],i,μ_prop.N,sim.L_squared_σ,sim.r_cut_squared_box)
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
end #hw_N_metropolis