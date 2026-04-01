using Test
using StaticArrays
using Random
using Plots

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using segc_wl   

println("")
println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Running Function Unit Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
println("")

############ Tests for things in utils_module ############
@testset "Utils" begin
    rng = MersenneTwister(42)
    r1 = MVector{3,Float64}(100*rand(rng, 3))
    r2 = MVector{3,Float64}(100*rand(rng, 3))
    @test euclidean_distance(r1, r2) ≈ sqrt(sum((r1 .- r2).^2))

    r1_box = MVector{3,Float64}(0, 0, 0.4)
    r2_box = MVector{3,Float64}(0, 0, -0.4)
    @test euclidean_distance_squared_pbc(r1_box, r2_box) ≈ 0.04

    # translate_by_random_vector! is now in-place and requires a SimCache
    input_path = joinpath(@__DIR__, "cube_vertices_home_made.inp")
    T_σ = 1.0
    sim_u = SimulationParams(N_max=8, N_min=0, T_σ=T_σ, Λ_σ=argon_deBroglie(T_σ),
                             λ_max=99, r_cut_σ=3.0,
                             input_filename=input_path,
                             save_directory_path=@__DIR__,
                             rng=MersenneTwister(1))
    μ_u = init_microstate(sim=sim_u, filename=input_path)
    c_u = init_cache(sim_u, μ_u)

    r = MVector{3,Float64}(0.0, 0.0, 0.0)
    translate_by_random_vector!(r, 0.1, sim_u.rng, c_u)
    @test sqrt(sum(r.^2)) ≤ sqrt(3)*0.1 # test that it moves by no more than √3*δr_max from sqrt(0.1^2 + 0.1^2 + 0.1^2)
end

@testset "Utils Metropolis acceptance rates" begin
    @test metropolis(100.0, 1.0) == false # reject large increase
    @test metropolis(-0.1,  1.0) == true  # accept downhill move
    @test metropolis(0.0,   1.0) == true  # accept neutral move

    rng = MersenneTwister(1234) # now we test acceptance of slightly uphill move

    # For ΔE/T = 1, acceptance should be ≈ exp(-1) ≈ 36.8%
    ΔE = 1.0
    T_σ = 1.0
    n_trials = 10000
    n_accepted = sum(metropolis(ΔE, T_σ, rng) for _ in 1:n_trials)
    acceptance_rate = n_accepted / n_trials
    expected_rate = exp(-ΔE / T_σ)

    # Test within reasonable statistical bounds (±3σ for binomial)
    σ = sqrt(n_trials * expected_rate * (1 - expected_rate)) / n_trials
    @test acceptance_rate ≈ expected_rate atol = 2*σ
end

############ Tests for things in lj_module ############
@testset "LJ_module" begin
    @test argon_deBroglie(2.0) ≈ 0.0530973 atol=0.000001 # verified against /test/mathematica_verification.nb

    # λ=0 means zero coupling → ϵ_ξ=0, σ_ξ_squared=0 → no fractional energy
    @test E_12_frac_LJ(2., 0, 99, 0.0, 0.0) == 0

    # E_12_frac_LJ now takes precomputed ϵ_ξ and σ_ξ_squared (previously computed internally)
    λ_t = 3; λ_max_t = 99; M_t = λ_max_t + 1
    ϵ_ξ_t    = (λ_t / M_t)^(1/3)
    σ_ξ_sq_t = (λ_t / M_t)^(1/2)
    fracs = [E_12_frac_LJ(r2, λ_t, λ_max_t, ϵ_ξ_t, σ_ξ_sq_t)
             for r2 in [0.0001, 1., 2., 10., 30., 5.6, 100000000000.0]]
    mathematica = [3.355811e19, -0.0064247, -0.000806758, -6.45823*10^-6, -2.39195*10^-7, -0.0000367738, -6.45826*10^-36]
    @test fracs[1] ≈ mathematica[1] atol=10^15
    @test all(isapprox.(fracs[2:end], mathematica[2:end]; atol=1e-6)) # tests E_12_frac_LJ

    # tested against allen and tildesly's potential_1 from mc_lj_module.py:
    # r_box is now Vector{MVector{3,Float64}} instead of a matrix; r_frac_box is MVector;
    # potential_1_normal now takes ϵ_ξ and σ_ξ_squared as extra args (0.0 since λ=0)
    cfg = joinpath(@__DIR__, "cnf_default.inp")
    N, L_σ, r_σ = load_configuration(cfg)
    r_box = r_σ ./ L_σ
    r_frac_box = @MVector [0.0, 0.0, 0.0]
    test_idx = 1 .+ [0, 2, 3, 56, 100, 34, 222, 255, 78, 89]
    E_test = [potential_1_normal(r_box, r_box[idx], idx, r_frac_box, 0, 99, N, L_σ^2, (3/L_σ)^2, 0.0, 0.0)
              for idx in test_idx]
    allen_tildesly_results = [-11.932319723716308, -11.932319753958657, -11.932319753958655,
                               -11.932319756099865, -11.9323197764367,   -11.932319756566486,
                               -11.932319756099865, -11.932319738837482, -11.932319771221039,
                               -11.9323197764367]
    @test all(E_test .≈ allen_tildesly_results) # this tests E_12_LJ, euclidean_distance_squared_pbc, the non-fractional-part of potential_1_normal
end

println("")

@testset "λ_metropolis" begin
    # λ_metropolis_pm1 now takes (μ, μ_prop, idx_deleted, wl, sim) instead of many positional args.
    # Build structs using init functions; use cache to get a properly allocated μ_prop.
    input_path = joinpath(@__DIR__, "cube_vertices_home_made.inp")
    T_σ = 1.0; Λ_σ = argon_deBroglie(T_σ); λ_max = 99; M = λ_max + 1

    sim = SimulationParams(N_max=450, N_min=0, T_σ=T_σ, Λ_σ=Λ_σ,
                           λ_max=λ_max, r_cut_σ=3.0,
                           input_filename=input_path,
                           save_directory_path=@__DIR__,
                           rng=MersenneTwister())

    # first lets handle the situation where only λ changes within the range s.t. N doesn't change
    # and there is no partition function bias
    λ = 34
    λ_proposed = 33 # energy goes down in this configuration so always accepts if propose 35
    μ = init_microstate(sim=sim, filename=input_path, λ=λ)
    c = init_cache(sim, μ)
    wl = init_WangLandauVars(sim)

    copy_microstate!(c.μ_prop, μ)
    c.μ_prop.λ           = λ_proposed
    c.μ_prop.ϵ_ξ         = (λ_proposed / M)^(1/3)
    c.μ_prop.σ_ξ_squared = (λ_proposed / M)^(1/2)

    # should have the same probability as metropolis()
    ΔE = potential_1_frac(μ.r_box, μ.r_frac_box, λ_proposed, λ_max, μ.N,
                           sim.L_squared_σ, sim.r_cut_squared_box,
                           c.μ_prop.ϵ_ξ, c.μ_prop.σ_ξ_squared) -
         potential_1_frac(μ.r_box, μ.r_frac_box, λ, λ_max, μ.N,
                           sim.L_squared_σ, sim.r_cut_squared_box,
                           μ.ϵ_ξ, μ.σ_ξ_squared)
    n_trials = 10000
    expected_rate = exp(-ΔE / T_σ)
    n_accepted = sum(λ_metropolis_pm1(μ, c.μ_prop, 0, wl, sim) for _ in 1:n_trials)
    acceptance_rate = n_accepted / n_trials
    σ = sqrt(n_trials * expected_rate * (1 - expected_rate)) / n_trials
    @test acceptance_rate ≈ expected_rate atol = 3*σ

    # BUT NOW LETS BIAS WANG LANDAU
    wl.logQ_λN[λ+1,      μ.N+1] = 2
    wl.logQ_λN[λ_proposed+1, μ.N+1] = 3

    n_accepted = sum(λ_metropolis_pm1(μ, c.μ_prop, 0, wl, sim) for _ in 1:n_trials)
    acceptance_rate_biased = n_accepted / n_trials
    @test acceptance_rate_biased < acceptance_rate
    @test acceptance_rate_biased ≈ exp(2)/exp(3)*acceptance_rate atol = 0.1

    # particle created, no partition bias should accept because having particle at 0,0,0 is lower energy for this config
    wl2 = init_WangLandauVars(sim)
    μ2 = init_microstate(sim=sim, filename=input_path, λ=λ_max)
    c2 = init_cache(sim, μ2)
    μ2.r_frac_box .= MVector{3,Float64}(0.0, 0.0, 0.0)

    copy_microstate!(c2.μ_prop, μ2)
    c2.μ_prop.N                   = μ2.N + 1
    c2.μ_prop.r_box[c2.μ_prop.N] .= μ2.r_frac_box  # promote ghost → full particle
    c2.μ_prop.r_frac_box          .= MVector{3,Float64}(0.1, 0.2, -0.3)
    c2.μ_prop.λ                   = 0
    c2.μ_prop.ϵ_ξ                 = 0.0
    c2.μ_prop.σ_ξ_squared         = 0.0

    N_proposed = c2.μ_prop.N
    E_old = potential_1_frac(μ2.r_box, μ2.r_frac_box, λ_max, λ_max, μ2.N,
                              sim.L_squared_σ, sim.r_cut_squared_box, μ2.ϵ_ξ, μ2.σ_ξ_squared)
    E_new = potential_1_normal(c2.μ_prop.r_box, c2.μ_prop.r_box[N_proposed], N_proposed,
                                c2.μ_prop.r_frac_box, 0, λ_max, N_proposed,
                                sim.L_squared_σ, sim.r_cut_squared_box, 0.0, 0.0)
    ΔE = E_new - E_old
    # V_Λ_prefactor = V^(N_prop - N - 1) * Λ^(3(N+1) - 3*N_prop) = 1 for this transition
    prefactor = (sim.V_σ^N_proposed / sim.V_σ^(μ2.N+1)) * (1/N_proposed) * (Λ_σ^(3*μ2.N + 3) / Λ_σ^(3*N_proposed))
    n_trials = 1000
    n_accepted = sum(λ_metropolis_pm1(μ2, c2.μ_prop, 0, wl2, sim) for _ in 1:n_trials)
    acceptance_rate = n_accepted / n_trials
    @test acceptance_rate ≈ prefactor*exp(-1*ΔE/T_σ) atol = 0.02

    # Particle DESTROYED
    wl3 = init_WangLandauVars(sim)
    μ3 = init_microstate(sim=sim, filename=input_path, λ=0)
    c3 = init_cache(sim, μ3)
    idx_deleted = 3 # just picked randomly
    μ3.r_frac_box .= MVector{3,Float64}(-0.08, 0.1, 0.1)

    copy_microstate!(c3.μ_prop, μ3)
    c3.μ_prop.N = μ3.N - 1
    if idx_deleted != μ3.N
        c3.μ_prop.r_box[idx_deleted] .= μ3.r_box[μ3.N]  # swap last particle into deleted slot
    end
    c3.μ_prop.r_frac_box  .= μ3.r_frac_box
    c3.μ_prop.λ            = λ_max
    c3.μ_prop.ϵ_ξ          = (λ_max / M)^(1/3)
    c3.μ_prop.σ_ξ_squared  = (λ_max / M)^(1/2)

    E_old = potential_1_normal(μ3.r_box, μ3.r_box[idx_deleted], idx_deleted,
                                μ3.r_frac_box, 0, λ_max, μ3.N,
                                sim.L_squared_σ, sim.r_cut_squared_box, μ3.ϵ_ξ, μ3.σ_ξ_squared)
    E_new = potential_1_frac(c3.μ_prop.r_box, c3.μ_prop.r_frac_box, λ_max, λ_max, c3.μ_prop.N,
                              sim.L_squared_σ, sim.r_cut_squared_box, c3.μ_prop.ϵ_ξ, c3.μ_prop.σ_ξ_squared)
    ΔE = E_new - E_old
    metrop_prob = exp(-ΔE/T_σ)
    # V_Λ_prefactor = V^(N_prop+1 - N) * Λ^(3N - 3(N_prop+1)) = 1 for this transition
    prefactor = (sim.V_σ^(c3.μ_prop.N+1 - μ3.N)) * (Λ_σ^(3*μ3.N - 3*c3.μ_prop.N - 3))
    factorial_prefactor = μ3.N
    expected_prob = metrop_prob * factorial_prefactor * prefactor

    n_trials = 10000
    n_accepted = sum(λ_metropolis_pm1(μ3, c3.μ_prop, idx_deleted, wl3, sim) for _ in 1:n_trials)
    acceptance_rate = n_accepted / n_trials
    σ = sqrt(n_trials * expected_prob * (1 - expected_prob)) / n_trials
    @test acceptance_rate ≈ expected_prob atol = 3*σ
end

@testset "translation_move!() tests" begin
    # translation_move! now takes a SimCache as a 4th argument
    input_path = joinpath(@__DIR__, "cube_vertices_home_made.inp")
    T_σ = 1.0; Λ_σ = argon_deBroglie(T_σ); λ_max = 99

    sim = SimulationParams(N_max=8, N_min=0, T_σ=T_σ, Λ_σ=Λ_σ,
                           λ_max=λ_max, r_cut_σ=3.0,
                           input_filename=input_path,
                           save_directory_path=@__DIR__,
                           rng=MersenneTwister(1234),
                           dynamic_δr_max_box=true)
    # microstate and WangLandauVars now built via init functions instead of direct constructors
    μ = init_microstate(sim=sim, filename=input_path, λ=34)
    wl = init_WangLandauVars(sim)
    c = init_cache(sim, μ)

    old_proposed = wl.translation_moves_proposed
    old_r_frac = deepcopy(μ.r_frac_box)
    old_r = deepcopy(μ.r_box)
    translation_move!(sim, μ, wl, c)
    @test wl.translation_moves_proposed == old_proposed + 1  # counter always increments

    # making sure δr_max gets updated properly — capture value before each call
    wl.translation_moves_accepted = 2
    wl.translation_moves_proposed = 3
    δr_before = wl.δr_max_box
    translation_move!(sim, μ, wl, c) # ratio after: ≥ 2/4 = 0.5, either way stays > 0.45, check increase condition
    # accepted = 2 or 3, proposed = 4: ratio is 0.5 or 0.75; 0.5 is in (0.45, 0.55) → no change only if exactly 0.5
    # to make this test deterministic: set counts so ratio is always > 0.55 regardless of accept/reject
    wl.translation_moves_accepted = 10
    wl.translation_moves_proposed = 15
    δr_before = wl.δr_max_box
    translation_move!(sim, μ, wl, c) # 10/16 or 11/16, both > 0.55 → δr_max increases
    @test wl.δr_max_box == δr_before * 1.05

    wl.δr_max_box = 0.15
    wl.translation_moves_accepted = 1
    wl.translation_moves_proposed = 10
    δr_before = wl.δr_max_box
    translation_move!(sim, μ, wl, c) # 1/11 or 2/11, both < 0.45 → δr_max decreases
    @test wl.δr_max_box == δr_before * 0.95
end

println("")
println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Now Running SimCache Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
println("")

############ SimCache Tests ############

@testset "SimCache: init_cache correctness" begin
    input_path = joinpath(@__DIR__, "cube_vertices_home_made.inp")
    T_σ = 1.0
    sim_c = SimulationParams(N_max=8, N_min=0, T_σ=T_σ, Λ_σ=argon_deBroglie(T_σ),
                             λ_max=99, r_cut_σ=3.0,
                             input_filename=input_path,
                             save_directory_path=@__DIR__,
                             rng=MersenneTwister(42))
    μ_c = init_microstate(sim=sim_c, filename=input_path, λ=5)
    c = init_cache(sim_c, μ_c)

    # Scalar fields match μ after init
    @test c.μ_prop.N == μ_c.N
    @test c.μ_prop.λ == μ_c.λ
    @test c.μ_prop.ϵ_ξ == μ_c.ϵ_ξ
    @test c.μ_prop.σ_ξ_squared == μ_c.σ_ξ_squared

    # r_frac_box values match
    @test c.μ_prop.r_frac_box == μ_c.r_frac_box

    # All N particle positions match
    for i in 1:μ_c.N
        @test c.μ_prop.r_box[i] == μ_c.r_box[i]
    end

    # Deep copy: no shared MVector references for any particle slot
    for i in 1:μ_c.N
        @test c.μ_prop.r_box[i] !== μ_c.r_box[i]  # different objects
    end
    @test c.μ_prop.r_frac_box !== μ_c.r_frac_box

    # Verify independence: mutating μ_c does not affect c.μ_prop
    old_val = μ_c.r_box[1][1]
    μ_c.r_box[1][1] += 99.0
    @test c.μ_prop.r_box[1][1] ≈ old_val  # cache not affected
    μ_c.r_box[1][1] = old_val  # restore

    # Scratch fields initialized to zero
    @test all(c.ζ_Mvec .== 0.0)
    @test c.ζ_idx == 0
    @test all(c.ri_proposed_box .== 0.0)

    # ri_proposed_box does not alias any particle position
    for i in 1:μ_c.N
        @test c.ri_proposed_box !== μ_c.r_box[i]
        @test c.ri_proposed_box !== c.μ_prop.r_box[i]
    end

    # μ_prop.r_box has N_max slots allocated
    @test length(c.μ_prop.r_box) == sim_c.N_max
end

@testset "SimCache: mirror invariant under translation moves" begin
    input_path = joinpath(@__DIR__, "cube_vertices_home_made.inp")
    T_σ = 1.5
    sim_t = SimulationParams(N_max=8, N_min=0, T_σ=T_σ, Λ_σ=argon_deBroglie(T_σ),
                             λ_max=99, r_cut_σ=3.0,
                             input_filename=input_path,
                             save_directory_path=@__DIR__,
                             rng=MersenneTwister(7))
    μ_t = init_microstate(sim=sim_t, filename=input_path, λ=10)
    wl_t = init_WangLandauVars(sim_t)
    c_t = init_cache(sim_t, μ_t)

    function check_mirror(μ, c)
        # After every complete move, c.μ_prop must mirror μ exactly
        c.μ_prop.N == μ.N || return false
        c.μ_prop.λ == μ.λ || return false
        c.μ_prop.ϵ_ξ == μ.ϵ_ξ || return false
        c.μ_prop.σ_ξ_squared == μ.σ_ξ_squared || return false
        c.μ_prop.r_frac_box == μ.r_frac_box || return false
        for i in 1:μ.N
            c.μ_prop.r_box[i] == μ.r_box[i] || return false
        end
        return true
    end

    for _ in 1:500
        translation_move!(sim_t, μ_t, wl_t, c_t)
        @test check_mirror(μ_t, c_t)
        @test 1 ≤ c_t.ζ_idx ≤ μ_t.N + 1
    end
end

@testset "SimCache: mirror invariant under lambda moves" begin
    input_path = joinpath(@__DIR__, "cube_vertices_home_made.inp")
    T_σ = 5.0  # high T to encourage acceptance of all move types
    sim_l = SimulationParams(N_max=8, N_min=0, T_σ=T_σ, Λ_σ=argon_deBroglie(T_σ),
                             λ_max=9, r_cut_σ=3.0,
                             input_filename=input_path,
                             save_directory_path=@__DIR__,
                             rng=MersenneTwister(13))
    # Start at λ=0 so N-decrement and N-increment moves are both reachable quickly
    μ_l = init_microstate(sim=sim_l, filename=input_path, λ=0)
    wl_l = init_WangLandauVars(sim_l)
    c_l = init_cache(sim_l, μ_l)

    function check_mirror_full(μ, c)
        c.μ_prop.N == μ.N || return false
        c.μ_prop.λ == μ.λ || return false
        c.μ_prop.ϵ_ξ == μ.ϵ_ξ || return false
        c.μ_prop.σ_ξ_squared == μ.σ_ξ_squared || return false
        c.μ_prop.r_frac_box == μ.r_frac_box || return false
        for i in 1:μ.N
            c.μ_prop.r_box[i] == μ.r_box[i] || return false
        end
        return true
    end

    λ_values_seen = Set{Int}()
    N_values_seen = Set{Int}()

    for _ in 1:2000
        λ_move!(sim_l, μ_l, wl_l, c_l)
        update_wl!(wl_l, μ_l)  # must call after every move as in the real simulation; without this, WL bias never builds and the system gets trapped near high-λ states
        push!(λ_values_seen, μ_l.λ)
        push!(N_values_seen, μ_l.N)
        @test check_mirror_full(μ_l, c_l)
        # λ and N always in bounds
        @test sim_l.N_min ≤ μ_l.N ≤ sim_l.N_max
        @test 0 ≤ μ_l.λ ≤ sim_l.λ_max
    end

    # At high T with 2000 WL-biased λ moves, all (λ,N) states are visited uniformly;
    # both N values and λ values must change.
    @test length(N_values_seen) > 1
    @test length(λ_values_seen) > 1
end

@testset "SimCache: mirror invariant through mixed move sequences" begin
    input_path = joinpath(@__DIR__, "cube_vertices_home_made.inp")
    T_σ = 3.0
    sim_m = SimulationParams(N_max=8, N_min=0, T_σ=T_σ, Λ_σ=argon_deBroglie(T_σ),
                             λ_max=9, r_cut_σ=3.0,
                             input_filename=input_path,
                             save_directory_path=@__DIR__,
                             rng=MersenneTwister(99))
    μ_m = init_microstate(sim=sim_m, filename=input_path, λ=0)
    wl_m = init_WangLandauVars(sim_m)
    c_m = init_cache(sim_m, μ_m)

    function check_mirror_m(μ, c)
        c.μ_prop.N == μ.N || return false
        c.μ_prop.λ == μ.λ || return false
        c.μ_prop.ϵ_ξ == μ.ϵ_ξ || return false
        c.μ_prop.σ_ξ_squared == μ.σ_ξ_squared || return false
        c.μ_prop.r_frac_box == μ.r_frac_box || return false
        for i in 1:μ.N
            c.μ_prop.r_box[i] == μ.r_box[i] || return false
        end
        return true
    end

    # Replicate the actual simulation loop move selection (75% translation, 25% λ)
    rng_loop = MersenneTwister(55)
    for _ in 1:1000
        ζ = rand(rng_loop)
        if ζ < 0.75
            translation_move!(sim_m, μ_m, wl_m, c_m)
        else
            λ_move!(sim_m, μ_m, wl_m, c_m)
        end
        update_wl!(wl_m, μ_m)
        @test check_mirror_m(μ_m, c_m)
    end

    # Verify r_box deep copy independence still holds after 1000 mixed moves
    # (mutating μ_m.r_box[1] must not change c_m.μ_prop.r_box[1])
    old_val = μ_m.r_box[1][1]
    μ_m.r_box[1][1] += 99.0
    @test c_m.μ_prop.r_box[1][1] ≈ old_val
    μ_m.r_box[1][1] = old_val
end

println("")
println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Now Running Initialization Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
println("")

@testset "Initialization: SimulationParams with direct L_σ" begin
    T_σ = 1.0
    L_σ = 8.0
    sim_L = SimulationParams(N_max=10, N_min=0, T_σ=T_σ, Λ_σ=argon_deBroglie(T_σ),
                             λ_max=99, r_cut_σ=3.0,
                             L_σ=L_σ,
                             save_directory_path=@__DIR__,
                             rng=MersenneTwister(1))
    @test sim_L.L_σ ≈ L_σ
    @test sim_L.V_σ ≈ L_σ^3
    @test sim_L.L_squared_σ ≈ L_σ^2
    @test sim_L.r_cut_box ≈ 3.0 / L_σ
    @test sim_L.r_cut_squared_box ≈ (3.0 / L_σ)^2

    # should throw if neither input_filename nor L_σ provided
    @test_throws ArgumentError SimulationParams(N_max=10, N_min=0, T_σ=T_σ, Λ_σ=argon_deBroglie(T_σ),
                                                λ_max=99, r_cut_σ=3.0,
                                                save_directory_path=@__DIR__)
end

@testset "Initialization: init_microstate vacuum (N=0, λ=0) state" begin
    T_σ = 1.0
    sim_v = SimulationParams(N_max=50, N_min=0, T_σ=T_σ, Λ_σ=argon_deBroglie(T_σ),
                             λ_max=99, r_cut_σ=3.0,
                             L_σ=8.0,
                             save_directory_path=@__DIR__,
                             rng=MersenneTwister(1))
    μ_v = init_microstate(sim_v)

    @test μ_v.N == 0
    @test μ_v.λ == 0
    @test μ_v.ϵ_ξ == 0.0
    @test μ_v.σ_ξ_squared == 0.0
    @test length(μ_v.r_box) == sim_v.N_max  # pre-allocated to N_max
    @test all(μ_v.r_frac_box .== 0.0)
end

@testset "Initialization: init_microstate from file pre-allocates N_max slots" begin
    input_path = joinpath(@__DIR__, "cube_vertices_home_made.inp")
    T_σ = 1.0
    sim_f = SimulationParams(N_max=50, N_min=0, T_σ=T_σ, Λ_σ=argon_deBroglie(T_σ),
                             λ_max=99, r_cut_σ=3.0,
                             input_filename=input_path,
                             save_directory_path=@__DIR__,
                             rng=MersenneTwister(1))
    μ_f = init_microstate(sim=sim_f, filename=input_path)

    # cube_vertices_home_made.inp has 8 particles but N_max=50
    @test μ_f.N == 8
    @test length(μ_f.r_box) == sim_f.N_max  # regression: was previously length N_file, not N_max
end

@testset "Initialization: init_cache with vacuum microstate" begin
    T_σ = 1.0
    sim_cv = SimulationParams(N_max=20, N_min=0, T_σ=T_σ, Λ_σ=argon_deBroglie(T_σ),
                              λ_max=99, r_cut_σ=3.0,
                              L_σ=8.0,
                              save_directory_path=@__DIR__,
                              rng=MersenneTwister(1))
    μ_cv = init_microstate(sim_cv)
    c_cv = init_cache(sim_cv, μ_cv)

    @test c_cv.μ_prop.N == 0
    @test c_cv.μ_prop.λ == 0
    @test length(c_cv.μ_prop.r_box) == sim_cv.N_max
    @test c_cv.μ_prop.r_frac_box == μ_cv.r_frac_box
    @test c_cv.μ_prop.r_frac_box !== μ_cv.r_frac_box  # deep copy, not aliased
end

println("")
println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Now Running Physics Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
println("")

@testset "Comparison to Analytic Values of Q(N=1),Q(N=2)" begin
    # simulation of up to 4 atoms with SEGC-WL currently takes about 15 seconds
    # so lets run this 5 times for statistics
    logQ_N1 = 0 
    logQ_N2 = 0
    for ii in 1:5
        input_path = joinpath(@__DIR__, "4_atom_cnf.inp")
        T_σ = 1.0 
        Λ_σ = argon_deBroglie(T_σ)
        sim = SimulationParams(
            N_max=4,
            N_min=0,
            T_σ=T_σ,
            Λ_σ = Λ_σ,
            λ_max = 99,
            r_cut_σ = 3.,
            input_filename=input_path,
            save_directory_path= @__DIR__ , 
            maxiter=100_000_000)
        μstate = init_microstate(sim=sim,filename=input_path)

        wl = init_WangLandauVars(sim)
        cache = init_cache(sim,μstate)

        run_simulation!(sim,μstate,wl,cache)
        # post_run(sim,μstate,wl)
        logQ = correct_logQ(wl)
        logQ_N1 += logQ[2]
        logQ_N2 += logQ[3]
    end
    logQ_N1 *= (1/5) # average value of Q(N=1,λ=0|V=512σ,T*=1) over five monte carlo runs
    logQ_N2 *= (1/5)
    println(logQ_N1)
    println(logQ_N2)
    #= the technically and statistically best thing to do would be run the simulation m independent times - independent meaning they must start from different initializaiton states, which is not the FCC focused workflow we have now
    Then observe the sample standard error from the m trials.  (from normal Standard Error = Standard Deviation/sqrt(m) formula)
    then you would use that SE to form a confidence interval and assert the analytic value falls inside of it:
        abs(mean_logQ - logQ_analytic) ≤ k *SE where k is the confidence interval such as k=1.96 corresponding to 95% confidence interval
    can come back and do this later but because this is going to run often we're just going to do it quick and dirty:
    =#
    mathematica_logQ_N1 = 14.0055 # analytically computed in /tests/mathematica_verification.nb a mathematica notebook
    mathematica_logQ_N2 = 27.3179 # analytically computed in /tests/mathematica_verification.nb a mathematica notebook

    @test logQ_N1 ≈ mathematica_logQ_N1 atol = (0.05*mathematica_logQ_N1) # within 5% seems good
    @test logQ_N2 ≈ mathematica_logQ_N2 atol = (0.05*mathematica_logQ_N2)

end


@testset "Comparison to Analytic Ideal Gas Limit" begin
    # in the high temperature limit, we expect our system to become the ideal gas which has an analytically computable partition function

    logQ_avg = [0.,0.,0.,0.,0.]
    for ii in 1:5
        input_path = joinpath(@__DIR__, "4_atom_cnf.inp")
        T_σ = 1_000_000.0 
        Λ_σ = argon_deBroglie(T_σ)
        sim = SimulationParams(
            N_max=4,
            N_min=0,
            T_σ=T_σ,
            Λ_σ = Λ_σ,
            λ_max = 99,
            r_cut_σ = 3.,
            input_filename=input_path,
            save_directory_path= @__DIR__ , 
            maxiter=100_000_000)
        μstate = init_microstate(sim=sim,filename=input_path)
        wl = init_WangLandauVars(sim)
        cache = init_cache(sim,μstate)

        run_simulation!(sim,μstate,wl,cache)
        logQ = correct_logQ(wl)
        for ii in 1:5
            logQ_avg[ii] += logQ[ii]
        end
    end
    logQ_avg = logQ_avg.*(1/5) # average value of partition functions over five monte carlo runs
    println(logQ_avg)
    #println(logQ_avg)

    #= the technically and statistically best thing to do would be run the simulation m independent times - independent meaning they must start from different initializaiton states, which is not the FCC focused workflow we have now
    Then observe the sample standard error from the m trials.  (from normal Standard Error = Standard Deviation/sqrt(m) formula)
    then you would use that SE to form a confidence interval and assert the analytic value falls inside of it:
        abs(mean_logQ - logQ_analytic) ≤ k *SE where k is the confidence interval such as k=1.96 corresponding to 95% confidence interval
    can come back and do this later but because this is going to run often we're just going to do it quick and dirty:
    =#
    mathematica_logQ_N2 = 68.7644 # analytically computed in /tests/mathematica_verification.nb a mathematica notebook
    mathematica_logQ_N3 = 102.395
    mathematica_logQ_N4 = 135.737
    @test logQ_avg[3] ≈ mathematica_logQ_N2 atol = (0.05*mathematica_logQ_N2) # within 5% seems good
    @test logQ_avg[4] ≈ mathematica_logQ_N3 atol = (0.05*mathematica_logQ_N3) # within 5% seems good
    @test logQ_avg[5] ≈ mathematica_logQ_N4 atol = (0.05*mathematica_logQ_N4) # within 5% seems good
end


@testset "High-N Ideal Gas Limit: logQ(N=10 to 108) vs analytic (diagnostic for high-N drift)" begin
    #= This test is designed to expose the systematic upward drift in logQ(N) at high N that has
       been observed across many conditions. It can catch bugs in two distinct layers:

       1. Acceptance criterion: at T*=1e6 the Boltzmann factor exp(-ΔE/T)→1 for all moves, so
          the acceptance criterion reduces to partition_ratio × V_Λ_prefactor × factorial_prefactor.
          Any wrong exponent or off-by-one in these terms produces a per-step multiplicative error
          that accumulates linearly with N (i.e. logQ_WL(N) ≈ logQ_true(N) + N×log(f) for some f≠1).

       2. Cache / microstate scaffolding: bugs in copy_microstate!, shallow-copy aliasing between
          μ.r_box and c.μ_prop.r_box, incorrect rollback of c.μ_prop on rejection, or wrong particle
          positions in c.μ_prop.r_box during N-increment / N-decrement energy evaluation will all
          introduce a systematic energy error in λ_metropolis_pm1. At T*=1e6 these would manifest
          as wrong acceptance rates for N-change moves (since even a tiny absolute energy error
          becomes a large relative error when divided by T=1), biasing logQ(N) at high N.

       λ_max=3 is used instead of 99 so the WL histogram has only (λ_max+1)×(N_max+1) = 4×109 = 436
       bins rather than 10,900. This makes convergence ~25× faster while remaining physically valid
       at T*=1e6 (all alchemical moves accept trivially since ΔE/T→0).

       Comparison uses ideal_gas_logQ_loggamma (exact, no Stirling approximation) and starts at N=10.
       Comparison tolerance is 5% of |logQ_analytic|.
    =#
    input_path = joinpath(@__DIR__, "../initial_configs/N108_L8.inp")
    T_σ = 1_000_000.0
    Λ_σ = argon_deBroglie(T_σ)
    sim = SimulationParams(
        N_max=108,
        N_min=0,
        T_σ=T_σ,
        Λ_σ=Λ_σ,
        λ_max=3,
        r_cut_σ=3.0,
        input_filename=input_path,
        save_directory_path=@__DIR__,
        maxiter=50_000_000)
    μstate = init_microstate(sim=sim, filename=input_path)
    wl = init_WangLandauVars(sim)
    cache = init_cache(sim, μstate)

    run_simulation!(sim, μstate, wl, cache)
    logQ_wl = correct_logQ(wl)

    Ns            = collect(10:sim.N_max)
    logQ_analytic = [ideal_gas_logQ_loggamma(N, sim.V_σ, sim.Λ_σ) for N in Ns]
    logQ_sim      = [logQ_wl[N+1] for N in Ns]  # 1-indexed: index N+1 = N
    rel_err       = (logQ_sim .- logQ_analytic) ./ abs.(logQ_analytic)

    println("High-N ideal gas test: logQ_wl vs analytic for N=10:108")
    for (i, N) in enumerate(Ns)
        println("  N=$N: WL=$(round(logQ_sim[i],digits=2))  analytic=$(round(logQ_analytic[i],digits=2))  diff=$(round(logQ_sim[i]-logQ_analytic[i],digits=3))  rel_err=$(round(100*rel_err[i],digits=2))%")
        @test logQ_sim[i] ≈ logQ_analytic[i] atol = (0.05 * abs(logQ_analytic[i]))
    end

    # Save comparison plots
    p1 = plot(Ns, logQ_analytic, label="analytic (loggamma)", lw=2, color=:blue,
              xlabel="N", ylabel="log Q(N,V,T)", title="High-N Ideal Gas: WL vs Analytic",
              legend=:topleft)
    plot!(p1, Ns, logQ_sim, label="WL simulation", lw=2, color=:red, linestyle=:dash)

    p2 = plot(Ns, 100 .* rel_err, label="relative error (%)", lw=2, color=:green,
              xlabel="N", ylabel="relative error (%)",
              title="High-N Ideal Gas: Relative Error (WL − analytic) / |analytic|",
              legend=:topleft)
    hline!(p2, [5.0, -5.0], label="±5% tolerance", color=:red, linestyle=:dot)
    hline!(p2, [0.0], label="zero", color=:black, linestyle=:dash)

    combined = plot(p1, p2, layout=(2,1), size=(800, 800))
    savefig(combined, joinpath(@__DIR__, "high_N_ideal_gas_logQ_comparison.png"))
    println("Plot saved to test/high_N_ideal_gas_logQ_comparison.png")
end