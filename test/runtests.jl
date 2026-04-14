using Test
using StaticArrays
using Random
using Plots

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using gc_wl

println("")
println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Running Function Unit Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
println("")

############ Utils ############

@testset "Utils" begin
    rng = MersenneTwister(42)
    r1 = MVector{3,Float64}(100*rand(rng, 3))
    r2 = MVector{3,Float64}(100*rand(rng, 3))
    @test euclidean_distance(r1, r2) ≈ sqrt(sum((r1 .- r2).^2))

    r1_box = MVector{3,Float64}(0, 0,  0.4)
    r2_box = MVector{3,Float64}(0, 0, -0.4)
    @test euclidean_distance_squared_pbc(r1_box, r2_box) ≈ 0.04

    T_σ = 1.0
    sim_u = SimulationParams(N_max=8, N_min=0, T_σ=T_σ, Λ_σ=argon_deBroglie(T_σ),
                             r_cut_σ=3.0,
                             input_filename=joinpath(@__DIR__, "cube_vertices_home_made.inp"),
                             save_directory_path=@__DIR__,
                             rng=MersenneTwister(1))
    μ_u = init_microstate(sim=sim_u, filename=joinpath(@__DIR__, "cube_vertices_home_made.inp"))
    c_u = init_cache(sim_u, μ_u)

    r = MVector{3,Float64}(0.0, 0.0, 0.0)
    translate_by_random_vector!(r, 0.1, sim_u.rng, c_u)
    @test sqrt(sum(r.^2)) ≤ sqrt(3)*0.1
end

@testset "Utils Metropolis acceptance rates" begin
    @test metropolis(100.0, 1.0) == false
    @test metropolis(-0.1,  1.0) == true
    @test metropolis(0.0,   1.0) == true

    rng = MersenneTwister(1234)
    ΔE = 1.0; T_σ = 1.0; n_trials = 10000
    n_accepted = sum(metropolis(ΔE, T_σ, rng) for _ in 1:n_trials)
    acceptance_rate = n_accepted / n_trials
    expected_rate   = exp(-ΔE / T_σ)
    σ = sqrt(n_trials * expected_rate * (1 - expected_rate)) / n_trials
    @test acceptance_rate ≈ expected_rate atol = 2*σ
end

############ LJ module ############

@testset "LJ_module" begin
    @test argon_deBroglie(2.0) ≈ 0.0530973 atol=1e-6

    # E_12_LJ spot checks against algebraic formula 4*(1/r^12 - 1/r^6)
    for r2 in [0.5, 1.0, 2.0, 4.0]
        @test E_12_LJ(r2) ≈ 4*((1/r2)^6 - (1/r2)^3) atol=1e-12
    end

    # potential_1: compare against Allen & Tildesly mc_lj_module.py reference values.
    # With no fractional particle, this is a clean all-normal-particle LJ sum.
    cfg = joinpath(@__DIR__, "cnf_default.inp")
    N, L_σ, r_σ = load_configuration(cfg)
    r_box = r_σ ./ L_σ
    test_idx = 1 .+ [0, 2, 3, 56, 100, 34, 222, 255, 78, 89]
    E_test = [potential_1(LennardJones(), r_box, r_box[idx], idx, N, L_σ^2, (3/L_σ)^2)
              for idx in test_idx]
    allen_tildesly_results = [-11.932319723716308, -11.932319753958657, -11.932319753958655,
                               -11.932319756099865, -11.9323197764367,   -11.932319756566486,
                               -11.932319756099865, -11.932319738837482, -11.932319771221039,
                               -11.9323197764367]
    @test all(E_test .≈ allen_tildesly_results)
end

println("")
println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Running N_metropolis Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
println("")

############ N_metropolis ############

#=
  Design for insertion/deletion rate tests: choose V and Λ so that
  the V/Λ³ ratio gives a non-trivial acceptance probability (between 0 and 1).
  Use T*=1e6 (ideal gas limit) so ΔU/T → 0 and exp(-βΔU) → 1, making the
  acceptance purely determined by the V/Λ³ and factorial prefactors.

  cube_vertices_home_made.inp: N=8, L=5σ, V=125σ³.

  Insertion N=0→1: acc = min(1, V/Λ³).
    Choose Λ so V/Λ³ = 0.5  →  Λ = (125/0.5)^(1/3) = 250^(1/3) ≈ 6.30σ.
    Insert at box centre (0,0,0): ΔU=0 (no other particles). Expected acc = 0.5.

  Deletion N=2→1: acc = min(1, 2Λ³/V).
    Choose Λ so 2Λ³/V = 0.4  →  Λ³ = 0.2V = 25  →  Λ = 25^(1/3) ≈ 2.924σ.
    Place 2 particles >3σ apart so E_old=0. Expected acc = 0.4.
=#

@testset "N_metropolis: insertion acceptance rate (flat logQ, ideal gas)" begin
    T_σ = 1e6
    L_σ = 5.0;  V_σ = L_σ^3          # 125 σ³ (matches cube_vertices box)
    Λ_σ = (V_σ / 0.5)^(1/3)          # chosen so V/Λ³ = 0.5
    r_cut_σ = 3.0

    sim = SimulationParams(N_max=10, N_min=0, T_σ=T_σ, Λ_σ=Λ_σ,
                           r_cut_σ=r_cut_σ, L_σ=L_σ,
                           save_directory_path=@__DIR__,
                           rng=MersenneTwister(42))
    μ   = init_microstate(sim)           # vacuum N=0
    wl  = init_WangLandauVars(sim)
    c   = init_cache(sim, μ)

    copy_microstate!(c.μ_prop, μ)
    c.μ_prop.N = 1
    c.μ_prop.r_box[1] .= MVector{3,Float64}(0.0, 0.0, 0.0)  # centre of box, no overlap

    # E_proposed = 0 (N=0, no existing particles); expected acc = V/((0+1)*Λ³) = 0.5
    expected_rate = min(1.0, sim.V_σ / (1 * sim.Λ_σ^3))
    n_trials  = 10_000
    n_accepted = sum(N_metropolis(μ, c.μ_prop, 0, wl, sim) for _ in 1:n_trials)
    acceptance_rate = n_accepted / n_trials
    σ = sqrt(n_trials * expected_rate * (1 - expected_rate)) / n_trials
    @test acceptance_rate ≈ expected_rate atol = 3*σ
end

@testset "N_metropolis: deletion acceptance rate (flat logQ, ideal gas)" begin
    T_σ = 1e6
    L_σ = 5.0;  V_σ = L_σ^3
    Λ_σ = (V_σ / (0.4/2))^(1/3)      # 2Λ³/V = 0.4 → Λ³ = 0.2V
    r_cut_σ = 3.0

    sim = SimulationParams(N_max=10, N_min=0, T_σ=T_σ, Λ_σ=Λ_σ,
                           r_cut_σ=r_cut_σ, L_σ=L_σ,
                           save_directory_path=@__DIR__,
                           rng=MersenneTwister(7))

    # Two particles placed 4σ apart (> r_cut=3σ), so they don't interact → E_old=0
    # In box units: 4σ / 5σ = 0.8; place at ±0.4 on x-axis
    μ = init_microstate(sim)
    μ.N = 2
    μ.r_box[1] .= MVector{3,Float64}( 0.4, 0.0, 0.0)
    μ.r_box[2] .= MVector{3,Float64}(-0.4, 0.0, 0.0)

    wl = init_WangLandauVars(sim)
    c  = init_cache(sim, μ)

    # Delete particle 1 (E_old=0 since particles don't interact at 4σ separation)
    idx_deleted = 1
    copy_microstate!(c.μ_prop, μ)
    c.μ_prop.N = μ.N - 1
    c.μ_prop.r_box[idx_deleted] .= μ.r_box[μ.N]  # swap last into deleted slot

    expected_rate = min(1.0, μ.N * sim.Λ_σ^3 / sim.V_σ)  # = 2 * Λ³/V = 0.4
    n_trials   = 10_000
    n_accepted = sum(N_metropolis(μ, c.μ_prop, idx_deleted, wl, sim) for _ in 1:n_trials)
    acceptance_rate = n_accepted / n_trials
    σ = sqrt(n_trials * expected_rate * (1 - expected_rate)) / n_trials
    @test acceptance_rate ≈ expected_rate atol = 3*σ
end

@testset "N_metropolis: logQ bias changes acceptance in correct direction" begin
    T_σ = 1e6
    L_σ = 5.0
    Λ_σ = (L_σ^3 / 0.5)^(1/3)
    sim = SimulationParams(N_max=10, N_min=0, T_σ=T_σ, Λ_σ=Λ_σ,
                           r_cut_σ=3.0, L_σ=L_σ,
                           save_directory_path=@__DIR__,
                           rng=MersenneTwister(99))
    μ  = init_microstate(sim)
    wl = init_WangLandauVars(sim)
    c  = init_cache(sim, μ)

    copy_microstate!(c.μ_prop, μ)
    c.μ_prop.N = 1
    c.μ_prop.r_box[1] .= MVector{3,Float64}(0.0, 0.0, 0.0)

    # Flat logQ baseline
    n_trials = 5000
    n_flat = sum(N_metropolis(μ, c.μ_prop, 0, wl, sim) for _ in 1:n_trials)

    # Bias: penalise proposed state (N=1) relative to current (N=0) → fewer acceptances
    wl.logQ_N[1] = 1.0   # logQ(N=0)
    wl.logQ_N[2] = 4.0   # logQ(N=1): higher → partition_ratio = exp(1-4) < 1 → harder to accept
    n_biased = sum(N_metropolis(μ, c.μ_prop, 0, wl, sim) for _ in 1:n_trials)

    @test n_biased < n_flat
    # Biased rate ≈ flat_rate × exp(1-4) = flat_rate × exp(-3)
    rate_flat   = n_flat   / n_trials
    rate_biased = n_biased / n_trials
    @test rate_biased ≈ rate_flat * exp(-3) atol=0.05
end

println("")
println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Running translation_move! Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
println("")

############ translation_move! ############

@testset "translation_move!: counter increments and δr_max tuning" begin
    input_path = joinpath(@__DIR__, "cube_vertices_home_made.inp")
    T_σ = 1.0
    sim = SimulationParams(N_max=8, N_min=0, T_σ=T_σ, Λ_σ=argon_deBroglie(T_σ),
                           r_cut_σ=3.0,
                           input_filename=input_path,
                           save_directory_path=@__DIR__,
                           rng=MersenneTwister(1234),
                           dynamic_δr_max_box=true)
    μ  = init_microstate(sim=sim, filename=input_path)
    wl = init_WangLandauVars(sim)
    c  = init_cache(sim, μ)

    old_proposed = wl.translation_moves_proposed
    translation_move!(sim, μ, wl, c)
    @test wl.translation_moves_proposed == old_proposed + 1

    # δr_max increases when acceptance > 0.55
    wl.translation_moves_accepted = 10
    wl.translation_moves_proposed = 15
    δr_before = wl.δr_max_box
    translation_move!(sim, μ, wl, c)  # 10/16 or 11/16, both > 0.55
    @test wl.δr_max_box == δr_before * 1.05

    # δr_max decreases when acceptance < 0.45
    wl.δr_max_box = 0.15
    wl.translation_moves_accepted = 1
    wl.translation_moves_proposed = 10
    δr_before = wl.δr_max_box
    translation_move!(sim, μ, wl, c)  # 1/11 or 2/11, both < 0.45
    @test wl.δr_max_box == δr_before * 0.95
end

@testset "translation_move!: N=0 guard (no crash, no particle change)" begin
    T_σ = 1.0
    sim = SimulationParams(N_max=10, N_min=0, T_σ=T_σ, Λ_σ=argon_deBroglie(T_σ),
                           r_cut_σ=3.0, L_σ=8.0,
                           save_directory_path=@__DIR__,
                           rng=MersenneTwister(5))
    μ  = init_microstate(sim)          # N=0
    wl = init_WangLandauVars(sim)
    c  = init_cache(sim, μ)

    # Should return early without crashing
    @test_nowarn translation_move!(sim, μ, wl, c)
    @test μ.N == 0                     # N unchanged
end

println("")
println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Running N_move! Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
println("")

############ N_move! ############

@testset "N_move!: counter increments and N stays in bounds" begin
    input_path = joinpath(@__DIR__, "cube_vertices_home_made.inp")
    T_σ = 100.0   # high T → many moves accepted
    sim = SimulationParams(N_max=8, N_min=0, T_σ=T_σ, Λ_σ=argon_deBroglie(T_σ),
                           r_cut_σ=3.0,
                           input_filename=input_path,
                           save_directory_path=@__DIR__,
                           rng=MersenneTwister(13))
    μ  = init_microstate(sim=sim, filename=input_path)
    wl = init_WangLandauVars(sim)
    c  = init_cache(sim, μ)

    N_values_seen = Set{Int}()
    for _ in 1:2000
        N_move!(sim, μ, wl, c)
        update_wl!(wl, μ)
        push!(N_values_seen, μ.N)
        @test sim.N_min ≤ μ.N ≤ sim.N_max
    end
    # With WL updates at high T, the walker should visit multiple N values
    @test length(N_values_seen) > 1
end

@testset "N_move!: proposed counter always increments" begin
    T_σ = 1.0
    sim = SimulationParams(N_max=8, N_min=0, T_σ=T_σ, Λ_σ=argon_deBroglie(T_σ),
                           r_cut_σ=3.0,
                           input_filename=joinpath(@__DIR__, "cube_vertices_home_made.inp"),
                           save_directory_path=@__DIR__,
                           rng=MersenneTwister(3))
    μ  = init_microstate(sim=sim, filename=joinpath(@__DIR__, "cube_vertices_home_made.inp"))
    wl = init_WangLandauVars(sim)
    c  = init_cache(sim, μ)

    for step in 1:100
        N_move!(sim, μ, wl, c)
        update_wl!(wl, μ)
        @test wl.N_moves_proposed == step
    end
end

println("")
println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Running SimCache Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
println("")

############ SimCache ############

@testset "SimCache: init_cache correctness" begin
    input_path = joinpath(@__DIR__, "cube_vertices_home_made.inp")
    T_σ = 1.0
    sim = SimulationParams(N_max=8, N_min=0, T_σ=T_σ, Λ_σ=argon_deBroglie(T_σ),
                           r_cut_σ=3.0,
                           input_filename=input_path,
                           save_directory_path=@__DIR__,
                           rng=MersenneTwister(42))
    μ = init_microstate(sim=sim, filename=input_path)
    c = init_cache(sim, μ)

    @test c.μ_prop.N == μ.N
    for i in 1:μ.N
        @test c.μ_prop.r_box[i] == μ.r_box[i]
    end
    # Deep copy: no shared MVector references
    for i in 1:μ.N
        @test c.μ_prop.r_box[i] !== μ.r_box[i]
    end
    # Mutating μ does not affect cache
    old_val = μ.r_box[1][1]
    μ.r_box[1][1] += 99.0
    @test c.μ_prop.r_box[1][1] ≈ old_val
    μ.r_box[1][1] = old_val

    @test all(c.ζ_Mvec   .== 0.0)
    @test c.ζ_idx == 0
    @test all(c.ri_proposed_box .== 0.0)
    @test length(c.μ_prop.r_box) == sim.N_max
end

@testset "SimCache: mirror invariant under translation moves" begin
    input_path = joinpath(@__DIR__, "cube_vertices_home_made.inp")
    T_σ = 1.5
    sim = SimulationParams(N_max=8, N_min=0, T_σ=T_σ, Λ_σ=argon_deBroglie(T_σ),
                           r_cut_σ=3.0,
                           input_filename=input_path,
                           save_directory_path=@__DIR__,
                           rng=MersenneTwister(7))
    μ  = init_microstate(sim=sim, filename=input_path)
    wl = init_WangLandauVars(sim)
    c  = init_cache(sim, μ)

    function check_mirror(μ, c)
        c.μ_prop.N == μ.N || return false
        for i in 1:μ.N
            c.μ_prop.r_box[i] == μ.r_box[i] || return false
        end
        return true
    end

    for _ in 1:500
        translation_move!(sim, μ, wl, c)
        @test check_mirror(μ, c)
        @test 1 ≤ c.ζ_idx ≤ μ.N
    end
end

@testset "SimCache: mirror invariant under N_moves" begin
    input_path = joinpath(@__DIR__, "cube_vertices_home_made.inp")
    T_σ = 5.0
    sim = SimulationParams(N_max=8, N_min=0, T_σ=T_σ, Λ_σ=argon_deBroglie(T_σ),
                           r_cut_σ=3.0,
                           input_filename=input_path,
                           save_directory_path=@__DIR__,
                           rng=MersenneTwister(13))
    μ  = init_microstate(sim=sim, filename=input_path)
    wl = init_WangLandauVars(sim)
    c  = init_cache(sim, μ)

    function check_mirror(μ, c)
        c.μ_prop.N == μ.N || return false
        for i in 1:μ.N
            c.μ_prop.r_box[i] == μ.r_box[i] || return false
        end
        return true
    end

    N_values_seen = Set{Int}()
    for _ in 1:2000
        N_move!(sim, μ, wl, c)
        update_wl!(wl, μ)
        push!(N_values_seen, μ.N)
        @test check_mirror(μ, c)
        @test sim.N_min ≤ μ.N ≤ sim.N_max
    end
    @test length(N_values_seen) > 1
end

@testset "SimCache: mirror invariant through mixed move sequences" begin
    input_path = joinpath(@__DIR__, "cube_vertices_home_made.inp")
    T_σ = 3.0
    sim = SimulationParams(N_max=8, N_min=0, T_σ=T_σ, Λ_σ=argon_deBroglie(T_σ),
                           r_cut_σ=3.0,
                           input_filename=input_path,
                           save_directory_path=@__DIR__,
                           rng=MersenneTwister(99))
    μ  = init_microstate(sim=sim, filename=input_path)
    wl = init_WangLandauVars(sim)
    c  = init_cache(sim, μ)

    function check_mirror(μ, c)
        c.μ_prop.N == μ.N || return false
        for i in 1:μ.N
            c.μ_prop.r_box[i] == μ.r_box[i] || return false
        end
        return true
    end

    rng_loop = MersenneTwister(55)
    for _ in 1:1000
        if rand(rng_loop) < 0.75
            translation_move!(sim, μ, wl, c)
        else
            N_move!(sim, μ, wl, c)
        end
        update_wl!(wl, μ)
        @test check_mirror(μ, c)
    end

    # Deep copy independence still holds after 1000 mixed moves
    old_val = μ.r_box[1][1]
    μ.r_box[1][1] += 99.0
    @test c.μ_prop.r_box[1][1] ≈ old_val
    μ.r_box[1][1] = old_val
end

println("")
println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Running Initialization Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
println("")

############ Initialization ############

@testset "Initialization: SimulationParams with direct L_σ" begin
    T_σ = 1.0; L_σ = 8.0
    sim = SimulationParams(N_max=10, N_min=0, T_σ=T_σ, Λ_σ=argon_deBroglie(T_σ),
                           r_cut_σ=3.0, L_σ=L_σ,
                           save_directory_path=@__DIR__,
                           rng=MersenneTwister(1))
    @test sim.L_σ         ≈ L_σ
    @test sim.V_σ         ≈ L_σ^3
    @test sim.L_squared_σ ≈ L_σ^2
    @test sim.r_cut_box   ≈ 3.0 / L_σ
    @test sim.r_cut_squared_box ≈ (3.0 / L_σ)^2

    @test_throws ArgumentError SimulationParams(N_max=10, N_min=0, T_σ=T_σ,
                                                Λ_σ=argon_deBroglie(T_σ),
                                                r_cut_σ=3.0,
                                                save_directory_path=@__DIR__)
end

@testset "Initialization: init_microstate vacuum (N=0) state" begin
    T_σ = 1.0
    sim = SimulationParams(N_max=50, N_min=0, T_σ=T_σ, Λ_σ=argon_deBroglie(T_σ),
                           r_cut_σ=3.0, L_σ=8.0,
                           save_directory_path=@__DIR__,
                           rng=MersenneTwister(1))
    μ = init_microstate(sim)
    @test μ.N == 0
    @test length(μ.r_box) == sim.N_max
end

@testset "Initialization: init_microstate from file pre-allocates N_max slots" begin
    input_path = joinpath(@__DIR__, "cube_vertices_home_made.inp")
    T_σ = 1.0
    sim = SimulationParams(N_max=50, N_min=0, T_σ=T_σ, Λ_σ=argon_deBroglie(T_σ),
                           r_cut_σ=3.0,
                           input_filename=input_path,
                           save_directory_path=@__DIR__,
                           rng=MersenneTwister(1))
    μ = init_microstate(sim=sim, filename=input_path)
    @test μ.N == 8                          # cube_vertices_home_made.inp has 8 atoms
    @test length(μ.r_box) == sim.N_max      # pre-allocated to N_max, not N_file
end

@testset "Initialization: init_cache with vacuum microstate" begin
    T_σ = 1.0
    sim = SimulationParams(N_max=20, N_min=0, T_σ=T_σ, Λ_σ=argon_deBroglie(T_σ),
                           r_cut_σ=3.0, L_σ=8.0,
                           save_directory_path=@__DIR__,
                           rng=MersenneTwister(1))
    μ = init_microstate(sim)
    c = init_cache(sim, μ)
    @test c.μ_prop.N == 0
    @test length(c.μ_prop.r_box) == sim.N_max
end


println("")
println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Now Running Detailed Balance Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
println("")

#=
Detailed balance requires that the raw acceptance argument product for a forward/reverse
move pair equals 1 (independent of any Q bias, which cancels exactly):

    [V_Λ_fwd × factorial_fwd × exp(-βΔU_fwd)] × [V_Λ_rev × factorial_rev × exp(-βΔU_rev)] = 1

This reduces to two independent checks:
  (a) V_Λ_fwd × V_Λ_rev × factorial_fwd × factorial_rev = 1   (prefactor symmetry)
  (b) ΔU_fwd + ΔU_rev = 0                                      (energy antisymmetry)
=#

@testset "Detailed balance: translation move ΔU antisymmetry" begin
    input_path = joinpath(@__DIR__, "cube_vertices_home_made.inp")
    T_σ = 1.0
    sim = SimulationParams(N_max=8, N_min=0, T_σ=T_σ, Λ_σ=argon_deBroglie(T_σ),
                           r_cut_σ=3.0,
                           input_filename=input_path,
                           save_directory_path=@__DIR__,
                           rng=MersenneTwister(1))
    μ = init_microstate(sim=sim, filename=input_path)

    i     = 2
    r_old = MVector{3,Float64}(μ.r_box[i])
    r_new = MVector{3,Float64}(r_old .+ 0.05)

    E_at_r_old = potential_1(sim.potential, μ.r_box, r_old, i, μ.N, sim.L_squared_σ, sim.r_cut_squared_box)
    E_at_r_new = potential_1(sim.potential, μ.r_box, r_new, i, μ.N, sim.L_squared_σ, sim.r_cut_squared_box)

    ΔU_fwd = E_at_r_new - E_at_r_old
    ΔU_rev = E_at_r_old - E_at_r_new
    @test ΔU_fwd + ΔU_rev ≈ 0.0 atol=1e-12
end

@testset "Detailed balance: insertion/deletion V_Λ prefactor symmetry" begin
    #= For direct GCWL insertion/deletion the acceptance arguments must be reciprocals:
         insertion  (N → N+1): V/Λ³  × 1/(N+1)
         deletion   (N+1 → N): Λ³/V  × (N+1)
       Product = 1. A wrong exponent in V_Λ_prefactor would break this. =#
    T_σ = 1.0; L_σ = 8.0
    sim = SimulationParams(N_max=10, N_min=0, T_σ=T_σ, Λ_σ=argon_deBroglie(T_σ),
                           r_cut_σ=3.0, L_σ=L_σ,
                           save_directory_path=@__DIR__,
                           rng=MersenneTwister(1))
    for N in [0, 1, 4, 9]
        V_Λ_ins = sim.V_σ^(1) * sim.Λ_σ^(-3)          # insertion N→N+1: V/Λ³
        f_ins   = 1.0 / (N + 1)
        V_Λ_del = sim.V_σ^(-1) * sim.Λ_σ^(3)          # deletion N+1→N: Λ³/V
        f_del   = Float64(N + 1)

        @test V_Λ_ins ≈ sim.V_σ / sim.Λ_σ^3           atol=1e-10
        @test V_Λ_del ≈ sim.Λ_σ^3 / sim.V_σ           atol=1e-10
        @test V_Λ_ins * f_ins * V_Λ_del * f_del ≈ 1.0 atol=1e-10
    end
end

@testset "Detailed balance: insertion/deletion ΔU antisymmetry" begin
    #= Insert a particle at r_new into an N-particle system.
       Then delete that same particle. The ΔU values must be exact negatives.
       This is the GCWL equivalent of the r_frac detailed balance test in SEGC-WL:
       ΔU_ins = potential_1(new particle at r_new with N existing)
       ΔU_del = 0 - potential_1(same particle at r_new with N existing) = -ΔU_ins
       Sum = 0.  If the energy function is inconsistent this will not hold. =#
    input_path = joinpath(@__DIR__, "cube_vertices_home_made.inp")
    T_σ = 1.0
    sim = SimulationParams(N_max=9, N_min=0, T_σ=T_σ, Λ_σ=argon_deBroglie(T_σ),
                           r_cut_σ=3.0,
                           input_filename=input_path,
                           save_directory_path=@__DIR__,
                           rng=MersenneTwister(1))
    μ = init_microstate(sim=sim, filename=input_path)  # N=8

    r_new = MVector{3,Float64}(0.05, -0.03, 0.02)     # interior position, no overlap

    # Insertion: new particle is slot N+1 = 9 in a 9-particle proposed state
    r_box_Np1 = [MVector{3,Float64}(μ.r_box[j]) for j in 1:sim.N_max]
    r_box_Np1[μ.N+1] .= r_new
    ΔU_ins = potential_1(sim.potential, r_box_Np1, r_new, μ.N+1, μ.N+1, sim.L_squared_σ, sim.r_cut_squared_box)

    # Deletion: delete the particle just inserted (particle μ.N+1 at r_new from N+1 system)
    # E_old = energy of that particle with remaining N particles
    E_del = potential_1(sim.potential, r_box_Np1, r_new, μ.N+1, μ.N+1, sim.L_squared_σ, sim.r_cut_squared_box)
    ΔU_del = 0.0 - E_del

    @test ΔU_ins + ΔU_del ≈ 0.0 atol=1e-12
    @test exp(-(ΔU_ins + ΔU_del) / T_σ) ≈ 1.0 atol=1e-10
end


println("")
println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Now Running Physics Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
println("")

############ Physics: comparison to analytic values ############

@testset "Comparison to Analytic Values of Q(N=1), Q(N=2) for Lennard Jones system" begin
    #= Run 5 independent WL simulations on a small 4-atom box and average.
       Reference values computed analytically in test/mathematica_verification.nb. =#
    logQ_N1 = 0.0; logQ_N2 = 0.0
    for _ in 1:5
        input_path = joinpath(@__DIR__, "4_atom_cnf.inp")
        T_σ = 1.0
        sim = SimulationParams(N_max=4, N_min=0, T_σ=T_σ, Λ_σ=argon_deBroglie(T_σ),
                               r_cut_σ=3.0,
                               input_filename=input_path,
                               save_directory_path=@__DIR__,
                               maxiter=100_000_000)
        μ     = init_microstate(sim=sim, filename=input_path)
        wl    = init_WangLandauVars(sim)
        cache = init_cache(sim, μ)
        run_simulation!(sim, μ, wl, cache)
        logQ   = correct_logQ(wl)
        logQ_N1 += logQ[2]
        logQ_N2 += logQ[3]
    end
    logQ_N1 /= 5; logQ_N2 /= 5

    mathematica_logQ_N1 = 14.0055
    mathematica_logQ_N2 = 27.3179
    @test logQ_N1 ≈ mathematica_logQ_N1 atol = 0.05*mathematica_logQ_N1
    @test logQ_N2 ≈ mathematica_logQ_N2 atol = 0.05*mathematica_logQ_N2
end

@testset "Comparison to Analytic Ideal Gas Limit (T*=1e6, N_max=4)" begin
    logQ_avg = zeros(5)
    for _ in 1:5
        input_path = joinpath(@__DIR__, "4_atom_cnf.inp")
        T_σ = 1_000_000.0
        sim = SimulationParams(N_max=4, N_min=0, T_σ=T_σ, Λ_σ=argon_deBroglie(T_σ),
                               r_cut_σ=3.0,
                               input_filename=input_path,
                               save_directory_path=@__DIR__,
                               maxiter=100_000_000)
        μ     = init_microstate(sim=sim, filename=input_path)
        wl    = init_WangLandauVars(sim)
        cache = init_cache(sim, μ)
        run_simulation!(sim, μ, wl, cache)
        logQ = correct_logQ(wl)
        logQ_avg .+= logQ
    end
    logQ_avg ./= 5

    mathematica_logQ_N2 = 68.7644
    mathematica_logQ_N3 = 102.395
    mathematica_logQ_N4 = 135.737
    @test logQ_avg[3] ≈ mathematica_logQ_N2 atol = 0.05*mathematica_logQ_N2
    @test logQ_avg[4] ≈ mathematica_logQ_N3 atol = 0.05*mathematica_logQ_N3
    @test logQ_avg[5] ≈ mathematica_logQ_N4 atol = 0.05*mathematica_logQ_N4
end

@testset "High-N Ideal Gas Limit: logQ(N=10 to 108) vs analytic" begin
    #= At T*=1e6 the system is ideal gas. WL should recover logQ(N) = N log(V/Λ³) - log(N!)
       to within 5% for all N in [10, 108]. This test is sensitive to wrong V/Λ prefactors
       and cache aliasing bugs, both of which produce N-proportional drift in logQ. =#
    input_path = joinpath(@__DIR__, "../initial_configs/N108_L8.inp")
    T_σ = 1_000_000.0
    sim = SimulationParams(N_max=108, N_min=0, T_σ=T_σ, Λ_σ=argon_deBroglie(T_σ),
                           r_cut_σ=3.0,
                           input_filename=input_path,
                           save_directory_path=@__DIR__,
                           maxiter=50_000_000)
    μ     = init_microstate(sim=sim, filename=input_path)
    wl    = init_WangLandauVars(sim)
    cache = init_cache(sim, μ)
    run_simulation!(sim, μ, wl, cache)
    logQ_wl = correct_logQ(wl)

    Ns            = collect(10:sim.N_max)
    logQ_analytic = [ideal_gas_logQ_loggamma(N, sim.V_σ, sim.Λ_σ) for N in Ns]
    logQ_sim      = [logQ_wl[N+1] for N in Ns]
    rel_err       = (logQ_sim .- logQ_analytic) ./ abs.(logQ_analytic)

    println("High-N ideal gas test: logQ_wl vs analytic for N=10:108")
    for (i, N) in enumerate(Ns)
        @test logQ_sim[i] ≈ logQ_analytic[i] atol = 0.05 * abs(logQ_analytic[i])
    end

    p1 = plot(Ns, logQ_analytic, label="analytic", lw=2, color=:blue,
              xlabel="N", ylabel="log Q(N,V,T)", title="High-N Ideal Gas: WL vs Analytic")
    plot!(p1, Ns, logQ_sim, label="WL", lw=2, color=:red, linestyle=:dash)
    p2 = plot(Ns, 100 .* rel_err, label="relative error (%)", lw=2, color=:green,
              xlabel="N", ylabel="relative error (%)", title="Relative Error")
    hline!(p2, [5.0, -5.0], label="±5%", color=:red, linestyle=:dot)
    savefig(plot(p1, p2, layout=(2,1), size=(800,800)),
            joinpath(@__DIR__, "high_N_ideal_gas_logQ_comparison.png"))
end

# Julia scoping requires these be globals and thus must be defined outside of @testset which has a local scope so gcwl.jl can find them
struct HardSphere <: PairPotential end 
# because cutoff check happens in potential_1, we can just set SimulationParams.r_cut_σ = 1 and return +∞ energy if pair_energ(HardSphere, r2_σ) ever gets called
function gc_wl.pair_energy(p::HardSphere, r2_σ::Float64)::Float64 # have to have gc_wl prefix so gc_wl.jl and associated functions can find this
    # ONLY WORKS IF YOU SET SimulationParams.r_cut_σ = 1 
    return(typemax(Float64))
end

@testset "Hard sphere partition functions" begin
    logQ_avg = zeros(13)
    for _ in 1:5
        T_σ = 1.0
        sim = SimulationParams(N_max=12, N_min=0, T_σ=T_σ, Λ_σ=1.0, # Λ = 1 just for ease
                               r_cut_σ=1.0, # CRITICAL FOR HARD SPHERE
                               L_σ = 20.0, # dilute
                               save_directory_path=@__DIR__,
                               maxiter=100_000_000, potential = HardSphere())
        μ     = init_microstate(sim)
        wl    = init_WangLandauVars(sim)
        cache = init_cache(sim, μ)
        run_simulation!(sim, μ, wl, cache)
        logQ   = correct_logQ(wl)
        logQ_avg = logQ_avg .+ logQ 
    end
    logQ_avg = logQ_avg ./ 5

    println("Hard sphere partition functions: ",logQ_avg )

    mathematica_hardsphere_logQs = [0.0,8.9872,16.5881,23.378,29.5925,35.3607,40.7642,45.8594,
                                    50.6875,55.2799,59.6617,63.8527,67.8698] # starts from N=0 and goes to N=12, computed from known virial coefficients in mathematica_verification.nb
    
    for ii in range(1,length(logQ_avg))
        @test logQ_avg[ii] ≈ mathematica_hardsphere_logQs[ii] atol = 0.05*mathematica_hardsphere_logQs[ii]
    end
end