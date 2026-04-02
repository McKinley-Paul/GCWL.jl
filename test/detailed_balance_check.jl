"""
Concrete demonstration that the buggy deletion scheme violates detailed balance
while the fixed scheme satisfies it.

Detailed balance requires: for any pair of states A and B,
    g(A) × T(A→B) = g(B) × T(B→A)

With a flat WL DoS (g = 1 everywhere), the acceptance criterion alone must satisfy
that the product of the forward and reverse "raw" Boltzmann factors equals 1:

    [N × exp(-β ΔU_delete)] × [1/N × exp(-β ΔU_insert)] = 1
    ↔  ΔU_delete + ΔU_insert = 0

FIXED:  r_frac ← r_i (deleted particle's position)
        ΔU_delete = U_frac(r_i, λ_max) - U_particle(r_i)   with N-1 others
        ΔU_insert = U_particle(r_i)    - U_frac(r_i, λ_max) = -ΔU_delete  ✓

BUGGY:  r_frac left at ghost position (stale, unrelated to r_i)
        ΔU_delete = U_frac(ghost, λ_max) - U_particle(r_i)
        ΔU_insert = U_particle(ghost)    - U_frac(ghost, λ_max)
        These energies involve different positions → sum ≠ 0 in general  ✗
"""

import Pkg
Pkg.activate("/Users/mckinleypaul/Documents/montecarlo/segc_wl")
using segc_wl
using StaticArrays
using Printf
import Random

input_path = joinpath(@__DIR__, "cube_vertices_home_made.inp")
T_σ   = 1.0
Λ_σ   = argon_deBroglie(T_σ)
λ_max = 9   # M = 10

sim = SimulationParams(N_max=8, N_min=0, T_σ=T_σ, Λ_σ=Λ_σ,
                       λ_max=λ_max, r_cut_σ=3.0,
                       input_filename=input_path,
                       save_directory_path=@__DIR__,
                       rng=Random.MersenneTwister(1),
                       maxiter=1)

# State A: N=8 particles at cube vertices, λ=0
# Ghost position chosen to be near box centre — well away from all 8 cube vertices
# (vertices are at ±0.2 box units; centre is at 0,0,0 so nearest vertex is ~0.35 box = 1.73σ)
μ = init_microstate(sim=sim, filename=input_path, λ=0)
r_ghost = MVector{3,Float64}(0.05, -0.03, 0.02)   # near box centre
μ.r_frac_box .= r_ghost

idx_del   = 3                             # delete particle 3
r_deleted = MVector{3,Float64}(μ.r_box[idx_del])  # save its position

println("="^60)
println("State A: N=$(μ.N) particles, λ=0")
println("  Deleted particle (idx=$idx_del) at r_box = $r_deleted")
println("  Ghost r_frac_box at r_box = $r_ghost")
println("="^60)

# ─── Build the N-1 particle box ────────────────────────────────────────────
# Deep-copy so we don't modify the original
r_box_Nm1 = [MVector{3,Float64}(μ.r_box[j]) for j in 1:sim.N_max]
if idx_del != μ.N
    r_box_Nm1[idx_del] .= μ.r_box[μ.N]   # swap last into deleted slot
end
N_m1 = μ.N - 1

# Coupling at λ_max
ϵ_ξ_max  = (λ_max / (λ_max + 1))^(1/3)
σ_ξ²_max = (λ_max / (λ_max + 1))^(1/2)

# ─── Energy helpers ────────────────────────────────────────────────────────
# U_particle(r_i): energy of particle idx_del at r_i with the other N-1 normal particles.
# We compute this from the ORIGINAL r_box (N particles) using idx_del so the j==i check
# excludes the particle itself. We set ϵ_ξ=0 so the fractional term contributes nothing.
function U_particle_ri()
    potential_1_normal(μ.r_box, r_deleted, idx_del,
                       r_ghost,            # r_frac_box (ignored: ϵ_ξ=0 below)
                       0, λ_max, μ.N,
                       sim.L_squared_σ, sim.r_cut_squared_box,
                       0.0, 1.0)           # ϵ_ξ=0 → frac contributes 0
end

# U_frac_at(pos): energy of the frac particle at pos (coupling λ_max) with the N-1 real particles.
function U_frac_at(pos::MVector{3,Float64})
    potential_1_frac(r_box_Nm1, pos, λ_max, λ_max, N_m1,
                     sim.L_squared_σ, sim.r_cut_squared_box,
                     ϵ_ξ_max, σ_ξ²_max)
end

# Dummy r_frac position: placed at a corner of the box, far from everything.
# Used only so potential_1_normal has a valid r_frac_box != pos (ϵ_ξ=0 means
# the fractional particle contributes zero energy, but r_frac ≠ pos is required
# to avoid the zero-distance early-return).
r_frac_dummy = MVector{3,Float64}(-0.48, -0.48, -0.48)

# U_particle_at(pos): energy of a FULL particle at pos with the N-1 real particles.
# We place it at a "virtual" index N_m1+1 so no j==i exclusion fires for real particles.
# (r_box_Nm1 slots N_m1+1..end are unused; we never access them in the j=1..N_m1 loop)
function U_particle_at(pos::MVector{3,Float64})
    potential_1_normal(r_box_Nm1, pos, N_m1+1,
                       r_frac_dummy,       # r_frac_box != pos; ϵ_ξ=0 → contributes 0
                       0, λ_max, N_m1,     # N = N_m1: loop runs j=1..N_m1
                       sim.L_squared_σ, sim.r_cut_squared_box,
                       0.0, 1.0)           # ϵ_ξ=0
end

# ─── Compute energies ──────────────────────────────────────────────────────
U_part_ri    = U_particle_ri()
U_frac_ri    = U_frac_at(r_deleted)
U_frac_ghost = U_frac_at(r_ghost)
U_part_ghost = U_particle_at(r_ghost)

println("\nEnergies (LJ units) — all interactions with the N-1 remaining particles:")
@printf "  U_particle(r_i)          = %12.6f\n"  U_part_ri
@printf "  U_frac(r_i,    λ_max)    = %12.6f\n"  U_frac_ri
@printf "  U_frac(r_ghost, λ_max)   = %12.6f\n"  U_frac_ghost
@printf "  U_particle(r_ghost)      = %12.6f\n"  U_part_ghost

# ─── FIXED scheme ─────────────────────────────────────────────────────────
# Delete: particle at r_i removed, frac appears at r_i
# Insert: frac at r_i promoted to full particle at r_i (exact reverse)
ΔU_del_fixed = U_frac_ri   - U_part_ri
ΔU_ins_fixed = U_part_ri   - U_frac_ri   # = -ΔU_del_fixed exactly

raw_fwd_fixed = μ.N       * exp(-ΔU_del_fixed / T_σ)
raw_rev_fixed = (1/μ.N)   * exp(-ΔU_ins_fixed / T_σ)
product_fixed = raw_fwd_fixed * raw_rev_fixed

println("\n─── FIXED scheme (r_frac ← r_deleted position) ───")
@printf "  ΔU_delete  = %12.6f\n"  ΔU_del_fixed
@printf "  ΔU_insert  = %12.6f   (= -ΔU_delete: exact reverse)\n"  ΔU_ins_fixed
@printf "  fwd factor = N   × exp(-β ΔU_del) = %12.6f\n"  raw_fwd_fixed
@printf "  rev factor = 1/N × exp(-β ΔU_ins) = %12.6f\n"  raw_rev_fixed
@printf "  product (must = 1): %.10f  ← error = %.2e\n"  product_fixed  abs(product_fixed-1)

# ─── BUGGY scheme ─────────────────────────────────────────────────────────
# Delete: particle at r_i removed, frac LEFT at ghost (wrong position)
# "Insert": frac at ghost promoted to full particle at ghost  ← different state!
ΔU_del_buggy = U_frac_ghost - U_part_ri    # forward: particle gone, frac at ghost
ΔU_ins_buggy = U_part_ghost - U_frac_ghost # "reverse": frac at ghost → particle at ghost

raw_fwd_buggy = μ.N     * exp(-ΔU_del_buggy / T_σ)
raw_rev_buggy = (1/μ.N) * exp(-ΔU_ins_buggy / T_σ)
product_buggy = raw_fwd_buggy * raw_rev_buggy

println("\n─── BUGGY scheme (r_frac left at ghost) ───")
@printf "  ΔU_delete  = U_frac(ghost) - U_particle(r_i) = %12.6f\n"  ΔU_del_buggy
@printf "  ΔU_insert  = U_particle(ghost) - U_frac(ghost) = %12.6f\n"  ΔU_ins_buggy
@printf "  fwd factor = N   × exp(-β ΔU_del) = %12.6f\n"  raw_fwd_buggy
@printf "  rev factor = 1/N × exp(-β ΔU_ins) = %12.6f\n"  raw_rev_buggy
@printf "  product (must = 1): %.10f  ← error = %.2e\n"  product_buggy  abs(product_buggy-1)

println("\n─── Summary ───")
@printf "  FIXED: product - 1 = %+.2e  ✓\n"  product_fixed  - 1.0
@printf "  BUGGY: product - 1 = %+.2e  ✗\n"  product_buggy  - 1.0

println("""
\nNote: even if the acceptance probabilities happened to be similar for one
specific pair of states, the buggy scheme also connects DIFFERENT state pairs:
  delete(A): (r_1..r_i..r_N, λ=0) → (r_1..r_{N-1}, λ_max, r_frac=ghost)
  insert(B): (r_1..r_{N-1}, λ_max, r_frac=ghost) → (r_1..r_{N-1}+ghost, λ=0, r_frac=NEW_RANDOM)
The inserted particle is placed at the ghost position, not at r_i.
A→B and B→A are not reverses of each other, so no g can satisfy detailed balance.""")
