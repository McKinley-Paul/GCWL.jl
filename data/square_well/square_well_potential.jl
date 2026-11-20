# ═══════════════════════════════════════════════════════════════════════════
#  Square-well pair potential for the GCWL package.
#
#      U(r) =  +∞        r < σ          (hard core)
#           =  -ε        σ ≤ r < λσ     (well)
#           =   0        r ≥ λσ
#
#  Reduced units throughout: σ = ε = 1, so the well depth is exactly -1.
#
#  This file is `include`d by every square-well main.jl so that the struct has
#  an identical definition everywhere (JLD2 needs the type to exist with the
#  same name when a checkpointed SimulationParams is loaded back).
#
#  NOTE ON TRUNCATION: the potential is *exactly* zero beyond λσ, so setting
#  r_cut_σ = λ is the full range, not a truncation. No long-range correction is
#  applicable or needed — do NOT call apply_lrc_to_logQ on square-well data.
# ═══════════════════════════════════════════════════════════════════════════

using GCWL

struct SquareWell <: PairPotential
    λ::Float64      # well width in units of σ
end

# Pair potentials must accept SQUARED distances in σ = 1 units — see the note at
# the top of src/initialization.jl and the r_cut test inside potential_1().
function GCWL.pair_energy(p::SquareWell, r2_σ::Float64)::Float64
    if r2_σ < 1.0
        return Inf                      # hard-core overlap
    elseif r2_σ < p.λ * p.λ
        return -1.0                     # inside the well, depth -ε = -1
    else
        return 0.0                      # unreachable when r_cut_σ = λ
    end
end

# ── Why Inf is safe in every acceptance path ───────────────────────────────
#   translation_move!  : E_old is the energy of an existing particle in a valid
#                        (overlap-free) configuration, hence always finite. Only
#                        E_proposed can be Inf ⇒ ΔE = Inf ⇒ metropolis() takes
#                        exponent = -Inf, exp(-Inf) = 0.0 > ζ is false ⇒ reject.
#   N_metropolis (ins) : E_old = 0, E_proposed may be Inf ⇒ exp(-Inf) = 0.0 and
#                        prob_ratio = finite * 0.0 = 0.0 ⇒ reject.
#   N_metropolis (del) : E_old is finite (valid state), E_proposed = 0.
#   Because no overlapping configuration is ever accepted, Inf - Inf (→ NaN)
#   cannot arise.
