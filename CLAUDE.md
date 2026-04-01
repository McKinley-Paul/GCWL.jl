# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Run tests:**
```bash
julia test/runtests.jl
# or from the Julia REPL:
# using Pkg; Pkg.test()
```

**Run a simulation (from an example directory):**
```julia
using segc_wl
sim = SimulationParams(N_max=..., N_min=..., T_σ=..., Λ_σ=..., λ_max=..., r_cut_σ=...,
                       input_filename=..., save_directory_path=..., maxiter=...)
μstate = init_microstate(sim=sim, filename=input_path)
wl     = init_WangLandauVars(sim)
cache  = init_cache(sim, μstate)
initialization_check(sim, μstate, wl)
run_simulation!(sim, μstate, wl, cache)
post_run(sim, μstate, wl)
logQ   = correct_logQ(wl)
```

**Resume from checkpoint:**
```julia
wl     = load_wanglandau_jld2(path)
μstate = load_microstate_jld2(path)
```

**Generate initial configs:** see `initial_configs/how_to_gen_initial_configs.txt` and `initialize.py` (Allen & Tildesly FCC lattice).

---

## Architecture

The package computes expanded canonical partition functions Q(N,V,T,λ) for Lennard-Jones fluids using the **Single-Particle Extended Grand Canonical (SEGC) Wang-Landau** algorithm (Desgranges & Delhommelle 2012/2016).

### Source files (`src/`)

| File | Role |
|------|------|
| `segc_wl.jl` | Module entry, main loop `run_simulation!`, move proposals (`translation_move!`, `λ_move!`), `update_wl!`, `post_run` |
| `initialization.jl` | All structs (`microstate`, `SimulationParams`, `WangLandauVars`, `SimCache`), `init_*` functions, checkpointing (JLD2), PBC wrapping |
| `lj.jl` | LJ energy functions (`E_12_LJ`, `E_12_frac_LJ`, `potential_1_normal`, `potential_1_frac`), λ-move acceptance criterion `λ_metropolis_pm1` |
| `utils.jl` | PBC distance, random translation, standard Metropolis criterion |
| `thermo.jl` | `correct_logQ` (normalization via Q(0,λ=0)=1), ideal gas partition function (Stirling and exact via loggamma) |

### Key data structures

**`SimulationParams`** (immutable) — set once at startup. Input fields: `N_max/N_min`, `T_σ`, `Λ_σ`, `λ_max`, `r_cut_σ`, `maxiter`, `dynamic_δr_max_box`. Derived fields computed in inner constructor: `L_σ`, `V_σ`, `r_cut_box`, etc.

**`microstate`** (mutable) — current configuration: `N` (particle count), `λ` (coupling integer 0…λ_max), `r_box` (positions in box=1 units), `r_frac_box`, precomputed couplings `ϵ_ξ` and `σ_ξ_squared`.

**`WangLandauVars`** (mutable) — WL state: `logf`, `H_λN` histogram, `logQ_λN` density-of-states matrix, move counters, `δr_max_box`.

**`SimCache`** (mutable) — pre-allocated scratch buffers (`ζ_Mvec`, `ri_proposed_box`, `μ_prop`) to avoid heap allocations in the inner loop.

### Units

- **Box units**: box side = 1; PBC at ±0.5. Convert: `r_LJ = r_box × L_σ`.
- **LJ reduced units**: σ=1, ϵ=1; temperature `T_σ = k_B T / ϵ`.

### Algorithm summary

Two MC move types each step (75/25 split):
- **Translation**: Metropolis accept/reject on ΔE for one particle.
- **λ-move**: Change coupling by ±1 (fractional particle appears/disappears); acceptance uses the SEGC criterion in `λ_metropolis_pm1` (includes factorial and V/Λ³ prefactors).

Wang-Landau updates `logQ_λN` and `H_λN` after every accepted move. Each *epoch* ends when all (λ,N) states are visited ≥ threshold; then `logf → logf/2`. Simulation converges when `logf < 1e-8`.

### Input config format (`.inp`)

```
N
L
x1 y1 z1
x2 y2 z2
...
```
