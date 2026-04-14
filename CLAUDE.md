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
using gc_wl
sim = SimulationParams(N_max=..., N_min=..., T_σ=..., Λ_σ=..., r_cut_σ=...,
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

The package computes canonical partition functions Q(N,V,T) for Lennard-Jones fluids using the **Grand Canonical (GC) Wang-Landau** algorithm. Particles are inserted and deleted directly (integer ΔN = ±1); there is no fractional particle or λ-staging.

### Source files (`src/`)

| File | Role |
|------|------|
| `gc_wl.jl` | Module entry, main loop `run_simulation!`, move proposals (`translation_move!`, `N_move!`), `update_wl!`, `post_run`, `N_metropolis` |
| `initialization.jl` | All structs (`microstate`, `SimulationParams`, `WangLandauVars`, `SimCache`), `init_*` functions, checkpointing (JLD2), PBC wrapping |
| `lj.jl` | LJ energy functions (`E_12_LJ`, `potential_1`) |
| `utils.jl` | PBC distance, random translation, standard Metropolis criterion |
| `thermo.jl` | `correct_logQ` (normalization via Q(N=0)=1), ideal gas partition function (Stirling and exact via loggamma) |

### Key data structures

**`SimulationParams`** (immutable) — set once at startup. Input fields: `N_max/N_min`, `T_σ`, `Λ_σ`, `r_cut_σ`, `maxiter`, `dynamic_δr_max_box`. Derived fields computed in inner constructor: `L_σ`, `V_σ`, `r_cut_box`, etc.

**`microstate`** (mutable) — current configuration: `N` (particle count), `r_box` (positions in box=1 units, pre-allocated to `N_max` slots).

**`WangLandauVars`** (mutable) — WL state: `logf`, `H_N` histogram (length `N_max+1`), `logQ_N` density-of-states vector (length `N_max+1`), move counters, `δr_max_box`, `phase2::Bool`, `flat::Bool`.

**`SimCache`** (mutable) — pre-allocated scratch buffers (`ζ_Mvec`, `ri_proposed_box`, `μ_prop`) to avoid heap allocations in the inner loop.

### Units

- **Box units**: box side = 1; PBC at ±0.5. Convert: `r_LJ = r_box × L_σ`.
- **LJ reduced units**: σ=1, ϵ=1; temperature `T_σ = k_B T / ϵ`.

### Algorithm summary

Two MC move types each step (75/25 split):
- **Translation**: Metropolis accept/reject on ΔE for one particle.
- **N-move**: Insert or delete one full particle (ΔN = ±1); acceptance uses `N_metropolis` (Ganzmuller & Camp 2007, Eq. 19).

Wang-Landau updates `logQ_N` and `H_N` after every MC step (accepted or rejected).

#### Metropolis acceptance criterion for N-moves

Reference: Ganzmuller & Camp, *J. Chem. Phys.* **127**, 154504 (2007), Eq. 19. All cases written in terms of Q(N_old)/Q(N_new) from the WL density of states.

**Insertion** (N → N+1):
```
acc = min(1,  Q(N)/Q(N+1) × (V/Λ³) × 1/(N+1) × exp(−βΔU) )
```
ΔU = interaction energy of the new particle with all N existing ones.

**Deletion** (N → N−1):
```
acc = min(1,  Q(N)/Q(N-1) × (Λ³/V) × N × exp(−βΔU) )
```
ΔU = −(interaction energy of the deleted particle with all N−1 remaining).

Both cases are implemented in `N_metropolis` (`gc_wl.jl`). The function computes `V_Λ_prefactor = V^(ΔN) × Λ^(−3ΔN)` and `factorial_prefactor` (N for deletion, 1/(N+1) for insertion) and multiplies them with `partition_ratio = exp(logQ_N[N_old+1] − logQ_N[N_new+1])`.

#### 1/t Wang-Landau update schedule (Pereyra 2007 hybrid)

The implementation uses a two-phase 1/t WL schedule rather than pure halving to logf < 1e-8:

**Phase 1** (standard WL halving): logf starts at 1 and is halved each epoch. An epoch ends when `min(H_N) ≥ 1` over all active N bins (checked every 1000 Monte Carlo time steps). After each halving, compute the Monte Carlo time `t = wl.iters / num_active_bins` (where `num_active_bins = N_max − N_min + 1`). Phase 1 ends and phase 2 begins when `logf ≤ 1/t`.

**Phase 2** (1/t schedule): `logf = num_active_bins / wl.iters` is set continuously each MC step. No halving or histogram reset occurs. Phase 2 exits — and the simulation terminates — when `min(H_N) / mean(H_N) > 0.8` (80% flatness criterion), checked every 1,000,000 steps. `wl.iters` is never reset; it counts total MC steps from the start of the run.

**Why**: The 1/t schedule eliminates the asymptotic saturation error inherent in fixed-logf WL and exits once the histogram is genuinely flat rather than after an arbitrary number of halvings.

**References**: Pereyra et al., *J. Chem. Phys.* **126**, 124111 (2007); Shchur & Janke, *Phys. Rev. E* **96**, 043307 (2017).

### Input config format (`.inp`)

```
N
L
x1 y1 z1
x2 y2 z2
...
```

---

## Reference: Desgranges & Delhommelle 2012

**Full citation**: C. Desgranges and J. Delhommelle, "Evaluation of the grand-canonical partition function using expanded Wang-Landau simulations. I. Thermodynamic properties in the bulk and at the liquid-vapor phase boundary," *J. Chem. Phys.* **136**, 184107 (2012). DOI: 10.1063/1.4712023

Files in `literature/`: `desgranges2012.pdf`, `desgranges2012.txt` (copy-paste with artifacts), `correspondence.txt` (email exchange with Prof. Desgranges, Jul–Oct 2025), `figure1adata/*.csv` (plot-digitized data from Fig. 1a; x = N, y = ln Q*; subject to pixel-picking error).

This paper uses the SEGC algorithm (fractional-particle λ-staging), which is the predecessor approach. The current code targets the same Q(N,V,T) but via direct GCMC insertion/deletion.

---

### Exact argon simulation parameters (Figure 1a, Desgranges 2012)

| Parameter | Value |
|-----------|-------|
| Box       | V = 512 σ³ → L = 8σ |
| N_max     | 450 |
| r_cut     | 3σ |
| ε/k_B     | 117.05 K |
| σ         | 3.4 Å |
| logf₀     | 1 (i.e. f = e) |
| Statistics| std dev over **4 independent** runs |
| MC split  | 75% translation / 25% N-move |

**Temperatures** (10 isotherms, T* = kT/ε):

| T (K)  | T* (= T / 117.05) |
|--------|-------------------|
| 87.79  | 0.750 |
| 93.64  | 0.800 |
| 99.49  | 0.850 |
| 105.35 | 0.900 |
| 111.20 | 0.950 |
| 117.05 | 1.000 |
| 122.90 | 1.050 |
| 128.76 | 1.100 |
| 134.61 | 1.150 |
| 140.46 | 1.200 |

---

### Q* definition and normalization

**Definition** (Desgranges 2012, Eq. 2 + Fig. 1a caption):
```
Q*(N,V,T) = Λ^(3N) × Q(N,V,T)
           = V^N / N! × <exp(-βU)>_config      (ideal gas → V^N / N!)
```

Confirmed by Desgranges (email Aug 12, 2025): "we multiply our value Q(N,V,T) by Λ³ to obtain Q*(N,V,T)."  Note this means Λ³ *cancels* the Λ^(3N) denominator inside Q, leaving only configurational integrals.

**Normalization**: Q(N=0) = 1  →  logQ*(N=0) = 0 (anchor for the WL DoS). The `correct_logQ` function in `thermo.jl` enforces this.

**Approximate scale at N=450, T*=0.75**: ln Q* ≈ 2527.  At T*=1.2, N≈349: ln Q* ≈ 1071 (from digitized data).  Curves are nearly linear in N but slope is ~10–40× the ideal-gas slope due to attractive LJ interactions.

---

### correct_logQ output units

`correct_logQ` returns **logQ(N,V,T)** — the full thermodynamic partition function including Λ^(−3N) factors. This is NOT the same as the paper's Q*(N).

To compare to Desgranges Fig. 1a, apply the Λ correction:
```julia
logQ_star(N) = logQ_raw[N+1] + 3*N*log(Λ_σ)
```
For argon at T*=1.2: `logQ_raw[N+1] ≈ logQ_star + 8.04*N` (Λ³ ≈ 3.22e-4 σ³).

---

### Key numerical results from the paper (argon at coexistence)

Chemical potential μ_coex (kJ/kg) and ln z_sat:

| T (K)  | μ_coex | ln z_sat |
|--------|--------|----------|
| 87.79  | −237.68 | −5.684 |
| 99.49  | −254.44 | −4.775 |
| 117.05 | −282.35 | −3.833 |
| 140.46 | −323.27 | −3.028 |

Liquid densities at coexistence (g/cm³) match experiment to within 0.002 g/cm³ over the full range 87–140 K.
