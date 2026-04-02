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

---

## Reference: Desgranges & Delhommelle 2012

**Full citation**: C. Desgranges and J. Delhommelle, "Evaluation of the grand-canonical partition function using expanded Wang-Landau simulations. I. Thermodynamic properties in the bulk and at the liquid-vapor phase boundary," *J. Chem. Phys.* **136**, 184107 (2012). DOI: 10.1063/1.4712023

Files in `literature/`: `desgranges2012.pdf`, `desgranges2012.txt` (copy-paste with artifacts), `correspondence.txt` (email exchange with Prof. Desgranges, Jul–Oct 2025), `figure1adata/*.csv` (plot-digitized data from Fig. 1a; x = N, y = ln Q*; subject to pixel-picking error).

---

### Exact argon simulation parameters (Figure 1a)

| Parameter | Value |
|-----------|-------|
| Box       | V = 512 σ³ → L = 8σ |
| N_max     | 450 |
| r_cut     | 3σ |
| M (stages)| 100 (l = 0 … 99, so `λ_max = 99`) |
| ε/k_B     | 117.05 K |
| σ         | 3.4 Å |
| logf₀     | 1 (i.e. f = e) |
| logf_min  | 10⁻⁸ |
| Flatness  | every (N, l) visited ≥ 1000 times per epoch |
| Statistics| std dev over **4 independent** runs |
| MC split  | 75% translation / 25% λ-move |

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

**Definition** (Eq. 2 + Fig. 1a caption):
```
Q*(N,V,T) = Λ^(3N) × Q(N,V,T)
           = V^N / N! × <exp(-βU)>_config      (ideal gas → V^N / N!)
```

Confirmed by Desgranges (email Aug 12, 2025): "we multiply our value Q(N,V,T) by Λ³ to obtain Q*(N,V,T)."  Note this means Λ³ *cancels* the Λ^(3N) denominator inside Q, leaving only configurational integrals.

**Normalization**: Q(N=0, l=0) = 1  →  logQ*(N=0) = 0 (anchor for the WL DoS). The `correct_logQ` function in `thermo.jl` enforces this.

**Final output**: Desgranges confirmed — only the l=0 slice is used: Q(N,V,T) ≡ Q(N,V,T,l=0).

**Approximate scale at N=450, T*=0.75**: ln Q* ≈ 2527.  At T*=1.2, N≈349: ln Q* ≈ 1071 (from digitized data).  Curves are nearly linear in N but slope is ~10–40× the ideal-gas slope due to attractive LJ interactions.

---

### Metropolis acceptance criteria (Eqs. 10–12)

All written in terms of the *ratio* Q(old state) / Q(new state), which is what the WL density-of-states tracks.

**Eq. 10** — translation move OR λ-move where both l_old and l_new are either both 0 or both > 0 (N unchanged):
```
acc = min(1,  Q(N, l_o) / Q(N, l_n) × exp(−βΔU) )
```
For a pure translation (l_o = l_n, N unchanged) this reduces to the standard Metropolis criterion.

**Eq. 11** — λ-move from l=0 → l=1 (fractional particle *created*, N stays same):
```
acc = min(1,  Q(N, 0) / Q(N, 1) × (V / Λ³) × exp(−βΔU) )
```
The `V/Λ³` factor arises from the V^(N+1) vs V^N and Λ^(3N) vs Λ^(3(N+1)) difference.

**Eq. 12** — λ-move from l>0 → l=0 (fractional particle *destroyed* or *fully inserted*):

*Case A*: l=1 → l=0, N unchanged (fractional disappears):
```
acc = min(1,  Q(N, 1) / Q(N, 0) × (Λ³ / V) × exp(−βΔU) )
```

*Case B*: l=λ_max → l=0, N → N+1 (fractional becomes a full particle, complete insertion):
```
acc = min(1,  Q(N, λ_max) / Q(N+1, 0) × 1/(N+1) × exp(−βΔU) )
```
The `1/(N+1)` is the factorial ratio N!/(N+1)!.  The V/Λ³ factors cancel when No=N, Nn=N+1.

*Case C* (reverse of B): l=0, N → N−1, l → λ_max (begin deletion):
```
acc = min(1,  Q(N, 0) / Q(N−1, λ_max) × N × exp(−βΔU) )
```

The `λ_metropolis_pm1` function in `lj.jl` implements all these cases.

---

### Fractional-particle LJ coupling (Eq. 16)

Coupling exponents from Ref. 35 (Kaminski 1994):
```
ε_ξ(l)        = (l/M)^(1/3) × ε        [well depth scales as l^(1/3)]
σ_ξ²(l)       = (l/M)^(1/2) × σ²       [exclusion diameter: σ_ξ = (l/M)^(1/4) σ]
```
These correspond to `ϵ_ξ` and `σ_ξ_squared` in the `microstate` struct.

---

### correct_logQ output units

`correct_logQ` returns **logQ(N,V,T)** — the full thermodynamic partition function including Λ^(−3N) factors. This is NOT the same as the paper's Q*(N).

To compare to Desgranges Fig. 1a, apply the Λ correction:
```julia
logQ_star(N) = logQ_raw[N+1] + 3*N*log(Λ_σ)
```
For argon at T*=1.2: `logQ_raw[N+1] ≈ logQ_star + 8.04*N` (Λ³ ≈ 3.22e-4 σ³).

---

### Deletion bug (fixed Apr 2026) — detailed balance violation

**Root cause**: In `λ_move!` (`segc_wl.jl`), the deletion branch (λ_proposed == -1) left `r_frac_box` at the stale ghost position instead of setting it to the deleted particle's position. This violated detailed balance: the forward move (delete particle at r_i, frac appears at ghost) and the reverse move (insert from ghost, particle appears at ghost ≠ r_i) connected *different* pairs of microstates. No choice of WL DoS can satisfy detailed balance for such a scheme.

**Effect**: Systematic overestimation of logQ* at high N. Grew linearly with N and with M (number of λ stages): M=3 gave −39% (underestimate), M=10 gave +14%, M=100 gave +97% at N=350.

**Fix** (3 lines in `λ_move!`):
```julia
# Before λ_metropolis_pm1 call:
c.μ_prop.r_frac_box .= μ.r_box[idx_deleted]
# In acceptance block, BEFORE the r_box swap:
μ.r_frac_box .= μ.r_box[idx_deleted]
# In rejection block:
c.μ_prop.r_frac_box .= μ.r_frac_box
```

**Validation**: M=10, N=108, T*=1.2 → logQ*(108) = 347 vs reference 350 (−0.8%). M=100, N=50, T*=1.2 → logQ*(50) = 180.1 vs reference 182 (−1.1%).

**Regression tests**: `test/runtests.jl` contains a "Detailed balance: insertion/deletion ΔU antisymmetry" testset that directly catches this bug if reintroduced (checks ΔU_del + ΔU_ins = 0, which only holds when r_frac is set to r_deleted). The high-N ideal gas test (T*=1e6) also catches it.

**Key principle**: In any extended-ensemble MC move that changes N, the forward and reverse moves must connect the *same* pair of microstates. For deletion: the fractional particle must appear at the deleted particle's position so that insertion (the reverse) recreates the original state exactly. This is a microstate-level detailed balance requirement, invisible if you only check the (N,λ) macrostate histogram.

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
