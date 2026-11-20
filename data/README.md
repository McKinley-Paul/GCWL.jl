<!-- VS Code preview: ⇧⌘V (Cmd+Shift+V) in this tab, or ⌘K then V for side by side -->

# Simulation data

The 76 GC-WL runs of the paper, and the intermediate analysis files that `../figures/make_figures.ipynb` reads.
Every run started from the empty box (N = 0) and stopped on the phase-2 flatness criterion.

## Lennard-Jones fluid, `lennard_jones/`

Argon parameters (ε/k_B = 117.05 K, σ = 3.4 Å, m = 39.948 u), V = 512σ³ (L = 8σ), N_max = 450, r_c = 3σ, Λ from
`argon_deBroglie(T*)`. Ten isotherms, named by T in kelvin, from 87.79 K (T* = 0.75) to 140.46 K (T* = 1.20) in
steps of ΔT* = 0.05, with four independent runs each. The runs were seeded from system entropy
(`MersenneTwister()`). The `main.jl` scripts have no resume logic; one run, 87.79/run1, was interrupted and resumed
by hand from its checkpoints (`wl_checkpoint.jld2`, `microstate_checkpoint.jld2`) with the same parameters.

**The long-range correction is not in the stored ln Q.** Every analysis adds it with
`apply_lrc_to_logQ(logQ, sim.T_σ, sim.V_σ, sim.r_cut_σ)` after loading a run.

## Square-well fluid, `square_well/`

Hard core at σ, well depth −ε for σ ≤ r < λσ, zero beyond. Reduced units with Λ = 1, V = 729σ³ (L = 9σ),
r_c = λ (the full range of the potential, so there is no long-range correction). Four runs per state point, with the
seeds in each `main.jl`.

| λ | T* | N_max |
|---|---|---|
| 1.50 | 0.95, 1.00, 1.05, 1.10, 1.15 | 590, 565, 530, 505, 470 |
| 1.75 | 1.40, 1.55, 1.62, 1.68 | 565, 530, 505, 470 |

The potential is defined in `square_well_potential.jl`, outside the package, and `include`d by every run script.
The stored ln Q already has ln Q(0) = 0 (from `correct_logQ!`); the analysis subtracts ln Q(0) again after loading,
which changes nothing.

## Files of each run

| File | Contents |
|---|---|
| `main.jl` | the run script |
| `final_wl.jld2` | the final Wang–Landau state, normalized to Q(0) = 1 (`load_wanglandau_jld2`); `wl.logQ_N[N+1]` is ln Q(N) |
| `logQ.csv` | the same ln Q(N) as CSV, without long-range correction, with the number of moves and the final ln f in its header |
| `sim.jld2` | the `SimulationParams` of the run (Lennard-Jones only; `load_sim_jld2`) |
| `wl_progress_log.txt` | the halvings of ln f, the switch to phase 2 and the flatness checks |

`export_csv.jl` writes the `logQ.csv` files from the `final_wl.jld2` files.

## Intermediate analysis files

| File | Contents | Made by |
|---|---|---|
| `lennard_jones/output/yang_lee_zeros_avg_T*.jld2` | the four-run mean ln Q (with long-range correction) and the 450 zeros of its Ξ(z), Aberth–Ehrlich at 1024 bits | `lennard_jones/yang_lee_all_temps.jl` |
| `lennard_jones/analysis/lee_expansion_lj.jld2` | the cumulant jumps and series coefficients of Lee's expansion at the ten isotherms, and the error of each order on the first 40 ring zeros (Figs. 7–8) | `lennard_jones/analysis/lee_expansion_lj.jl` |
| `lennard_jones/analysis/per_run_roots.jld2` | the zeros of each run's ln Q (with long-range correction) at T* = 0.75, 1.00 and 1.20, `yang_lee_zeros` at 1024 bits, `tol = 1e-10` (Fig. 5 and the run-to-run spread in the text) | `yang_lee_zeros(lnQ, 0:450; prec_bits = 1024, tol = 1e-10)` on each run's ln Q, by a script not included here; if this file is absent, `make_figures.ipynb` computes the same zeros with the same call and caches them in `figures/cache/` |
| `square_well/output/sw_zeros_lam*_T*_avg.jld2` | zeros of the four-run mean ln Q at each square-well state point, Aberth–Ehrlich at 1024 bits; `make_figures.ipynb` finds the state points from these file names, checks the residuals of the zeros against the mean ln Q and re-solves if they fail | `square_well/sw_zeros_all.jl` |
| `square_well/output/lee_expansion_sw.jld2` | Lee's expansion and its errors, as in `lee_expansion_lj.jld2`, at the nine square-well state points (Figs. 7–8) | `square_well/lee_expansion_sw.jl` |

The four scripts recompute their files from the runs, and overwrite the tracked copies. Run them from the repository
root:

```bash
julia data/lennard_jones/yang_lee_all_temps.jl              # about 12 min
julia data/lennard_jones/analysis/lee_expansion_lj.jl --force   # about 11 min
julia data/square_well/lee_expansion_sw.jl --force              # about 10 min
julia data/square_well/sw_zeros_all.jl                          # about 10 min; an output directory can be given
```

`yang_lee_all_temps.jl` runs in the `figures/` environment (it also draws an overlay of the zeros with Plots), so
instantiate that first (`julia --project=figures -e 'using Pkg; Pkg.instantiate()'`). Without `--force` the two Lee
scripts only load their file and print its summary; `make_figures.ipynb` runs them if their file is missing.
