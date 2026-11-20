<!-- VS Code preview: ⇧⌘V (Cmd+Shift+V) in this tab, or ⌘K then V for side by side -->

# GCWL.jl

A Julia implementation of grand canonical Wang–Landau sampling (GC-WL) with a 1/t schedule. A run gives the canonical
partition functions ln Q(N, V, T) for every N from 0 to N_max at fixed V and T, starting from the empty box. The
package also finds the Yang–Lee zeros of the grand canonical partition function
Ξ(z) = Σ_N Q(N, V, T) z^N in arbitrary precision, and computes coexistence properties from ln Q(N).

This repository holds the code, the simulation data and the analysis behind

> M. K. Paul, "Yang–Lee zeros of condensation in three-dimensional off-lattice fluids" (submitted to *Phys. Rev. E*).

Every table and figure of the paper can be rebuilt from the data here; see [Reproducing the paper](#reproducing-the-paper).

## Method

- **Moves.** 75% are translations of one particle, accepted with the Metropolis criterion. 25% insert or delete one
  whole particle (ΔN = ±1), with the acceptance rule of Ganzenmüller and Camp [1], Eq. 19:
  - insertion, N → N+1: acc = min(1, Q(N)/Q(N+1) · (V/Λ³) · 1/(N+1) · exp(−βΔU));
  - deletion, N → N−1: acc = min(1, Q(N)/Q(N−1) · (Λ³/V) · N · exp(−βΔU)).
- **Wang–Landau schedule** (Belardinelli and Pereyra [2]).
  - ln Q(N) is raised by ln f after every attempted move, starting from ln f = 1.
  - Phase 1: ln f is halved, and the histogram reset, once every N bin has been visited at least once (checked every
    1000 units of Monte Carlo time t, the number of moves divided by the number of bins). Phase 1 ends when
    ln f ≤ 1/t.
  - Phase 2: ln f = 1/t, updated after every move, with no further resets. The run stops when the histogram of
    phase 2 is flat, min H / mean H > 0.8 (checked every 10⁶ moves).
- **Normalization.** `post_run` (or `correct_logQ!`) shifts ln Q so that Q(N = 0, V, T) = 1.
- **Box and potential.** Cubic box with periodic boundary conditions, minimum image and a spherical cutoff r_c. The
  standard Lennard-Jones long-range correction is not applied during sampling; `apply_lrc_to_logQ` adds it to ln Q(N)
  afterwards.
- **Displacements.** The maximum displacement is tuned towards 50% acceptance from the translations at
  N ≥ 0.9 N_max (×1.05 above 55%, ×0.95 below 45%), throughout the run. `dynamic_δr_max_box = false` keeps it fixed.

**Units.** Lengths are in σ, energies in ε, and the temperature is T* = k_BT/ε. Positions are stored in box units,
in [−0.5, 0.5]³. The thermal de Broglie wavelength Λ (in σ) enters the insertion and deletion rules, so ln Q depends
on the mass. `argon_deBroglie(T*)` gives Λ for argon (ε/k_B = 117.05 K, σ = 3.4 Å, m = 39.948 u). The returned ln Q
includes the factor Λ^(−3N), and z = e^(βμ). The reduced partition function Q* = Λ^(3N) Q of Desgranges and
Delhommelle [3] is `logQ[N+1] + 3N*log(Λ_σ)`.

## Installation

Julia ≥ 1.11 is required; the Manifest was resolved with Julia 1.12.1.

```bash
git clone https://github.com/McKinley-Paul/GCWL.jl.git
cd GCWL.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Usage

```julia
using GCWL, Random

T = 1.2
sim = SimulationParams(N_max = 40, N_min = 0, T_σ = T, Λ_σ = argon_deBroglie(T),
                       L_σ = 4.0, r_cut_σ = 2.0,
                       save_directory_path = "out", maxiter = 10^13, rng = MersenneTwister(1))
μ     = init_microstate(sim)          # vacuum start, N = 0
wl    = init_WangLandauVars(sim)
cache = init_cache(sim, μ)
run_simulation!(sim, μ, wl, cache)
post_run(sim, μ, wl)                  # normalizes, then writes final_wl.jld2 and final_microstate.jld2
logQ  = wl.logQ_N                     # logQ[N+1] = ln Q(N, V, T), Q(0) = 1
```

`examples/lj_argon_small.jl` is this example as a script; it takes a few seconds. A run can also start from a
configuration file (`init_microstate(sim = sim, filename = ...)`): N on the first line, the box length in σ on the
second, then one `x y z` line per particle, in σ.

**Checkpoints.** `run_simulation!` writes `wl_checkpoint.jld2` and `microstate_checkpoint.jld2` to
`save_directory_path` every 10⁶ moves, and appends progress to `wl_progress_log.txt`. To resume, rebuild `sim` with
exactly the same parameters as the interrupted run, then:

```julia
wl    = load_wanglandau_jld2("wl_checkpoint.jld2")
μ     = load_microstate_jld2("microstate_checkpoint.jld2")
cache = init_cache(sim, μ)
run_simulation!(sim, μ, wl, cache)
```

**Other pair potentials.** Define a subtype of `PairPotential` and a method of `GCWL.pair_energy` that takes the
*squared* distance in σ², and pass it as `SimulationParams(...; potential = ...)`. The square-well potential used in
the paper, `data/square_well/square_well_potential.jl`, is an example; the tests define a hard-sphere one.

**Analysis of ln Q(N).**

- Coexistence (equal area of the two peaks of P(N)): `find_μ_coex`, `compute_ln_y_sat`, `compute_phase_densities`,
  `compute_pN`, `compute_logΞ`, `compute_Psat_bar`.
- Yang–Lee zeros: `yang_lee_zeros(lnQ, 0:N_max; prec_bits = 1024)` finds the roots of Ξ(z) = Σ_N e^(lnQ[N+1]) z^N
  by the Aberth–Ehrlich method in ArbNumerics arbitrary precision. It returns a NamedTuple; the roots are in `.roots`
  (sorted by |Im z|, with non-finite roots and roots with |z| ≥ 10⁶ dropped). `verify_yang_lee_zeros` checks their
  residuals.
  `mean_N_from_roots`, `var_N_from_roots`, `density_from_roots` and `compressibility_from_roots` give thermodynamics
  from the roots.

## Tests

```bash
julia --project=. test/runtests.jl
```

`Pkg.test()` (or `] test`) currently fails on Julia 1.12 with "expected package Test to be registered"; run the
script as shown, as CI does.

The tests (about 10 seconds) check:

- unit tests of the energy functions, the Metropolis criterion and the insertion and deletion acceptance rules;
- that the cache stays consistent with the microstate, and the initialization of the structs;
- the ideal-gas V/Λ³ and N! prefactors of insertion and deletion, and the sign of ΔU for a move and its reverse;
- short end-to-end runs whose ln Q agree with reference values to within 5%: N = 1 and 2 of the Lennard-Jones fluid
  (numerical integration), the ideal gas up to N = 108 (exact), and dilute hard spheres and the Uhlenbeck–Ford model
  up to N = 12 (virial series). `test/mathematica_verification.nb` holds the reference values. These comparisons are
  at dilute or ideal-gas (T* = 10⁶) conditions, where the interaction contribution to ln Q is far smaller than the
  5% tolerance, so they check the sampler and the normalization, not the pair potential.

## Reproducing the paper

| Step | Where | What |
|---|---|---|
| 1. Simulations | [`data/`](data/README.md) | run scripts and ln Q(N) of all 76 runs: 40 Lennard-Jones (ten isotherms, four runs each) and 36 square-well (nine state points, four runs each) |
| 2. Zeros and tables | [`figures/make_figures.ipynb`](figures/README.md) | Julia: coexistence, Tables I–II, the zeros and every number in the text; writes `figures/data/figure_data.jld2` |
| 3. Figures | [`figures/make_figures_mpl.ipynb`](figures/README.md) | Python: draws Figs. 1–8 from `figure_data.jld2` |
| Benchmark | [`benchmarks/`](benchmarks/README.md) | run scripts and analysis of the comparison with SEGC-WL; the data are in [SEGC_WangLandau](https://github.com/McKinley-Paul/SEGC_WangLandau) |

Steps 2 and 3 take about a minute each and reproduce the paper's tables and figures exactly from the data in this
repository (the figure PDFs differ only in their embedded creation date). Step 1 took about 196 CPU-hours for the
Lennard-Jones runs.

## Repository layout

| Path | Contents |
|---|---|
| `src/` | the package: `GCWL.jl` (main loop, moves, Wang–Landau update), `initialization.jl` (structs, input, checkpoints), `lj.jl`, `utils.jl`, `thermo.jl` (normalization, coexistence, long-range correction), `yang_lee_zeros.jl` |
| `test/` | unit and regression tests |
| `examples/` | a small worked example |
| `initial_configs/` | an fcc configuration (N = 108, L = 8σ) used by the tests |
| `data/` | the simulation runs of the paper and the intermediate analysis files |
| `figures/` | the notebooks that make the paper's tables and figures, their caches, and the figures |
| `benchmarks/` | run scripts and analysis of the SEGC-WL comparison |

## Citation

If you use this code or data, please cite the archived release (see `CITATION.cff`) and the paper above.

## References

1. G. Ganzenmüller and P. J. Camp, "Applications of Wang–Landau sampling to determine phase equilibria in complex
   fluids," *J. Chem. Phys.* **127**, 154504 (2007). doi:10.1063/1.2794042
2. R. E. Belardinelli and V. D. Pereyra, "Fast algorithm to calculate density of states," *Phys. Rev. E* **75**,
   046701 (2007). doi:10.1103/PhysRevE.75.046701
3. C. Desgranges and J. Delhommelle, "Evaluation of the grand-canonical partition function using expanded
   Wang-Landau simulations. I. Thermodynamic properties in the bulk and at the liquid-vapor phase boundary,"
   *J. Chem. Phys.* **136**, 184107 (2012). doi:10.1063/1.4712023

## License

MIT; see `LICENSE`.
