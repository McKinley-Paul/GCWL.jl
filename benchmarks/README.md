<!-- VS Code preview: ⇧⌘V (Cmd+Shift+V) in this tab, or ⌘K then V for side by side -->

# Benchmark: 1/t GC-WL against SEGC-WL

The paper compares the CPU time and accuracy of 1/t GC-WL (this package) with the expanded grand canonical
Wang–Landau method (SEGC-WL) of Desgranges and Delhommelle (2012), for the Lennard-Jones fluid with argon parameters
at T* = 1.20 (N_max = 350) and T* = 0.75 (N_max = 450), V = 512σ³, r_c = 3σ. Each method had four independent runs
per state point on single cores of the same CPU model (Intel Xeon 6972P). At T* = 0.75 the SEGC-WL runs were
stopped after 504–599 CPU-hours, before reaching their ln f < 10⁻⁸ stopping criterion; their results use ln Q at the
last checkpoint.

The data of both methods (ln Q(N) of every run, CPU times, logs and the results) are in the
[SEGC_WangLandau](https://github.com/McKinley-Paul/SEGC_WangLandau) repository, under `benchmarks/`; the 1/t GC-WL
runs are in its `gcwl_reference/` folders. A summary table is in that repository's README.

This folder holds the GC-WL side of the benchmark:

| File | Contents |
|---|---|
| `segc_vs_gcwl_T*/run1–run4/main.jl` | the 1/t GC-WL run scripts; each resumes from its checkpoint and writes `logQ_final.csv` on convergence |
| `segc_vs_gcwl_T*/job.sbatch` | Slurm array job for the four runs (set the account, QOS and CPU constraint) |
| `segc_vs_gcwl_T*/analyze.jl` | the analysis: long-range correction, coexistence by equal weight of the two peaks of P(N), comparison with NIST SAT-TMMC |

To repeat the analysis, clone SEGC_WangLandau anywhere (here next to GCWL.jl) and run, from the GCWL.jl root:

```bash
git clone https://github.com/McKinley-Paul/SEGC_WangLandau.git ../SEGC_WangLandau
cd benchmarks/segc_vs_gcwl_T1.20
julia --project=../.. analyze.jl ../../../SEGC_WangLandau/benchmarks/segc_vs_gcwl_T1.20
```

It writes `results.md` in this folder; the same file is in SEGC_WangLandau. The same commands with `T0.75` give the
other state point.
