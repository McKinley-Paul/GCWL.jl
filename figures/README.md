<!-- VS Code preview: ⇧⌘V (Cmd+Shift+V) in this tab, or ⌘K then V for side by side -->

# Tables and figures of the paper

Two notebooks rebuild every table and figure of the paper from `../data/`.

1. **`make_figures.ipynb`** (Julia). It computes coexistence, Tables I–II, the zeros and their classification, and
   every number quoted in the text, and writes `data/figure_data.jld2`. Its last cell runs `scripts/sw_psat.jl`,
   which adds the square-well saturation pressures to that file. It takes about a minute with the caches in
   `cache/`.
2. **`make_figures_mpl.ipynb`** (Python). It draws Figs. 1–8 (`fig*.pdf`, `fig*.png`) from `data/figure_data.jld2`.

From a clean checkout, the two notebooks reproduce the CSV tables in `cache/` and `data/figure_data.jld2` exactly,
and the figures up to the creation date embedded in each PDF (the drawn content is identical).

## Setup

The Julia notebook uses the environment in this folder (the package from `..`, plus Plots). From the repository
root:

```bash
julia --project=figures -e 'using Pkg; Pkg.instantiate()'
julia -e 'using Pkg; Pkg.add("IJulia")'         # once, with Julia 1.12: registers the Jupyter kernel
```

The notebook asks for a Jupyter kernel named `julia-1.12`, which is the name IJulia gives its kernel under Julia 1.12.
Check with `jupyter kernelspec list`; if it is missing, `julia -e 'using IJulia; installkernel("Julia")'` registers
it. Under another Julia version the kernel is named after that version, and the notebook's kernel has to be changed
to match.

The Python notebook needs the packages in `requirements.txt` (`pip install -r figures/requirements.txt`, from the
repository root). Upgrade pip first (`python -m pip install --upgrade pip`): older versions of pip fail to build
pyzmq. SciencePlots' `science` style typesets all text with LaTeX, so a LaTeX installation (with `latex`
and `dvipng` on the path) is also needed.

Then run both notebooks with `figures/` as the working directory (both find their files relative to it):

```bash
cd figures
jupyter nbconvert --to notebook --execute --inplace make_figures.ipynb       # Julia kernel julia-1.12
jupyter nbconvert --to notebook --execute --inplace make_figures_mpl.ipynb   # Python kernel
```

`--inplace` writes the outputs into the notebooks; they are committed without outputs.

## Files

| Path | Contents |
|---|---|
| `fig1_validation.pdf` … `fig8_series_breakdown.pdf` | the figures of the paper (the PNG copies are not tracked) |
| `data/figure_data.jld2` | every array the figures are drawn from (HDF5, readable with `h5py`) |
| `cache/table1_lnysat.csv` | Table I: μ_coex and ln y_sat per isotherm, mean and standard deviation over four runs, with the SEGC-WL and TMMC values |
| `cache/table2_densities.csv` | Table II: coexistence densities and saturation pressure, with experiment and TMMC |
| `cache/numbers_for_manuscript.csv`, `cache/numbers_square_well.csv` | the other numbers quoted in the text, per state point |
| `cache/polished_roots_LJ_T*.jld2` | the LJ zeros refined at 2048 bits; `FORCE = true` in the notebook recomputes them |
| `scripts/sw_psat.jl` | square-well saturation pressure, run by the last cell of `make_figures.ipynb` |
| `Project.toml`, `Manifest.toml` | the Julia environment of `make_figures.ipynb` |
| `requirements.txt` | the Python packages of `make_figures_mpl.ipynb` |
| `julia/` | Plots.jl previews written by `make_figures.ipynb` (not tracked) |

The literature values in the tables are from Desgranges and Delhommelle, *J. Chem. Phys.* **136**, 184107 (2012)
(SEGC-WL and experiment), and from the NIST Standard Reference Simulation Website (LJ SAT-TMMC, V = 512σ³,
r_c = 3σ with long-range corrections).
