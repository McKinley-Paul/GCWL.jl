import Pkg
Pkg.activate("/Users/mckinleypaul/Documents/montecarlo/gc_wl")

using gc_wl
using JLD2
using Statistics
using Printf
using Plots

# ─── Physical constants and unit conversions for argon ────────────────────────

const k_B     = 1.380649e-23   # J/K  (exact by SI 2019)
const N_A     = 6.02214076e23  # mol^-1
const ε_kB_Ar = 117.05         # K  (LJ depth / k_B for argon)
const M_Ar    = 39.948e-3      # kg/mol
const σ_Ar    = 3.4e-10        # m

# 1 LJ reduced energy unit = ε/molecule = ε_kB * k_B J/molecule
# Per unit mass: ε/m_Ar = ε_kB * k_B * N_A / M_Ar
const LJ_TO_JPERKG  = ε_kB_Ar * k_B * N_A / M_Ar          # J/kg per reduced unit
const LJ_TO_KJKG    = LJ_TO_JPERKG / 1000.0                # kJ/kg per reduced unit

# ρ_LJ (molecules/σ³) → g/cm³
const m_Ar_g        = M_Ar * 1000.0 / N_A                  # g/molecule
const σ_Ar_cm       = σ_Ar * 100.0                          # cm
const LJDENS_TO_GCM3 = m_Ar_g / σ_Ar_cm^3                   # g/cm³ per (molecules/σ³)

# ε/σ³ → bar conversion (for pressure)
const LJPRES_TO_BAR = (ε_kB_Ar * k_B) / σ_Ar^3 / 1.0e5    # bar per LJ reduced pressure unit

# ─── Data layout ──────────────────────────────────────────────────────────────

const BASE = joinpath(@__DIR__, "..")   # figure1/

const TEMPERATURES = ["87.79", "93.64", "99.49", "105.35", "111.20",
                      "117.05", "122.90", "128.76", "134.61", "140.46"]

const TEMP_K = [87.79, 93.64, 99.49, 105.35, 111.20, 117.05, 122.90, 128.76, 134.61, 140.46]

# Reference values from Desgranges & Delhommelle (2012) Table I
const REF_SEGC_MUCX = [-237.68, -245.87, -254.44, -263.41, -272.73,
                        -282.35, -292.16, -302.32, -312.65, -323.27]  # kJ/kg
const REF_SEGC_LNZSAT = [-5.684, -5.194, -4.775, -4.415, -4.104,
                          -3.833, -3.592, -3.382, -3.193, -3.028]
const REF_TMMC_LNZSAT  = [-5.683, -5.196, -4.779, -4.419, -4.107,
                           -3.834, -3.595, -3.384, -3.197, -3.031]

# ─── Data loading ─────────────────────────────────────────────────────────────

function load_run(T_str::String, run_idx::Int)
    dir = joinpath(BASE, T_str, "run$run_idx")
    wl_path = isfile(joinpath(dir, "final_wl.jld2")) ?
              joinpath(dir, "final_wl.jld2") :
              joinpath(dir, "wl_checkpoint.jld2")
    isfile(wl_path) || return nothing, nothing
    wl  = load(wl_path, "wl")
    sim = load(joinpath(dir, "sim.jld2"), "sim")
    return wl, sim
end

# Average logQ_N across all 4 runs for a temperature; returns (logQ_mean, logQ_std, sim, n_runs)
function load_averaged_data(T_str::String)
    arrays = Vector{Float64}[]
    sim_ref = nothing
    for run in 1:4
        wl, sim = load_run(T_str, run)
        wl === nothing && continue
        push!(arrays, collect(Float64, wl.logQ_N))
        sim_ref === nothing && (sim_ref = sim)
    end
    isempty(arrays) && error("No data for T=$T_str")
    mat = reduce(hcat, arrays)                          # (N_max+1) × n_runs
    logQ_mean = vec(mean(mat, dims=2))
    logQ_std  = size(mat, 2) > 1 ? vec(std(mat, dims=2)) : zeros(size(mat, 1))
    return logQ_mean, logQ_std, sim_ref, length(arrays)
end

# ─── Load everything ──────────────────────────────────────────────────────────

println("Loading GC-WL data from figure1/...")
flush(stdout)

all_logQ  = Dict{String, Vector{Float64}}()
all_std   = Dict{String, Vector{Float64}}()
all_sim   = Dict{String, Any}()
all_nruns = Dict{String, Int}()

for T_str in TEMPERATURES
    logQ_mean, logQ_std, sim, n = load_averaged_data(T_str)
    all_logQ[T_str]  = logQ_mean
    all_std[T_str]   = logQ_std
    all_sim[T_str]   = sim
    all_nruns[T_str] = n
    @printf("  T=%-6s K  N_max=%-3d  T*=%.2f  Λ_σ=%.5f  runs=%d\n",
            T_str, sim.N_max, sim.T_σ, sim.Λ_σ, n)
end

# ─── Equal-area scan: find μ_coex and ln z_sat ────────────────────────────────

println("\nSearching for μ_coex via equal-peak-area bisection...")
flush(stdout)

gcwl_μstar  = Float64[]   # reduced units
gcwl_μkJkg  = Float64[]   # kJ/kg
gcwl_lnzsat = Float64[]

for (i, T_str) in enumerate(TEMPERATURES)
    logQ = all_logQ[T_str]
    sim  = all_sim[T_str]
    T_σ  = sim.T_σ
    Λ_σ  = sim.Λ_σ
    N_min = sim.N_min

    try
        μ_star = find_μ_coex(logQ, T_σ; N_min=N_min, μ_lo=-20.0, μ_hi=-5.0, tol=1e-7)
        lnz    = compute_lnzsat(μ_star, T_σ, Λ_σ)
        push!(gcwl_μstar,  μ_star)
        push!(gcwl_μkJkg,  μ_star * LJ_TO_KJKG)
        push!(gcwl_lnzsat, lnz)
        @printf("  T=%-6s K: μ*=%-9.4f  μ=%-8.2f kJ/kg  ln z_sat=%-7.3f\n",
                T_str, μ_star, μ_star * LJ_TO_KJKG, lnz)
    catch e
        push!(gcwl_μstar,  NaN)
        push!(gcwl_μkJkg,  NaN)
        push!(gcwl_lnzsat, NaN)
        @printf("  T=%-6s K: ERROR — %s\n", T_str, e)
    end
end

# ─── Print comparison table ───────────────────────────────────────────────────

println("\n" * "="^85)
println("TABLE 1 COMPARISON: ln(z_sat) at vapour-liquid coexistence")
println("="^85)
@printf("%-8s  %-14s  %-10s  %-14s  %-10s  %-10s  %s\n",
        "T (K)", "μ_coex GCWL", "ln z GCWL", "μ_coex SEGC", "SEGC", "TMMC", "Δ(GCWL-TMMC)")
@printf("%-8s  %-14s  %-10s  %-14s  %-10s  %-10s  %s\n",
        "", "(kJ/kg)", "", "(kJ/kg)", "", "", "")
println("-"^85)
for i in eachindex(TEMPERATURES)
    T_K = TEMP_K[i]
    Δ = isnan(gcwl_lnzsat[i]) ? NaN : gcwl_lnzsat[i] - REF_TMMC_LNZSAT[i]
    @printf("%-8.2f  %-14.2f  %-10.3f  %-14.2f  %-10.3f  %-10.3f  %+.3f\n",
            T_K, gcwl_μkJkg[i], gcwl_lnzsat[i],
            REF_SEGC_MUCX[i], REF_SEGC_LNZSAT[i], REF_TMMC_LNZSAT[i], Δ)
end
println("="^85)
println("SEGC-WL and TMMC references from Desgranges & Delhommelle, J. Chem. Phys. 136, 184107 (2012), Table I.")
println("Note: TMMC entry at T=128.76 K is -3.384 (paper has -4.384, assumed typo).")
flush(stdout)

# ─── FIGURE 2: Chemical potential μ vs <N> ────────────────────────────────────
# Reproduces Desgranges Fig. 2 for T=87.79, 105.35, 140.46 K

println("\nGenerating Figure 2 (μ vs ⟨N⟩)...")
flush(stdout)

fig2_temps  = ["87.79", "105.35", "140.46"]
fig2_colors = [:navy, :crimson, :darkgreen]
fig2_styles = [:solid, :dash, :dot]

μ_star_vals = range(-18.0, -7.0, length=600)

plt2 = plot(
    xlabel = "⟨N⟩",
    ylabel = "μ (kJ/kg)",
    title  = "Chemical potential vs. average particle number",
    legend = :bottomright,
    size   = (700, 500),
    xlims  = (0, 430),
    ylims  = (-400, -200),
    framestyle = :box,
    grid   = true,
    gridalpha = 0.3,
)

for (T_str, col, sty) in zip(fig2_temps, fig2_colors, fig2_styles)
    logQ = all_logQ[T_str]
    sim  = all_sim[T_str]
    T_σ  = sim.T_σ
    N_min = sim.N_min
    T_K  = parse(Float64, T_str)

    N_mean_vals = Float64[]
    μ_kJkg_vals = Float64[]
    for μ_star in μ_star_vals
        N_mean = compute_mean_N(logQ, μ_star, T_σ; N_min=N_min)
        push!(N_mean_vals, N_mean)
        push!(μ_kJkg_vals, μ_star * LJ_TO_KJKG)
    end

    plot!(plt2, N_mean_vals, μ_kJkg_vals,
          label     = "T = $(T_K) K  (T*=$(T_σ))",
          color     = col,
          linestyle = sty,
          linewidth = 2)
end

# Mark the coexistence μ for each of the three temperatures
for (j, T_str) in enumerate(fig2_temps)
    idx = findfirst(==(T_str), TEMPERATURES)
    idx === nothing && continue
    μ_c = gcwl_μkJkg[idx]
    isnan(μ_c) && continue
    hline!(plt2, [μ_c], color=fig2_colors[j], linestyle=:dash, alpha=0.4, linewidth=1, label="")
end

fig2_path = joinpath(@__DIR__, "figure2_mu_vs_N.pdf")
savefig(plt2, fig2_path)
println("  Saved: $fig2_path")

# ─── FIGURE 4: p(N) at T=140.46 K for three μ values ─────────────────────────
# Reproduces Desgranges Fig. 4 (a), (b), (c)

println("Generating Figure 4 (p(N) at T=140.46 K)...")
flush(stdout)

T_str = "140.46"
logQ  = all_logQ[T_str]
sim   = all_sim[T_str]
T_σ   = sim.T_σ
N_min = sim.N_min
N_max = sim.N_max
N_vals = N_min:N_max

# Three μ values from the paper (kJ/kg) → reduced units
μ_liquid_star = -316.62 / LJ_TO_KJKG   # bulk liquid
μ_vapor_star  = -361.68 / LJ_TO_KJKG   # bulk vapor
μ_coex_paper  = -323.27 / LJ_TO_KJKG   # liquid-vapor coexistence (paper)
μ_coex_gcwl   = gcwl_μstar[findfirst(==("140.46"), TEMPERATURES)]   # our value

pN_liquid = compute_pN(logQ, μ_liquid_star, T_σ; N_min=N_min)
pN_vapor  = compute_pN(logQ, μ_vapor_star,  T_σ; N_min=N_min)
pN_coex_paper = compute_pN(logQ, μ_coex_paper, T_σ; N_min=N_min)
pN_coex_gcwl  = isnan(μ_coex_gcwl) ? nothing :
                compute_pN(logQ, μ_coex_gcwl, T_σ; N_min=N_min)

# Panel (a): bulk liquid
plt4a = plot(N_vals, pN_liquid,
             xlabel="N", ylabel="p(N)",
             title="(a) μ = −316.62 kJ/kg  (bulk liquid)",
             label="GC-WL", color=:navy, linewidth=1.5,
             framestyle=:box, legend=:topleft)

# Panel (b): bulk vapor
plt4b = plot(N_vals, pN_vapor,
             xlabel="N", ylabel="p(N)",
             title="(b) μ = −361.68 kJ/kg  (bulk vapor)",
             label="GC-WL", color=:crimson, linewidth=1.5,
             framestyle=:box, legend=:topright)

# Panel (c): coexistence — paper value and our GC-WL value
plt4c = plot(N_vals, pN_coex_paper,
             xlabel="N", ylabel="p(N)",
             title="(c) Coexistence (T = 140.46 K)",
             label="μ = −323.27 kJ/kg (SEGC-WL)", color=:darkgreen, linewidth=1.5,
             framestyle=:box, legend=:top)
if pN_coex_gcwl !== nothing
    plot!(plt4c, N_vals, pN_coex_gcwl,
          label=@sprintf("μ = %.2f kJ/kg (GC-WL)", gcwl_μkJkg[findfirst(==("140.46"), TEMPERATURES)]),
          color=:orange, linestyle=:dash, linewidth=1.5)
end

# Find and mark the valley N_b for the coexistence panel
if pN_coex_gcwl !== nothing
    idx_b = find_Nb_idx(pN_coex_gcwl)
    N_b   = N_min + idx_b - 1
    vline!(plt4c, [N_b], color=:gray, linestyle=:dot, linewidth=1, label="N_b=$(N_b)")
end

plt4 = plot(plt4a, plt4b, plt4c, layout=(3,1), size=(700, 900))

fig4_path = joinpath(@__DIR__, "figure4_pN.pdf")
savefig(plt4, fig4_path)
println("  Saved: $fig4_path")

# ─── BONUS: p(N) at coexistence for all temperatures ─────────────────────────

println("Generating coexistence p(N) for all temperatures...")
flush(stdout)

plt_all = plot(
    xlabel    = "N",
    ylabel    = "p(N)",
    title     = "p(N) at coexistence — all temperatures",
    legend    = :outerright,
    size      = (900, 500),
    framestyle = :box,
    grid      = true,
    gridalpha = 0.3,
)

colors_all = cgrad(:plasma, length(TEMPERATURES), categorical=true)

for (i, T_str) in enumerate(TEMPERATURES)
    isnan(gcwl_μstar[i]) && continue
    logQ = all_logQ[T_str]
    sim  = all_sim[T_str]
    T_σ  = sim.T_σ
    N_min = sim.N_min
    N_max = sim.N_max
    μ_star = gcwl_μstar[i]
    pN = compute_pN(logQ, μ_star, T_σ; N_min=N_min)
    T_K = TEMP_K[i]
    plot!(plt_all, N_min:N_max, pN,
          label="T=$(T_K) K", color=colors_all[i], linewidth=1.2)
end

fig_all_path = joinpath(@__DIR__, "figure_pN_allT.pdf")
savefig(plt_all, fig_all_path)
println("  Saved: $fig_all_path")

# ─── BONUS: μ vs ⟨N⟩ for all temperatures ────────────────────────────────────

println("Generating μ vs ⟨N⟩ for all temperatures...")
flush(stdout)

μ_scan = range(-18.0, -7.0, length=400)

plt_mu_all = plot(
    xlabel    = "⟨N⟩",
    ylabel    = "μ (kJ/kg)",
    title     = "μ vs. ⟨N⟩ — all temperatures",
    legend    = :bottomright,
    size      = (800, 550),
    xlims     = (0, 450),
    ylims     = (-380, -200),
    framestyle = :box,
    grid      = true,
    gridalpha = 0.3,
)

for (i, T_str) in enumerate(TEMPERATURES)
    logQ  = all_logQ[T_str]
    sim   = all_sim[T_str]
    T_σ   = sim.T_σ
    N_min = sim.N_min
    T_K   = TEMP_K[i]

    N_means = [compute_mean_N(logQ, μ_star, T_σ; N_min=N_min) for μ_star in μ_scan]
    μ_kJkg  = collect(μ_scan) .* LJ_TO_KJKG

    plot!(plt_mu_all, N_means, μ_kJkg,
          label="T=$(T_K) K", color=colors_all[i], linewidth=1.5)
end

fig_mu_all_path = joinpath(@__DIR__, "figure2_allT.pdf")
savefig(plt_mu_all, fig_mu_all_path)
println("  Saved: $fig_mu_all_path")

println("\nAnalysis complete.")
