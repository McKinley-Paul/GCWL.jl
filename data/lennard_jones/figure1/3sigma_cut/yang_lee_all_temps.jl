"""
Multi-temperature Yang–Lee zero analysis for figure1/3sigma_cut.

For each of the 10 argon isotherms:
  1. Load all completed runs (up to 4), apply long-range corrections.
  2. Average lnQ(N) across runs.
  3. Compute Yang–Lee zeros of Ξ(z) = Σ_N Q(N,V,T) z^N via Aberth–Ehrlich
     (uses gc_wl.src.yang_lee_zeros library).
  4. Save the averaged zeros to output/yang_lee_zeros_avg_<T>.jld2.
  5. Produce an overlay figure showing all zeros for all temperatures.

Output files
  output/yang_lee_zeros_avg_<T>.jld2   — per-temperature zeros (averaged lnQ)
  output/yang_lee_all_temps.csv         — leading-zero summary table
  figures/yang_lee_all_temps.pdf/.png  — overlay figure
"""

import Pkg
Pkg.activate("/Users/mckinleypaul/Documents/montecarlo/gc_wl")

using gc_wl
using JLD2
using Statistics
using Printf
using Plots
using Plots: RGB, RGBA

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

const BASE_DIR    = @__DIR__
const OUTPUT_DIR  = joinpath(BASE_DIR, "output")
const FIGURES_DIR = joinpath(BASE_DIR, "figures")
mkpath(OUTPUT_DIR)
mkpath(FIGURES_DIR)

const ε_kB_Ar  = 117.05    # K — LJ ε/k_B for argon
const N_RUNS   = 4
const N_MAX    = 450        # expected; actual read from sim
const V_SIGMA3 = 512.0      # box volume in σ³

# Temperatures in K (T* = T / 117.05)
const TEMP_TABLE = [
    (T_K=87.79,  T_star=87.79/117.05,  label="T*=0.75"),
    (T_K=93.64,  T_star=93.64/117.05,  label="T*=0.80"),
    (T_K=99.49,  T_star=99.49/117.05,  label="T*=0.85"),
    (T_K=105.35, T_star=105.35/117.05, label="T*=0.90"),
    (T_K=111.20, T_star=111.20/117.05, label="T*=0.95"),
    (T_K=117.05, T_star=1.0,           label="T*=1.00"),
    (T_K=122.90, T_star=122.90/117.05, label="T*=1.05"),
    (T_K=128.76, T_star=128.76/117.05, label="T*=1.10"),
    (T_K=134.61, T_star=134.61/117.05, label="T*=1.15"),
    (T_K=140.46, T_star=140.46/117.05, label="T*=1.20"),
]

const PREC_BITS = 1024
const MAX_ITER  = 400
const TOL       = 1e-10

# Colorblind-friendly sequential palette: blue (cold) → red (hot), 10 steps
function temp_colors(n::Int)
    # Interpolate between a cool blue and a warm red via a perceptually uniform path
    cold = (0.173, 0.486, 0.725)   # steel blue
    mid  = (0.400, 0.761, 0.647)   # teal-green (midpoint)
    hot  = (0.835, 0.243, 0.310)   # brick red
    colors = Vector{RGB{Float64}}(undef, n)
    for i in 1:n
        t = (i - 1) / (n - 1)
        if t < 0.5
            s = 2t
            r = cold[1] + s * (mid[1] - cold[1])
            g = cold[2] + s * (mid[2] - cold[2])
            b = cold[3] + s * (mid[3] - cold[3])
        else
            s = 2(t - 0.5)
            r = mid[1] + s * (hot[1] - mid[1])
            g = mid[2] + s * (hot[2] - mid[2])
            b = mid[3] + s * (hot[3] - mid[3])
        end
        colors[i] = RGB(r, g, b)
    end
    return colors
end

# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

function load_and_average(T_K::Float64)
    T_str = @sprintf("%.2f", T_K)
    runs_lnQ = Vector{Float64}[]
    sim_ref  = nothing
    N_min_common = 0
    N_max_common = typemax(Int)

    for run in 1:N_RUNS
        dir     = joinpath(BASE_DIR, T_str, "run$run")
        wl_path = joinpath(dir, "final_wl.jld2")
        sp_path = joinpath(dir, "sim.jld2")
        isfile(wl_path) && isfile(sp_path) || continue

        wl  = load(wl_path, "wl")
        sim = load(sp_path, "sim")

        logQ_lrc = apply_lrc_to_logQ(collect(Float64, wl.logQ_N),
                                      sim.T_σ, sim.V_σ, sim.r_cut_σ; N_min=sim.N_min)
        N_max_common = min(N_max_common, sim.N_max)
        N_min_common = max(N_min_common, sim.N_min)
        push!(runs_lnQ, logQ_lrc)
        if sim_ref === nothing
            sim_ref = sim
        end
    end

    isempty(runs_lnQ) && error("No completed runs for T=$T_str K")

    # Trim all runs to the common N range
    n_keep = N_max_common - N_min_common + 1
    i_lo   = N_min_common - sim_ref.N_min + 1
    i_hi   = i_lo + n_keep - 1
    mat    = hcat([lnQ[i_lo:i_hi] for lnQ in runs_lnQ]...)

    lnQ_mean = vec(mean(mat, dims=2))
    lnQ_std  = size(mat, 2) > 1 ? vec(std(mat, dims=2)) : zeros(n_keep)
    N_values = N_min_common:N_max_common
    n_loaded = length(runs_lnQ)

    return (lnQ=lnQ_mean, lnQ_std=lnQ_std, N_values=N_values,
            n_runs=n_loaded, sim=sim_ref, T_K=T_K,
            T_star=T_K / ε_kB_Ar)
end

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════════

println("="^70)
println("Multi-temperature Yang–Lee zero analysis")
println("  Temperatures: $(length(TEMP_TABLE)) isotherms, $(N_RUNS) runs each")
println("  Precision:    $(PREC_BITS) bits (ArbNumerics Aberth–Ehrlich)")
println("="^70)
println()

results = []   # (T_K, T_star, label, roots, z1, z_ref, n_runs)

for row in TEMP_TABLE
    T_K    = row.T_K
    T_star = row.T_star
    label  = row.label
    T_str  = @sprintf("%.2f", T_K)

    println("-"^70)
    @printf("T = %.2f K  (%s)\n", T_K, label)

    # ── Load & average ───────────────────────────────────────────────────────
    d = load_and_average(T_K)
    @printf("  Loaded %d runs  N=[%d,%d]\n",
            d.n_runs, first(d.N_values), last(d.N_values))

    # ── Yang-Lee zeros ───────────────────────────────────────────────────────
    result = yang_lee_zeros(d.lnQ, d.N_values;
                             prec_bits = PREC_BITS,
                             max_iter  = MAX_ITER,
                             tol       = TOL,
                             verbose   = true)

    z1 = leading_zero(result.roots)
    if z1 !== nothing
        @printf("  Leading zero:  Re(z₁) = %.6e   Im(z₁) = %.6e\n",
                real(z1), imag(z1))
    else
        println("  Leading zero:  none found on positive real side")
    end

    # ── Verification ─────────────────────────────────────────────────────────
    vf = verify_yang_lee_zeros(result)
    ok = vf.max_relative_residual < 1e-10
    @printf("  Residual check: max rel = %.2e  %s\n",
            vf.max_relative_residual, ok ? "ok" : "WARN")
    @printf("  Conjugate violations: %d\n", vf.conjugate_pair_violations)

    # ── Save JLD2 ─────────────────────────────────────────────────────────────
    out_path = joinpath(OUTPUT_DIR, "yang_lee_zeros_avg_T$(T_str).jld2")
    jldsave(out_path;
            T_K       = T_K,
            T_star    = T_star,
            N_values  = collect(d.N_values),
            lnQ_mean  = d.lnQ,
            lnQ_std   = d.lnQ_std,
            n_runs    = d.n_runs,
            roots     = result.roots,
            z_ref     = result.z_ref,
            converged = result.converged,
            final_corr = result.final_corr,
            prec_bits = result.prec_bits,
            leading_zero_re = z1 !== nothing ? real(z1) : NaN,
            leading_zero_im = z1 !== nothing ? imag(z1) : NaN)
    println("  Saved: $out_path")

    push!(results, (T_K=T_K, T_star=T_star, label=label,
                    roots=result.roots, z1=z1, z_ref=result.z_ref,
                    n_runs=d.n_runs, converged=result.converged))
    println()
end

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════════

println("="^70)
println("Leading-zero summary")
println("="^70)
@printf("%-10s  %-8s  %-14s  %-14s  %s\n",
        "T (K)", "T*", "Re(z₁)", "Im(z₁)", "status")
println("-"^70)
for r in results
    z1 = r.z1
    re_s = z1 !== nothing ? @sprintf("%.6e", real(z1)) : "—"
    im_s = z1 !== nothing ? @sprintf("%.6e", imag(z1)) : "—"
    status = r.converged ? "ok" : "not converged"
    @printf("%-10.2f  %-8.4f  %-14s  %-14s  %s\n",
            r.T_K, r.T_star, re_s, im_s, status)
end

# Save CSV
csv_path = joinpath(OUTPUT_DIR, "yang_lee_all_temps.csv")
open(csv_path, "w") do io
    println(io, "T_K,T_star,label,re_z1,im_z1,z_ref,n_runs,converged")
    for r in results
        z1 = r.z1
        re_z1 = z1 !== nothing ? real(z1) : NaN
        im_z1 = z1 !== nothing ? imag(z1) : NaN
        println(io, "$(r.T_K),$(r.T_star),$(r.label),$(re_z1),$(im_z1),$(r.z_ref),$(r.n_runs),$(r.converged)")
    end
end
println("Table saved: $csv_path")
println()

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE
# ═══════════════════════════════════════════════════════════════════════════════

println("Building overlay figure...")

colors = temp_colors(length(results))

# Determine axis limits from all roots across all temperatures.
# Focus on a sensible Re/Im window — exclude the far-out roots that are
# preconditioning artefacts and not physically relevant.
all_re = Float64[]
all_im = Float64[]
for r in results
    for z in r.roots
        re = real(z); im = imag(z)
        if re > 0 && im >= 0
            push!(all_re, re)
            push!(all_im, im)
        end
    end
end

# Collect only leading zeros from subcritical temperatures (Im small, near real axis)
lead_re = [real(r.z1) for r in results if r.z1 !== nothing]
lead_im = [imag(r.z1) for r in results if r.z1 !== nothing]

# Panel (a): full upper-half-plane view of all zeros
# Use a percentile clip to avoid a few stray outlier roots stretching the axis
re_95 = sort(all_re)[min(end, round(Int, 0.90 * length(all_re)))]
im_95 = sort(all_im)[min(end, round(Int, 0.90 * length(all_im)))]
re_lo = 0.0
re_hi = re_95 * 1.25
im_hi = im_95 * 1.25

pa = plot(; xlabel="Re(z)", ylabel="Im(z)",
           title="Yang–Lee zeros — all temperatures",
           xlims=(re_lo, re_hi), ylims=(0, im_hi),
           legend=:topright, framestyle=:box, size=(700, 500),
           tickfontsize=8, guidefontsize=10, legendfontsize=7)

for (idx, r) in enumerate(results)
    c  = colors[idx]
    pts = [(real(z), imag(z)) for z in r.roots if real(z) >= 0 && imag(z) >= 0]
    isempty(pts) && continue
    xs = first.(pts); ys = last.(pts)
    scatter!(pa, xs, ys;
             label        = r.label,
             color        = c,
             markerstrokecolor = c,
             markersize   = 3,
             markerstrokewidth = 0,
             alpha        = 0.65)
end
# Overlay leading zeros as larger filled symbols
for (idx, r) in enumerate(results)
    r.z1 === nothing && continue
    scatter!(pa, [real(r.z1)], [imag(r.z1)];
             label        = "",
             color        = colors[idx],
             markershape  = :star5,
             markersize   = 8,
             markerstrokecolor = :black,
             markerstrokewidth = 0.8,
             alpha        = 1.0)
end

# Panel (b): zoom on the leading zeros only — shows the T-dependence trajectory
if length(lead_re) >= 2
    re_lo_z = minimum(lead_re) * 0.98
    re_hi_z = maximum(lead_re) * 1.02
    im_hi_z = maximum(lead_im) * 3.0
    re_spread = max(re_hi_z - re_lo_z, im_hi_z * 0.5)
    re_ctr    = (re_lo_z + re_hi_z) / 2

    pb = plot(; xlabel="Re(z₁)", ylabel="Im(z₁)",
               title="Leading zeros  z₁(T)",
               xlims=(re_ctr - re_spread, re_ctr + re_spread),
               ylims=(0, im_hi_z),
               legend=false, framestyle=:box, size=(400, 400),
               tickfontsize=8, guidefontsize=10)

    # Draw a trajectory arrow connecting leading zeros in temperature order
    for i in 1:length(lead_re)-1
        plot!(pb, [lead_re[i], lead_re[i+1]], [lead_im[i], lead_im[i+1]];
              color=:grey70, linewidth=1, linestyle=:dash, label="")
    end

    for (idx, r) in enumerate(results)
        r.z1 === nothing && continue
        scatter!(pb, [real(r.z1)], [imag(r.z1)];
                 color=colors[idx], markershape=:circle, markersize=7,
                 markerstrokecolor=:black, markerstrokewidth=0.8, alpha=1.0)
        annotate!(pb, real(r.z1), imag(r.z1) + im_hi_z * 0.06,
                  Plots.text(r.label, 6, :center, colors[idx]))
    end
end

# Assemble final figure
if length(lead_re) >= 2
    fig = plot(pa, pb;
               layout   = @layout([a{0.62w} b]),
               size     = (1100, 480),
               left_margin  = 6Plots.mm,
               bottom_margin = 6Plots.mm)
else
    fig = pa
end

for ext in ("pdf", "png")
    path = joinpath(FIGURES_DIR, "yang_lee_all_temps.$ext")
    savefig(fig, path)
    println("  Saved: $path")
end

println()
println("Done.")
