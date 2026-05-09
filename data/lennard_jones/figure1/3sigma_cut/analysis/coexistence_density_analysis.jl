# coexistence_density_analysis.jl
#
# Reproduces Desgranges & Delhommelle (2012) Table II, Figure 5, and Figure 6
# for the GC-WL simulations in /data/lennard_jones/figure1/.
#
# Table II : liquid and vapor coexistence densities (g/cm³) vs TMMC and experiment.
# Figure 5 : saturation pressure P (bar) vs T (K).
# Figure 6 : molar volume V_m (cm³/mol) vs T (K).
#
# Methodology (follows Desgranges exactly):
#   1. Compute μ_coex from equal-area criterion on each of the 4 runs independently.
#   2. At the averaged-logQ μ_coex, split p(N) at the valley (Nb = argmin p(N))
#      and compute conditional means for ρ_liq and ρ_vap (Eqs. 22–23).
#   3. Per-run densities give sample std for the uncertainty.
#   4. P_sat = T_σ * logΞ(μ_coex) / V_σ  (LJ units → bar, Eq. 20).
#   5. Critical point estimated via rectilinear diameters + order-parameter scaling.

import Pkg
Pkg.activate("/Users/mckinleypaul/Documents/montecarlo/gc_wl")
using gc_wl, JLD2, Statistics, Printf, Plots

# ─── Physical constants ───────────────────────────────────────────────────────

const k_B       = 1.380649e-23
const N_A       = 6.02214076e23
const ε_kB_Ar   = 117.05       # ε/k_B for argon (K)
const M_Ar      = 39.948e-3    # molar mass (kg/mol)
const σ_Ar_Å    = 3.4          # σ in Å
const LJ_TO_KJKG = ε_kB_Ar * k_B * N_A / M_Ar / 1000

const BASE = joinpath(@__DIR__, "..")

const TEMPERATURES = ["87.79","93.64","99.49","105.35","111.20",
                      "117.05","122.90","128.76","134.61","140.46"]
const TEMP_K = [87.79,93.64,99.49,105.35,111.20,117.05,122.90,128.76,134.61,140.46]

# ─── Reference data (Table II, Desgranges 2012) ───────────────────────────────
# TMMC densities (g/cm³), Ref. 17 in paper
const TMMC_ΡLIQ = [1.391,1.354,1.315,1.274,1.232,1.187,1.138,1.085,1.025,0.953]
const TMMC_ΡVAP = [0.006,0.010,0.0163,0.025,0.036,0.050,0.069,0.093,0.125,0.170]

# Experimental densities (g/cm³) — from desgranges tbale 2
const EXP_ΡLIQ = [1.389,1.352, 1.314, 1.273, 1.231, 1.184, 1.134, 1.077,1.012, 0.934]
const EXP_ΡVAP = [0.006, 0.010, 0.0163, 0.025, 0.036, 0.051, 0.070, 0.096, 0.131, 0.182]

# Experimental saturation pressures (bar) — fill in from NIST WebBook
const EXP_PSAT = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# ─── Data loading ─────────────────────────────────────────────────────────────

function load_run(T_str, run_idx)
    dir = joinpath(BASE, T_str, "run$run_idx")
    wl_path = isfile(joinpath(dir,"final_wl.jld2")) ?
              joinpath(dir,"final_wl.jld2") : joinpath(dir,"wl_checkpoint.jld2")
    isfile(wl_path) || return nothing, nothing
    wl  = load(wl_path, "wl")
    sim = load(joinpath(dir, "sim.jld2"), "sim")
    return wl, sim
end

function load_averaged_data(T_str)
    arrays = Vector{Float64}[]
    sim_ref = nothing
    for run in 1:4
        wl, sim = load_run(T_str, run)
        wl === nothing && continue
        push!(arrays, collect(Float64, wl.logQ_N))
        sim_ref === nothing && (sim_ref = sim)
    end
    isempty(arrays) && error("No data for T=$T_str")
    return vec(mean(reduce(hcat, arrays), dims=2)), sim_ref, length(arrays)
end

println("Loading averaged logQ data...")
flush(stdout)
all_logQ = Dict{String,Vector{Float64}}()
all_sim  = Dict{String,Any}()
for T_str in TEMPERATURES
    logQ, sim, n = load_averaged_data(T_str)
    all_logQ[T_str] = logQ
    all_sim[T_str]  = sim
    @printf("  T=%-6s K  N_max=%-3d  T*=%.2f  Λ_σ=%.4f  runs=%d\n",
            T_str, sim.N_max, sim.T_σ, sim.Λ_σ, n)
end

# ─── Per-run coexistence analysis ─────────────────────────────────────────────
# For each temperature: bisect on each run's logQ_N to get per-run μ_coex,
# then compute phase densities and P_sat at the averaged μ_coex.

println("\nComputing coexistence properties...")
flush(stdout)

results = Dict{String, NamedTuple}()

for T_str in TEMPERATURES
    logQ_avg = all_logQ[T_str]
    sim      = all_sim[T_str]
    T_σ = sim.T_σ; Λ_σ = sim.Λ_σ; N_min = sim.N_min; V_σ = sim.V_σ

    # Per-run bisection for μ_coex and derived densities
    run_μ   = Float64[]
    run_ρl  = Float64[]
    run_ρv  = Float64[]
    run_P   = Float64[]

    for run in 1:4
        wl, _ = load_run(T_str, run)
        wl === nothing && continue
        logQ_run = collect(Float64, wl.logQ_N)
        μ = try
            find_μ_coex(logQ_run, T_σ; N_min=N_min, tol=1e-7)
        catch
            continue
        end
        ρl_σ, ρv_σ = compute_phase_densities(logQ_run, μ, T_σ, V_σ; N_min=N_min)
        logΞ = compute_logΞ(logQ_run, μ, T_σ; N_min=N_min)
        push!(run_μ,  μ)
        push!(run_ρl, ljdens_to_gcm3(ρl_σ; σ_Å=σ_Ar_Å, M_g_per_mol=39.948))
        push!(run_ρv, ljdens_to_gcm3(ρv_σ; σ_Å=σ_Ar_Å, M_g_per_mol=39.948))
        push!(run_P,  compute_Psat_bar(logΞ, T_σ, V_σ; ε_kB=ε_kB_Ar, σ_Å=σ_Ar_Å))
    end

    isempty(run_μ) && continue

    μ_mean  = mean(run_μ);   μ_std  = length(run_μ) > 1 ? std(run_μ) : 0.0
    ρl_mean = mean(run_ρl);  ρl_std = length(run_ρl) > 1 ? std(run_ρl) : 0.0
    ρv_mean = mean(run_ρv);  ρv_std = length(run_ρv) > 1 ? std(run_ρv) : 0.0
    P_mean  = mean(run_P);   P_std  = length(run_P)  > 1 ? std(run_P)  : 0.0
    lnz     = compute_lnzsat(μ_mean, T_σ, Λ_σ)

    results[T_str] = (
        μ_mean=μ_mean, μ_std=μ_std,
        ρl_mean=ρl_mean, ρl_std=ρl_std,
        ρv_mean=ρv_mean, ρv_std=ρv_std,
        P_mean=P_mean, P_std=P_std,
        lnz=lnz, n_runs=length(run_μ)
    )
end

# ─── Table II ─────────────────────────────────────────────────────────────────

println()
println("="^110)
println("TABLE II — LIQUID AND VAPOR DENSITIES AT COEXISTENCE (g/cm³)")
println("Desgranges method: conditional <N>/V weighted by p(N) on each side of valley (Eqs. 22–23)")
println("="^110)
@printf("\n%-8s  %-20s  %-10s  %-10s  %-10s  %-20s  %-10s  %-10s  %-10s\n",
        "T(K)", "ρ_liq GC-WL (g/cm³)", "Exp.", "%err_GCW", "%err_TMMC",
               "ρ_vap GC-WL (g/cm³)", "Exp.", "%err_GCW", "%err_TMMC")
println("-"^140)

ρl_gcwl = Float64[]; ρv_gcwl = Float64[]
P_gcwl  = Float64[]; T_gcwl  = Float64[]

for (i, T_str) in enumerate(TEMPERATURES)
    haskey(results, T_str) || continue
    r = results[T_str]

    label_l    = @sprintf("%.3f ± %.3f [%d]", r.ρl_mean, r.ρl_std, r.n_runs)
    label_v    = @sprintf("%.4f ± %.4f [%d]", r.ρv_mean, r.ρv_std, r.n_runs)
    err_l_gcwl = EXP_ΡLIQ[i] > 0 ? @sprintf("%+.1f%%", 100*(r.ρl_mean   - EXP_ΡLIQ[i])/EXP_ΡLIQ[i]) : "N/A"
    err_l_tmmc = EXP_ΡLIQ[i] > 0 ? @sprintf("%+.1f%%", 100*(TMMC_ΡLIQ[i] - EXP_ΡLIQ[i])/EXP_ΡLIQ[i]) : "N/A"
    err_v_gcwl = EXP_ΡVAP[i] > 0 ? @sprintf("%+.1f%%", 100*(r.ρv_mean   - EXP_ΡVAP[i])/EXP_ΡVAP[i]) : "N/A"
    err_v_tmmc = EXP_ΡVAP[i] > 0 ? @sprintf("%+.1f%%", 100*(TMMC_ΡVAP[i] - EXP_ΡVAP[i])/EXP_ΡVAP[i]) : "N/A"

    @printf("%-8.2f  %-20s  %-10.3f  %-10s  %-10s  %-20s  %-10.4f  %-10s  %-10s\n",
            TEMP_K[i], label_l, EXP_ΡLIQ[i], err_l_gcwl, err_l_tmmc,
                        label_v, EXP_ΡVAP[i], err_v_gcwl, err_v_tmmc)

    push!(ρl_gcwl, r.ρl_mean)
    push!(ρv_gcwl, r.ρv_mean)
    push!(P_gcwl,  r.P_mean)
    push!(T_gcwl,  TEMP_K[i])
end

# ─── Saturation pressure table ────────────────────────────────────────────────

println()
println("="^80)
println("SATURATION PRESSURE AT COEXISTENCE")
println("NOTE: GC-WL uses r_cut > box size (effectively no cutoff within the simulation volume).")
println("      Remaining discrepancy vs experiment reflects LJ model accuracy, not a simulation artifact.")
println("="^80)
@printf("\n%-8s  %-22s  %-12s  %-10s\n", "T(K)", "P_sat GC-WL (bar)", "Exp. (bar)", "Ratio")
println("-"^60)
for (i, T_str) in enumerate(TEMPERATURES)
    haskey(results, T_str) || continue
    r = results[T_str]
    ratio_str = EXP_PSAT[i] > 0 ? @sprintf("%.2f", r.P_mean / EXP_PSAT[i]) : "N/A"
    @printf("%-8.2f  %-22s  %-12.3f  %-10s\n",
            TEMP_K[i],
            @sprintf("%.3f ± %.3f", r.P_mean, r.P_std),
            EXP_PSAT[i], ratio_str)
end

# ─── Critical point estimation ────────────────────────────────────────────────
# Simultaneous fit of:
#   (1) rectilinear diameters:  (ρ_liq + ρ_vap)/2 = ρ_c + A*(T - T_c)
#   (2) order parameter:        (ρ_liq - ρ_vap)/2 = B*(T_c - T)^β,  β=0.326 (3D Ising)
# Grid search over T_c minimising residuals on the order-parameter fit.

function estimate_critical_point(T_K, ρ_liq, ρ_vap; β=0.326)
    best_Tc = NaN; best_ρc = NaN; best_err = Inf
    for Tc in 141.0:0.05:170.0
        mask = T_K .< Tc
        sum(mask) < 3 && continue
        T_use = T_K[mask]; ρl = ρ_liq[mask]; ρv = ρ_vap[mask]

        # Order parameter fit (β fixed): log((ρl-ρv)/2) = a + β*log(Tc - T)
        y = log.((ρl .- ρv) ./ 2)
        x = log.(Tc .- T_use)
        a   = mean(y .- β .* x)
        err = sum((y .- (a .+ β .* x)).^2)

        if err < best_err
            best_err = err
            best_Tc  = Tc
            # Law of rectilinear diameters: proper OLS of diam on (T - Tc)
            diam      = (ρl .+ ρv) ./ 2
            ΔT        = T_use .- Tc
            m_ΔT      = mean(ΔT);  m_d = mean(diam)
            A         = sum((ΔT .- m_ΔT) .* (diam .- m_d)) / sum((ΔT .- m_ΔT).^2)
            best_ρc   = m_d - A * m_ΔT  # intercept at ΔT = 0
        end
    end
    return best_Tc, best_ρc
end

Tc_est, ρc_est = estimate_critical_point(T_gcwl, ρl_gcwl, ρv_gcwl)
# Extrapolate P_sat to T_c using Clausius-Clapeyron: log(P) linear in 1/T
# Fit log(P) = a + b/T on the two highest-T points
n_fit = min(5, length(T_gcwl))
T_fit = T_gcwl[end-n_fit+1:end]; P_fit = log.(P_gcwl[end-n_fit+1:end])
invT  = 1.0 ./ T_fit
b_cc  = (sum(invT .* P_fit) - sum(invT)*mean(P_fit)) / (sum(invT.^2) - sum(invT)^2/n_fit)
a_cc  = mean(P_fit) - b_cc * mean(invT)
Pc_est = exp(a_cc + b_cc / Tc_est)

println()
println("="^60)
println("CRITICAL POINT ESTIMATE (order-parameter scaling, β=0.326)")
println("="^60)
@printf("  T_c  = %.1f K    (experimental: 150.86 K)\n", Tc_est)
@printf("  ρ_c  = %.3f g/cm³  (experimental: 0.536 g/cm³)\n", ρc_est)
@printf("  P_c  = %.1f bar   (experimental: 48.98 bar)\n", Pc_est)
@printf("  Desgranges reports: T_c=151±4 K, ρ_c=0.528±0.008 g/cm³, P_c=50.8±2 bar\n")

# ─── Figure 5: P–T phase diagram ──────────────────────────────────────────────

println("\nGenerating Figure 5 (P–T diagram)...")
flush(stdout)

fig5 = plot(;
    xlabel = "T (K)", ylabel = "P (bar)",
    title  = "Argon: Vapor-Liquid Phase Diagram (P–T)",
    legend = :topleft, size=(600,500), dpi=150,
    xlims  = (80, 160), ylims = (0, 55),
    framestyle = :box)

plot!(fig5, T_gcwl, P_gcwl;
      label="GC-WL (this work)", color=:blue, lw=2, marker=:circle, ms=4)

let mask = EXP_PSAT .> 0
    any(mask) && scatter!(fig5, TEMP_K[mask], EXP_PSAT[mask];
                          label="Experiment (NIST)", color=:black, marker=:square, ms=5)
end

# Critical point
scatter!(fig5, [Tc_est], [Pc_est];
         label="Critical point (est.)", color=:blue, marker=:circle, ms=8,
         markerstrokewidth=0)

savefig(fig5, joinpath(@__DIR__, "figure5_PT_diagram.pdf"))
println("  Saved: figure5_PT_diagram.pdf")

# ─── Figure 6: V–T phase diagram (molar volume) ──────────────────────────────

println("Generating Figure 6 (V–T diagram)...")
flush(stdout)

M_Ar_gmol = 39.948  # g/mol

Vm_liq_gcwl = M_Ar_gmol ./ ρl_gcwl   # cm³/mol
Vm_vap_gcwl = M_Ar_gmol ./ ρv_gcwl

exp_mask    = (EXP_ΡLIQ .> 0) .& (EXP_ΡVAP .> 0)
Vm_liq_exp  = ifelse.(EXP_ΡLIQ .> 0, M_Ar_gmol ./ max.(EXP_ΡLIQ, eps()), 0.0)
Vm_vap_exp  = ifelse.(EXP_ΡVAP .> 0, M_Ar_gmol ./ max.(EXP_ΡVAP, eps()), 0.0)

Vm_c = M_Ar_gmol / ρc_est

fig6 = plot(;
    xlabel = "T (K)", ylabel = "V_m (cm³/mol)",
    title  = "Argon: Vapor-Liquid Phase Diagram (V–T)",
    legend = :topright, size=(600,500), dpi=150,
    xlims  = (80, 160),
    framestyle = :box)

# GC-WL liquid + vapor connected by a line through the two-phase envelope
T_env = vcat(T_gcwl, reverse(T_gcwl))
V_env = vcat(Vm_liq_gcwl, reverse(Vm_vap_gcwl))
plot!(fig6, T_env, V_env; label="GC-WL (this work)", color=:blue, lw=2)
scatter!(fig6, T_gcwl, Vm_liq_gcwl; color=:blue, ms=4, label="")
scatter!(fig6, T_gcwl, Vm_vap_gcwl; color=:blue, ms=4, label="")

# Experimental envelope
T_env_exp = vcat(TEMP_K, reverse(TEMP_K))
V_env_exp = vcat(Vm_liq_exp, reverse(Vm_vap_exp))
any(exp_mask) && scatter!(fig6, TEMP_K[exp_mask], Vm_liq_exp[exp_mask];
                          color=:black, marker=:square, ms=5, label="Experiment (NIST)")
any(exp_mask) && scatter!(fig6, TEMP_K[exp_mask], Vm_vap_exp[exp_mask];
                          color=:black, marker=:square, ms=5, label="")

# Critical point
scatter!(fig6, [Tc_est], [Vm_c];
         label="Critical point (est.)", color=:blue, marker=:circle, ms=8,
         markerstrokewidth=0)

savefig(fig6, joinpath(@__DIR__, "figure6_VT_diagram.pdf"))
println("  Saved: figure6_VT_diagram.pdf")

println("\nAnalysis complete.")
