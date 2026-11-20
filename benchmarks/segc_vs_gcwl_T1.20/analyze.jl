# Analysis of the SEGC-WL vs 1/t GC-WL benchmark (LJ argon, T* = 1.20, V = 512 σ³, N_max = 350, r_c = 3σ).
# Usage, from this directory:
#     julia --project=../.. analyze.jl /path/to/SEGC_WangLandau/benchmarks/segc_vs_gcwl_T1.20
# The run data of both methods are in the SEGC_WangLandau repository (https://github.com/McKinley-Paul/SEGC_WangLandau):
# the SEGC-WL runs in that directory and the 1/t GC-WL runs in its gcwl_reference/ subdirectory.
# Reads runK/logQ_final.csv (written by each runK/main.jl on convergence) for both methods and cpu_time.csv in each
# benchmark directory, applies the standard LJ long-range correction, locates
# coexistence by equal weight of the two peaks of P(N), and compares with NIST SAT-TMMC for the same system.
# Writes results.md next to this script.
import Pkg; Pkg.activate(joinpath(@__DIR__, "..", ".."); io=devnull)
using GCWL, Printf, Statistics

const T, V, RC = 1.20, 512.0, 3.0
const Λ = argon_deBroglie(T)
# NIST SAT-TMMC (GC-TMMC, Errington 2003), LJ r_c = 3σ with LRC, V = 512σ³ (67% confidence limits from five runs),
# https://mmlapps.nist.gov/srs/LJ_PURE/sattmmc.htm, the TMMC values of Table I of the paper. NIST's ln z_sat is ln y_sat here.
const NIST = (rho_vap = (1.00339e-1, 3.00e-5), rho_liq = (5.63158e-1, 5.22e-5), psat = (7.72251e-2, 1.42e-5), lnzsat = (-3.03072, 7.61e-5))

function read_logQ(path)
    rows = filter(l -> !startswith(l, "#") && !startswith(l, "N"), readlines(path))
    return [parse(Float64, split(r, ",")[2]) for r in rows]
end

# Phase split. GCWL's find_μ_coex / compute_phase_densities split P(N) at the valley between its two largest local
# maxima >= 10 apart. At T* = 1.2 the valley is shallow, and in these runs (ln Q noise 0.1-0.6) a bump inside the
# liquid peak was taken as the second peak, giving ρ_vap ≈ 0.4-0.6. Here the two peaks must be
# >= SEP = 100 apart (gas peak N ≈ 51, liquid N ≈ 288); otherwise the same equal-area rule and Nb definition.
const SEP = 100
split_idx(p) = GCWL.find_Nb_idx(p; min_separation = SEP)
function coex_μ(lq; μ_lo = -20.0, μ_hi = -5.0, tol = 1e-12)
    function area(μ)
        p = compute_pN(lq, μ, T); i = split_idx(p)
        i == 0 && return sum((j - 1) * p[j] for j in eachindex(p)) < (length(p) - 1) / 2 ? 1.0 : -1.0
        return 2sum(p[1:i]) - 1
    end
    flo = area(μ_lo)
    while μ_hi - μ_lo > tol
        m = (μ_lo + μ_hi) / 2; fm = area(m)
        flo * fm < 0 ? (μ_hi = m) : (μ_lo = m; flo = fm)
    end
    return (μ_lo + μ_hi) / 2
end

"Coexistence observables from one logQ vector (no LRC applied yet)."
function observables(logQ_raw)
    lq = apply_lrc_to_logQ(logQ_raw, T, V, RC)
    μc = coex_μ(lq)
    p = compute_pN(lq, μc, T); i = split_idx(p); N = 0:length(p)-1
    ρv = sum(N[1:i] .* p[1:i]) / (sum(p[1:i]) * V); ρl = sum(N[i+1:end] .* p[i+1:end]) / (sum(p[i+1:end]) * V)
    psat = T * (compute_logΞ(lq, μc, T) - log(2)) / V      # phase pressure: ln Ξ_gas = ln Ξ - ln 2 at equal weight
    return (lnzsat = compute_ln_y_sat(μc, T, Λ), rho_liq = ρl, rho_vap = ρv, psat = psat)
end

function read_cpu(dir)
    f = joinpath(dir, "cpu_time.csv"); isfile(f) || return Dict{Int,Float64}()
    d = Dict{Int,Float64}()
    for l in readlines(f)[2:end]
        p = split(l, ","); d[parse(Int, p[1])] = parse(Float64, p[2])
    end
    return d
end

function method_block(name, dir)
    runs = sort(filter(k -> isfile(joinpath(dir, "run$k", "logQ_final.csv")), 1:4))
    obs = [observables(read_logQ(joinpath(dir, "run$k", "logQ_final.csv"))) for k in runs]
    cpu = read_cpu(dir)
    return (name = name, runs = runs, obs = obs, cpu = [get(cpu, k, NaN) for k in runs])
end

ms(x) = length(x) > 1 ? @sprintf("%.6g ± %.2g", mean(x), std(x)) : (length(x) == 1 ? @sprintf("%.6g", x[1]) : "—")
"Deviation of the mean from NIST: relative (%) and in units of the combined uncertainty √(σ_NIST² + sem²)."
function dev(x, ref)
    isempty(x) && return "—"
    d = mean(x) - ref[1]; sem = length(x) > 1 ? std(x) / sqrt(length(x)) : 0.0
    return @sprintf("%+.3f%% (%.1fσ)", 100d / abs(ref[1]), d / sqrt(ref[2]^2 + sem^2))
end

length(ARGS) ≥ 1 || error("usage: julia --project=../.. analyze.jl /path/to/SEGC_WangLandau/benchmarks/segc_vs_gcwl_T1.20")
segc_dir = ARGS[1]
blocks = [method_block("1/t GC-WL", joinpath(segc_dir, "gcwl_reference"))]
push!(blocks, method_block("SEGC-WL (M = 100)", segc_dir))

io = IOBuffer()
println(io, "<!-- VS Code preview: ⇧⌘V (Cmd+Shift+V) in this tab, or ⌘K then V for side by side -->\n")
println(io, "# SEGC-WL vs 1/t GC-WL: LJ argon, T* = 1.20, V = 512σ³, N_max = 350, r_c = 3σ + LRC\n")
println(io, "Mean ± sample standard deviation over the four runs of each method. CPU-hours = wall time × 1 core summed over every job of a run (cpu_time.csv). ln y_sat here is the saturation activity called ln z_sat by NIST and by Desgranges and Delhommelle.\n")
println(io, "| Method | runs | CPU-hours per run (mean; min–max) | ln y_sat | ρ_liq | ρ_vap | P_sat |")
println(io, "|---|---|---|---|---|---|---|")
for b in blocks
    c = filter(!isnan, b.cpu)
    cs = isempty(c) ? "—" : @sprintf("%.1f; %.1f–%.1f", mean(c), minimum(c), maximum(c))
    g(f) = [getfield(o, f) for o in b.obs]
    @printf(io, "| %s | %d | %s | %s | %s | %s | %s |\n", b.name, length(b.runs), cs, ms(g(:lnzsat)), ms(g(:rho_liq)), ms(g(:rho_vap)), ms(g(:psat)))
end
@printf(io, "| NIST SAT-TMMC | 5 | — | %.4f ± %.1g | %.5f ± %.1g | %.4e ± %.1g | %.4e ± %.1g |\n\n",
        NIST.lnzsat..., NIST.rho_liq..., NIST.rho_vap..., NIST.psat...)
println(io, "Deviation of the mean from NIST, relative and in units of the combined uncertainty √(σ_NIST² + sem²):\n")
println(io, "| Method | ln y_sat | ρ_liq | ρ_vap | P_sat |"); println(io, "|---|---|---|---|---|")
for b in blocks
    g(f) = [getfield(o, f) for o in b.obs]
    @printf(io, "| %s | %s | %s | %s | %s |\n", b.name, dev(g(:lnzsat), NIST.lnzsat), dev(g(:rho_liq), NIST.rho_liq), dev(g(:rho_vap), NIST.rho_vap), dev(g(:psat), NIST.psat))
end
# relative uncertainty (run-to-run sd / |mean|; NIST: its 67% limit / |value|) and relative error of the mean vs TMMC
relunc(x) = length(x) > 1 ? @sprintf("%.3g%%", 100std(x) / abs(mean(x))) : "—"
relerr(x, ref) = isempty(x) ? "—" : @sprintf("%.3g%%", 100abs(mean(x) - ref[1]) / abs(ref[1]))
println(io, "\nRelative uncertainty (standard deviation over runs / |mean|) and relative error of the mean vs NIST SAT-TMMC, |mean − TMMC|/|TMMC|:\n")
println(io, "| Method | quantity | ln y_sat | ρ_liq | ρ_vap | P_sat |"); println(io, "|---|---|---|---|---|---|")
for b in blocks
    g(f) = [getfield(o, f) for o in b.obs]
    @printf(io, "| %s | rel. uncertainty | %s | %s | %s | %s |\n", b.name, relunc(g(:lnzsat)), relunc(g(:rho_liq)), relunc(g(:rho_vap)), relunc(g(:psat)))
    @printf(io, "| %s | rel. error vs TMMC | %s | %s | %s | %s |\n", b.name, relerr(g(:lnzsat), NIST.lnzsat), relerr(g(:rho_liq), NIST.rho_liq), relerr(g(:rho_vap), NIST.rho_vap), relerr(g(:psat), NIST.psat))
end
@printf(io, "| NIST SAT-TMMC | rel. uncertainty | %.3g%% | %.3g%% | %.3g%% | %.3g%% |\n", (100 .* [NIST.lnzsat[2] / abs(NIST.lnzsat[1]), NIST.rho_liq[2] / NIST.rho_liq[1], NIST.rho_vap[2] / NIST.rho_vap[1], NIST.psat[2] / NIST.psat[1]])...)
println(io, "\nPer run:\n"); println(io, "| Method | run | CPU-h | ln y_sat | ρ_liq | ρ_vap | P_sat |"); println(io, "|---|---|---|---|---|---|---|")
for b in blocks, (k, o, c) in zip(b.runs, b.obs, b.cpu)
    @printf(io, "| %s | %d | %.2f | %.5f | %.5f | %.5e | %.5e |\n", b.name, k, c, o.lnzsat, o.rho_liq, o.rho_vap, o.psat)
end
s = String(take!(io)); print(s); write(joinpath(@__DIR__, "results.md"), s)
