# threshold_analysis.jl
#
# Addresses two questions:
#
# (1) Ambiguity in peak definition for the equal-area coexistence criterion.
#
#     Current approach (analyze_figure1.jl):
#       Find N_b = argmin p(N) in the interior of the N range ("valley splitting").
#       Area_vap = sum p(N) for N <= N_b, Area_liq = sum p(N) for N > N_b.
#       This is also what Desgranges Eqs. 22–23 specify explicitly.
#       Bisect on μ until Area_vap = Area_liq = 0.5 exactly.
#
#     Alternative approach tested here ("threshold"):
#       Define each peak as the connected region where p(N) > f * max(p(N)).
#       Only integrate over those regions; exclude the exponentially suppressed
#       inter-phase valley entirely.
#       Bisect on μ until peak1_area = peak2_area.
#
#     For a deep first-order transition the valley is so suppressed that the
#     two methods should agree; the threshold approach is more robust when the
#     valley is shallow or ambiguous.
#
# (2) At the TMMC (and SEGC-WL) reference μ_coex values, does our GC-WL p(N)
#     actually show two peaks and are their areas equal?
#     If not, that quantifies the systematic offset.

import Pkg
Pkg.activate("/Users/mckinleypaul/Documents/montecarlo/gc_wl")
using gc_wl, JLD2, Statistics, Printf

# ─── Physical constants ───────────────────────────────────────────────────────

const k_B       = 1.380649e-23
const N_A       = 6.02214076e23
const ε_kB_Ar   = 117.05
const M_Ar      = 39.948e-3
const LJ_TO_KJKG = ε_kB_Ar * k_B * N_A / M_Ar / 1000

const BASE = joinpath(@__DIR__, "..")

const TEMPERATURES = ["87.79","93.64","99.49","105.35","111.20",
                      "117.05","122.90","128.76","134.61","140.46"]
const TEMP_K = [87.79,93.64,99.49,105.35,111.20,117.05,122.90,128.76,134.61,140.46]

# Reference Table I (Desgranges 2012)
const REF_SEGC_MUCX   = [-237.68,-245.87,-254.44,-263.41,-272.73,
                          -282.35,-292.16,-302.32,-312.65,-323.27]
const REF_SEGC_LNZSAT = [-5.684,-5.194,-4.775,-4.415,-4.104,
                          -3.833,-3.592,-3.382,-3.193,-3.028]
const REF_TMMC_LNZSAT = [-5.683,-5.196,-4.779,-4.419,-4.107,
                          -3.834,-3.595,-3.384,-3.197,-3.031]

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
    @printf("  T=%-6s K  N_max=%-3d  T*=%.2f  runs=%d\n",
            T_str, sim.N_max, sim.T_σ, n)
end

# ─── Threshold-based peak finder ──────────────────────────────────────────────
#
# For a given p(N) and relative threshold f:
#   - mark all N where p(N) > f * max(p(N)) as "in a peak"
#   - collect contiguous in-peak regions
#   - return metadata for each region

struct Peak
    start_idx :: Int     # 1-based index into pN
    stop_idx  :: Int
    N_start   :: Int     # physical N value
    N_stop    :: Int
    N_peak    :: Int     # physical N at max of p(N) within region
    max_prob  :: Float64
    area      :: Float64 # sum p(N) over region
end

function find_peaks_threshold(pN::AbstractVector{Float64}, N_min::Int;
                               rel_threshold::Float64 = 1e-5)
    thresh = rel_threshold * maximum(pN)
    peaks  = Peak[]
    start  = nothing
    for i in eachindex(pN)
        if pN[i] > thresh && start === nothing
            start = i
        elseif (pN[i] <= thresh || i == lastindex(pN)) && start !== nothing
            stop   = (pN[i] > thresh && i == lastindex(pN)) ? i : i - 1
            region = pN[start:stop]
            lm     = argmax(region)
            push!(peaks, Peak(start, stop,
                               N_min + start - 1, N_min + stop - 1,
                               N_min + start + lm - 2,
                               region[lm], sum(region)))
            start = nothing
        end
    end
    return peaks
end

# ─── Threshold equal-area bisection ──────────────────────────────────────────

function area_diff_threshold(logQ_N, μ_star, T_σ, N_min; rel_threshold=1e-5)
    pN    = compute_pN(logQ_N, μ_star, T_σ; N_min=N_min)
    peaks = find_peaks_threshold(pN, N_min; rel_threshold=rel_threshold)
    length(peaks) == 2 || return NaN
    return peaks[1].area - peaks[2].area  # >0 = vapor-dominated
end

function find_μ_coex_threshold(logQ_N, T_σ;
                                N_min=0, μ_lo=-20.0, μ_hi=-5.0,
                                rel_threshold=1e-5, tol=1e-7, max_iter=300)
    # Coarse scan to locate the bracket where two peaks exist and area_diff changes sign
    μ_scan = range(μ_lo, μ_hi, length=400)
    f_scan = [area_diff_threshold(logQ_N, μ, T_σ, N_min; rel_threshold=rel_threshold)
              for μ in μ_scan]

    valid = findall(!isnan, f_scan)
    length(valid) < 2 && error("No two-peak region found in [$μ_lo, $μ_hi]")

    # Find first sign change within the two-peak region
    bl = nothing; bh = nothing; fl = nothing; fh = nothing
    for k in 1:length(valid)-1
        i, j = valid[k], valid[k+1]
        if f_scan[i] * f_scan[j] < 0
            bl, bh = μ_scan[i], μ_scan[j]
            fl, fh = f_scan[i], f_scan[j]
            break
        end
    end
    bl === nothing && error("No sign change in area_diff within two-peak region")

    # Bisection on the refined bracket
    for _ in 1:max_iter
        μ_mid = (bl + bh) / 2
        abs(bh - bl) < tol && return μ_mid
        f_mid = area_diff_threshold(logQ_N, μ_mid, T_σ, N_min; rel_threshold=rel_threshold)
        if isnan(f_mid) || fl * f_mid < 0
            bh = μ_mid; fh = isnan(f_mid) ? fh : f_mid
        else
            bl = μ_mid; fl = f_mid
        end
    end
    return (bl + bh) / 2
end

# ─── Part 1: compare threshold vs valley μ_coex ──────────────────────────────

println("\n" * "="^100)
println("PART 1 — THRESHOLD vs. VALLEY APPROACH TO EQUAL-AREA COEXISTENCE")
println("Relative threshold: p(N) > 1e-5 * max(p(N)) defines a peak region")
println("Valley: splits at argmin p(N) in interior [20%, 80%] of N range (current approach)")
println("="^100)
@printf("\n%-8s  %-16s  %-12s  %-16s  %-12s  %-10s  %-10s\n",
        "T(K)", "μ threshold", "lnz thresh", "μ valley", "lnz valley",
        "Δμ(T-V)", "Δlnz(T-V)")
@printf("%-8s  %-16s  %-12s  %-16s  %-12s  %-10s  %-10s\n",
        "", "(kJ/kg)", "", "(kJ/kg)", "", "(kJ/kg)", "")
println("-"^100)

for (i, T_str) in enumerate(TEMPERATURES)
    logQ  = all_logQ[T_str]
    sim   = all_sim[T_str]
    T_σ   = sim.T_σ;  Λ_σ = sim.Λ_σ;  N_min = sim.N_min

    # Threshold approach
    μt = NaN; lnzt = NaN
    try
        μt   = find_μ_coex_threshold(logQ, T_σ; N_min=N_min, rel_threshold=1e-5, tol=1e-7)
        lnzt = compute_lnzsat(μt, T_σ, Λ_σ)
    catch e
        @warn "T=$T_str threshold failed: $e"
    end

    # Valley approach (gc_wl.jl find_μ_coex)
    μv = NaN; lnzv = NaN
    try
        μv   = find_μ_coex(logQ, T_σ; N_min=N_min, tol=1e-7)
        lnzv = compute_lnzsat(μv, T_σ, Λ_σ)
    catch e
        @warn "T=$T_str valley failed: $e"
    end

    Δμ   = isnan(μt) || isnan(μv) ? NaN : (μt - μv) * LJ_TO_KJKG
    Δlnz = isnan(lnzt) || isnan(lnzv) ? NaN : lnzt - lnzv

    @printf("%-8.2f  %-16.3f  %-12.4f  %-16.3f  %-12.4f  %-10.4f  %-10.4f\n",
            TEMP_K[i],
            isnan(μt) ? NaN : μt * LJ_TO_KJKG, lnzt,
            isnan(μv) ? NaN : μv * LJ_TO_KJKG, lnzv,
            Δμ, Δlnz)
end

# ─── Part 2: p(N) peak areas at reference μ_coex values ─────────────────────

function peak_report(pN, N_min; rel_threshold=1e-5)
    peaks = find_peaks_threshold(pN, N_min; rel_threshold=rel_threshold)
    if length(peaks) == 0
        return "no peaks above threshold"
    elseif length(peaks) == 1
        return @sprintf("1 peak: N_peak=%d  area=%.6f", peaks[1].N_peak, peaks[1].area)
    elseif length(peaks) == 2
        ratio = peaks[1].area / (peaks[1].area + peaks[2].area)
        return @sprintf("2 peaks: N_vap=%d (%.4f)  N_liq=%d (%.4f)  A_vap%%=%.1f%%",
                        peaks[1].N_peak, peaks[1].area,
                        peaks[2].N_peak, peaks[2].area,
                        100*ratio)
    else
        return @sprintf("%d peaks (fragmented; threshold may be too high)", length(peaks))
    end
end

println("\n" * "="^100)
println("PART 2 — p(N) PEAK AREAS AT REFERENCE μ_coex VALUES")
println()
println("For each reference method (TMMC and SEGC-WL), back-compute μ_coex from")
println("their reported ln(z_sat) via:  μ* = (lnz + 3 ln Λ_σ) * T_σ")
println("Then evaluate our GC-WL p(N) at that μ and check whether the peaks are equal.")
println()
println("If A_vap% >> 50%: our logQ favors vapor at the reference μ →")
println("  our equal-area μ must shift to less negative (higher) to balance peaks")
println("  → our μ_coex is less negative than TMMC's  (consistent with Table 1)")
println("="^100)

for (label, lnzsat_ref) in [("TMMC", REF_TMMC_LNZSAT), ("SEGC-WL", REF_SEGC_LNZSAT)]
    println("\n── At $label μ_coex ──")
    @printf("%-8s  %-12s  %-10s  %s\n", "T(K)", "μ_ref (kJ/kg)", "μ* (red.)", "Peak areas from our GC-WL p(N)")
    println("-"^90)

    for (i, T_str) in enumerate(TEMPERATURES)
        logQ  = all_logQ[T_str]
        sim   = all_sim[T_str]
        T_σ   = sim.T_σ;  Λ_σ = sim.Λ_σ;  N_min = sim.N_min

        # Back-compute reference μ_coex from ln(z_sat)
        lnz_ref    = lnzsat_ref[i]
        μ_star_ref = (lnz_ref + 3*log(Λ_σ)) * T_σ

        pN     = compute_pN(logQ, μ_star_ref, T_σ; N_min=N_min)
        report = peak_report(pN, N_min; rel_threshold=1e-5)

        @printf("%-8.2f  %-12.2f  %-10.4f  %s\n",
                TEMP_K[i], μ_star_ref * LJ_TO_KJKG, μ_star_ref, report)
    end
end

# ─── Part 3: sensitivity to threshold choice ──────────────────────────────────

println("\n" * "="^100)
println("PART 3 — SENSITIVITY OF μ_coex TO THRESHOLD CHOICE")
println("Tests multiple thresholds; near-zero sensitivity confirms the valley is")
println("deeply suppressed and the threshold choice is immaterial.")
println("="^100)
@printf("\n%-8s", "T(K)")
for f in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8]
    @printf("  %-12s", "f=$(f)")
end
println("  (ln z_sat values)")
println("-"^120)

for (i, T_str) in enumerate(TEMPERATURES)
    logQ  = all_logQ[T_str]
    sim   = all_sim[T_str]
    T_σ   = sim.T_σ;  Λ_σ = sim.Λ_σ;  N_min = sim.N_min

    @printf("%-8.2f", TEMP_K[i])
    for f in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8]
        lnz = NaN
        try
            μ   = find_μ_coex_threshold(logQ, T_σ; N_min=N_min, rel_threshold=f, tol=1e-7)
            lnz = compute_lnzsat(μ, T_σ, Λ_σ)
        catch
        end
        @printf("  %-12s", isnan(lnz) ? "FAIL" : @sprintf("%.4f", lnz))
    end
    println()
end

println("\nAnalysis complete.")
