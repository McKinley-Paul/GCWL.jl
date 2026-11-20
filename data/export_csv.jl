# Writes runK/logQ.csv next to every final_wl.jld2 under data/: ln Q(N, V, T) for N = 0…N_max as found by the
# simulation, normalized so that Q(0) = 1. No long-range correction is applied (the LJ analysis adds it with
# apply_lrc_to_logQ; the square well has none). Run from the repository root:  julia --project=. data/export_csv.jl
using GCWL, JLD2, Printf

for (root, _, files) in walkdir(@__DIR__)
    "final_wl.jld2" in files || continue
    wl = load_wanglandau_jld2(joinpath(root, "final_wl.jld2"))
    lq = collect(Float64, wl.logQ_N) .- wl.logQ_N[1]
    rel = relpath(root, @__DIR__)
    open(joinpath(root, "logQ.csv"), "w") do io
        println(io, "# $rel: ln Q(N,V,T) with Q(0) = 1, no long-range correction; MC moves = $(wl.iters), final ln f = $(wl.logf)")
        println(io, "N,logQ")
        for (i, x) in enumerate(lq)
            @printf(io, "%d,%.17g\n", i - 1, x)
        end
    end
end
