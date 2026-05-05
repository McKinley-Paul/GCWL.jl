import Pkg
Pkg.activate("/Users/mckinleypaul/Documents/montecarlo/gc_wl")
using gc_wl
using Random
using Statistics
using JLD2

results_path = "/Users/mckinleypaul/Documents/montecarlo/gc_wl/data/gaussian_gas/error_investigation/9000run/error_stats_9000runs.jld2"
# adjust path as needed

results = JLD2.load(results_path, "results")

println("═"^70)
println("Results from $(results.n_runs) runs")
println("═"^70)

# Top-level scalars and vectors
for key in keys(results)
    val = getfield(results, key)
    if val isa AbstractDict
        println("\n── $key ──")
        for k in sort(collect(keys(val)))
            println("  [k=$k]: $(val[k])")
        end
    elseif val isa AbstractVector
        println("\n── $key (length $(length(val))) ──")
        for (i, x) in enumerate(val)
            println("  [$i] $x")
        end
    else
        println("\n── $key ──")
        println("  $val")
    end
end