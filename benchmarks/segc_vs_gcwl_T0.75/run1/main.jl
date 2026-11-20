# Benchmark run: 1/t GC-WL (this paper's method), Lennard-Jones argon, T* = 0.75 (87.79 K), V = 512 σ³ (L = 8σ), N_max = 450,
# r_c = 3σ (the long-range correction is applied afterwards, in the analysis), Λ = argon thermal wavelength.
# Schedule: 1/t two-phase schedule, stopping on phase-2 flatness min H / mean H > 0.8 (src/GCWL.jl).
#
# Run from this directory:   julia --project=../../.. -t 1 main.jl >> julia_out.txt 2>&1
# The script resumes automatically from wl_checkpoint.jld2 / microstate_checkpoint.jld2 (written every 10^6
# moves), so a job killed by the scheduler's time limit is continued by resubmitting it. Do not change the parameters between resubmissions.
import Pkg; Pkg.activate(joinpath(@__DIR__, "..", "..", ".."); io=devnull)
using GCWL, Random, Printf, Dates
cd(@__DIR__)

const RUN_ID = parse(Int, match(r"run(\d+)$", basename(@__DIR__))[1])
nrestart = isfile("restarts.txt") ? parse(Int, strip(read("restarts.txt", String))) + 1 : 0
const SEED = 1000 * (1000 + RUN_ID) + nrestart          # GC-WL seeds 1001xxx–1004xxx; new stream on every resume
write("restarts.txt", string(nrestart))

T_σ = 0.75
Λ_σ = argon_deBroglie(T_σ)
sim = SimulationParams(N_max = 450, N_min = 0, T_σ = T_σ, Λ_σ = Λ_σ, L_σ = 8.0, r_cut_σ = 3.0,
                       save_directory_path = @__DIR__, maxiter = 100_000_000_000_000, rng = MersenneTwister(SEED))

resumed = isfile("wl_checkpoint.jld2") && isfile("microstate_checkpoint.jld2")
if resumed
    wl = load_wanglandau_jld2("wl_checkpoint.jld2"); μstate = load_microstate_jld2("microstate_checkpoint.jld2")
    println("Resuming (restart $nrestart, seed $SEED) at iters = $(wl.iters), ln f = $(wl.logf), phase2 = $(wl.phase2)")
else
    μstate = init_microstate(sim)            # start from the vacuum, N = 0
    wl     = init_WangLandauVars(sim)
end
cache = init_cache(sim, μstate)
resumed || initialization_check(sim, μstate, wl)
flush(stdout)

jobid = get(ENV, "SLURM_JOB_ID", "local"); host = gethostname()
t0 = time(); start_stamp = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
run_simulation!(sim, μstate, wl, cache)
wall = time() - t0
open("timing_log.txt", "a") do io
    @printf(io, "job %s host %s start %s wall_s %.0f iters %d logf %.3e converged %s\n", jobid, host, start_stamp, wall, wl.iters, wl.logf, wl.flat)
end

if wl.flat                                    # 1/t GC-WL stopping criterion (phase-2 flatness) met
    post_run(sim, μstate, wl)
    correct_logQ!(wl)                         # Q(0) = 1
    open("logQ_final.csv", "w") do io
        println(io, "# method=GC-WL_1/t run=$RUN_ID T_star=$T_σ Lambda_sigma=$Λ_σ V=512 r_cut=3 LRC=not_applied iters=$(wl.iters)")
        println(io, "N,logQ")
        for N in 0:sim.N_max; @printf(io, "%d,%.17g\n", N, wl.logQ_N[N+1]); end
    end
    write("DONE", "converged $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS")) iters $(wl.iters)\n")
end
