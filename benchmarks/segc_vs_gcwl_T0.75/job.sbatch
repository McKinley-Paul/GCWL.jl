#!/bin/bash
# Slurm array job: 1/t GC-WL, LJ argon, T* = 0.75, V = 512σ³, N_max = 450; four independent runs (run1–run4).
# Submit from this directory:  sbatch job.sbatch
# The benchmark ran every run on one CPU model (Intel Xeon 6972P). Set the account, QOS and CPU constraint for your
# cluster. A run that reaches the time limit resumes from its checkpoint when resubmitted (sbatch --array=K job.sbatch).
#SBATCH -J gcwl_T0.75
#SBATCH --array=1-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=2-00:00:00
##SBATCH -A <account>
##SBATCH -q <qos>
##SBATCH -C <cpu feature>
#SBATCH -o run%a/slurm-%A_%a.out

export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR/run${SLURM_ARRAY_TASK_ID}" || exit 1
if [ -f DONE ]; then echo "run${SLURM_ARRAY_TASK_ID} already converged"; exit 0; fi
JULIA=${JULIA:-julia}
"$JULIA" --project=../../.. -t 1 main.jl >> julia_out.txt 2>&1
