#!/bin/bash

# ============================================
# SLURM Job Configuration Section
# ============================================
# These lines starting with #SBATCH are directives for the job scheduler.
# They tell SLURM how to run your job and what resources you need.

#SBATCH --job-name=spin_test_N3      # A descriptive name for your job
#SBATCH --account= shaas_31          # Our research group's account goes here. Please reach out to Haas to find out the name of the account for his group!
#SBATCH --partition=main              # Queue for short jobs (< 2 hours)
#SBATCH --time=00:30:00              # Maximum runtime (HH:MM:SS)
#SBATCH --nodes=1                    # Number of compute nodes
#SBATCH --ntasks=1                   # Number of tasks (usually 1 for serial jobs)
#SBATCH --cpus-per-task=1           # CPU cores per task
#SBATCH --mem=4G                    # Memory allocation
#SBATCH --output=job_%j.log         # Output file (%j = Job ID)

# ============================================
# Job Execution Section
# ============================================
# Everything below runs on the compute node when your job starts

echo "=========================================="
echo "Job Information:"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Job Name: $SLURM_JOB_NAME"
echo "  Node: $(hostname)"
echo "  Start Time: $(date)"
echo "=========================================="

# Load required software modules
module load python/3.11.9

# Set the number of spins for this simulation
# This environment variable will be read by our Python script
export N_SPINS=3

# Run the simulation
echo "Starting simulation for N = $N_SPINS spins..."
python run_simulation.py

# Check if the simulation was successful
if [ $? -eq 0 ]; then
    echo "✓ Simulation completed successfully"
else
    echo "✗ Simulation failed with error code $?"
fi

echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="