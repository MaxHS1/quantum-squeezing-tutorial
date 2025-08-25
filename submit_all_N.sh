#!/bin/bash
# submit_all_N.sh - Submit jobs for N=2 through N=8

echo "Submitting spin squeezing simulations for Figure 5 reproduction"
echo "================================================"

# Loop through N values and both d values
for N in 2 3 4 5 6 7 8; do
    for D_MHZ in 0 1; do
        echo "Submitting: N=$N, d/(2π)=$D_MHZ MHz"
        
        # Create a custom batch script for this specific job
        cat > temp_N${N}_d${D_MHZ}.sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=spin_N${N}_d${D_MHZ}
#SBATCH --account=xyz_group          
#SBATCH --partition=main
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=N${N}_d${D_MHZ}_%j.log

module load python/3.11.9

export N_SPINS=$N
export D_MHZ=$D_MHZ

echo "Running: N=\$N_SPINS, d/(2π)=\$D_MHZ MHz"
python run_spin_squeezing_hpc.py
EOF
        
        # Submit the job
        sbatch temp_N${N}_d${D_MHZ}.sbatch
        
        # Small delay to avoid overwhelming the scheduler
        sleep 0.5
    done
done

echo "================================================"
echo "All jobs submitted! Monitor with: squeue -u $USER"