#!/usr/bin/env python3
"""
run_simulation.py - HPC Test Script for Spin Squeezing Calculations

This script demonstrates how to run simulations on the HPC cluster.
It reads the N_SPINS environment variable to determine the system size,
allowing the same script to be used for different configurations.
"""

import os
import sys
import numpy as np
import time

def calculate_sql_sigma(N):
    """Calculate the Standard Quantum Limit (SQL) for N spins.
    
    SQL represents the minimum uncertainty achievable with 
    uncorrelated spins: σ_SQL = 1/√N
    """
    return 1.0 / np.sqrt(N)

def main():
    # Read the number of spins from environment variable
    # This allows the SLURM script to control what we simulate
    try:
        N = int(os.environ.get('N_SPINS', 3))  # Default to 3 if not set
    except ValueError:
        print("Error: N_SPINS must be an integer")
        sys.exit(1)
    
    print(f"="*50)
    print(f"Starting simulation for N = {N} spins")
    print(f"Job ID: {os.environ.get('SLURM_JOB_ID', 'local')}")
    print(f"="*50)
    
    # Calculate SQL for this system size
    sql = calculate_sql_sigma(N)
    print(f"Standard Quantum Limit (SQL) for N={N}: {sql:.4f}")
    
    # Simulate some work (replace with actual DH_spin_squeezing_sim.py call)
    print(f"Simulating spin dynamics...")
    time.sleep(2)  # Placeholder for actual calculation
    
    # In practice, you would run your actual simulation here:
    # from DH_spin_squeezing_sim import SpinSqueezingSimulator
    # simulator = SpinSqueezingSimulator(N_spins=N, ...)
    # results = simulator.comprehensive_analysis(...)
    
    # Save results to a file
    output_file = f"results_N{N}.txt"
    with open(output_file, 'w') as f:
        f.write(f"N_spins,SQL_sigma\n")
        f.write(f"{N},{sql:.6f}\n")
    
    print(f"Results saved to: {output_file}")
    print(f"Simulation complete for N = {N}")

if __name__ == "__main__":
    main()