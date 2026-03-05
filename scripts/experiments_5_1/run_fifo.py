"""
Driver Arrival Rate Sensitivity Analysis: STRICT_FIFO Dispatching
===============================================================

This script analyzes system performance under strict FIFO dispatching across
a range of driver arrival rates. The experiment sweeps λ from 1 to 15.

"""

import numpy as np

import numpy as np
from os.path import dirname, abspath, join
import sys
sys.path.append(abspath(join(dirname(__file__), "../..")))

from src.setting_class import Setting
from src.system_class import System
from src.dispatching_class import Dispatching
from src.simulator_class import *
from src.reporting_class import *
from src.utilities import *
from src.get_economies import *


def main():
    """
    Main experimental function for driver arrival rate sensitivity analysis.
    """
    # =============================================================================
    # EXPERIMENTAL SETUP
    # =============================================================================
    
    print("="*80)
    print("STRICT FIFO DISPATCHING: DRIVER ARRIVAL RATE SENSITIVITY ANALYSIS")
    print("="*80)
    
    # Load calibrated economic parameters from O'Hare airport data
    print("Loading O'Hare economy parameters...")
    economy = get_ohare_economy()
    
    # Extract key economic parameters
    earnings = economy.w.tolist()           # Driver earnings by job type
    job_rates = economy.mu_jobs.tolist()    # Job arrival rates by type
    waiting_cost = economy.c                # Driver waiting cost
    patience = economy.patience             # Driver patience level
    
    # =============================================================================
    # EXPERIMENTAL CONFIGURATION
    # =============================================================================
    
    # System parameters
    reneging = 0                           # No reneging behavior
    dispatch_rule = "STRICT_FIFO"         # Strict FIFO job assignment
    
    # Algorithm parameters - optimized for high precision
    tolerance = 1e-4                       # Convergence tolerance
    max_iterations = 10000                 # Maximum iterations (high for precision)
    saving_multiple = 10000                # Save only final results
    algorithm = '0-MIX'               # Hybrid algorithm for best convergence
    
    # Reproducibility
    random_seed = 4
    
    # Experimental range
    arrival_rate_min = 1
    arrival_rate_max = 15
    total_experiments = arrival_rate_max - arrival_rate_min + 1
    
    print(f"\nExperimental Configuration:")
    print(f"  Dispatching Rule: {dispatch_rule}")
    print(f"  Algorithm: {algorithm}")
    print(f"  Convergence Tolerance: {tolerance}")
    print(f"  Maximum Iterations: {max_iterations:,}")
    print(f"  Arrival Rate Range: λ ∈ [{arrival_rate_min}, {arrival_rate_max}]")
    print(f"  Total Experiments: {total_experiments}")
    print(f"  Random Seed: {random_seed}")
    
    # =============================================================================
    # ARRIVAL RATE SENSITIVITY ANALYSIS
    # =============================================================================
    
    successful_runs = 0
    failed_runs = []
    
    for driver_arrival_rate in range(arrival_rate_min, arrival_rate_max + 1):
        
        # Set reproducible random seed for each experiment
        np.random.seed(random_seed)
        
        print(f"\n{'='*80}")
        print(f"EXPERIMENT {driver_arrival_rate - arrival_rate_min + 1}/{total_experiments}")
        print(f"Driver Arrival Rate (λ): {driver_arrival_rate}")
        print(f"{'='*80}")
        
        try:
            # -------------------------------------------------------------------------
            # Determine queue capacity for current arrival rate
            # -------------------------------------------------------------------------
            Qmax = ohare_get_Qmax(driver_arrival_rate, dispatch_rule)
            print(f"Queue capacity (Qmax): {Qmax}")
            
            # -------------------------------------------------------------------------
            # Initialize system configuration
            # -------------------------------------------------------------------------
            print("Initializing system configuration...")
            setting = Setting(
                earnings=earnings,
                job_rates=job_rates,
                driver_arrival_rate=driver_arrival_rate,
                waiting_cost=waiting_cost,
                patience=patience,
                Qmax=Qmax,
                dispatch_rule=dispatch_rule,
                reneging=reneging,
                initial_alpha='RAND'  # Random initial acceptance strategy
            )
            
            # -------------------------------------------------------------------------
            # Configure advanced algorithmic parameters
            # -------------------------------------------------------------------------
            setting.beta = 0                      # Momentum parameter
            setting.gamma_cnst = 10000                 # Base step size (acceptance)
            setting.gamma_exp = 0                   # Step size exponent (0 = constant)
            setting.gamma_phi_divide_by = 1       # Joining strategy step size divisor
            setting.momentum_start_iter = 20        # Momentum activation iteration
            setting.num_parallel_workers = 10       # Parallel processing threads
            setting.gamma_cnst_grad = 20            # Gradient phase step size
            
            print(f"Algorithm Parameters:")
            print(f"  Momentum (β): {setting.beta}")
            print(f"  Base step size: {setting.gamma_cnst}")
            print(f"  Gradient step size: {setting.gamma_cnst_grad}")
            print(f"  Parallel workers: {setting.num_parallel_workers}")
            
            # -------------------------------------------------------------------------
            # Execute simulation
            # -------------------------------------------------------------------------
            print(f"\nInitializing {algorithm} simulator...")
            system = System(setting)
            simulator = Simulator(system, max_iterations, tolerance, algorithm, saving_multiple)
            
            print("Running simulation...")
            print(f"  Target tolerance: {tolerance}")
            print(f"  Maximum iterations: {max_iterations:,}")
            
            # Run the simulation
            outcome_list, V, alpha, phi, num_iter, inspect, convergence_flag = simulator.run_iterations()
            
            # -------------------------------------------------------------------------
            # Save and report results
            # -------------------------------------------------------------------------
            filename = save_run_history(
                simulator.setting,
                outcome_list,
                simulator.version,
                setting.dispatching_rule, directory='completed_runs/strict_fifo'
            )
            
            # Report results
            print(f"\n{'─'*60}")
            print(f"RESULTS SUMMARY")
            print(f"{'─'*60}")
            print(f"Convergence: {'✓ SUCCESS' if convergence_flag else '⚠ MAX ITERATIONS'}")
            print(f"Iterations completed: {num_iter:,}")
            print(f"Final tolerance achieved: {simulator.current_tolerance if hasattr(simulator, 'current_tolerance') else 'N/A'}")
            print(f"Results saved to: {filename}")
            print(f"{'─'*60}")
            
            successful_runs += 1
            
        except Exception as e:
            print(f"\n❌ ERROR in experiment λ={driver_arrival_rate}")
            print(f"Error message: {str(e)}")
            failed_runs.append((driver_arrival_rate, str(e)))
            continue
    
    # =============================================================================
    # EXPERIMENTAL SUMMARY
    # =============================================================================
    
    print(f"\n{'='*80}")
    print("EXPERIMENTAL SUMMARY")
    print(f"{'='*80}")
    print(f"Total experiments: {total_experiments}")
    print(f"Successful runs: {successful_runs}")
    print(f"Failed runs: {len(failed_runs)}")
    
    if failed_runs:
        print(f"\nFailed experiments:")
        for arrival_rate, error in failed_runs:
            print(f"  λ={arrival_rate}: {error}")

    print(f"\nDispatching rule: {dispatch_rule}")
    print(f"Algorithm: {algorithm}")
    print(f"Results directory: ./completed_runs/strict_fifo/")
    
    print(f"{'='*80}")


if __name__ == "__main__":
    main()