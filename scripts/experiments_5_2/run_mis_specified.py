"""
Job Rate Sensitivity Analysis: DYNAMIC_RAND_FIFO vs DIRECT_FIFO
================================================================

This script analyzes system performance under two dispatching strategies across
varying job arrival rates. The experiment sweeps normalized job rates μ from 1 to 20
while maintaining proportional job type distributions.

Part 1: DYNAMIC_RAND_FIFO dispatching (dynamic priority with randomization)
Part 2: DIRECT_FIFO dispatching (traditional first-in-first-out)

"""

import numpy as np
import copy
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.setting_class import Setting
from src.system_class import System
from src.simulator_class import Simulator
from src.utilities import save_run_history
from src.get_economies import get_ohare_economy


def get_qmax_dynamic_rand(mu: int) -> int:
    """
    Determine queue capacity for DYNAMIC_RAND_FIFO based on arrival rate.
    
    Args:
        mu: Job arrival rate
        
    Returns:
        Maximum queue length (Qmax)
    """
    if mu <= 14:
        return 800
    elif mu == 15:
        return 900
    elif mu == 16:
        return 1000
    elif mu == 17:
        return 1000
    elif mu == 18:
        return 1100
    elif mu == 19:
        return 1100
    elif mu == 20:
        return 1200
    else:
        return 800  # Default


def get_qmax_direct_fifo(mu: int) -> int:
    """
    Determine queue capacity for DIRECT_FIFO based on arrival rate.
    
    Args:
        mu: Job arrival rate
        
    Returns:
        Maximum queue length (Qmax)
    """
    if mu <= 9:
        return 500
    elif mu == 10:
        return 600
    elif mu == 11:
        return 700
    elif mu == 12:
        return 700
    else:  # mu >= 13
        return 800

# WARNING: The following experiments require 80GB of peak memory 
def main():
    """
    Main experimental function for job rate sensitivity analysis.
    """
    # =============================================================================
    # EXPERIMENTAL SETUP
    # =============================================================================
    
    print("="*80)
    print("JOB RATE SENSITIVITY ANALYSIS: DYNAMIC_RAND_FIFO vs DIRECT_FIFO")
    print("="*80)
    
    # Load calibrated economic parameters from O'Hare airport data
    print("Loading O'Hare economy parameters...")
    economy = get_ohare_economy()
    
    # Extract key economic parameters
    earnings = economy.w.tolist()           # Driver earnings by job type
    job_rates_base = economy.mu_jobs       # Base job arrival rates by type (unnormalized)
    waiting_cost = economy.c                # Driver waiting cost
    patience = economy.patience             # Driver patience level
    
    # =============================================================================
    # EXPERIMENTAL CONFIGURATION
    # =============================================================================
    
    # System parameters
    reneging = 0                           # No reneging behavior
    
    # Algorithm parameters
    tolerance = 1e-4                       # Convergence tolerance
    max_iterations = 10000                  # Maximum iterations
    saving_multiple = 100                  # Save every 100 iterations
    algorithm = '0-MIX-GRAD'                    # Mixed algorithm
    
    # Experimental range
    mu_min = 1
    mu_max = 20
    total_experiments = mu_max - mu_min + 1
    
    print(f"\nExperimental Configuration:")
    print(f"  Algorithm: {algorithm}")
    print(f"  Convergence Tolerance: {tolerance}")
    print(f"  Maximum Iterations: {max_iterations:,}")
    print(f"  Job Rate Range: μ ∈ [{mu_min}, {mu_max}]")
    print(f"  Total Experiments per dispatching rule: {total_experiments}")
    
    # =============================================================================
    # PART 1: DYNAMIC_RAND_FIFO DISPATCHING
    # =============================================================================
    
    print(f"\n{'='*80}")
    print("PART 1: DYNAMIC_RAND_FIFO DISPATCHING")
    print(f"{'='*80}")
    
    dispatching_rule = "DYNAMIC_RAND_FIFO"
    successful_runs_dynamic = 0
    failed_runs_dynamic = []
    
    # WARNING: The following experiments require 80GB of peak memory 
    for mu in range(mu_min, mu_max + 1):
        
        print(f"\n{'='*80}")
        print(f"EXPERIMENT {mu}/{mu_max} - DYNAMIC_RAND_FIFO")
        print(f"Normalized Job Rate (μ): {mu}")
        print(f"{'='*80}")
        
        try:
            # -------------------------------------------------------------------------
            # Determine queue capacity for current job rate
            # -------------------------------------------------------------------------
            Qmax = get_qmax_dynamic_rand(mu)
            print(f"Queue capacity (Qmax): {Qmax}")
            
            # -------------------------------------------------------------------------
            # Normalize job rates to sum to μ
            # -------------------------------------------------------------------------
            driver_arrival_rate = mu
            job_rates_normalized = (job_rates_base / np.sum(job_rates_base) * mu).tolist()
            print(f"Total job rate (verification): {np.sum(job_rates_normalized):.4f}")
            
            # -------------------------------------------------------------------------
            # Initialize system configuration
            # -------------------------------------------------------------------------
            print("Initializing system configuration...")
            setting = Setting(
                earnings=earnings,
                job_rates=job_rates_normalized,
                driver_arrival_rate=driver_arrival_rate,
                waiting_cost=waiting_cost,
                patience=patience,
                Qmax=Qmax,
                dispatch_rule=dispatching_rule,
                reneging=reneging,
                initial_alpha='1'
            )
            
            # -------------------------------------------------------------------------
            # Configure advanced algorithmic parameters
            # -------------------------------------------------------------------------
            setting.gamma_cnst = 10000               # Base step size (acceptance)
            setting.gamma_exp = 0                   # Step size exponent (0 = constant)
            setting.gamma_phi_divide_by = 1        # Joining strategy step size divisor
            setting.momentum_start_iter = 20        # Momentum activation iteration
            setting.num_parallel_workers = 10       # Parallel processing threads
            setting.gamma_cnst_grad = 10000            # Gradient phase step size
            
            print(f"Algorithm Parameters:")
            print(f"  Base step size: {setting.gamma_cnst}")
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
                setting.dispatching_rule,
                directory='completed_runs/mis_dynamic'
            )
            
            # Report results
            print(f"\n{'─'*60}")
            print(f"RESULTS SUMMARY")
            print(f"{'─'*60}")
            print(f"Convergence: {'✓ SUCCESS' if convergence_flag else '⚠ MAX ITERATIONS'}")
            print(f"Iterations completed: {num_iter:,}")
            print(f"Results saved to: {filename}")
            print(f"{'─'*60}")
            
            successful_runs_dynamic += 1
            
        except Exception as e:
            print(f"\n❌ ERROR in experiment μ={mu}")
            print(f"Error message: {str(e)}")
            failed_runs_dynamic.append((mu, str(e)))
            continue
    
    # =============================================================================
    # PART 2: DIRECT_FIFO DISPATCHING
    # =============================================================================
    
    print(f"\n{'='*80}")
    print("PART 2: DIRECT_FIFO DISPATCHING")
    print(f"{'='*80}")
    
    dispatching_rule = "DIRECT_FIFO"
    algorithm = '0-MIX' 
    successful_runs_direct = 0
    failed_runs_direct = []
    
    for mu in range(mu_min, mu_max + 1):
        
        print(f"\n{'='*80}")
        print(f"EXPERIMENT {mu}/{mu_max} - DIRECT_FIFO")
        print(f"Normalized Job Rate (μ): {mu}")
        print(f"{'='*80}")
        
        try:
            # -------------------------------------------------------------------------
            # Determine queue capacity for current job rate
            # -------------------------------------------------------------------------
            Qmax = get_qmax_direct_fifo(mu)
            print(f"Queue capacity (Qmax): {Qmax}")
            
            # -------------------------------------------------------------------------
            # Create temporary setting to extract dispatch distribution
            # -------------------------------------------------------------------------
            # DIRECT_FIFO dispatch distribution depends on Qmax, so we need to
            # create a setting first to get the correct dispatch_dist
            temp_setting = Setting(
                earnings=earnings,
                job_rates=job_rates_base.tolist(),
                driver_arrival_rate=mu,
                waiting_cost=waiting_cost,
                patience=patience,
                Qmax=Qmax,
                dispatch_rule=dispatching_rule,
                reneging=reneging,
                initial_alpha='1'
            )
            dispatch_dist = copy.deepcopy(temp_setting.dispatch_dist)
            
            # -------------------------------------------------------------------------
            # Normalize job rates to sum to μ
            # -------------------------------------------------------------------------
            driver_arrival_rate = mu
            job_rates_normalized = (job_rates_base / np.sum(job_rates_base) * mu).tolist()
            print(f"Total job rate (verification): {np.sum(job_rates_normalized):.4f}")
            
            # -------------------------------------------------------------------------
            # Initialize system configuration with normalized rates
            # -------------------------------------------------------------------------
            print("Initializing system configuration...")
            setting = Setting(
                earnings=earnings,
                job_rates=job_rates_normalized,
                driver_arrival_rate=driver_arrival_rate,
                waiting_cost=waiting_cost,
                patience=patience,
                Qmax=Qmax,
                dispatch_rule=dispatching_rule,
                reneging=reneging,
                initial_alpha='1'
            )
            
            # Override dispatch distribution with the pre-computed one
            setting.dispatch_dist = dispatch_dist
            
            # -------------------------------------------------------------------------
            # Configure advanced algorithmic parameters
            # -------------------------------------------------------------------------
            setting.gamma_cnst = 1000               # Base step size (acceptance)
            
            print(f"Algorithm Parameters:")
            print(f"  Base step size: {setting.gamma_cnst}")
            print(f"  Job rates sum: {np.sum(setting.job_rates):.4f}")
            print(f"  Driver arrival rate: {setting.driver_arrival_rate}")
            
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
                setting.dispatching_rule,
                directory='completed_runs/mis_direct'
            )
            
            # Report results
            print(f"\n{'─'*60}")
            print(f"RESULTS SUMMARY")
            print(f"{'─'*60}")
            print(f"Convergence: {'✓ SUCCESS' if convergence_flag else '⚠ MAX ITERATIONS'}")
            print(f"Iterations completed: {num_iter:,}")
            print(f"Results saved to: {filename}")
            print(f"{'─'*60}")
            
            successful_runs_direct += 1
            
        except Exception as e:
            print(f"\n❌ ERROR in experiment μ={mu}")
            print(f"Error message: {str(e)}")
            failed_runs_direct.append((mu, str(e)))
            continue
    
    # =============================================================================
    # EXPERIMENTAL SUMMARY
    # =============================================================================
    
    print(f"\n{'='*80}")
    print("EXPERIMENTAL SUMMARY")
    print(f"{'='*80}")
    
    print(f"\nDYNAMIC_RAND_FIFO:")
    print(f"  Total experiments: {total_experiments}")
    print(f"  Successful runs: {successful_runs_dynamic}")
    print(f"  Failed runs: {len(failed_runs_dynamic)}")
    
    if failed_runs_dynamic:
        print(f"  Failed experiments:")
        for mu, error in failed_runs_dynamic:
            print(f"    μ={mu}: {error}")
    
    print(f"\nDIRECT_FIFO:")
    print(f"  Total experiments: {total_experiments}")
    print(f"  Successful runs: {successful_runs_direct}")
    print(f"  Failed runs: {len(failed_runs_direct)}")
    
    if failed_runs_direct:
        print(f"  Failed experiments:")
        for mu, error in failed_runs_direct:
            print(f"    μ={mu}: {error}")
    
    print(f"\nAlgorithm: {algorithm}")
    print(f"Results directories:")
    print(f"  DYNAMIC_RAND_FIFO: ./completed_runs/mis_dynamic/")
    print(f"  DIRECT_FIFO: ./completed_runs/mis_direct/")
    
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
