"""
Queue Joining Behavior Experiment: Sequential Dispatching
==========================================================

This script analyzes the effect of queue joining behavior on system performance
under Sequential dispatching. The experiment varies the percentage of drivers
who join at the front of the queue from 0% to 100% in 10% increments.

"""
# %%
import numpy as np
import copy
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

# WARNING: The following experiments require 34GB of peak memory 
def main():
    """
    Main experimental function analyzing queue joining behavior effects.
    """
    # =============================================================================
    # EXPERIMENTAL SETUP
    # =============================================================================
    
    # Load calibrated economic parameters from O'Hare airport data
    print("Loading O'Hare economy parameters...")
    economy = get_ohare_economy()
    
    # Extract key economic parameters
    earnings = economy.w.tolist()           # Driver earnings by job type
    job_rates = economy.mu_jobs.tolist()    # Job arrival rates by type
    waiting_cost = economy.c                # Driver waiting cost
    patience = economy.patience             # Driver patience level
    
    # =============================================================================
    # SYSTEM CONFIGURATION
    # =============================================================================
    
    # System parameters
    reneging = 0                           # No reneging behavior
    driver_arrival_rate = 10               # Driver arrival rate (λ)
    dispatching_rule = "STRICT_FIFO"       # Dispatching policy
    
    # Queue configuration
    Qmax = 500                             # Maximum queue capacity
    
    # Simulation parameters
    tolerance = 1e-4                       # Convergence tolerance
    max_iterations = 1000                  # Maximum iterations
    saving_multiple = 50                  # Save progress every N iterations
    algorithm = '0-MIX'                    # Solution algorithm
    
    # =============================================================================
    # EXPERIMENTAL LOOP: QUEUE JOINING BEHAVIOR ANALYSIS
    # =============================================================================
    
    print(f"\nStarting experiment with {dispatching_rule} dispatching")
    print(f"Driver arrival rate (λ): {driver_arrival_rate}")
    print(f"Maximum queue capacity: {Qmax}")
    print(f"Convergence tolerance: {tolerance}")
    print("="*80)
    
    # Experiment: Vary percentage of drivers joining at front (0% to 100%)
    # WARNING: The following experiments require 34GB of peak memory 
    for experiment_run in range(0, 11):
        
        # Calculate current joining behavior parameter
        front_joining_percentage = experiment_run * 10  # 0%, 10%, 20%, ..., 100%
        join_at_front_rate = 0.1 * experiment_run       # 0.0, 0.1, 0.2, ..., 1.0
        
        print(f"\n{'='*80}")
        print(f"EXPERIMENT RUN {experiment_run + 1}/11")
        print(f"Front-joining percentage: {front_joining_percentage}%")
        print(f"Driver arrival rate (λ): {driver_arrival_rate}")
        print(f"{'='*80}")
        
        # -------------------------------------------------------------------------
        # Initialize system setting
        # -------------------------------------------------------------------------
        setting = Setting(
            earnings=earnings,
            job_rates=job_rates, 
            driver_arrival_rate=driver_arrival_rate,
            waiting_cost=waiting_cost,
            patience=patience,
            Qmax=Qmax,
            dispatch_rule=dispatching_rule,
            reneging=reneging,
            initial_alpha='RAND'  # Random initial acceptance strategy
        )
        
        # -------------------------------------------------------------------------
        # Configure algorithmic parameters
        # -------------------------------------------------------------------------
        setting.beta = 0.9                      # Momentum parameter
        setting.gamma_cnst = 100                 # Constant step size
        setting.gamma_exp = 0                   # Step size exponent (0 = constant)
        setting.gamma_phi_divide_by = 0.1       # Joining strategy step size
        setting.momentum_start_iter = 20        # Start momentum after N iterations
        setting.num_parallel_workers = 1        # Number of parallel workers
        setting.gamma_cnst_grad = 10            # Gradient algorithm step size
        
        # KEY EXPERIMENTAL VARIABLE: Queue joining behavior
        setting.join_at_front = join_at_front_rate
        
        # -------------------------------------------------------------------------
        # Run simulation
        # -------------------------------------------------------------------------
        print(f"Initializing system and simulator...")
        system = System(setting)
        simulator = Simulator(system, max_iterations, tolerance, algorithm, saving_multiple)
        
        print(f"Running {algorithm} algorithm...")
        print(f"Maximum iterations: {max_iterations}")
        print(f"Tolerance: {tolerance}")
        
        # Execute simulation
        outcome_list, V, alpha, phi, num_iter, inspect, convergence_flag = simulator.run_iterations()
        
        # -------------------------------------------------------------------------
        # Save results
        # -------------------------------------------------------------------------
        filename = save_run_history(
            simulator.setting, 
            outcome_list, 
            simulator.version, 
            setting.dispatching_rule, directory='completed_runs/front_of_queue_joining'
        )
        
        # Report run completion
        print(f"\nRun completed in {num_iter} iterations")
        print(f"Results saved to: {filename}")
        
        if convergence_flag:
            print("✓ Algorithm converged successfully")
        else:
            print("⚠ Algorithm reached maximum iterations without convergence")
    
    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETED")
    print(f"All {11} runs finished successfully")
    print(f"Results available in completed_runs/ directory")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
