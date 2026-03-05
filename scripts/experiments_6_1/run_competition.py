"""
Competition Simulation: Market Share Sweep
================================================

Compares STRICT_FIFO vs DYNAMIC_RAND_FIFO dispatching across varying
market share allocations from 10% to 100% shared market.
"""

import numpy as np
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

# WARNING: The following experiments require 16GB of peak memory 
def main():
    """Run competition mode simulations across market share range."""
    
    # =============================================================================
    # Load Economy Parameters
    # =============================================================================
    print("="*80)
    print("COMPETITION MODE SIMULATION")
    print("="*80)
    
    economy = get_ohare_economy()
    earnings = economy.w.tolist()
    job_rates = economy.mu_jobs.tolist()
    waiting_cost = economy.c
    
    # =============================================================================
    # System Configuration
    # =============================================================================
    driver_arrival_rate = 12
    Qmax = 800
    patience = 12
    reneging = 0
    dispatching_rule = "COMP"
    
    # Firm dispatching strategies
    firm1_dispatch_rule = 'STRICT_FIFO'
    firm2_dispatch_rule = 'DYNAMIC_RAND_FIFO'
    tie_breaking = 'RAND'
    
    # =============================================================================
    # Algorithm Configuration
    # =============================================================================
    tolerance = 1e-4
    max_iterations = 1000
    saving_multiple = 100
    algorithm = '0-COMP'
    
    # Optimization parameters
    beta = 0.9
    gamma_cnst = 1000
    gamma_exp = 0
    gamma_phi_divide_by = 1
    momentum_start_iter = 50
    
    print(f"\nConfiguration:")
    print(f"  Firm 1: {firm1_dispatch_rule}")
    print(f"  Firm 2: {firm2_dispatch_rule}")
    print(f"  Tie-breaking: {tie_breaking}")
    print(f"  Algorithm: {algorithm}")
    print(f"  Tolerance: {tolerance}")
    print(f"  Max iterations: {max_iterations:,}")
    
    # =============================================================================
    # Run Experiments
    # =============================================================================
    total_experiments = 10
    results = []
    
    # WARNING: The following experiments require 16GB of peak memory 
    for i in range(1, total_experiments + 1):
        # Calculate market shares
        total_share = 0.1 * i
        firm1_share = (1 - total_share) / 2
        firm2_share = (1 - total_share) / 2
        
        print(f"\n{'='*80}")
        print(f"EXPERIMENT {i}/{total_experiments}")
        print(f"Total shared market: {total_share:.1%}")
        print(f"Firm 1 exclusive: {firm1_share:.1%}, Firm 2 exclusive: {firm2_share:.1%}")
        print(f"{'='*80}")
        
        try:
            # -------------------------------------------------------------------------
            # Initialize Setting
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
                initial_alpha='1',
                firm1_dispatch_rule=firm1_dispatch_rule,
                firm2_dispatch_rule=firm2_dispatch_rule,
                tie_breaking=tie_breaking
            )
            
            # Set optimization parameters
            setting.beta = beta
            setting.gamma_cnst = gamma_cnst
            setting.gamma_exp = gamma_exp
            setting.gamma_phi_divide_by = gamma_phi_divide_by
            setting.momentum_start_iter = momentum_start_iter
            
            # Set market shares
            setting.total_share = total_share
            setting.firm1_share = firm1_share
            setting.firm2_share = firm2_share
            
            # -------------------------------------------------------------------------
            # Run Simulation
            # -------------------------------------------------------------------------
            system = System(setting)
            print(f"Inspection cost: {system.setting.inspection_cost}")
            
            simulator = Simulator(
                system=system,
                max_iterations=max_iterations,
                tolerance=tolerance,
                version=algorithm,
                saving_multiple=saving_multiple
            )
            
            print("Running simulation...")
            outcome_list, V, alpha, phi, num_iter, inspect, converged = simulator.run_iterations()
            
            # -------------------------------------------------------------------------
            # Save Results
            # -------------------------------------------------------------------------
            filename = save_run_history(
                setting=simulator.setting,
                outcome_list=outcome_list,
                version=simulator.version,
                dispatching_rule=setting.dispatching_rule,
                directory='completed_runs/competition'
            )
            
            # -------------------------------------------------------------------------
            # Record Summary
            # -------------------------------------------------------------------------
            final_outcome = outcome_list[-1]
            results.append({
                'experiment': i,
                'total_share': total_share,
                'converged': converged,
                'iterations': num_iter,
                'net_revenue': getattr(final_outcome, 'net_revenue', None),
                'throughput': getattr(final_outcome, 'throughput', None)
            })
            
            print(f"\n{'─'*60}")
            print(f"Result: {'✓ CONVERGED' if converged else '⚠ MAX ITERATIONS'}")
            print(f"Iterations: {num_iter:,}")
            if hasattr(final_outcome, 'net_revenue'):
                print(f"Net revenue: {final_outcome.net_revenue:.4f}")
            print(f"Saved to: {filename}")
            print(f"{'─'*60}")
            
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'experiment': i,
                'total_share': total_share,
                'error': str(e)
            })
    
    # =============================================================================
    # Summary
    # =============================================================================
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    successful = [r for r in results if 'error' not in r]
    failed = [r for r in results if 'error' in r]
    
    print(f"Total: {len(results)}, Success: {len(successful)}, Failed: {len(failed)}")
    
    if successful:
        print(f"\n{'Exp':<6} {'Share':<10} {'Conv':<8} {'Iter':<8} {'Revenue':<12}")
        print("─" * 60)
        for r in successful:
            conv = "✓" if r['converged'] else "⚠"
            rev = f"{r['net_revenue']:.4f}" if r['net_revenue'] else "N/A"
            print(f"{r['experiment']:<6} {r['total_share']:<10.1%} {conv:<8} "
                  f"{r['iterations']:<8} {rev:<12}")
    
    if failed:
        print(f"\nFailed experiments:")
        for r in failed:
            print(f"  {r['experiment']}: {r['error']}")
    
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
