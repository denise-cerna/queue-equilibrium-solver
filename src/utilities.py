import numpy.core as np_core
import os
import dill
import sys
sys.modules['numpy._core'] = np_core  # Monkey patch to resolve missing module
import dill
from datetime import datetime
import numpy as np
import pytz
from pathlib import Path


# load iteration outcome objects into list
def read_dill(file_name):
    print(f"Attempting to read from: {file_name}")
    if os.path.getsize(file_name) == 0:
        print("The file is empty.")
        raise EOFError("File is empty")
    with open(file_name, 'rb') as f:
        loaded_objects = dill.load(f)
    print("File loaded successfully.")
    return loaded_objects

def ohare_get_Qmax(driver_arrival_rate, dispatching_rule):
    """
    This function gets the default maximum queue lengths for the O'Hare simulations for the paper
    """
    if dispatching_rule == "PURE_RAND" or dispatching_rule == "DYNAMIC_RAND_FIFO":
        if driver_arrival_rate <= 8:
            Qmax = 150
        elif driver_arrival_rate == 9 or driver_arrival_rate == 10:
            Qmax = 200
        elif driver_arrival_rate == 11:
            Qmax = 300
            # Qmax = 350                    # This was for ad hoc testing only 
        elif driver_arrival_rate >= 12:
            Qmax = 800

    elif dispatching_rule == "STRICT_FIFO":
        Qmax = 300

    elif dispatching_rule == "DIRECT_FIFO":
        if driver_arrival_rate <= 11:
            Qmax = 350
        elif driver_arrival_rate == 12:
            Qmax = 800
        elif driver_arrival_rate >= 13:
            Qmax = 800
            
    elif dispatching_rule == "LIFO":
        Qmax = 200
    return Qmax


"""
Save simulation run history to disk with automatic versioning.
"""

import os
import dill
import pytz
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List


def save_run_history(
    setting: object,
    outcome_list: List,
    version: str,
    dispatching_rule: str,
    directory: str = 'intermediate_saved_results'
) -> str:
    """
    Save simulation outcome history to a pickle file.
    
    Automatically manages file versioning by overwriting previous runs with
    identical parameters. Compresses outcome list by removing intermediate
    data to reduce file size.
    
    Args:
        setting: Setting object containing simulation parameters
        outcome_list: List of Outcome objects from simulation
        version: Algorithm version identifier (e.g., '0-MIX', '0-PAR', '0-COMP')
        dispatching_rule: Dispatching policy used (e.g., 'FIFO', 'PURE_RAND')
        directory: Directory to save results (default: 'run_history')
        
    Returns:
        Full path to saved file as string
        
    Example:
        >>> filepath = save_run_history(setting, outcomes, '0-MIX', 'FIFO')
        Deleting existing file: run_history/FIFO_mu=10.50_lambda=12_p=15_Qmax=800_iter=1000_20250115_120000.pkl
        Results saved to: run_history/FIFO_mu=10.50_lambda=12_p=15_Qmax=800_iter=1234_20250115_123045.pkl
    """
    # =========================================================================
    # Generate timestamp in Pacific Time
    # =========================================================================
    pacific_tz = pytz.timezone("America/Los_Angeles")
    local_time = datetime.now(pacific_tz)
    timestamp = local_time.strftime("%Y%m%d_%H%M%S")
    
    # =========================================================================
    # Extract key parameters
    # =========================================================================
    num_iter = outcome_list[-1].iter
    total_mu = np.sum(setting.job_rates)
    lambda_val = setting.driver_arrival_rate
    patience = setting.patience
    qmax = setting.Qmax
    
    # =========================================================================
    # Generate base filename based on version and parameters
    # =========================================================================
    if setting.inspection_cost > 0:
        # Include inspection cost
        base_filename = (
            f"{dispatching_rule}_mu={total_mu:.2f}_lambda={lambda_val}"
            f"_p={patience}_Qmax={qmax}_insp_cost={setting.inspection_cost}_v{version}"
        )
    
    elif version == '0-COMP':
        # Competition mode: include shared market rate
        # Note: Setting object uses 'total_share' attribute for shared market
        shared_market = setting.total_share
        base_filename = (
            f"{dispatching_rule}_mu={total_mu:.2f}_lambda={lambda_val}"
            f"_p={patience}_Qmax={qmax}_shared={shared_market:.2f}_v{version}"
        )
    
    elif hasattr(setting, 'join_at_front') and setting.join_at_front > 0:
        # Priority joining: include join_at_front parameter
        base_filename = (
            f"{dispatching_rule}_mu={total_mu:.2f}_lambda={lambda_val}"
            f"_p={patience}_Qmax={qmax}_priority={setting.join_at_front:.2f}_v{version}"
        )
    
    else:
        # Standard mode: include inspection cost
        base_filename = (
            f"{dispatching_rule}_mu={total_mu:.2f}_lambda={lambda_val}"
            f"_p={patience}_Qmax={qmax}_insp_cost={setting.inspection_cost}_v{version}"
        )
    
    # =========================================================================
    # Setup directory and remove old files with same parameters
    # =========================================================================
    save_dir = Path(directory)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Remove existing files with same base parameters
    for file_path in save_dir.glob(f"{base_filename}*.pkl"):
        print(f"Deleting existing file: {file_path}")
        file_path.unlink()
    
    # =========================================================================
    # Create full filename and save
    # =========================================================================
    filename = f"{base_filename}_iter={num_iter}_{timestamp}.pkl"
    file_path = save_dir / filename
    
    # Compress outcome list: remove large arrays from intermediate outcomes
    for outcome in outcome_list[:-1]:
        outcome.earnings = None
        outcome.alpha = None
        outcome.phi = None
        outcome.V = None
        outcome.job_rates = None
    
    # Save to file
    with open(file_path, 'wb') as f:
        dill.dump(outcome_list, f)
    
    print(f"Results saved to: {file_path}")
    return str(file_path)