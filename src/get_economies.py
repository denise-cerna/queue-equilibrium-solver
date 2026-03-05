
import pandas as pd
from src.economy_class import Economy
import os
from os.path import dirname, abspath, join
import sys
from pathlib import Path

path_to_project = dirname(dirname(abspath(__file__)))  # nopep8
sys.path.append(join(path_to_project, 'src'))  # nopep8

### O'Hare
def get_ohare_economy():

    ROOT = Path(__file__).resolve().parents[1]
    file_path = ROOT / "data" / "trips_from_OHare_by_dropoff_community_area_for_Python.csv"

    # directory = r'input_data'
    # filename = 'trips_from_OHare_by_dropoff_community_area_for_Python.csv'

    # Ensure the directory exists
    # os.makedirs(directory, exist_ok=True)

    # # Full path to the file
    # file_path = os.path.join(directory, filename)

    df = pd.read_csv(file_path)

    L = df.shape[0]
    J = 1
    w = df['net_earnings'].to_numpy()
    W = w.reshape((1, L))
    c_platform      = 1/3 
    C               = [1/3] * J 
    lambda_drivers = [10]
    patience = 12

    rider_arrivasl_rate = 12
    df['mu_array'] = df['job_fraction'] * rider_arrivasl_rate
    mu_jobs = df['mu_array'].to_numpy()

    econ_ohare = Economy(L, J, w, W, c_platform, C, lambda_drivers, 
                        mu_jobs, patience)
    return econ_ohare


### Midway
def get_midway_economy():

    directory = r'input_data'
    filename = 'trips_from_Midway_by_dropoff_community_area_for_Python.csv'

    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)

    # Full path to the file
    file_path = os.path.join(directory, filename)

    df = pd.read_csv(file_path)

    L = df.shape[0]
    J = 1
    w = df['net_earnings'].to_numpy()
    W = w.reshape((1, L))
    c_platform      = 1/3 
    C               = [1/3] * J 
    lambda_drivers = [4]
    patience = 12

    rider_arrivasl_rate = 5
    df['mu_array'] = df['job_fraction'] * rider_arrivasl_rate
    mu_jobs = df['mu_array'].to_numpy()

    econ_midway = Economy(L, J, w, W, c_platform, C, lambda_drivers, 
                        mu_jobs, patience)
    return econ_midway