# %%
from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt
from ipywidgets import widgets
from IPython.display import display

def add_project_root(marker="src"):
    here = Path.cwd()
    for p in (here, *here.parents):
        if (p / marker).exists() and (p / "completed_runs").exists():
            sys.path.insert(0, str(p))
            return p
    raise RuntimeError("Could not locate project root")

PROJECT_ROOT = add_project_root()

print("Project root:", PROJECT_ROOT)
from src.outcome_class import Outcome
import numpy as np
import dill
import os
import re
from mpl_toolkits.mplot3d import Axes3D
from src.get_economies import *
from src.utilities import *


# %%
direct_net_revenue_list = []
direct_waiting_time_list = []
direct_throughput_list = []
direct_ss_list = []
direct_iterations_list = []


dynamic_net_revenue_list = []
dynamic_waiting_time_list = []
dynamic_throughput_list = []
dynamic_ss_list = []
dynamic_iterations_list = []

# %%
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

# Function to extract the lambda value from the filename
def extract_mu(filename):
    match = re.search(r'mu=(\d+)', filename)
    if match:
        return int(match.group(1))  # Extracts the numeric value of lambda
    return None

folder_path = Path(PROJECT_ROOT) / "completed_runs" / "mis_direct"
files = sorted([f for f in os.listdir(folder_path) if f != '.gitkeep'], key=lambda f: extract_mu(f))

for i, filename in enumerate(files):
    file_path = os.path.join(folder_path,filename)
    outcome_list = read_dill(file_path)
    direct_net_revenue_list.append(outcome_list[-1].net_revenue)
    direct_throughput_list.append(outcome_list[-1].throughput)
    direct_iterations_list.append(outcome_list[-1].iter)
    ss = np.sum(outcome_list[-1].steady_state *  list(range(0, len(outcome_list[-1].steady_state))))
    direct_ss_list.append(ss)
    direct_waiting_time_list.append(ss/outcome_list[-1].throughput)

folder_path = Path(PROJECT_ROOT) / "completed_runs" / "mis_dynamic"
files = sorted([f for f in os.listdir(folder_path) if f != '.gitkeep'], key=lambda f: extract_mu(f))

for i, filename in enumerate(files):
    file_path = os.path.join(folder_path,filename)
    outcome_list = read_dill(file_path)
    dynamic_net_revenue_list.append(outcome_list[-1].net_revenue)
    dynamic_throughput_list.append(outcome_list[-1].throughput)
    dynamic_iterations_list.append(outcome_list[-1].iter)
    ss = np.sum(outcome_list[-1].steady_state *  list(range(0, len(outcome_list[-1].steady_state))))
    dynamic_ss_list.append(ss)
    dynamic_waiting_time_list.append(ss/outcome_list[-1].throughput)


output_dir = Path(PROJECT_ROOT) / "figures" / "experiments_5_2_plots"
output_dir.mkdir(parents=True, exist_ok=True)

# %%
x = list(range(1, 21))
# Plot settings for exact styles
plt.figure(figsize=(10, 8))  # Square figure

# Direct FIFO (solid line, circle markers)
plt.plot(
    x, direct_ss_list, color='purple', linestyle='-', marker='o',
    markersize=5, linewidth=1, markerfacecolor='none', label='Direct FIFO'
)

# Dynamic rand FIFO (dashed line, square markers)
plt.plot(
    x, dynamic_ss_list, color='blue', linestyle='-', marker='d',
    markersize=5, linewidth=1, markerfacecolor='none', label='Dynamic RFIFO'
)

# Labels and ticks
plt.xlabel('Trip/Driver Arrival Rate (μ = λ)', fontsize=32)
plt.ylabel('# Drivers in Queue', fontsize=32)
plt.xticks(range(2, 21, 2), fontsize=26)  # Show only even numbers on x-axis
plt.yticks(fontsize=26)

# Add vertical dashed line at x = 12
plt.axvline(x=12, color='grey', linestyle=(0, (15, 15)), linewidth=1)

# Add a legend with smaller font size
plt.legend(fontsize=28, loc='upper left', framealpha=0.0)

# Light grid for clarity
plt.grid(True, linestyle='--', alpha=0.5)

# Tight layout to optimize spacing
plt.tight_layout()

# Save plot as eps file locally
plt.savefig(output_dir / 'queue_length_plot_mis.eps', format='eps', dpi=1000)
plt.savefig(output_dir / 'queue_length_plot_mis.png', format='png', dpi=300)



# %%
x = list(range(1, 21))
# Plot settings for exact styles
plt.figure(figsize=(10, 8))  # Square figure

# Direct FIFO (solid line, circle markers)
plt.plot(
    x, direct_net_revenue_list, color='purple', linestyle='-', marker='o',
    markersize=5, linewidth=1, markerfacecolor='none', label='Direct FIFO'
)
# Strict FIFO (dashed line, square markers)
plt.plot(
    x, dynamic_net_revenue_list, color='blue', linestyle='-', marker='d',
    markersize=5, linewidth=1, markerfacecolor='none', label='Dynamic RFIFO'
)

# Labels and ticks
plt.xlabel('Trip/Driver Arrival Rate (μ = λ)', fontsize=32)
plt.ylabel('Net Revenue ($ per min)', fontsize=32)
plt.xticks(range(2, 21, 2), fontsize=26)  # Show only even numbers on x-axis
plt.yticks(fontsize=26)

# Add vertical dashed line at x = 12
plt.axvline(x=12, color='grey', linestyle=(0, (15, 15)), linewidth=1)

# Add a legend with smaller font size
plt.legend(fontsize=28, loc='upper left', framealpha=0.0)

# Light grid for clarity
plt.grid(True, linestyle='--', alpha=0.5)

# Tight layout to optimize spacing
plt.tight_layout()

#save plot as eps file locally
plt.savefig(output_dir / 'net_revenue_plot_mis.eps', format='eps', dpi=1000)
plt.savefig(output_dir / 'net_revenue_plot_mis.png', format='png', dpi=300)


# %%
x = list(range(1, 21))
# Plot settings for exact styles
plt.figure(figsize=(10, 8))  # Square figure

# Direct FIFO (solid line, circle markers)
plt.plot(
    x, direct_throughput_list, color='purple', linestyle='-', marker='o',
    markersize=5, linewidth=1, markerfacecolor='none', label='Direct FIFO'
)

# Strict FIFO (dashed line, square markers)
plt.plot(
    x, dynamic_throughput_list, color='blue', linestyle='-', marker='d',
    markersize=5, linewidth=1, markerfacecolor='none', label='Dynamic RFIFO'
)

# Labels and ticks
plt.xlabel('Trip/Driver Arrival Rate (μ = λ)', fontsize=32)
plt.ylabel('Throughput (trips per min)', fontsize=32)
plt.xticks(range(2, 21, 2), fontsize=26)  # Show only even numbers on x-axis
plt.yticks(fontsize=26)

# Add a legend with smaller font size
plt.legend(fontsize=28, loc='upper left', framealpha=0.0)
# Add vertical dashed line at x = 12
plt.axvline(x=12, color='grey', linestyle=(0, (15, 15)), linewidth=1)

# Light grid for clarity
plt.grid(True, linestyle='--', alpha=0.5)

# Tight layout to optimize spacing
plt.tight_layout()

#save plot as eps file locally
plt.savefig(output_dir / 'throughput_plot_mis.eps', format='eps', dpi=1000)
plt.savefig(output_dir / 'throughput_plot_mis.png', format='png', dpi=300)




# %%
x = list(range(1, 21))
# Plot settings for exact styles
plt.figure(figsize=(10, 8))  # Square figure

# Direct FIFO (solid line, circle markers)
plt.plot(
    x, direct_waiting_time_list, color='purple', linestyle='-', marker='o',
    markersize=5, linewidth=1, markerfacecolor='none', label='Direct FIFO'
)

# Strict FIFO (dashed line, square markers)
plt.plot(
    x, dynamic_waiting_time_list, color='blue', linestyle='-', marker='d',
    markersize=5, linewidth=1, markerfacecolor='none', label='Dynamic RFIFO'
)
# Labels and ticks
plt.xlabel('Trip/Driver Arrival Rate (μ = λ)', fontsize=32)
plt.ylabel('Average Waiting Time (mins)', fontsize=32)
plt.xticks(range(2, 21, 2), fontsize=26)  # Show only even numbers on x-axis
plt.yticks(fontsize=26)
# Add vertical dashed line at x = 12
plt.axvline(x=12, color='grey', linestyle=(0, (15, 15)), linewidth=1)

# Add a legend with smaller font size
plt.legend(fontsize=28, loc='upper left', framealpha=0.0)

# Light grid for clarity
plt.grid(True, linestyle='--', alpha=0.5)

# Tight layout to optimize spacing
plt.tight_layout()

#save plot as eps file locally
plt.savefig(output_dir / 'waiting_time_plot_mis.eps', format='eps', dpi=1000)
plt.savefig(output_dir / 'waiting_time_plot_mis.png', format='png', dpi=300)




