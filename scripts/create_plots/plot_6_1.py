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
firm1_nr = []
firm2_nr = []
total_nr = []

firm1_tpt = []
firm2_tpt = []
total_tpt = []

firm1_ql = []
firm2_ql = []
total_ql = []

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

# Function to extract the priority parameter (as float) from the filename
def extract_shared_rates(filename):
    # matches e.g. "priority=0.0", "priority=0.1", ..., "priority=1"
    match = re.search(r'shared=(\d+(?:\.\d+)?)', filename)
    if match:
        return float(match.group(1))
    else:
        # put unmatched files at the front
        return 0

folder_path = Path(PROJECT_ROOT) / "completed_runs" / "competition"

# only keep non‐.gitkeep files, sort by numeric priority
files = sorted(
    [f for f in os.listdir(folder_path) if f != '.gitkeep'],
    key=extract_shared_rates
)

for i, filename in enumerate(files):
    file_path = os.path.join(folder_path, filename)
    outcome_list = read_dill(file_path)
    firm1_nr.append(outcome_list[-1].net_revenue_1)
    firm2_nr.append(outcome_list[-1].net_revenue_2)
    total_nr.append(outcome_list[-1].net_revenue)

    firm1_tpt.append(outcome_list[-1].throughput_1)
    firm2_tpt.append(outcome_list[-1].throughput_2)
    total_tpt.append(outcome_list[-1].throughput)
    queue_length = outcome_list[-1].queue_length
    total_ql.append(queue_length)

# Folder where you want to save the plot
output_dir = Path(PROJECT_ROOT) / "figures" / "experiments_6_1_plots"
output_dir.mkdir(parents=True, exist_ok=True)

# %%
x = list(np.arange(0, 1.1, 0.1))

# File path

# Plot settings for exact styles
plt.figure(figsize=(10, 8))  # Square figure

plt.plot(
    x, firm1_nr, color='hotpink', linestyle='-', marker='o',
    markersize=5, linewidth=1, markerfacecolor='none', label='Platform 1 (Strict FIFO)'
)

plt.plot(
    x, firm2_nr, color='black', linestyle='-', marker='o',
    markersize=5, linewidth=1, markerfacecolor='none', label='Platform 2 (Dynamic RFIFO)'
)

plt.plot(
    x, total_nr, color='blue', linestyle='-', marker='o',
    markersize=5, linewidth=1, markerfacecolor='none', label='Total'
)

# Labels and ticks
plt.xlabel('Share of Multihoming Trips', fontsize=32)
plt.ylabel('Net Revenue ($ per min)', fontsize=32)
plt.xticks(x, fontsize=26)  # Show only even numbers on x-axis
plt.yticks(fontsize=26)

# Add a legend with smaller font size
plt.legend(fontsize=28, loc='upper left', framealpha=0.0)

# Light grid for clarity
plt.grid(True, linestyle='--', alpha=0.5)

# Tight layout to optimize spacing
plt.tight_layout()

# Save plot to chosen folder
plt.savefig(output_dir / 'comp_net_revenue_plot.eps', format='eps', dpi=1000)
plt.savefig(output_dir / 'comp_net_revenue_plot.png', format='png', dpi=300)



# %%
x = list(np.arange(0, 1.1, 0.1))

# Plot settings for exact styles
plt.figure(figsize=(10, 8))  # Square figure

plt.plot(
    x, firm1_tpt, color='hotpink', linestyle='-', marker='o',
    markersize=5, linewidth=1, markerfacecolor='none', label='Platform 1 (Strict FIFO)'
)

# Direct FIFO (solid line, circle markers)
plt.plot(
    x, firm2_tpt, color='black', linestyle='-', marker='o',
    markersize=5, linewidth=1, markerfacecolor='none', label='Platform 2 (Dynamic RFIFO)'
)

plt.plot(
    x, total_tpt, color='blue', linestyle='-', marker='o',
    markersize=5, linewidth=1, markerfacecolor='none', label='Total'
)

# Labels and ticks
plt.xlabel('Share of Multihoming Trips', fontsize=32)
plt.ylabel('Throughput (trips per min)', fontsize=32)
plt.xticks(x, fontsize=26)
plt.yticks(fontsize=26)

# Add a legend with smaller font size
plt.legend(fontsize=28, loc='center right', framealpha=0.0)

# Light grid for clarity
plt.grid(True, linestyle='--', alpha=0.5)

# Tight layout to optimize spacing
plt.tight_layout()

plt.savefig(output_dir / 'comp_throughput_plot.eps', format='eps', dpi=1000)
plt.savefig(output_dir / 'comp_throughput_plot.png', format='png', dpi=300)



# %%
x = list(np.arange(0, 1.1, 0.1))


# Plot settings for exact styles
plt.figure(figsize=(10, 8))  # Square figure

# Total queue length
plt.plot(
    x, total_ql, color='blue', linestyle='-', marker='o',
    markersize=5, linewidth=1, markerfacecolor='none', label='Total'
)

# Labels and ticks
plt.xlabel('Share of Multihoming Trips', fontsize=32)
plt.ylabel('# Drivers in Queue', fontsize=32)
plt.xticks(x, fontsize=26)  # Show only even numbers on x-axis
plt.yticks(fontsize=26)

# Legend, grid, layout
plt.legend(fontsize=28, loc='upper left', framealpha=0.0)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

plt.savefig(output_dir / 'comp_queue_length_plot.eps', format='eps', dpi=1000)
plt.savefig(output_dir / 'comp_queue_length_plot.png', format='png', dpi=300)



