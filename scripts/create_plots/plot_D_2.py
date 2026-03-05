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
net_revenue_list = []
waiting_time_list = []
throughput_list = []
ss_list = []
V_1Q_list = []
V_12Q_list = []

# %%
economy = get_ohare_economy()
earnings = economy.w.tolist()
job_rates = economy.mu_jobs.tolist()
output_dir = Path(PROJECT_ROOT) / "figures" / "experiments_D_2_plots"
output_dir.mkdir(parents=True, exist_ok=True)

# %%
import matplotlib.pyplot as plt

# Plot earnings on x-axis and job rates on y-axis
fig, ax = plt.subplots(figsize=(7, 5))

markerline, stemlines, baseline = ax.stem(
    earnings, job_rates,
    markerfmt='o', basefmt=" ", linefmt='purple'
)

# Make stems thicker and markers larger
plt.setp(stemlines, linewidth=1)
plt.setp(markerline, markersize=4)

ax.set_xlabel('Trip Earnings', fontsize=14)
ax.set_ylabel('Trip Arrival Rates', fontsize=14)
# ax.set_title('Earnings vs Job Rates')

ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(output_dir / 'earnings_vs_job_rates.eps', format='eps', dpi=1000)
plt.savefig(output_dir / 'earnings_vs_job_rates.png', format='png', dpi=300)
plt.show()


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
def extract_priority(filename):
    # matches e.g. "priority=0.0", "priority=0.1", ..., "priority=1"
    match = re.search(r'priority=(\d+(?:\.\d+)?)', filename)
    if match:
        return float(match.group(1))
    else:
        # put unmatched files at the front
        return 0

folder_path = Path(PROJECT_ROOT) / "completed_runs" / "front_of_queue_joining"
# only keep non‐.gitkeep files, sort by numeric priority
files = sorted(
    [f for f in os.listdir(folder_path) if f != '.gitkeep'],
    key=extract_priority
)

for i, filename in enumerate(files):
    file_path = os.path.join(folder_path,filename)
    outcome_list = read_dill(file_path)
    net_revenue_list.append(outcome_list[-1].net_revenue)
    throughput_list.append(outcome_list[-1].throughput)
    V_1Q_list.append(outcome_list[-1].V[1, 100])
    V_12Q_list.append(outcome_list[-1].V[12, 100])
    ss = np.sum(outcome_list[-1].steady_state *  list(range(0, len(outcome_list[-1].steady_state))))
    ss_list.append(ss)
    waiting_time_list.append(ss/outcome_list[-1].throughput)

output_dir = Path(PROJECT_ROOT) / "figures" / "experiments_D_2_plots"
output_dir.mkdir(parents=True, exist_ok=True)

# %%
x = list(np.arange(0, 1.1, 0.1))
# Plot settings for exact styles
plt.figure(figsize=(10, 8))  # Square figure

# Direct FIFO (solid line, circle markers)
plt.plot(
    x, V_12Q_list, color='red', linestyle='-', marker='o',
    markersize=5, linewidth=1, markerfacecolor='none'
)

# Labels and ticks
plt.xlabel('Priority Parameter', fontsize=30)
plt.ylabel('V(12, Q)', fontsize=30)
plt.xticks(x, fontsize=24)  # Show only even numbers on x-axis
plt.yticks(fontsize=24)

# Add a legend with smaller font size
plt.legend(fontsize=24, loc='upper right', framealpha=0.0)

# Light grid for clarity
plt.grid(True, linestyle='--', alpha=0.5)

# Tight layout to optimize spacing
plt.tight_layout()

# Save plot as eps file locally
plt.savefig(output_dir / 'V_12_priority.eps', format='eps', dpi=1000)
plt.savefig(output_dir / 'V_12_priority.png', format='png', dpi=300)




# %%
x = list(np.arange(0, 1.1, 0.1))
# Plot settings for exact styles
plt.figure(figsize=(10, 8))  # Square figure

# Direct FIFO (solid line, circle markers)
plt.plot(
    x, ss_list, color='red', linestyle='-', marker='o',
    markersize=5, linewidth=1, markerfacecolor='none'
)

# Labels and ticks
plt.xlabel('Priority Parameter', fontsize=32)
plt.ylabel('# Drivers in Queue', fontsize=32)
plt.xticks(x, fontsize=26)  # Show only even numbers on x-axis
plt.yticks(fontsize=26)

# Add a legend with smaller font size
plt.legend(fontsize=28, loc='upper right', framealpha=0.0)

# Light grid for clarity
plt.grid(True, linestyle='--', alpha=0.5)

# Tight layout to optimize spacing
plt.tight_layout()

# Save plot as eps file locally
plt.savefig(output_dir / 'priority_queue_length_plot.eps', format='eps', dpi=1000)
plt.savefig(output_dir / 'priority_queue_length_plot.png', format='png', dpi=300)




# %%
x = list(np.arange(0, 1.1, 0.1))
# Plot settings for exact styles
plt.figure(figsize=(10, 8))  # Square figure

# Direct FIFO (solid line, circle markers)
plt.plot(
    x, net_revenue_list, color='red', linestyle='-', marker='o',
    markersize=5, linewidth=1, markerfacecolor='none'
)


# Labels and ticks
plt.xlabel('Priority Parameter', fontsize=32)
plt.ylabel('Net revenue ($ per min)', fontsize=32)
plt.xticks(x, fontsize=26)  # Show only even numbers on x-axis
plt.yticks(fontsize=26)

# Add a legend with smaller font size
plt.legend(fontsize=28, loc='upper left', framealpha=0.0)

# Light grid for clarity
plt.grid(True, linestyle='--', alpha=0.5)

# Tight layout to optimize spacing
plt.tight_layout()

#save plot as eps file locally
plt.savefig(output_dir / 'priority_net_revenue_plot.eps', format='eps', dpi=1000)
plt.savefig(output_dir / 'priority_net_revenue_plot.png', format='png', dpi=300)



# %%
x = list(np.arange(0, 1.1, 0.1))
# Plot settings for exact styles
plt.figure(figsize=(10, 8))  # Square figure

# Direct FIFO (solid line, circle markers)
plt.plot(
    x, throughput_list, color='red', linestyle='-', marker='o',
    markersize=5, linewidth=1, markerfacecolor='none'
)


# Labels and ticks
plt.xlabel('Priority Parameter', fontsize=32)
plt.ylabel('Throughput (trips per min)', fontsize=32)
plt.xticks(x, fontsize=26)  # Show only even numbers on x-axis
plt.yticks(fontsize=26)

# Add a legend with smaller font size
plt.legend(fontsize=28, loc='upper left', framealpha=0.0)

# Light grid for clarity
plt.grid(True, linestyle='--', alpha=0.5)

# Tight layout to optimize spacing
plt.tight_layout()

#save plot as eps file locally
plt.savefig(output_dir / 'priority_throughput_plot.eps', format='eps', dpi=1000)
plt.savefig(output_dir / 'priority_throughput_plot.png', format='png', dpi=300)

# Show the plot
plt.show()


# %%
x = list(np.arange(0, 1.1, 0.1))
# Plot settings for exact styles
plt.figure(figsize=(10, 8))  # Square figure

# Direct FIFO (solid line, circle markers)
plt.plot(
    x, waiting_time_list, color='red', linestyle='-', marker='o',
    markersize=5, linewidth=1, markerfacecolor='none'
)


# Labels and ticks
plt.xlabel('Priority Parameter', fontsize=32)
plt.ylabel('Average Waiting Time (mins)', fontsize=32)
plt.xticks(x, fontsize=26)  # Show only even numbers on x-axis
plt.yticks(fontsize=26)


# Add a legend with smaller font size
plt.legend(fontsize=28, loc='upper right', framealpha=0.0)

# Light grid for clarity
plt.grid(True, linestyle='--', alpha=0.5)

# Tight layout to optimize spacing
plt.tight_layout()

#save plot as eps file locally
plt.savefig(output_dir / 'priority_waiting_time_plot.eps', format='eps', dpi=1000)
plt.savefig(output_dir / 'priority_waiting_time_plot.png', format='png', dpi=300)



