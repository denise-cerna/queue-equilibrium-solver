# %%
from pathlib import Path
import sys
import matplotlib.pyplot as plt
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
import numpy as np
import dill
import os
import re
from mpl_toolkits.mplot3d import Axes3D
from src.get_economies import *
from src.utilities import *


# # %%
fb_path = Path(PROJECT_ROOT) / "completed_runs" / "fb_data" / "fb_net_rev.npy"
print("Loading:", fb_path)  # sanity check
fb_net_revenue_list = np.load(fb_path, allow_pickle=True)

fb_path = Path(PROJECT_ROOT) / "completed_runs" / "fb_data" / "fb_throughput.npy"
print("Loading:", fb_path)  # sanity check
fb_throughput_list = np.load(fb_path, allow_pickle=True)

fb_path = Path(PROJECT_ROOT) / "completed_runs" / "fb_data" / "fb_queue_length.npy"
print("Loading:", fb_path)  # sanity check
fb_avg_queue_length_list = np.load(fb_path, allow_pickle=True)

fb_path = Path(PROJECT_ROOT) / "completed_runs" / "fb_data" / "fb_waiting_time.npy"
print("Loading:", fb_path)  # sanity check
fb_waiting_time_list = np.load(fb_path, allow_pickle=True)


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

economy = get_ohare_economy()
earnings = economy.w.tolist()
job_rates = economy.mu_jobs.tolist()

# %%
strict_net_revenue_list = []
direct_net_revenue_list = []
lifo_net_revenue_list = []
pure_rand_net_revenue_list = []
rand_fifo_net_revenue_list = []

strict_ss_list = []
direct_ss_list = []
lifo_ss_list = []
pure_rand_ss_list = []
rand_fifo_ss_list = []

strict_throughput_list = []
direct_throughput_list = []
lifo_throughput_list = []
pure_rand_throughput_list = []
rand_fifo_throughput_list = []

strict_waiting_time_list = []
direct_waiting_time_list = []
lifo_waiting_time_list = []
pure_rand_waiting_time_list = []
rand_fifo_waiting_time_list = []

strict_variance_list = []
direct_variance_list = []
lifo_variance_list = []
rand_fifo_variance_list = []
pure_rand_variance_list = []

strict_earnings_list = []
direct_earnings_list = []
lifo_earnings_list = []
rand_fifo_earnings_list = []
pure_rand_earnings_list = []

# Function to extract the lambda value from the filename
def extract_lambda(filename):
    match = re.search(r'lambda=(\d+)', filename)
    if match:
        return int(match.group(1))  # Extracts the numeric value of lambda
    return None

folder_path = Path(PROJECT_ROOT) / "completed_runs" / "direct_fifo"
files = sorted([f for f in os.listdir(folder_path) if f != '.gitkeep'], key=lambda f: extract_lambda(f))

for i, filename in enumerate(files):
    file_path = os.path.join(folder_path,filename)
    outcome_list = read_dill(file_path)
    direct_net_revenue_list.append(outcome_list[-1].net_revenue)
    direct_ss_list.append(outcome_list[-1].queue_length)
    direct_throughput_list.append(outcome_list[-1].throughput)
    direct_waiting_time_list.append(outcome_list[-1].queue_length/outcome_list[-1].throughput)
    direct_variance_list.append(outcome_list[-1].variance)
    direct_earnings_list.append(outcome_list[-1].ave_earnings)

folder_path = Path(PROJECT_ROOT) / "completed_runs" / "strict_fifo"
files = sorted([f for f in os.listdir(folder_path) if f != '.gitkeep'], key=lambda f: extract_lambda(f))

for i, filename in enumerate(files):
    file_path = os.path.join(folder_path,filename)
    outcome_list = read_dill(file_path)
    strict_net_revenue_list.append(outcome_list[-1].net_revenue)
    strict_ss_list.append(outcome_list[-1].queue_length)
    strict_throughput_list.append(outcome_list[-1].throughput)
    strict_waiting_time_list.append(outcome_list[-1].queue_length/outcome_list[-1].throughput)
    strict_variance_list.append(outcome_list[-1].variance)
    strict_earnings_list.append(outcome_list[-1].ave_earnings)

folder_path = Path(PROJECT_ROOT) / "completed_runs" / "lifo"
files = sorted([f for f in os.listdir(folder_path) if f != '.gitkeep'], key=lambda f: extract_lambda(f))

for i, filename in enumerate(files):
    file_path = os.path.join(folder_path,filename)
    outcome_list = read_dill(file_path)
    lifo_net_revenue_list.append(outcome_list[-1].net_revenue)
    lifo_ss_list.append(outcome_list[-1].queue_length)
    lifo_throughput_list.append(outcome_list[-1].throughput)
    lifo_waiting_time_list.append(outcome_list[-1].queue_length/outcome_list[-1].throughput)
    lifo_variance_list.append(outcome_list[-1].variance)
    lifo_earnings_list.append(outcome_list[-1].ave_earnings)  

folder_path = Path(PROJECT_ROOT) / "completed_runs" / "pure_rand"
files = sorted([f for f in os.listdir(folder_path) if f != '.gitkeep'], key=lambda f: extract_lambda(f))

for i, filename in enumerate(files):
    file_path = os.path.join(folder_path,filename)
    outcome_list = read_dill(file_path)
    pure_rand_net_revenue_list.append(outcome_list[-1].net_revenue)
    pure_rand_ss_list.append(outcome_list[-1].queue_length)
    pure_rand_throughput_list.append(outcome_list[-1].throughput)
    pure_rand_waiting_time_list.append(outcome_list[-1].queue_length/outcome_list[-1].throughput)
    pure_rand_variance_list.append(outcome_list[-1].variance)
    pure_rand_earnings_list.append(outcome_list[-1].ave_earnings)

folder_path = Path(PROJECT_ROOT) / "completed_runs" / "dynamic_rand_fifo"
files = sorted([f for f in os.listdir(folder_path) if f != '.gitkeep'], key=lambda f: extract_lambda(f))

for i, filename in enumerate(files):
    file_path = os.path.join(folder_path,filename)
    outcome_list = read_dill(file_path)
    rand_fifo_net_revenue_list.append(outcome_list[-1].net_revenue)
    rand_fifo_ss_list.append(outcome_list[-1].queue_length)
    rand_fifo_throughput_list.append(outcome_list[-1].throughput)
    rand_fifo_waiting_time_list.append(outcome_list[-1].queue_length/outcome_list[-1].throughput)
    rand_fifo_variance_list.append(outcome_list[-1].variance)
    rand_fifo_earnings_list.append(outcome_list[-1].ave_earnings)

output_dir = Path(PROJECT_ROOT) / "figures" / "experiments_5_1_plots"
output_dir.mkdir(parents=True, exist_ok=True)

# %%
x = list(range(1, 16))
# Plot settings for exact styles
plt.figure(figsize=(10, 8))  # Square figure

# fb_net_revenue_list (thin dotted black line)
plt.plot(
    x, fb_net_revenue_list, color='black', linestyle=':', linewidth=3, label='First Best'
)

#DRFIFO (solid line, diamond markers)
plt.plot(
    x, rand_fifo_net_revenue_list, color='blue', linestyle='-', marker='d',
    markersize=5, linewidth=1, markerfacecolor='none', label='Dynamic RFIFO'
)

# Direct FIFO (solid line, circle markers)
plt.plot(
    x, direct_net_revenue_list, color='purple', linestyle='-', marker='o',
    markersize=5, linewidth=1, markerfacecolor='none', label='Direct FIFO'
)

# LIFO (dash-dot line, triangle-up markers)
plt.plot(
    x, lifo_net_revenue_list, color='green', linestyle='-.', marker='^',
    markersize=5, linewidth=1, markerfacecolor='none', label='LIFO'
)

# Strict FIFO (dashed line, square markers)
plt.plot(
    x, strict_net_revenue_list, color='red', linestyle='--', marker='s',
    markersize=5, linewidth=1, markerfacecolor='none', label='Strict FIFO'
)

# # Pure Random (dotted line, x markers)
plt.plot(
    x, pure_rand_net_revenue_list, color='#40E0D0', linestyle=':', marker='x',
    markersize=5, linewidth=1, label='Pure Random'  # 'x' markers are naturally outlined
)


# Labels and ticks
plt.xlabel('Driver Arrival Rate (λ)', fontsize=30)
plt.ylabel('Net Revenue ($ per min)', fontsize=30)
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)

# Add a legend with smaller font size
plt.legend(fontsize=22, loc='upper left', framealpha=0.0)

# Light grid for clarity
plt.grid(True, linestyle='--', alpha=0.5)

# Tight layout to optimize spacing
plt.tight_layout()

# save plot as eps file locally

plt.savefig(output_dir / 'net_revenue_plot.eps', format='eps', dpi=1000)
plt.savefig(output_dir / 'net_revenue_plot.png', format='png', dpi=300)


# %%
x = list(range(1, 16))
plt.figure(figsize=(10, 8))  # Square figure

# fb_ave_waiting_time_list (thin dotted black line)
plt.plot(
    x, fb_waiting_time_list, color='black', linestyle=':', linewidth=3, label='First Best'
)

# DRFIFO (solid line, diamond markers)
plt.plot(
    x, rand_fifo_ss_list, color='blue', linestyle='-', marker='d',
    markersize=5, linewidth=1, markerfacecolor='none', label='Dynamic RFIFO'
)

# Direct FIFO (solid line, circle markers)
plt.plot(
    x, direct_ss_list, color='purple', linestyle='-', marker='o',
    markersize=5, linewidth=1, markerfacecolor='none', label='Direct FIFO'
)

# LIFO (dash-dot line, triangle-up markers)
plt.plot(
    x, lifo_ss_list, color='green', linestyle='-.', marker='^',
    markersize=5, linewidth=1, markerfacecolor='none', label='LIFO'
)

# Strict FIFO (dashed line, square markers)
plt.plot(
    x, strict_ss_list, color='red', linestyle='--', marker='s',
    markersize=5, linewidth=1, markerfacecolor='none', label='Strict FIFO'
)

# # Pure Random (dotted line, x markers)
plt.plot(
    x, pure_rand_ss_list, color='#40E0D0', linestyle=':', marker='x',
    markersize=5, linewidth=1, label='Pure Random'
)

plt.xlabel('Driver Arrival Rate (λ)', fontsize=30)
plt.ylabel('# Drivers in Queue', fontsize=30)
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
plt.legend(fontsize=22, loc='upper left', framealpha=0.0)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# save plot as eps file locally

plt.savefig(output_dir / 'queue_length_plot.eps', format='eps', dpi=1000)
plt.savefig(output_dir / 'queue_length_plot.png', format='png', dpi=300)



# %%
x = list(range(1, 16))
plt.figure(figsize=(10, 8))  # Square figure

# fb_throughput_list (thin dotted black line)
plt.plot(
    x, fb_throughput_list, color='black', linestyle=':', linewidth=3, label='First Best'
)

# DRFIFO (solid line, diamond markers)
plt.plot(
    x, rand_fifo_throughput_list, color='blue', linestyle='-', marker='d',
    markersize=5, linewidth=1, markerfacecolor='none', label='Dynamic RFIFO'
)

# Direct FIFO (solid line, circle markers)
plt.plot(
    x, direct_throughput_list, color='purple', linestyle='-', marker='o',
    markersize=5, linewidth=1, markerfacecolor='none', label='Direct FIFO'
)

# LIFO (dash-dot line, triangle-up markers)
plt.plot(
    x, lifo_throughput_list, color='green', linestyle='-.', marker='^',
    markersize=5, linewidth=1, markerfacecolor='none', label='LIFO'
)

# Strict FIFO (dashed line, square markers)
plt.plot(
    x, strict_throughput_list, color='red', linestyle='--', marker='s',
    markersize=5, linewidth=1, markerfacecolor='none', label='Strict FIFO'
)

# Pure Random (dotted line, x markers)
plt.plot(
    x, pure_rand_throughput_list, color='#40E0D0', linestyle=':', marker='x',
    markersize=5, linewidth=1, label='Pure Random'
)

plt.xlabel('Driver Arrival Rate (λ)', fontsize=30)
plt.ylabel('Throughput (trips per min)', fontsize=30)
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)

plt.legend(fontsize=22, loc='upper left', framealpha=0.0)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()


plt.savefig(output_dir / 'throughput_plot.eps', format='eps', dpi=1000)
plt.savefig(output_dir / 'throughput_plot.png', format='png', dpi=300)



# %%
x = list(range(1, 16))
plt.figure(figsize=(10, 8))  # Square figure

# fb_ave_waiting_time_list (thin dotted black line)
plt.plot(
    x, fb_waiting_time_list, color='black', linestyle=':', linewidth=3, label='First Best'
)

# DRFIFO (solid line, diamond markers)
plt.plot(
    x, rand_fifo_ss_list, color='blue', linestyle='-', marker='d',
    markersize=5, linewidth=1, markerfacecolor='none', label='Dynamic RFIFO'
)

# Direct FIFO (solid line, circle markers)
plt.plot(
    x, direct_ss_list, color='purple', linestyle='-', marker='o',
    markersize=5, linewidth=1, markerfacecolor='none', label='Direct FIFO'
)

# LIFO (dash-dot line, triangle-up markers)
plt.plot(
    x, lifo_ss_list, color='green', linestyle='-.', marker='^',
    markersize=5, linewidth=1, markerfacecolor='none', label='LIFO'
)

# Strict FIFO (dashed line, square markers)
plt.plot(
    x, strict_ss_list, color='red', linestyle='--', marker='s',
    markersize=5, linewidth=1, markerfacecolor='none', label='Strict FIFO'
)

# Pure Random (dotted line, x markers)
plt.plot(
    x, pure_rand_ss_list, color='#40E0D0', linestyle=':', marker='x',
    markersize=5, linewidth=1, label='Pure Random'
)

plt.xlabel('Driver Arrival Rate (λ)', fontsize=30)
plt.ylabel('Average Waiting Time (min)', fontsize=30)
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
plt.legend(fontsize=22, loc='lower left', bbox_to_anchor=(0.05, 0.15), framealpha=0.0)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

plt.savefig(output_dir / 'waiting_time_plot.eps', format='eps', dpi=1000)
plt.savefig(output_dir / 'waiting_time_plot.png', format='png', dpi=300)


x = list(range(1, 16))
# Plot settings for exact styles
plt.figure(figsize=(10, 8))  # Square figure

# fb_net_revenue_list (thin dotted black line)
plt.plot(
    x, fb_net_revenue_list, color='black', linestyle=':', linewidth=3, label='First Best'
)

sq = np.sqrt(strict_variance_list)

#DRFIFO (solid line, diamond markers)
plt.plot(
    x, np.sqrt(rand_fifo_variance_list), color='blue', linestyle='-', marker='d',
    markersize=5, linewidth=1, markerfacecolor='none', label='Dynamic RFIFO'
)

# Direct FIFO (solid line, circle markers)
plt.plot(
    x, np.sqrt(direct_variance_list), color='purple', linestyle='-', marker='o',
    markersize=5, linewidth=1, markerfacecolor='none', label='Direct FIFO'
)

# LIFO (dash-dot line, triangle-up markers)
plt.plot(
    x, np.sqrt(lifo_variance_list), color='green', linestyle='-.', marker='^',
    markersize=5, linewidth=1, markerfacecolor='none', label='LIFO'
)

# Strict FIFO (dashed line, square markers)
plt.plot(
    x, np.sqrt(strict_variance_list), color='red', linestyle='--', marker='s',
    markersize=5, linewidth=1, markerfacecolor='none', label='Strict FIFO'
)

# Pure Random (dotted line, x markers)
plt.plot(
    x, np.sqrt(pure_rand_variance_list), color='#40E0D0', linestyle=':', marker='x',
    markersize=5, linewidth=1, label='Pure Random'  # 'x' markers are naturally outlined
)


# Labels and ticks
plt.xlabel('Driver Arrival Rate (λ)', fontsize=32)
plt.ylabel('SD of Driver Earnings', fontsize=32)
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)

# Add a legend with smaller font size
plt.legend(fontsize=24, loc='upper left', framealpha=0.0)

# Light grid for clarity
plt.grid(True, linestyle='--', alpha=0.5)

# Tight layout to optimize spacing
plt.tight_layout()

# save plot as eps file locally
plt.savefig(output_dir / 'standard_deviation_plot.eps', format='eps', dpi=1000)
plt.savefig(output_dir / 'standard_deviation_plot.png', format='png', dpi=300)



x = list(range(1, 16))
# Plot settings for exact styles
plt.figure(figsize=(10, 8))  # Square figure

# fb_net_revenue_list (thin dotted black line)
plt.plot(
    x, fb_net_revenue_list, color='black', linestyle=':', linewidth=3, label='First Best'
)

#DRFIFO (solid line, diamond markers)
plt.plot(
    x, rand_fifo_earnings_list, color='blue', linestyle='-', marker='d',
    markersize=5, linewidth=1, markerfacecolor='none', label='Dynamic RFIFO'
)

# Direct FIFO (solid line, circle markers)
plt.plot(
    x, direct_earnings_list, color='purple', linestyle='-', marker='o',
    markersize=5, linewidth=1, markerfacecolor='none', label='Direct FIFO'
)

# LIFO (dash-dot line, triangle-up markers)
plt.plot(
    x, lifo_earnings_list, color='green', linestyle='-.', marker='^',
    markersize=5, linewidth=1, markerfacecolor='none', label='LIFO'
)

# Strict FIFO (dashed line, square markers)
plt.plot(
    x, strict_earnings_list, color='red', linestyle='--', marker='s',
    markersize=5, linewidth=1, markerfacecolor='none', label='Strict FIFO'
)

# Pure Random (dotted line, x markers)
plt.plot(
    x, pure_rand_earnings_list, color='#40E0D0', linestyle=':', marker='x',
    markersize=5, linewidth=1, label='Pure Random'  # 'x' markers are naturally outlined
)


# Labels and ticks
plt.xlabel('Driver Arrival Rate (λ)', fontsize=32)
plt.ylabel('Ave Driver Earnings', fontsize=32)
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)

# Add a legend with smaller font size
plt.legend(fontsize=24, loc='center left', framealpha=0.0)

# Light grid for clarity
plt.grid(True, linestyle='--', alpha=0.5)

# Tight layout to optimize spacing
plt.tight_layout()

# save plot as eps file locally
plt.savefig(output_dir / 'driver_earnings_plot.eps', format='eps', dpi=1000)
plt.savefig(output_dir / 'driver_earnings_plot.png', format='png', dpi=300)
