import pandas as pd
import matplotlib.pyplot as plt
from ipywidgets import widgets
from IPython.display import display
from src.outcome_class import Outcome
import numpy as np
import dill
import os
from mpl_toolkits.mplot3d import Axes3D
from src.utilities import *
from typing import List, Tuple, Optional
from pathlib import Path


class Reporting:
    """
    Generates reports and visualizations from simulation outcomes.
    
    Loads outcome data from a dill file and provides methods to plot
    convergence metrics, steady-state distributions, and policy decisions.
    """
    
    def __init__(self, filename: str):
        """
        Initialize reporting with outcome data from file.
        
        Args:
            filename: Path to the dill file containing outcome list
        """
        self.outcome_list = read_dill(filename)
        
        # Extract and organize historical data
        (self.net_revenue_history,
         self.throughput_history,
         self.norm_history,
         self.steady_state_history,
         self.phi_history,
         self.V_history,
         self.saved_iters,
         self.max_loss_V_history,
         self.max_loss_phi_history,
         self.max_loss_inspect_history,
         self.alpha_history) = self._extract_data_from_outcomes()

    def _extract_data_from_outcomes(self) -> Tuple[np.ndarray, ...]:
        """
        Extract historical data from outcome list and save to files.
        
        Returns:
            Tuple of numpy arrays containing historical metrics
        """
        # Initialize collectors
        collectors = {
            'alpha': [],
            'V': [],
            'phi': [],
            'net_revenue': [],
            'throughput': [],
            'steady_state': [],
            'norm': [],
            'iter': [],
            'max_loss_V': [],
            'max_loss_phi': [],
            'max_loss_inspect': []
        }
        
        # Collect data from each outcome
        for outcome in self.outcome_list:
            for key in collectors:
                if hasattr(outcome, key):
                    collectors[key].append(getattr(outcome, key))
        
        # Calculate expected steady state from last outcome
        steady_state_exp = []
        last_outcome = self.outcome_list[-1]
        if hasattr(last_outcome, 'steady_state'):
            steady_state = last_outcome.steady_state
            steady_state_exp.append(
                sum(steady_state[i] * i for i in range(len(steady_state)))
            )
        
        # Save data to files
        self._save_data_to_files(collectors, steady_state_exp, last_outcome)
        
        # Convert to numpy arrays
        return self._convert_to_arrays(collectors)

    def _save_data_to_files(
        self, 
        collectors: dict, 
        steady_state_exp: List[float], 
        last_outcome: Outcome
    ) -> None:
        """
        Save collected data to CSV and NPY files.
        
        Args:
            collectors: Dictionary of collected metric lists
            steady_state_exp: Expected steady state values
            last_outcome: Last outcome object for final values
        """
        output_dir = Path('plot_a_run_data')
        output_dir.mkdir(exist_ok=True)
        
        # Save scalar histories
        scalar_files = {
            'throughput_by_iter.csv': collectors['throughput'],
            'net_revenue_by_iter.csv': collectors['net_revenue'],
            'steady_state_exp_by_iter.csv': steady_state_exp,
            'norm_by_iter.csv': collectors['norm'],
            'saved_iterations.csv': collectors['iter']
        }
        
        for filename, data in scalar_files.items():
            if data:
                np.savetxt(output_dir / filename, data, delimiter=",")
        
        # Save final vectors
        final_data = {
            'final_alpha.npy': ('alpha', np.save),
            'final_V.csv': ('V', lambda path, data: np.savetxt(path, data, delimiter=",")),
            'final_phi.csv': ('phi', lambda path, data: np.savetxt(path, data, delimiter=",")),
            'final_inspect.csv': ('inspect', lambda path, data: np.savetxt(path, data, delimiter=",")),
            'final_steady_state.csv': ('steady_state', lambda path, data: np.savetxt(path, data, delimiter=","))
        }
        
        for filename, (attr, save_func) in final_data.items():
            if hasattr(last_outcome, attr):
                save_func(output_dir / filename, getattr(last_outcome, attr))

    def _convert_to_arrays(self, collectors: dict) -> Tuple[np.ndarray, ...]:
        """
        Convert collected data lists to numpy arrays.
        
        Args:
            collectors: Dictionary of collected metric lists
            
        Returns:
            Tuple of numpy arrays for each metric
        """
        return (
            np.array(collectors['net_revenue']),
            np.array(collectors['throughput']),
            np.array(collectors['norm']),
            np.array(collectors['steady_state']),
            np.array(collectors['phi'], dtype=object),
            np.array(collectors['V'], dtype=object),
            np.array(collectors['iter']),
            np.array(collectors['max_loss_V']),
            np.array(collectors['max_loss_phi']),
            np.array(collectors['max_loss_inspect']),
            np.array(collectors['alpha'], dtype=object)
        )

    def print_alpha_matrix(self, iteration: int = -1) -> None:
        """
        Print the alpha matrix for a given iteration.
        
        Args:
            iteration: Iteration index (default: -1 for last iteration)
        """
        print(f"Alpha for iteration {iteration}:")
        alpha = self.alpha_history[iteration]
        
        for l in range(alpha.shape[2]):
            print(f"\nAlpha for job type {l}:")
            print(alpha[:, :, l])
    
    def print_V_matrix(self, iteration: int = -1) -> None:
        """
        Print the V matrix for a given iteration.
        
        Args:
            iteration: Iteration index (default: -1 for last iteration)
        """
        print(f"V for iteration {iteration}:")
        print(self.V_history[iteration])

    def plot_alpha(self, q: int, Q: int, l: int) -> None:
        """
        Plot alpha value over iterations for specific indices.
        
        Args:
            q: Queue position index
            Q: Queue length index
            l: Job type index
        """
        alpha = self.alpha_history
        
        x = list(range(len(alpha)))
        y = [alpha[i, q, Q, l] for i in range(len(alpha))]
        
        plt.figure(figsize=(10, 5))
        plt.plot(x, y)
        plt.xlabel('Iteration')
        plt.ylabel('Alpha')
        plt.title(f'Alpha vs Iteration (q={q}, Q={Q}, l={l})')
        plt.grid(True)
        plt.show()

    def display_alpha(self) -> None:
        """Display interactive sliders for exploring alpha values."""
        q_slider = widgets.IntSlider(
            min=0, 
            max=self.alpha_history.shape[1] - 1, 
            step=1, 
            description='q:',
            continuous_update=False
        )
        
        Q_slider = widgets.IntSlider(
            min=0, 
            max=self.alpha_history.shape[2] - 1, 
            step=1, 
            description='Q:',
            continuous_update=False
        )
        
        l_slider = widgets.IntSlider(
            min=0, 
            max=self.alpha_history.shape[3] - 1, 
            step=1, 
            description='l:',
            continuous_update=False
        )
        
        plot = widgets.interactive(
            self.plot_alpha, 
            q=q_slider, 
            Q=Q_slider, 
            l=l_slider
        )
        
        display(plot)

    def plot_V(self, q: int, Q: int) -> None:
        """
        Plot V value over iterations for specific indices.
        
        Args:
            q: Queue position index
            Q: Queue length index
        """
        V = self.V_history
        
        x = list(range(len(V)))
        y = [V[i, q, Q] for i in range(len(V))]
        
        plt.figure(figsize=(10, 5))
        plt.plot(x, y)
        
        # Add reference lines for earnings if available
        last_outcome = self.outcome_list[-1]
        if hasattr(last_outcome, 'earnings'):
            for earning in last_outcome.earnings:
                plt.axhline(y=earning, color='r', linestyle='--', alpha=0.5)
        
        plt.xlabel('Iteration')
        plt.ylabel('V')
        plt.title(f'V vs Iteration (q={q}, Q={Q})')
        plt.grid(True)
        plt.show()

    def display_V(self) -> None:
        """Display interactive sliders for exploring V values."""
        q_slider = widgets.IntSlider(
            min=0, 
            max=self.alpha_history.shape[1] - 1, 
            step=1, 
            description='q:',
            continuous_update=False
        )
        
        Q_slider = widgets.IntSlider(
            min=0, 
            max=self.alpha_history.shape[2] - 1, 
            step=1, 
            description='Q:',
            continuous_update=False
        )
        
        plot = widgets.interactive(
            self.plot_V, 
            q=q_slider, 
            Q=Q_slider
        )
        
        display(plot)

    def plot_phi(self, iteration: int) -> None:
        """
        Plot joining decision probabilities for a given iteration.
        
        Args:
            iteration: Iteration index
        """
        plt.figure(figsize=(10, 5))
        
        x = list(range(self.phi_history.shape[1]))
        y = self.phi_history[iteration, :]
        
        plt.bar(x, y)
        plt.xlabel('Queue Length')
        plt.ylabel('Probability of Joining')
        plt.title(f'Joining Decision (Iteration {self.saved_iters[iteration]})')
        plt.grid(True)
        plt.show()

    def plot_steady_state(self, iteration: int) -> None:
        """
        Plot steady-state distribution and expected queue length.
        
        Args:
            iteration: Iteration index
        """
        # Plot steady-state distribution
        plt.figure(figsize=(10, 5))
        
        x = list(range(self.steady_state_history.shape[1]))
        y = self.steady_state_history[iteration, :]
        
        plt.bar(x, y)
        plt.xlabel('Queue Length')
        plt.ylabel('Probability')
        plt.title(f'Steady State Distribution (Iteration {self.saved_iters[iteration]})')
        plt.grid(True)
        plt.show()
        
        # Calculate and plot expected queue length
        expected_queue_length = np.sum(self.steady_state_history[iteration, :] * x)
        
        plt.figure(figsize=(10, 5))
        plt.bar([1], [expected_queue_length])
        plt.xlabel('Metric')
        plt.ylabel('Expected Queue Length')
        plt.title(f'Expected Queue Length: {expected_queue_length:.2f}')
        plt.xticks([1], ['Expected Queue Length'])
        plt.grid(True)
        plt.show()

    def plot_net_revenue(self) -> None:
        """Plot net revenue over iterations."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.saved_iters, self.net_revenue_history)
        plt.xlabel('Iteration')
        plt.ylabel('Net Revenue')
        plt.title('Net Revenue vs Iteration')
        plt.grid(True)
        plt.show()
    
    def plot_throughput(self) -> None:
        """Plot throughput over iterations."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.saved_iters, self.throughput_history)
        plt.xlabel('Iteration')
        plt.ylabel('Throughput')
        plt.title('Throughput vs Iteration')
        plt.grid(True)
        plt.show()

    def plot_norm(self) -> None:
        """Plot infinity norm over iterations (log scale)."""
        plt.figure(figsize=(10, 5))
        plt.yscale('log')
        plt.plot(self.saved_iters, self.norm_history)
        plt.xlabel('Iteration')
        plt.ylabel('Infinity Norm (log scale)')
        plt.title('V Infinity Norm vs Iteration')
        plt.grid(True)
        plt.show()

    def plot_max_loss_V(self) -> None:
        """Plot maximum V loss over iterations (log scale)."""
        plt.figure(figsize=(10, 5))
        plt.yscale('log')
        plt.plot(self.saved_iters, self.max_loss_V_history)
        plt.xlabel('Iteration')
        plt.ylabel('Max V Loss (log scale)')
        plt.title('Maximum V Loss vs Iteration')
        plt.grid(True)
        plt.show()

    def plot_max_loss_phi(self) -> None:
        """Plot maximum phi loss over iterations (log scale)."""
        plt.figure(figsize=(10, 5))
        plt.yscale('log')
        plt.plot(self.saved_iters, self.max_loss_phi_history)
        plt.xlabel('Iteration')
        plt.ylabel('Max Phi Loss (log scale)')
        plt.title('Maximum Phi Loss vs Iteration')
        plt.grid(True)
        plt.show()

    def plot_max_loss_inspect(self) -> None:
        """Plot maximum inspect loss over iterations (log scale)."""
        plt.figure(figsize=(10, 5))
        plt.yscale('log')
        plt.plot(self.saved_iters, self.max_loss_inspect_history)
        plt.xlabel('Iteration')
        plt.ylabel('Max Inspect Loss (log scale)')
        plt.title('Maximum Inspect Loss vs Iteration')
        plt.grid(True)
        plt.show()

    def _create_iteration_slider(self) -> widgets.IntSlider:
        """
        Create an iteration slider widget.
        
        Returns:
            IntSlider widget for selecting iterations
        """
        return widgets.IntSlider(
            value=0,
            min=0,
            max=len(self.saved_iters) - 1,
            step=1,
            description='Iteration:',
            continuous_update=False
        )

    def plot_run(self) -> None:
        """
        Generate comprehensive report with all key plots.
        
        Creates plots for:
        - Net revenue over iterations
        - Throughput over iterations
        - Norm convergence
        - Steady-state distribution (interactive)
        - Loss function values
        """
        # Main convergence metrics
        self.plot_net_revenue()
        self.plot_throughput()
        self.plot_norm()

        # Interactive steady-state plot
        steady_state_plot = widgets.interactive(
            self.plot_steady_state, 
            df=widgets.fixed(self.steady_state_history), 
            iteration=self._create_iteration_slider()
        )
        display(steady_state_plot)

        # Loss function plots
        self.plot_max_loss_V()
        self.plot_max_loss_phi()
        self.plot_max_loss_inspect()