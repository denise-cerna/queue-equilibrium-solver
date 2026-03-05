"""
Dispatching system for ride-sharing platforms.

This module implements various dispatching strategies for matching drivers
with riders in a queue-based system.
"""

import numpy as np
from typing import Optional, Tuple
from enum import Enum


class DispatchRule(Enum):
    """Enumeration of available dispatch rules."""
    STRICT_FIFO = "STRICT_FIFO"
    PURE_RAND = "PURE_RAND"
    LIFO = "LIFO"
    DYNAMIC_RAND_FIFO = "DYNAMIC_RAND_FIFO"
    DIRECT_FIFO = "DIRECT_FIFO"


class Dispatching:
    """
    Handles dispatching logic for ride-sharing platforms.
    
    Attributes:
        earnings: Array of earnings per job type
        job_rates: Array of arrival rates for each job type
        driver_arrival_rate: Rate at which drivers arrive
        waiting_cost: Cost per unit time of waiting
        patience: Maximum patience level for riders
        Qmax: Maximum queue length
        dispatch_rule: Strategy used for dispatching
    """
    
    def __init__(
        self,
        earnings: np.ndarray,
        job_rates: np.ndarray,
        driver_arrival_rate: float,
        waiting_cost: float,
        patience: int,
        Qmax: int,
        dispatch_rule: str,
        firm1_dispatch_rule: Optional[str] = None,
        firm2_dispatch_rule: Optional[str] = None
    ):
        """Initialize the Dispatching system."""
        # Validate inputs
        self._validate_inputs(earnings, job_rates, patience, Qmax)
        
        self.earnings = np.asarray(earnings)
        self.job_rates = np.asarray(job_rates)
        self.driver_arrival_rate = driver_arrival_rate
        self.waiting_cost = waiting_cost
        self.patience = patience
        self.Qmax = Qmax
        self.dispatch_rule = dispatch_rule
        
        # Optional firm-specific rules (currently unused)
        self.firm1_dispatch_rule = firm1_dispatch_rule
        self.firm2_dispatch_rule = firm2_dispatch_rule

    @staticmethod
    def _validate_inputs(
        earnings: np.ndarray,
        job_rates: np.ndarray,
        patience: int,
        Qmax: int
    ) -> None:
        """Validate input parameters."""
        if len(earnings) != len(job_rates):
            raise ValueError("earnings and job_rates must have the same length")
        if patience < 1:
            raise ValueError("patience must be >= 1")
        if Qmax < 1:
            raise ValueError("Qmax must be >= 1")

    def get_dispatch_prob(self) -> np.ndarray:
        """
        Compute dispatch probability tensor based on the dispatch rule.
        
        Returns:
            4D numpy array of shape (patience, Qmax+1, Qmax+1, num_jobs)
            representing dispatch probabilities.
            
        Raises:
            ValueError: If dispatch_rule is not one of the supported rules.
        """
        num_jobs = len(self.job_rates)
        shape = (self.patience, self.Qmax + 1, self.Qmax + 1, num_jobs)
        dispatch_tensor = np.zeros(shape)
        
        # Job 0: Zero-paying job (reneging option)
        dispatch_tensor[:, :, :, 0] = self._compute_reneging_policy()
        
        # Compute policy for paying jobs based on dispatch rule
        rule_handlers = {
            "STRICT_FIFO": self._compute_strict_fifo,
            "PURE_RAND": self._compute_pure_random,
            "LIFO": self._compute_lifo,
            "DYNAMIC_RAND_FIFO": self._compute_dynamic_rand_fifo,
            "DIRECT_FIFO": self._compute_direct_fifo,
        }
        
        handler = rule_handlers.get(self.dispatch_rule)
        if handler is None:
            valid_rules = ", ".join(rule_handlers.keys())
            raise ValueError(
                f"Invalid dispatch rule: '{self.dispatch_rule}'. "
                f"Must be one of: {valid_rules}"
            )
        
        dispatch_tensor = handler(dispatch_tensor)
        
        print(f"Dispatch Rule '{self.dispatch_rule}' computed successfully")
        return dispatch_tensor

    def _compute_reneging_policy(self) -> np.ndarray:
        """Compute reneging policy (job 0) - same across all patience levels."""
        policy = np.zeros((self.patience, self.Qmax + 1, self.Qmax + 1))
        policy[:, 0, :] = 0
        policy[:, :, 0] = 0
        policy[0, :, :] = np.ones((self.Qmax + 1, self.Qmax + 1))
        return policy

    def _compute_strict_fifo(self, dispatch_tensor: np.ndarray) -> np.ndarray:
        """Strict First-In-First-Out dispatching."""
        for p in range(self.patience):
            policy = np.zeros((self.Qmax + 1, self.Qmax + 1))
            policy[p + 1, 1:] = 1
            dispatch_tensor[p, :, :, 1:] = policy[:, :, None]
        return dispatch_tensor

    def _calculate_bin_range(self, p: int, Q: int) -> Tuple[int, int]:
        """Calculate the bin range for a given patience level and queue size."""
        if Q <= self.patience and p < Q:
            return p + 1, p + 2
        
        num_per_bin = Q // self.patience
        residual = Q % self.patience
        
        bin_start = (
            min(residual, p) * (num_per_bin + 1) +
            max(p - residual, 0) * num_per_bin
        )
        bin_end = bin_start + (1 if residual > p else 0) + num_per_bin
        
        return bin_start + 1, bin_end + 1

    def _compute_pure_random(self, dispatch_tensor: np.ndarray) -> np.ndarray:
        """Uniform random dispatching."""
        policy = np.zeros((self.Qmax + 1, self.Qmax + 1))
        for q in range(1, self.Qmax + 1):
            policy[q, q:] = 1.0 / np.arange(q, self.Qmax + 1)
        
        for p in range(self.patience):
            dispatch_tensor[p, :, :, 1:] = policy[:, :, None]
        return dispatch_tensor

    def _compute_lifo(self, dispatch_tensor: np.ndarray) -> np.ndarray:
        """Last-In-First-Out dispatching."""
        for p in range(self.patience):
            policy = np.zeros((self.Qmax + 1, self.Qmax + 1))
            np.fill_diagonal(policy[:, p:], 1)
            policy[0, :] = 0
            policy[:, 0] = 0
            dispatch_tensor[p, :, :, 1:] = policy[:, :, None]
        return dispatch_tensor

    def _compute_dynamic_rand_fifo(self, dispatch_tensor: np.ndarray) -> np.ndarray:
        """Dynamic random FIFO within bins."""
        for p in range(self.patience):
            policy = np.zeros((self.Qmax+1, self.Qmax+1))
            for Q in range(1, self.Qmax+1):
                if Q <= self.patience:
                    policy[p+1, Q] = 1
                else:
                    num_per_bin = Q // self.patience
                    residual    = Q % self.patience
                    bin_start = min(residual, p)*(num_per_bin+1) \
                            + max(p-residual, 0)*num_per_bin
                    bin_end   = bin_start + int(residual>p) + num_per_bin
                    for q in range(bin_start+1, bin_end+1):
                        policy[q, Q] = 1.0/(bin_end-bin_start)
            dispatch_tensor[p, :, :, 1:] = policy[:, :, None]
        return dispatch_tensor

    def _compute_direct_fifo(self, dispatch_tensor: np.ndarray) -> np.ndarray:
        """Direct FIFO with predetermined queue positions."""
        positions = np.floor(self._compute_direct_fifo_positions()).astype(int)
        total_rate = np.sum(self.job_rates)
        print(f'Direct FIFO positions for μ = {total_rate}: {positions}')
        
        num_jobs = len(self.job_rates)
        for i in range(1, num_jobs):
            start = positions[i - 1]
            for p in range(self.patience):
                policy = np.zeros((self.Qmax + 1, self.Qmax + 1))
                position = start + p + 1
                if position <= self.Qmax:
                    policy[position, 1:] = 1
                dispatch_tensor[p, :, :, i] = policy
        
        return dispatch_tensor

    def _compute_direct_fifo_positions(self) -> np.ndarray:
        """
        Calculate direct FIFO positions for each job type based on
        willingness to wait and throughput.
        
        Returns:
            Array of starting positions for each job type.
        """
        num_job_types = len(self.job_rates) - 1
        positions = np.zeros(num_job_types)
        
        for i in range(1, num_job_types):
            cumulative_throughput = np.sum(self.job_rates[1:i + 1])
            earnings_diff = self.earnings[i] - self.earnings[i + 1]
            willingness_to_wait = earnings_diff / self.waiting_cost
            
            positions[i] = (
                positions[i - 1] +
                willingness_to_wait * cumulative_throughput
            )
        
        return positions
