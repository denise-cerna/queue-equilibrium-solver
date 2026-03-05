"""
Simulator class for running analytical policy iteration algorithms.

Implements various equilibrium-finding algorithms for queueing systems with
strategic drivers and riders.
"""

import numpy as np
from src.outcome_class import Outcome
from src.utilities import *
import time
from typing import Tuple, List, Optional, Literal


class Simulator:
    """
    Runs policy iteration algorithms to find equilibria in queueing systems.
    
    Supports multiple algorithm variants:
    - '0-MIX': Approximate algorithm (ignores second-order cascading effects)
    - '0-GRAD': Gradient-based algorithm (finds precise equilibrium)
    - '0-PAR': Parallel dispatch algorithm
    - '0-MIX-GRAD': Hybrid approach (starts with MIX, refines with GRAD)
    - '0-COMP': Competition mode algorithm
    """
    
    def __init__(
        self,
        system: object,
        max_iterations: int,
        tolerance: float,
        version: Literal['0-MIX', '0-GRAD', '0-PAR', '0-MIX-GRAD', '0-COMP'] = '0-MIX',
        saving_multiple: int = 5000
    ):
        """
        Initialize simulator with system configuration and algorithm parameters.
        
        Args:
            system: System object containing dispatching rules and settings
            max_iterations: Maximum number of iterations to run
            tolerance: Convergence tolerance for equilibrium checks
            version: Algorithm variant to use
            saving_multiple: Save intermediate results every N iterations
        """
        # System configuration
        self.system = system
        self.setting = system.setting
        
        # Algorithm parameters
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.version = version
        self.saving_multiple = saving_multiple
        
        # Extract system parameters for convenience
        self.earnings = system.setting.earnings
        self.job_rates = system.setting.job_rates
        self.driver_arrival_rate = system.setting.driver_arrival_rate
        self.waiting_cost = system.setting.waiting_cost
        self.inspection_cost = system.setting.inspection_cost
        self.patience = system.patience
        self.Qmax = system.setting.Qmax
        self.flag_verbose = system.setting.flag_verbose
        
        # Iteration tracking
        self.iter = 0
        self.current_time = 0.0
        self.gamma = 0.0
        
        # Loss metrics
        self.loss_V_sum = 0.0
        self.loss_phi_sum = 0.0
        self.loss_inspect_sum = 0.0
        self.max_loss_V = 1e6
        self.max_loss_phi = 1e6
        self.max_loss_inspect = 1e6
        self.total_loss = 0.0
        
        # Loss arrays for detailed analysis
        self.loss_phi = None
        self.loss_inspect = None
        
        # Momentum tracking
        self.direction_history: List[np.ndarray] = []
        self.insp_direction_history: List[np.ndarray] = []
        
        # Initialize outcome history
        self.outcome_history = self._initialize_outcome_history()

    def _initialize_outcome_history(self) -> List[Outcome]:
        """
        Create initial outcome with zero iteration values.
        
        Returns:
            List containing the initial outcome
        """
        nu_q_Q = np.zeros((self.Qmax + 1, self.Qmax + 1, len(self.job_rates)))
        
        first_outcome = Outcome(
            V=self.setting.V,
            alpha=self.setting.alpha,
            norm=0,
            phi=self.setting.phi,
            inspect=self.setting.inspect,
            setting=self.setting,
            nu_q_Q=nu_q_Q,
            iter=self.iter,
            total_loss_V=self.loss_V_sum,
            total_loss_phi=self.loss_phi_sum,
            max_loss_V=self.max_loss_V,
            max_loss_phi=self.max_loss_phi,
            max_loss_inspect=self.max_loss_inspect
        )
        
        return [first_outcome]

    def update_history(
        self,
        V_t: np.ndarray,
        alpha_t: np.ndarray,
        phi: np.ndarray,
        inspect_t: np.ndarray,
        nu_q_Q: np.ndarray,
        nu_1: Optional[np.ndarray] = None,
        nu_2: Optional[np.ndarray] = None,
        converged: Optional[bool] = None
    ) -> None:
        """
        Add current iteration results to outcome history.
        
        Args:
            V_t: Current continuation values
            alpha_t: Current acceptance policy
            phi: Current joining policy
            inspect_t: Current inspection policy
            nu_q_Q: Queue-dependent service rates
            nu_1: Lower bound rates (for competition/parallel)
            nu_2: Upper bound rates (for competition/parallel)
            converged: Whether the algorithm has converged (for logging)
        """
        outcome = Outcome(
            V=V_t,
            alpha=alpha_t,
            norm=self._compute_norm(V_t),
            phi=phi,
            inspect=inspect_t,
            setting=self.setting,
            nu_q_Q=nu_q_Q,
            iter=self.iter,
            total_loss_V=self.total_loss,
            total_loss_phi=self.loss_phi_sum,
            max_loss_V=self.max_loss_V,
            max_loss_phi=self.max_loss_phi,
            max_loss_inspect=self.max_loss_inspect,
            nu_1=nu_1,
            nu_2=nu_2,
            converged=converged
        )
        self.outcome_history.append(outcome)

    def _compute_norm(self, V: np.ndarray) -> float:
        """
        Compute infinity norm of change in V from previous iteration.
        
        Args:
            V: Current continuation values
            
        Returns:
            Infinity norm of difference
        """
        if self.outcome_history[-1].V is None or len(self.outcome_history[-1].V) == 0:
            norm = 0.0
        else:
            norm = np.linalg.norm(V - self.outcome_history[-1].V, ord=np.inf)
        
        print(f"Infinity Norm: {norm}")
        return norm

    # ==================== Policy Updates ====================

    def update_phi(self, phi_old: np.ndarray, V: np.ndarray) -> np.ndarray:
        """
        Update joining policy based on continuation values.
        
        Args:
            phi_old: Current joining policy
            V: Continuation values
            
        Returns:
            Updated joining policy
        """
        # Direction for phi (accounts for joining at front or back)
        d_s = np.diag(V)*(1-self.setting.join_at_front) + V[1, :]*self.setting.join_at_front
        
        # Update and clip to [0, 1]
        phi_new = phi_old + self.gamma / self.setting.gamma_phi_divide_by * d_s
        clipped_phi = np.clip(phi_new, 0, 1)
        clipped_phi[0] = 0  # Can't join empty queue
        
        return clipped_phi

    def update_inspect(
        self,
        inspect_old: np.ndarray,
        alpha_old: np.ndarray,
        nu_q_Q: np.ndarray,
        tau_q_Q: np.ndarray,
        V_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Update inspection policy using gradient ascent.
        
        Args:
            inspect_old: Current inspection policy
            alpha_old: Current acceptance policy
            nu_q_Q: Service rates
            tau_q_Q: Inspection-adjusted rates
            V_matrix: Continuation values expanded for all job types
            
        Returns:
            Updated inspection policy
        """
        # Gradient direction
        d_s = (np.sum(tau_q_Q * alpha_old * (self.earnings - V_matrix), axis=2) - 
               np.sum(nu_q_Q[:, :, 1:] * self.inspection_cost, axis=2))
        
        # Add momentum if enabled
        if (self.setting.beta > 0 and 
            len(self.insp_direction_history) > 0 and 
            self.iter > self.setting.momentum_start_iter):
            print("Using momentum for inspect!")
            d_s = (self.setting.beta * self.insp_direction_history[-1] + 
                   (1 - self.setting.beta) * d_s)
        
        self.insp_direction_history.append(d_s)
        
        # Update and clip
        inspect_new = inspect_old + (self.gamma / self.setting.gamma_inspect_divide_by) * d_s
        clipped_inspect = np.clip(inspect_new, 0, 1)
        
        # Zero out unreachable states
        clipped_inspect = self._zero_unreachable_states(clipped_inspect)
        
        return clipped_inspect

    def update_alpha_mixed(
        self,
        alpha_old: np.ndarray,
        phi_old: np.ndarray,
        V: np.ndarray,
        inspect_old: np.ndarray,
        nu_q_Q: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Update acceptance policy using mixed algorithm.
        
        Args:
            alpha_old: Current acceptance policy
            phi_old: Current joining policy
            V: Continuation values
            inspect_old: Current inspection policy
            nu_q_Q: Service rates
            
        Returns:
            Tuple of (updated alpha, updated phi, updated inspect)
        """
        # Expand V for all job types
        V_matrix = np.stack([V.copy() for _ in range(len(self.job_rates))], axis=-1)
        
        # Gradient direction
        d_s = self.earnings - V_matrix
        
        # Add momentum if enabled
        if (self.setting.beta > 0 and 
            len(self.direction_history) > 0 and 
            self.iter > self.setting.momentum_start_iter):
            print("Using momentum for alpha!")
            d_s = (self.setting.beta * self.direction_history[-1] + 
                   (1 - self.setting.beta) * d_s)
        
        self.direction_history.append(d_s)
        
        # Update alpha
        alpha_new = alpha_old + self.gamma * d_s
        clipped_alpha = np.clip(alpha_new, 0, 1)
        clipped_alpha = self._zero_unreachable_states(clipped_alpha)
        
        # Update phi
        clipped_phi = self.update_phi(phi_old, V)
        
        # Update inspect if needed
        if self.inspection_cost > 0:
            clipped_inspect = self.update_inspect(
                inspect_old, alpha_old, nu_q_Q, nu_q_Q, V_matrix
            )
        else:
            clipped_inspect = inspect_old
        
        return clipped_alpha, clipped_phi, clipped_inspect

    def update_alpha_parallel(
        self,
        alpha_old: np.ndarray,
        phi_old: np.ndarray,
        inspect_old: np.ndarray,
        V: np.ndarray,
        nu_q_Q: np.ndarray,
        tau_q_Q: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Update policies for parallel dispatch.
        
        Args:
            alpha_old: Current acceptance policy
            phi_old: Current joining policy
            inspect_old: Current inspection policy
            V: Continuation values
            nu_q_Q: Service rates
            tau_q_Q: Inspection-adjusted rates
            
        Returns:
            Tuple of (updated alpha, updated phi, updated inspect)
        """
        # Expand V for all job types
        V_matrix = np.stack([V.copy() for _ in range(len(self.job_rates))], axis=-1)
        
        # Gradient direction
        d_s = self.earnings - V_matrix
        
        # Add momentum if enabled
        if (self.setting.beta > 0 and 
            len(self.direction_history) > 0 and 
            self.iter > self.setting.momentum_start_iter):
            print("Using momentum for alpha!")
            d_s = (self.setting.beta * self.direction_history[-1] + 
                   (1 - self.setting.beta) * d_s)
        
        self.direction_history.append(d_s)
        
        # Update alpha
        alpha_new = alpha_old + self.gamma * d_s
        clipped_alpha = np.clip(alpha_new, 0, 1)
        clipped_alpha = self._zero_unreachable_states(clipped_alpha)
        
        # Update phi
        clipped_phi = self.update_phi(phi_old, V)
        
        # Update inspect if needed
        if self.setting.update_inspect:
            clipped_inspect = self.update_inspect(
                inspect_old, alpha_old, nu_q_Q, tau_q_Q, V_matrix
            )
        else:
            clipped_inspect = inspect_old
        
        return clipped_alpha, clipped_phi, clipped_inspect

    def update_alpha_gradient(
        self,
        alpha_old: np.ndarray,
        phi_old: np.ndarray,
        V: np.ndarray,
        grad_direction: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update policies using gradient direction.
        
        Args:
            alpha_old: Current acceptance policy
            phi_old: Current joining policy
            V: Continuation values
            grad_direction: Gradient direction
            
        Returns:
            Tuple of (updated alpha, updated phi)
        """
        d_s = grad_direction
        
        # Add momentum if enabled
        if (self.setting.beta > 0 and 
            len(self.direction_history) > 0 and 
            self.iter > self.setting.momentum_start_iter):
            print("Using momentum!")
            d_s = (self.setting.beta * self.direction_history[-1] + 
                   (1 - self.setting.beta) * d_s)
        
        self.direction_history.append(d_s)
        
        # Update alpha
        alpha_new = alpha_old + self.gamma * d_s
        clipped_alpha = np.clip(alpha_new, 0, 1)
        clipped_alpha = self._zero_unreachable_states(clipped_alpha)
        
        # Update phi
        clipped_phi = self.update_phi(phi_old, V)
        
        return clipped_alpha, clipped_phi

    # ==================== Equilibrium Checking ====================

    def check_joining_br(
        self,
        phi: np.ndarray,
        V: np.ndarray,
        flag_verbose: bool = False
    ) -> bool:
        """
        Check if joining policy is a best response.
        
        Args:
            phi: Joining policy
            V: Continuation values
            flag_verbose: Whether to print detailed output
            
        Returns:
            True if joining policy satisfies equilibrium conditions
        """
        # Expected payoff from joining
        d_join_back = np.diag(V) * (1 - self.setting.join_at_front)
        d_join_front= V[1, :] * self.setting.join_at_front
        d_join = d_join_front + d_join_back
        
        # Compute loss
        positive_part = np.maximum(d_join, 0)
        negative_part = np.abs(np.minimum(d_join, 0))
        loss_phi = positive_part * (1 - phi) + negative_part * phi
        
        # Store for analysis
        self.loss_phi = loss_phi
        self.loss_phi_sum = np.sum(loss_phi[1:])
        self.max_loss_phi = np.max(loss_phi[1:])
        
        if flag_verbose:
            print(f'Sum phi loss: {self.loss_phi_sum}')
        print(f'Max phi loss: {self.max_loss_phi}')
        
        return np.all(loss_phi[1:] <= self.tolerance)

    def check_inspect_decision(
        self,
        inspect: np.ndarray,
        nu_q_Q: np.ndarray,
        tau_q_Q: np.ndarray,
        alpha: np.ndarray,
        V: np.ndarray
    ) -> bool:
        """
        Check if inspection policy is a best response.
        
        Args:
            inspect: Inspection policy
            nu_q_Q: Service rates
            tau_q_Q: Inspection-adjusted rates
            alpha: Acceptance policy
            V: Continuation values
            
        Returns:
            True if inspection policy satisfies equilibrium conditions
        """
        # Expand V for all job types
        V_matrix = np.stack([V.copy() for _ in range(len(self.job_rates))], axis=-1)
        
        # Compute gradient
        d_inspect = (np.sum(tau_q_Q * alpha * (self.earnings - V_matrix), axis=2) - 
                     np.sum(nu_q_Q[:, :, 1:] * self.inspection_cost, axis=2))
        
        # Compute loss
        positive_part = np.maximum(d_inspect, 0)
        negative_part = np.abs(np.minimum(d_inspect, 0))
        loss_inspect = positive_part * (1 - inspect) + negative_part * inspect
        
        # Zero out unreachable states
        loss_inspect = self._zero_unreachable_states(loss_inspect)
        
        # Store for analysis
        self.loss_inspect = loss_inspect
        self.loss_inspect_sum = np.sum(loss_inspect)
        self.max_loss_inspect = np.max(loss_inspect)
        
        print(f'Max inspect loss: {self.max_loss_inspect}')
        
        return np.all(loss_inspect <= self.tolerance)

    def check_equilibrium_mixed(
        self,
        alpha: np.ndarray,
        phi: np.ndarray,
        V: np.ndarray,
        inspect: np.ndarray,
        nu_q_Q: np.ndarray
    ) -> bool:
        """
        Check equilibrium conditions for mixed algorithm.
        
        Args:
            alpha: Acceptance policy
            phi: Joining policy
            V: Continuation values
            inspect: Inspection policy
            nu_q_Q: Service rates
            
        Returns:
            True if all equilibrium conditions are satisfied
        """
        # Check acceptance policy
        V_matrix = np.stack([V.copy() for _ in range(len(self.job_rates))], axis=-1)
        d_V = self.earnings - V_matrix
        
        positive_part = np.maximum(d_V, 0)
        negative_part = np.abs(np.minimum(d_V, 0))
        loss_V = positive_part * (1 - alpha) + negative_part * alpha
        
        # Zero out unreachable states
        loss_V = self._zero_unreachable_states(loss_V)
        
        # Store metrics
        self.loss_V_sum = np.sum(loss_V)
        self.max_loss_V = np.max(loss_V)
        
        if self.flag_verbose:
            print(f'Sum V loss: {self.loss_V_sum}')
        print(f'Max V loss: {self.max_loss_V}')
        
        flag_V = np.all(loss_V <= self.tolerance)
        
        # Check joining policy
        flag_phi = self.check_joining_br(phi, V)
        
        # Check inspection policy
        if self.inspection_cost > 0:
            flag_inspect = self.check_inspect_decision(
                inspect, nu_q_Q, nu_q_Q, alpha, V
            )
        else:
            flag_inspect = True
            self.loss_inspect_sum = 0
        
        # Calculate total loss
        self.total_loss = self.loss_V_sum + self.loss_phi_sum + self.loss_inspect_sum
        print(f'Total loss: {self.total_loss}')
        
        # Check convergence
        if (flag_V and flag_phi and flag_inspect) or self.total_loss <= self.tolerance:
            return True
        
        # Report violations if verbose
        if self.flag_verbose:
            self._report_equilibrium_violations(flag_V, flag_phi, flag_inspect, loss_V)
        
        return False

    def check_equilibrium_parallel(
        self,
        alpha: np.ndarray,
        phi: np.ndarray,
        inspect: np.ndarray,
        V: np.ndarray,
        nu_q_Q: np.ndarray,
        tau_q_Q: np.ndarray
    ) -> bool:
        """
        Check equilibrium conditions for parallel dispatch.
        
        Args:
            alpha: Acceptance policy
            phi: Joining policy
            inspect: Inspection policy
            V: Continuation values
            nu_q_Q: Service rates
            tau_q_Q: Inspection-adjusted rates
            
        Returns:
            True if all equilibrium conditions are satisfied
        """
        # Check acceptance policy
        V_matrix = np.stack([V.copy() for _ in range(len(self.job_rates))], axis=-1)
        d_V = self.earnings - V_matrix
        
        positive_part = np.maximum(d_V, 0)
        negative_part = np.abs(np.minimum(d_V, 0))
        loss_V = positive_part * (1 - alpha) + negative_part * alpha
        
        # Zero out unreachable states
        loss_V = self._zero_unreachable_states(loss_V)
        
        # Store metrics
        self.loss_V_sum = np.sum(loss_V)
        self.max_loss_V = np.max(loss_V)
        
        if self.flag_verbose:
            print(f'Sum V loss: {self.loss_V_sum}')
        print(f'Max V loss: {self.max_loss_V}')
        
        flag_V = np.all(loss_V <= self.tolerance)
        
        # Check joining policy
        flag_phi = self.check_joining_br(phi, V)
        
        # Check inspection policy
        if self.setting.update_inspect:
            flag_inspect = self.check_inspect_decision(
                inspect, nu_q_Q, tau_q_Q, alpha, V
            )
        else:
            flag_inspect = True
            self.loss_inspect_sum = 0
        
        # Calculate total loss
        self.total_loss = self.loss_V_sum + self.loss_phi_sum + self.loss_inspect_sum
        print(f'Total loss: {self.total_loss}')
        
        # Check convergence
        if (flag_V and flag_phi and flag_inspect) or self.total_loss <= self.tolerance:
            return True
        
        # Report violations if verbose
        if self.flag_verbose:
            self._report_equilibrium_violations(flag_V, flag_phi, flag_inspect, loss_V)
        
        return False

    def check_equilibrium_grad(
        self,
        grad_direction: np.ndarray,
        alpha: np.ndarray,
        phi: np.ndarray,
        V: np.ndarray
    ) -> bool:
        """
        Check equilibrium conditions for gradient-based algorithm.
        
        Args:
            grad_direction: Gradient direction
            alpha: Acceptance policy
            phi: Joining policy
            V: Continuation values
            
        Returns:
            True if equilibrium conditions are satisfied
        """
        # Check joining decisions
        flag_phi = self.check_joining_br(phi, V, self.flag_verbose)
        
        # Check acceptance decisions
        pure_strategy_tol = self.tolerance
        almost_pure_tol = self.tolerance
        
        # Find pure strategies (alpha ≈ 1 or alpha ≈ 0)
        pos_mask = (alpha > 1 - almost_pure_tol)
        neg_mask = (alpha < almost_pure_tol)
        
        # For alpha ≈ 1, gradient should be non-negative
        pos_indices = np.where(pos_mask)
        pos_grads = grad_direction[pos_indices]
        flag_grad_pos = np.all(pos_grads >= -pure_strategy_tol)
        
        # For alpha ≈ 0, gradient should be non-positive
        neg_indices = np.where(neg_mask)
        neg_grads = grad_direction[neg_indices]
        flag_grad_neg = np.all(neg_grads <= pure_strategy_tol)
        
        # For mixed strategies, gradient should be ≈ 0
        mixed_mask = np.logical_and(alpha > almost_pure_tol, alpha < 1 - almost_pure_tol)
        mixed_indices = np.where(mixed_mask)
        mixed_grads = grad_direction[mixed_indices]
        flag_grad_zero = np.all(np.abs(mixed_grads) <= self.tolerance)
        
        # Compute multiplicative error
        positive_part = np.maximum(grad_direction, 0)
        negative_part = np.abs(np.minimum(grad_direction, 0))
        loss_grad = positive_part * (1 - alpha) + negative_part * alpha
        
        self.loss_V_sum = np.sum(loss_grad[1:, 1:])
        self.max_loss_V = np.max(loss_grad[1:, 1:])
        
        print(f'Sum gradient loss: {self.loss_V_sum}')
        print(f'Max gradient loss: {self.max_loss_V}')
        
        flag_grad = np.all(loss_grad[1:, 1:] <= self.tolerance)
        
        self.total_loss = self.loss_V_sum + self.loss_phi_sum
        
        # Report violations if any
        if not (flag_grad and flag_phi):
            if not flag_grad_zero:
                max_grad = np.max(np.abs(mixed_grads)) if len(mixed_grads) > 0 else 0
                print(f"Max abs gradient for randomized strategy: {max_grad}")
            
            if not flag_grad_pos:
                max_grad = np.abs(np.min(pos_grads)) if len(pos_grads) > 0 else 0
                print(f"Worst incorrect positive direction (alpha ~= 1): {max_grad}")
            
            if not flag_grad_neg:
                max_grad = np.max(neg_grads) if len(neg_grads) > 0 else 0
                print(f"Worst incorrect negative direction (alpha ~= 0): {max_grad}")
        
        return flag_grad and flag_phi

    def _report_equilibrium_violations(
        self,
        flag_V: bool,
        flag_phi: bool,
        flag_inspect: bool,
        loss_V: np.ndarray
    ) -> None:
        """Report which equilibrium conditions are violated."""
        if not flag_V:
            mask = loss_V > self.tolerance
            indices = np.where(mask)
            print("Row indices where V differs:", indices[0])
            print("Column indices where V differs:", indices[1])
        
        if not flag_phi:
            mask = np.abs(self.loss_phi[1:]) > self.tolerance
            indices = np.where(mask)
            print("Indices where phi differs:", indices)
        
        if not flag_inspect:
            mask = self.loss_inspect > self.tolerance
            indices = np.where(mask)
            print("Row indices where inspect differs:", indices[0])
            print("Column indices where inspect differs:", indices[1])

    # ==================== Utility Methods ====================

    def _zero_unreachable_states(self, array: np.ndarray) -> np.ndarray:
        """
        Set unreachable states to zero in the policy array.
        
        Unreachable states are those where there are more drivers in queue
        than riders waiting (for i >= 2, j < i).
        
        Args:
            array: Policy array to modify
            
        Returns:
            Modified array with unreachable states zeroed
        """
        array = array.copy()
        array[0, :] = 0
        array[:, 0] = 0
        
        n = array.shape[0]
        rows = np.arange(n)[:, None]
        cols = np.arange(n)
        mask = (rows >= 2) & (rows > cols)
        array[mask] = 0
        
        return array

    def _mark_unreachable_states(self, array: np.ndarray) -> np.ndarray:
        """
        Mark unreachable states with -1 for visualization.
        
        Args:
            array: Policy array to modify
            
        Returns:
            Modified array with unreachable states marked as -1
        """
        array = array.copy()
        n = array.shape[0]
        rows = np.arange(n)[:, None]
        cols = np.arange(n)
        mask = (rows >= 2) & (rows > cols)
        array[mask] = -1
        
        return array

    def _cleanup_old_history(self) -> None:
        """Clean up old outcome history to save memory."""
        if self.iter >= 3 and (self.iter % self.saving_multiple != 0):
            self.outcome_history[-3].V = []
            self.outcome_history[-3].alpha = []
            self.outcome_history[-3].inspect = []
            self.outcome_history[-3].phi = []
            self.outcome_history[-3].loss_V = []
            self.outcome_history[-3].loss_phi = []
            self.outcome_history[-3].job_rates = []
            self.outcome_history[-3].earnings = []
            
            # Keep only recent momentum history
            self.direction_history = self.direction_history[-2:]
            self.insp_direction_history = self.insp_direction_history[-2:]

    def _check_degeneracy(
        self,
        alpha: np.ndarray,
        phi: np.ndarray,
        inspect: np.ndarray
    ) -> bool:
        """
        Check if system has degenerated to trivial state.
        
        Args:
            alpha: Acceptance policy
            phi: Joining policy
            inspect: Inspection policy
            
        Returns:
            True if system is degenerate
        """
        if np.all(inspect[1:, 1:] == 0):
            print("No one is inspecting, stopping the algorithm.")
            return True
        
        if np.all(alpha[1:, 1:] == 0):
            print("No one is accepting jobs, stopping the algorithm.")
            return True
        
        if np.all(phi[1:] == 0):
            print("No one is joining the queue, stopping the algorithm.")
            return True
        
        return False

    # ==================== Main Iteration Loop ====================

    def run_iterations(self) -> Tuple[List[Outcome], np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
        """
        Run policy iteration algorithm until convergence or max iterations.
        
        Returns:
            Tuple containing:
            - outcome_history: List of all outcomes
            - V: Final continuation values
            - alpha: Final acceptance policy
            - phi: Final joining policy
            - iterations: Number of iterations run
            - inspect: Final inspection policy
        """
        # Initialize policies
        alpha_t = self.outcome_history[-1].alpha
        phi_t = self.outcome_history[-1].phi
        inspect_t = self.outcome_history[-1].inspect
        
        for i in range(self.max_iterations + 1):
            self.iter += 1
            print('\n' + '='*60)
            print(f'Iteration: {self.iter}')
            print('='*60)
            
            self.current_time = time.time()
            
            # Update step size
            self.gamma = self.setting.gamma_cnst / (self.iter ** self.setting.gamma_exp)
            print(f'Step size (gamma): {self.gamma:.6f}')
            
            # Run algorithm based on version
            converged = self._run_single_iteration(i, alpha_t, phi_t, inspect_t)
            
            if converged:
                break
            
            # Update policies for next iteration
            alpha_t = self.outcome_history[-1].alpha
            phi_t = self.outcome_history[-1].phi
            inspect_t = self.outcome_history[-1].inspect
            
            # Check for degeneracy
            if self._check_degeneracy(alpha_t, phi_t, inspect_t):
                break
            
            # Save intermediate results
            if (i + 1) % self.saving_multiple == 0:
                save_run_history(
                    self.setting,
                    self.outcome_history,
                    self.version,
                    self.setting.dispatching_rule
                )
            
            # Clean up old history
            self._cleanup_old_history()
            
            # Report timing
            elapsed_time = time.time() - self.current_time
            print(f"\nIteration {self.iter} completed in {elapsed_time:.2f} seconds.")
        
        # Return final results
        final = self.outcome_history[-1]
        return (
            self.outcome_history,
            final.V,
            final.alpha,
            final.phi,
            self.iter,
            final.inspect,
            converged
        )

    def _run_single_iteration(
        self,
        i: int,
        alpha_t: np.ndarray,
        phi_t: np.ndarray,
        inspect_t: np.ndarray
    ) -> bool:
        """
        Run a single iteration of the selected algorithm.
        
        Args:
            i: Current iteration number
            alpha_t: Current acceptance policy
            phi_t: Current joining policy
            inspect_t: Current inspection policy
            
        Returns:
            True if converged, False otherwise
        """
        if self.version == '0-MIX':
            return self._run_mixed_iteration(i, alpha_t, phi_t, inspect_t)
        elif self.version == '0-PAR':
            return self._run_parallel_iteration(i, alpha_t, phi_t, inspect_t)
        elif self.version == '0-COMP':
            return self._run_competition_iteration(i, alpha_t, phi_t, inspect_t)
        elif self.version == '0-GRAD':
            return self._run_gradient_iteration(i, alpha_t, phi_t, inspect_t)
        elif self.version == '0-MIX-GRAD':
            return self._run_hybrid_iteration(i, alpha_t, phi_t, inspect_t)
        else:
            raise ValueError(f"Algorithm version {self.version} not recognized!")

    def _run_mixed_iteration(
        self,
        i: int,
        alpha_t: np.ndarray,
        phi_t: np.ndarray,
        inspect_t: np.ndarray
    ) -> bool:
        """Run single iteration of mixed algorithm."""
        # Evaluate strategy
        V_t, nu_q_Q, _ = self.system.solve_system(alpha_t, phi_t, inspect_t)
        
        # Check equilibrium
        converged = self.check_equilibrium_mixed(alpha_t, phi_t, V_t, inspect_t, nu_q_Q)
        
        if converged:
            print(f"Converged at iteration {i}")
            # Mark unreachable states
            alpha_t = self._mark_unreachable_states(alpha_t)
            inspect_t = self._mark_unreachable_states(inspect_t)
            self.update_history(V_t, alpha_t, phi_t, inspect_t, nu_q_Q, converged = converged)
            return True
        
        # Update policies
        alpha_t, phi_t, inspect_t = self.update_alpha_mixed(
            alpha_t, phi_t, V_t, inspect_t, nu_q_Q
        )
        
        # Save to history
        self.update_history(V_t, alpha_t, phi_t, inspect_t, nu_q_Q)
        return False


    def _run_competition_iteration(
        self,
        i: int,
        alpha_t: np.ndarray,
        phi_t: np.ndarray,
        inspect_t: np.ndarray
    ) -> bool:
        """Run single iteration of competition algorithm."""
        # Evaluate strategy
        V_t, nu_q_Q, nu_1, nu_2 = self.system.solve_system_comp(
            alpha_t, phi_t, inspect_t
        )
        
        # Check equilibrium
        converged = self.check_equilibrium_mixed(alpha_t, phi_t, V_t, inspect_t, nu_q_Q)
        
        if converged:
            print(f"Converged at iteration {i}")
            # Mark unreachable states
            alpha_t = self._mark_unreachable_states(alpha_t)
            inspect_t = self._mark_unreachable_states(inspect_t)
            self.update_history(V_t, alpha_t, phi_t, inspect_t, nu_q_Q, nu_1, nu_2)
            return True
        
        # Update policies
        alpha_t, phi_t, inspect_t = self.update_alpha_mixed(
            alpha_t, phi_t, V_t, inspect_t, nu_q_Q
        )
        
        # Save to history
        self.update_history(V_t, alpha_t, phi_t, inspect_t, nu_q_Q, nu_1, nu_2)
        return False

    def _run_gradient_iteration(
        self,
        i: int,
        alpha_t: np.ndarray,
        phi_t: np.ndarray,
        inspect_t: np.ndarray
    ) -> bool:
        """Run single iteration of gradient-based algorithm."""
        # Evaluate strategy and compute gradient
        V_t, nu_q_Q, G = self.system.solve_system(alpha_t, phi_t, inspect_t)
        grad_direction, grad_unscaled = self.system.solve_gradient_and_get_grad_direction(V_t, G)
        
        # Check equilibrium
        converged = self.check_equilibrium_grad(grad_unscaled, alpha_t, phi_t, V_t)
        
        if converged:
            print(f"Converged at iteration {i}")
            alpha_t = self._mark_unreachable_states(alpha_t)
            self.update_history(V_t, alpha_t, phi_t, inspect_t, nu_q_Q, converged = converged)
            return True
        
        # Update policies using gradient
        alpha_t, phi_t = self.update_alpha_gradient(alpha_t, phi_t, V_t, grad_unscaled)
        
        # Save to history
        self.update_history(V_t, alpha_t, phi_t, inspect_t, nu_q_Q)
        return False

    def _run_hybrid_iteration(
        self,
        i: int,
        alpha_t: np.ndarray,
        phi_t: np.ndarray,
        inspect_t: np.ndarray
    ) -> bool:
        """Run single iteration of hybrid algorithm (switches from mixed to gradient)."""
        # Evaluate strategy
        V_t, nu_q_Q, _ = self.system.solve_system(alpha_t, phi_t, inspect_t)
        
        # Check equilibrium
        converged = self.check_equilibrium_mixed(alpha_t, phi_t, V_t, inspect_t, nu_q_Q)
        
        if converged:
            print(f"Approximate algorithm converged at iteration {i}")
            self.update_history(V_t, alpha_t, phi_t, inspect_t, nu_q_Q, converged = converged)
            save_run_history(
                self.setting,
                self.outcome_history,
                self.version,
                self.setting.dispatching_rule
            )
            
            # Switch to gradient method
            print("Switching to gradient-based refinement...")
            self.version = '0-GRAD'
            self.setting.gamma_cnst = self.setting.gamma_cnst_grad
            return False  # Continue with gradient method
        
        # Update policies
        alpha_t, phi_t, inspect_t = self.update_alpha_mixed(
            alpha_t, phi_t, V_t, inspect_t, nu_q_Q
        )
        
        # Save to history
        self.update_history(V_t, alpha_t, phi_t, inspect_t, nu_q_Q)
        return False
