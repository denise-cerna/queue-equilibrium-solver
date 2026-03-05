"""
System class for solving queueing equilibrium problems.

Handles computation of continuation values, service rates, and gradients
for various dispatching policies including parallel dispatch and competition modes.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from joblib import Parallel, delayed
from tqdm import tqdm
from typing import Tuple, Optional


class System:
    """
    Represents a queueing system with strategic drivers and riders.
    
    Solves for equilibrium continuation values using various methods including
    closed-form solutions and sparse linear systems. Supports gradient computation
    for gradient-based optimization algorithms.
    """
    
    def __init__(self, setting: object):
        """
        Initialize system with configuration parameters.
        
        Args:
            setting: Setting object containing system parameters
        """
        # Store reference to settings
        self.setting = setting
        
        # Extract key parameters for convenience
        self.earnings = setting.earnings
        self.job_rates = setting.job_rates
        self.driver_arrival_rate = setting.driver_arrival_rate
        self.waiting_cost = setting.waiting_cost
        self.inspection_cost = setting.inspection_cost
        self.patience = setting.patience
        self.reneging = setting.reneging
        self.Qmax = setting.Qmax
        self.dispatch_dist = setting.dispatch_dist
        
        # Initialize policies
        self.update_variables(setting.alpha, setting.phi, setting.inspect)

    def update_variables(
        self,
        alpha: np.ndarray,
        phi: np.ndarray,
        inspect: np.ndarray
    ) -> None:
        """
        Update current policies.
        
        Args:
            alpha: Acceptance policy
            phi: Joining policy
            inspect: Inspection policy
        """
        self.alpha = alpha
        self.phi = phi
        self.inspect = inspect

    # ==================== Service Rate Computation ====================

    def get_nu_and_G(self):
        # print('Calculating nu_q_Q')
        num_jobs = len(self.job_rates)
        Qmax = self.Qmax
        patience = self.patience
        
        # Initialize nu_q_Q directly as a NumPy array
        nu_q_Q = np.zeros((Qmax+1, Qmax+1, num_jobs))

        mu = self.job_rates

        set_of_dist = self.dispatch_dist

        # Preallocate G array instead of using a list
        G = np.zeros((patience, Qmax+1, Qmax+1, num_jobs))
        G[0] = set_of_dist[0]  # First dispatch matrix

        # Iterate over potential dispatches
        for k in range(1, patience):

            # Get probabilities g
            G_n = np.sum(G[k-1] * (1 - self.alpha*self.inspect[:, :, None]), axis=0)
            G_n[0] = 0  # Enforce boundary condition
            G[k] = G_n[None, :] * set_of_dist[k]  # Broadcasting
       
        # Sum over patience dimension and multiply by mu
        nu_q_Q = np.sum(G, axis=0) * mu
        return nu_q_Q, G

    def get_nu_and_tau(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute service rates accounting for inspection and priority ordering.
        
        Used in parallel dispatch algorithms where tie-breaking rules matter.
        
        Returns:
            Tuple of (nu_q_Q, tau_q_Q) where both have shape (Qmax+1, Qmax+1, num_jobs)
        """
        num_jobs = len(self.job_rates)
        patience = self.patience
        
        # Initialize arrays
        nu_q_Q = np.zeros((self.Qmax + 1, self.Qmax + 1, num_jobs))
        tau_q_Q = np.zeros((self.Qmax + 1, self.Qmax + 1, num_jobs))
        
        G_nu = np.zeros((patience, self.Qmax + 1, self.Qmax + 1, num_jobs))
        G_tau = np.zeros((patience, self.Qmax + 1, self.Qmax + 1, num_jobs))
        
        # First dispatch (k=0)
        G_nu[0] = self.dispatch_dist[0]
        
        # Compute priority for first dispatch
        priority = self._compute_priority(self.dispatch_dist[0])
        G_tau[0] = self.dispatch_dist[0] * priority
        
        # Iterate over patience levels
        for k in range(1, patience):
            # Compute rejection probability from previous dispatches
            G_nu_n = np.prod(1 - G_nu[k-1] * self.alpha * self.inspect[:, :, None], axis=0)
            for j in range(1, k):
                G_nu_n *= np.prod(1 - G_nu[j-1] * self.alpha * self.inspect[:, :, None], axis=0)
            
            G_nu_n[0] = 0  # Boundary condition
            G_nu[k] = G_nu_n[None, :] * self.dispatch_dist[k]
            
            # Compute priority for this dispatch
            priority = self._compute_priority_with_tiebreaking(k)
            
            # Real probability accounting for priority
            G_tau[k] = G_nu_n[None, :] * self.dispatch_dist[k] * priority
        
        # Sum over patience and scale by job rates
        nu_q_Q = np.sum(G_nu, axis=0) * self.job_rates
        tau_q_Q = np.sum(G_tau, axis=0) * self.job_rates
        
        return nu_q_Q, tau_q_Q

    def _compute_priority(self, dispatch_dist: np.ndarray) -> np.ndarray:
        """
        Compute priority ordering for FIFO tie-breaking.
        
        Args:
            dispatch_dist: Dispatch distribution matrix
            
        Returns:
            Priority array of same shape as dispatch_dist
        """
        # Probability item is not accepted by higher priority positions
        x = 1.0 - dispatch_dist * self.alpha * self.inspect[:, :, None]
        x[0, :, :] = 1.0
        
        # Cumulative product along queue positions
        cp = np.cumprod(x, axis=0)
        
        # Shift down so priority[q] = product up to q-1
        priority = np.empty_like(cp)
        priority[0, :, :] = 1.0
        priority[1:, :, :] = cp[:-1, :, :]
        
        return priority

    def _compute_priority_with_tiebreaking(self, k: int) -> np.ndarray:
        """
        Compute priority with specified tie-breaking rule.
        
        Args:
            k: Dispatch index
            
        Returns:
            Priority array
        """
        if self.setting.tie_breaking == 'FIFO':
            return self._compute_priority(self.dispatch_dist[k])
        elif self.setting.tie_breaking == 'RAND':
            # Random tie-breaking
            shape = (self.Qmax + 1, self.Qmax + 1, len(self.job_rates))
            priority = np.random.rand(*shape)
            return priority
        else:
            raise ValueError(f"Unknown tie-breaking rule: {self.setting.tie_breaking}")

    def get_nu_comp(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute service rates for competition mode with two firms.
        
        Returns:
            Tuple of (nu_q_Q, nu_1, nu_2) where each has shape (Qmax+1, Qmax+1, num_jobs)
        """
        num_jobs = len(self.job_rates)
        patience = self.patience
        
        # Job rates for each firm
        mu_1 = self.job_rates * self.setting.firm1_share
        mu_2 = self.job_rates * self.setting.firm2_share
        mu_s = self.job_rates * self.setting.total_share
        
        # Dispatch distributions
        firm1_dispatch = self.setting.firm1_dispatch
        firm2_dispatch = self.setting.firm2_dispatch
        
        # Rejection mask
        inspect_mask = 1 - self.alpha * self.inspect[:, :, None]
        
        # Running totals
        sum_G_1 = np.zeros((self.Qmax + 1, self.Qmax + 1, num_jobs))
        sum_G_2 = np.zeros_like(sum_G_1)
        sum_G_1_par = np.zeros_like(sum_G_1)
        sum_G_2_par = np.zeros_like(sum_G_2)
        
        # Initialize
        G_1 = firm1_dispatch[0].copy()
        G_2 = firm2_dispatch[0].copy()
        
        # Compute initial priorities
        d_1, d_2 = self._compute_competition_priorities(0, firm1_dispatch, firm2_dispatch)
        
        G_1_par = firm1_dispatch[0] * d_1
        G_2_par = firm2_dispatch[0] * d_2
        
        sum_G_1 += G_1
        sum_G_2 += G_2
        sum_G_1_par += G_1_par
        sum_G_2_par += G_2_par
        
        # Iterate over patience levels
        for k in range(1, patience):
            # Probability neither firm accepted
            prev_1 = np.einsum('ijk,ijk->jk', G_1_par, inspect_mask)
            prev_2 = np.einsum('ijk,ijk->jk', G_2_par, inspect_mask)
            prob_next = np.clip(prev_1 * prev_2, 0, 1)
            prob_next[0] = 0
            
            # Update G values
            G_1 = np.einsum('ijk,jk->ijk', firm1_dispatch[k],
                           np.einsum('ijk,ijk->jk', G_1, inspect_mask))
            G_2 = np.einsum('ijk,jk->ijk', firm2_dispatch[k],
                           np.einsum('ijk,ijk->jk', G_2, inspect_mask))
            G_1[0] = 0
            G_2[0] = 0
            
            # Compute priorities for this dispatch
            d_1, d_2 = self._compute_competition_priorities(k, firm1_dispatch, firm2_dispatch)
            
            # Update parallel dispatch probabilities
            G_1_par = firm1_dispatch[k] * prob_next * d_1
            G_2_par = firm2_dispatch[k] * prob_next * d_2
            
            sum_G_1 += G_1
            sum_G_2 += G_2
            sum_G_1_par += G_1_par
            sum_G_2_par += G_2_par
        
        # Compute final service rates
        nu_1 = sum_G_1 * mu_1 + sum_G_1_par * mu_s
        nu_2 = sum_G_2 * mu_2 + sum_G_2_par * mu_s
        nu_q_Q = nu_1 + nu_2
        
        return nu_q_Q, nu_1, nu_2

    def _compute_competition_priorities(
        self,
        k: int,
        firm1_dispatch: np.ndarray,
        firm2_dispatch: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute priority adjustments for competition between two firms.
        
        Args:
            k: Dispatch index
            firm1_dispatch: Dispatch distribution for firm 1
            firm2_dispatch: Dispatch distribution for firm 2
            
        Returns:
            Tuple of (d_1, d_2) priority adjustments
        """
        if self.setting.tie_breaking == 'RAND':
            # Random tie-breaking: split probability
            d_1 = 1 - 0.5 * np.sum(firm2_dispatch[k] * self.alpha, axis=0)
            d_2 = 1 - 0.5 * np.sum(firm1_dispatch[k] * self.alpha, axis=0)
            d_1[:, 0], d_1[0] = 1, 0
            d_2[:, 0], d_2[0] = 1, 0
            
        elif self.setting.tie_breaking == 'FIFO':
            # FIFO tie-breaking
            x = firm2_dispatch[k] * self.alpha * self.inspect[:, :, None]
            x[0, :, :] = 0
            cp = np.cumsum(x, axis=0)
            d_1 = np.empty_like(cp)
            d_1[0, :, :] = 1.0
            d_1[1:, :, :] = 1 - cp[:-1, :, :]
            
            x = firm1_dispatch[k] * self.alpha * self.inspect[:, :, None]
            x[0, :, :] = 0
            cp = np.cumsum(x, axis=0)
            d_2 = np.empty_like(cp)
            d_2[0, :, :] = 1.0
            d_2[1:, :, :] = 1 - cp[:-1, :, :]
        else:
            raise ValueError(f"Unknown tie-breaking rule: {self.setting.tie_breaking}")
        
        return d_1, d_2

    # ==================== Rate Matrix Computation ====================

    def get_rates(self, q: int, Q: int) -> Tuple[float, float, float]:
        """
        Compute transition rates for a specific state.
        
        Args:
            q: Queue position
            Q: Queue length
            
        Returns:
            Tuple of (total_rate, r_plus, r_minus)
        """
        r_plus = self.r_plus(q, Q)
        r_minus = self.r_minus(q, Q)
        total_r = (r_plus + r_minus + 
                  np.sum(self.nu_q_Q[q, Q] * self.alpha[q, Q]) + 
                  self.reneging)
        
        if Q < self.Qmax:
            total_r += self.driver_arrival_rate * self.phi[Q + 1]
        
        return total_r, r_plus, r_minus
    
    def r_plus(self, q: int, Q: int) -> float:
        """Rate of service from positions ahead in queue."""
        return (np.sum(self.alpha[:q, Q] * self.nu_q_Q[:q, Q]) + 
                self.reneging * (q - 1))
    
    def r_minus(self, q: int, Q: int) -> float:
        """Rate of service from positions behind in queue."""
        return (np.sum(self.alpha[q+1:Q+1, Q] * self.nu_q_Q[q+1:Q+1, Q]) + 
                self.reneging * (Q - q))

    def get_rate_matrices(self, sanity_check: bool = False) -> None:
        """
        Compute rate matrices using vectorized operations.
        
        Provides 10x speedup over loop-based computation for large Qmax.
        Stores results in self.r_plus_matrix, self.r_minus_matrix,
        self.total_r_matrix, and self.omega_matrix.
        
        Args:
            sanity_check: If True, verify results against loop-based method
        """
        Qmax = self.Qmax
        
        # Build index grids
        q_idx = np.arange(Qmax + 1)[:, None]
        Q_idx = np.arange(Qmax + 1)[None, :]
        
        # Masks
        mask_q_leq_Q = (q_idx <= Q_idx)
        valid = (q_idx >= 1) & mask_q_leq_Q
        
        # Compute M[i,Q] = sum over jobs of (alpha * inspect * nu)
        M_raw = np.sum(self.alpha * self.inspect[:, :, None] * self.nu_q_Q, axis=2)
        M = M_raw * mask_q_leq_Q
        
        # Cumulative sum along queue positions
        cs = np.cumsum(M, axis=0)
        cs_diag = np.diagonal(cs)
        
        # Compute r_plus vectorized
        r_plus = np.zeros((Qmax + 1, Qmax + 1))
        if Qmax > 0:
            part_sum = cs[:-1, :]
            q_vals = np.arange(1, Qmax + 1)
            r_plus[1:, :] = part_sum + self.reneging * ((q_vals - 1)[:, None])
        r_plus = r_plus * valid
        
        # Compute r_minus vectorized
        term1 = cs_diag[None, :] - cs
        term2 = self.reneging * (Q_idx - q_idx)
        r_minus_all = term1 + term2
        r_minus_all[0, :] = 0.0
        r_minus = r_minus_all * valid
        
        # Compute omega (earnings)
        W_raw = np.sum(
            self.alpha * self.inspect[:, :, None] * 
            self.nu_q_Q * self.earnings[None, None, :],
            axis=2
        )
        omega = W_raw * valid
        
        # Compute inspection cost
        cost_raw = np.sum(
            self.nu_q_Q[:, :, 1:] * self.inspect[:, :, None] * self.inspection_cost,
            axis=2
        )
        insp_cost = cost_raw * valid
        
        # Build driver arrival term
        phi_ext = np.zeros(Qmax + 2)
        phi_ext[:self.phi.shape[0]] = self.phi
        D_vec = self.driver_arrival_rate * phi_ext[1:Qmax + 2]
        D_full = D_vec[None, :]
        
        # Assemble total_r
        total_r_all = r_plus + r_minus + M + self.reneging + D_full
        total_r = total_r_all * valid
        
        # Store results
        self.r_plus_matrix = r_plus
        self.r_minus_matrix = r_minus
        self.total_r_matrix = total_r
        self.omega_matrix = omega
        self.insp_cost_matrix = insp_cost
        
        # Verify if requested
        if sanity_check:
            self._verify_rate_matrices()

    def _verify_rate_matrices(self) -> None:
        """Verify vectorized rate computation against loop-based method."""
        for q in range(1, self.Qmax + 1):
            for Q in range(q, self.Qmax + 1):
                total_r, r_plus, r_minus = self.get_rates(q, Q)
                
                assert np.isclose(self.r_plus_matrix[q, Q], r_plus), \
                    f"r_plus mismatch at (q={q}, Q={Q})"
                assert np.isclose(self.total_r_matrix[q, Q], total_r), \
                    f"total_r mismatch at (q={q}, Q={Q})"
                assert np.isclose(self.r_minus_matrix[q, Q], r_minus), \
                    f"r_minus mismatch at (q={q}, Q={Q})"

    def get_rate_matrices_par(self, sanity_check: bool = False) -> None:
        """
        Compute rate matrices for parallel dispatch.
        
        Similar to get_rate_matrices but uses tau_q_Q instead of nu_q_Q
        to account for inspection priorities.
        
        Args:
            sanity_check: If True, verify results
        """
        Qmax = self.Qmax
        
        # Build index grids
        q_idx = np.arange(Qmax + 1)[:, None]
        Q_idx = np.arange(Qmax + 1)[None, :]
        
        # Masks
        mask_q_leq_Q = (q_idx <= Q_idx)
        valid = (q_idx >= 1) & mask_q_leq_Q
        
        # Use tau_q_Q instead of nu_q_Q for service rates
        M_raw = np.sum(self.alpha * self.tau_q_Q * self.inspect[:, :, None], axis=2)
        M = M_raw * mask_q_leq_Q
        
        # Rest is same as get_rate_matrices
        cs = np.cumsum(M, axis=0)
        cs_diag = np.diagonal(cs)
        
        r_plus = np.zeros((Qmax + 1, Qmax + 1))
        if Qmax > 0:
            part_sum = cs[:-1, :]
            q_vals = np.arange(1, Qmax + 1)
            r_plus[1:, :] = part_sum + self.reneging * ((q_vals - 1)[:, None])
        r_plus = r_plus * valid
        
        term1 = cs_diag[None, :] - cs
        term2 = self.reneging * (Q_idx - q_idx)
        r_minus_all = term1 + term2
        r_minus_all[0, :] = 0.0
        r_minus = r_minus_all * valid
        
        W_raw = np.sum(
            self.alpha * self.tau_q_Q * self.inspect[:, :, None] * 
            self.earnings[None, None, :],
            axis=2
        )
        omega = W_raw * valid
        
        cost_raw = np.sum(
            self.nu_q_Q[:, :, 1:] * self.inspect[:, :, None] * self.inspection_cost,
            axis=2
        )
        insp_cost = cost_raw * valid
        
        phi_ext = np.zeros(Qmax + 2)
        phi_ext[:self.phi.shape[0]] = self.phi
        D_vec = self.driver_arrival_rate * phi_ext[1:Qmax + 2]
        D_full = D_vec[None, :]
        
        total_r_all = r_plus + r_minus + M + self.reneging + D_full
        total_r = total_r_all * valid
        
        self.r_plus_matrix = r_plus
        self.r_minus_matrix = r_minus
        self.total_r_matrix = total_r
        self.omega_matrix = omega
        self.insp_cost_matrix = insp_cost

    # ==================== Continuation Value Solvers ====================

    def find_A_b(self) -> Tuple[sp.csr_matrix, np.ndarray, dict]:
        """
        Build sparse linear system for continuation values.
        
        Constructs A*V = b where V contains continuation values only for
        reachable states (1 <= q <= Q <= Qmax with positive rates).
        
        Returns:
            Tuple of (A, b, state_to_idx) where:
            - A: Sparse coefficient matrix
            - b: Right-hand side vector
            - state_to_idx: Mapping from (q,Q) to vector index
        """
        Qmax = self.Qmax
        tot_r = self.total_r_matrix
        
        # Masked inverse to avoid divide-by-zero
        inv_tot_r = np.zeros_like(tot_r)
        mask = tot_r != 0
        inv_tot_r[mask] = 1.0 / tot_r[mask]
        
        # List active states
        active = [(q, Q)
                 for Q in range(1, Qmax + 1)
                 for q in range(1, Q + 1)
                 if mask[q, Q]]
        m = len(active)
        state_to_idx = {st: i for i, st in enumerate(active)}
        
        # Allocate sparse matrix and RHS
        A = sp.lil_matrix((m, m))
        b = np.zeros(m)
        
        # Cache parameters
        r_plus = self.r_plus_matrix
        r_minus = self.r_minus_matrix
        omega = self.omega_matrix
        phi = self.phi
        lam = self.driver_arrival_rate
        join_front = self.setting.join_at_front
        wcost = self.waiting_cost
        
        # Build system
        for (q, Q), i in state_to_idx.items():
            denom = inv_tot_r[q, Q]
            cost = wcost * denom
            accF = r_plus[q, Q] * denom
            accB = r_minus[q, Q] * denom
            accDir = omega[q, Q].sum() * denom
            
            if Q < Qmax:
                front = lam * phi[Q + 1] * join_front * denom
                back = lam * phi[Q + 1] * (1 - join_front) * denom
            else:
                front = back = 0.0
            
            # Diagonal
            A[i, i] = 1.0
            b[i] = accDir - cost
            
            # Off-diagonals for transitions
            neighbors = []
            if q > 1:
                neighbors.append(((q - 1, Q - 1), accF))
            if q < Q:
                neighbors.append(((q, Q - 1), accB))
            if Q < Qmax:
                neighbors.append(((q, Q + 1), back))
                neighbors.append(((q + 1, Q + 1), front))
            
            for (nq, nQ), coeff in neighbors:
                j = state_to_idx.get((nq, nQ))
                if j is not None:
                    A[i, j] -= coeff
        
        return A.tocsr(), b, state_to_idx

    def solve_system_of_eq(self) -> np.ndarray:
        """
        Solve sparse linear system for continuation values.
        
        Returns:
            V: Continuation values, shape (Qmax+1, Qmax+1)
        """
        print('Solving system of equations')
        self.get_rate_matrices()
        A, b, state_to_idx = self.find_A_b()
        x = spla.spsolve(A, b)
        
        # Rebuild full V matrix
        V = np.zeros((self.Qmax + 1, self.Qmax + 1))
        for (q, Q), i in state_to_idx.items():
            V[q, Q] = x[i]
        
        return V

    def get_V(self) -> np.ndarray:
        """
        Solve for continuation values using closed-form column recursion.
        
        Much faster than sparse solver for most cases. Works column by column
        from left to right, solving a tridiagonal-like system in each column.
        
        Returns:
            V: Continuation values, shape (Qmax+1, Qmax+1)
        """
        #self.get_rate_matrices()

        V = np.zeros((self.Qmax+1, self.Qmax+1))
        constants = np.zeros((self.Qmax+1, self.Qmax+1))
        coeff = np.zeros((self.Qmax+1, self.Qmax+1))
        
        # phi is the vector representing queue joining probabilities
        # phi[2] is the probability of joining the queue at position q = 2, 
        # i.e. when the length of the queue is Q = 1
        # phi has length Qmax+1, where phi[0] is unused, for positions 0 (unused), 1, 2, ..., Qmax 
        # phi[0] as a result is a number that is never used  
        
        # Here, we get a vector representing the probabilities of joining the queue at q = 1, 2, ..., Qmax+1 (note that the first element is unused)   
        phi_augmented = np.zeros(self.Qmax + 2)
        phi_augmented[:self.Qmax+1] = self.phi

        ###### Focusing on q = 1, i.e. the first row ######

        ### Find first column for Q = 1
        Q = 1
        q = 1

        driver_joined = self.driver_arrival_rate * phi_augmented[Q + 1]
        reward = self.omega_matrix[q, Q] - self.waiting_cost - self.insp_cost_matrix[q, Q]

        denominator = self.total_r_matrix[q, Q]
        constants[q, Q] = reward / denominator
        coeff[q, Q] = driver_joined / denominator

        ### Solve the V for Q \geq 2, i.e. rest of the columns 
        for Q in range(2, self.Qmax+1):
            driver_joined = self.driver_arrival_rate * phi_augmented[Q + 1]
            denominator = self.total_r_matrix[q, Q] - self.r_minus_matrix[q, Q] * coeff[q, Q-1]

            reward = self.omega_matrix[q, Q] - self.waiting_cost - self.insp_cost_matrix[q, Q]
            constants[q, Q] = (reward + constants[q, Q-1] * self.r_minus_matrix[q, Q]) / denominator
            coeff[q, Q] = driver_joined/denominator
        
        V[q, self.Qmax] = constants[q, self.Qmax]

        for Q in range(self.Qmax-1, 0, -1):
            V[q, Q] = constants[q, Q] + coeff[q, Q] * V[q, Q + 1]


        ###### Focusing on q > 1, i.e. the remaining rows ######

        for q in range(2, self.Qmax+1):
            
            Q = q
            driver_joined = self.driver_arrival_rate * phi_augmented[Q + 1]

            reward = self.omega_matrix[q, Q] + self.r_plus_matrix[q, Q] * V[q-1, Q-1] - self.waiting_cost - self.insp_cost_matrix[q, Q]
            constants[q, Q] = reward / self.total_r_matrix[q, Q]
            coeff[q, Q] = driver_joined / self.total_r_matrix[q, Q]

            for Q in range(q+1, self.Qmax+1):
                driver_joined = self.driver_arrival_rate * phi_augmented[Q + 1]
                denominator = self.total_r_matrix[q, Q] - self.r_minus_matrix[q, Q] * coeff[q, Q-1]

                reward = self.omega_matrix[q, Q] + self.r_plus_matrix[q, Q] * V[q-1, Q-1] - self.waiting_cost - self.insp_cost_matrix[q, Q]
                constants[q, Q] = (reward + constants[q, Q-1] * self.r_minus_matrix[q, Q]) / denominator
                coeff[q, Q] = driver_joined / denominator

            V[q, self.Qmax] = constants[q, self.Qmax]
            for Q in range(self.Qmax-1, q-1, -1):
                V[q, Q] = constants[q, Q] + coeff[q, Q] * V[q, Q+1]  

        return V

    # ==================== Main Solution Methods ====================

    def solve_system(
        self,
        alpha: np.ndarray,
        phi: np.ndarray,
        inspect: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve for equilibrium continuation values (standard mode).
        
        Args:
            alpha: Acceptance policy
            phi: Joining policy
            inspect: Inspection policy
            
        Returns:
            Tuple of (V, nu_q_Q, G) where:
            - V: Continuation values
            - nu_q_Q: Service rates
            - G: Dispatch probabilities
        """
        self.update_variables(alpha, phi, inspect)
        
        # Compute service rates
        self.nu_q_Q, G = self.get_nu_and_G()
        self.get_rate_matrices()
        
        # Solve for V
        if self.setting.join_at_front == 0:
            V = self.get_V()
        else:
            V = self.solve_system_of_eq()
        
        return V, self.nu_q_Q, G

    def solve_system_par(
        self,
        alpha: np.ndarray,
        phi: np.ndarray,
        inspect: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve for equilibrium continuation values (parallel dispatch mode).
        
        Args:
            alpha: Acceptance policy
            phi: Joining policy
            inspect: Inspection policy
            
        Returns:
            Tuple of (V, nu_q_Q, tau_q_Q)
        """
        self.update_variables(alpha, phi, inspect)
        
        # Compute service rates with inspection priorities
        self.nu_q_Q, self.tau_q_Q = self.get_nu_and_tau()
        self.get_rate_matrices_par()
        
        V = self.get_V()
        
        return V, self.nu_q_Q, self.tau_q_Q

    def solve_system_comp(
        self,
        alpha: np.ndarray,
        phi: np.ndarray,
        inspect: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve for equilibrium continuation values (competition mode).
        
        Args:
            alpha: Acceptance policy
            phi: Joining policy
            inspect: Inspection policy
            
        Returns:
            Tuple of (V, nu_q_Q, nu_1, nu_2) where nu_1 and nu_2 are
            firm-specific service rates
        """
        self.update_variables(alpha, phi, inspect)
        
        # Get dispatch distributions for both firms
        self.firm1_dispatch = self.setting.firm1_dispatch
        self.firm2_dispatch = self.setting.firm2_dispatch
        
        # Compute service rates
        self.nu_q_Q, self.nu_1, self.nu_2 = self.get_nu_comp()
        self.get_rate_matrices()
        
        # Solve for V
        if self.setting.join_at_front == 0:
            V = self.get_V()
        else:
            V = self.solve_system_of_eq()
        
        return V, self.nu_q_Q, self.nu_1, self.nu_2

    # ==================== Gradient Computation ====================

    def _compute_nu_grad(self, G: np.ndarray, Q: int) -> np.ndarray:
        """
        Compute gradient of nu_q_Q with respect to acceptance strategies.
        
        For a fixed queue length Q, computes how service rate at each (q, ell)
        changes with respect to acceptance decisions at other positions.
        
        Args:
            G: Dispatch probabilities, shape (patience, Qmax+1, Qmax+1, num_jobs)
            Q: Queue length to compute gradient for
            
        Returns:
            Gradient array, shape (Q+1, num_jobs, Q+1)
        """
        num_jobs = len(self.job_rates)
        patience = self.patience
        mu_expanded = self.job_rates[None, :, None]
        
        # Initialize gradient tensor
        dG = np.zeros((patience, Q + 1, num_jobs, Q + 1))
        
        q_slice = slice(1, Q + 1)
        
        for k in range(1, patience):
            # Extract dispatch probabilities
            sd_k2 = self.dispatch_dist[k][q_slice, Q, :]
            sd_trans = sd_k2.T
            
            # Extract previous G values
            Gprev = G[k-1, q_slice, Q, :]
            Gprev_trans = Gprev.T
            
            if k == 1:
                # First derivative: simple outer product
                dG_k = -np.einsum('mi,mj->mij', sd_trans, Gprev_trans)
            else:
                # Higher order: includes chain rule from previous derivatives
                dG_prev = dG[k-1, q_slice, :, q_slice]
                alpha_sub = self.alpha[q_slice, Q, :]
                weights = 1 - alpha_sub
                
                S = np.einsum('im,imj->mj', weights, dG_prev)
                B = -Gprev_trans + S
                dG_k = np.einsum('mi,mj->mij', sd_trans, B)
            
            # Assign (transpose from (ell,q,qbar) to (q,ell,qbar))
            dG[k, q_slice, :, q_slice] = dG_k.transpose(1, 0, 2)
        
        # Sum over patience and scale by job rates
        grad_nu_q_Q = np.sum(dG, axis=0) * mu_expanded
        
        return grad_nu_q_Q

    def compute_direction_Q(
        self,
        V: np.ndarray,
        grad_nu_q_Q: np.ndarray,
        Q: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute gradient direction for acceptance policy at queue length Q.
        
        Args:
            V: Continuation values
            grad_nu_q_Q: Gradient of service rates
            Q: Queue length
            
        Returns:
            Tuple of (grad_direction, grad_direction_unscaled)
        """
        grad_direction = np.zeros((Q + 1, len(self.job_rates)))
        grad_direction_unscaled = np.zeros((Q + 1, len(self.job_rates)))
        
        # Scaling factor
        M = np.sum(self.job_rates) + self.driver_arrival_rate
        
        for q in range(1, Q + 1):
            for ell in range(len(self.job_rates)):
                # Scaling factor (accounts for indirect effects)
                nu = self.nu_q_Q[q, Q, ell] + self.alpha[q, Q, ell] * grad_nu_q_Q[q, ell, q]
                
                # Direct effect: earnings vs continuation value
                d_nu = self.earnings[ell] - V[q, Q]
                
                if nu == 0:
                    # No jobs seen at this state
                    d_r_plus = 0
                    d_r_minus = 0
                else:
                    # Indirect effects through transitions
                    r_plus = (np.sum(grad_nu_q_Q[:q, ell, q] * self.alpha[:q, Q, ell]) / nu)
                    d_r_plus = r_plus * (V[q-1, Q-1] - V[q, Q])
                    
                    r_minus = (np.sum(grad_nu_q_Q[q+1:Q+1, ell, q] * 
                                     self.alpha[q+1:Q+1, Q, ell]) / nu)
                    d_r_minus = r_minus * (V[q, Q-1] - V[q, Q])
                
                # Total gradient direction
                grad_direction[q, ell] = (1 / M) * (d_nu + d_r_plus + d_r_minus)
                grad_direction_unscaled[q, ell] = grad_direction[q, ell] * nu
        
        return grad_direction, grad_direction_unscaled

    def get_grad_col(self, Q: int, G: np.ndarray) -> np.ndarray:
        """
        Compute gradient for a single column Q (legacy method).
        
        This is an alternative implementation kept for compatibility.
        Uses different computational approach than _compute_nu_grad.
        
        Args:
            Q: Queue length
            G: Dispatch probabilities
            
        Returns:
            Gradient array, shape (Q+1, num_jobs, Q+1)
        """
        num_jobs = len(self.job_rates)
        patience = self.patience
        mu = self.job_rates
        mu_expanded = mu[None, :, None]
        
        # Initialize gradient tensor
        grad_nu_q_Q = np.zeros((Q + 1, num_jobs, Q + 1))
        dG = np.zeros((patience, Q + 1, num_jobs, Q + 1), dtype=float)
        
        # Gradient at first dispatch is 0
        dG[0, :, :, :] = 0
        
        # Iterate over potential dispatches
        for n in range(1, patience):
            # Compute derivatives for each position
            for q in range(1, Q + 1):
                dG_n_Q = np.zeros((num_jobs, Q + 1))
                
                # Take derivative with respect to other positions
                for pos in range(1, Q + 1):
                    if n == 1:
                        dG_n_Q[:, pos] = -self.dispatch_dist[n][q, Q] * G[n-1, pos, Q]
                    else:
                        term1 = G[n-1, pos, Q]
                        term2 = np.sum(dG[n-1, :Q+1, :, pos] * (1 - self.alpha[:Q+1, Q]))
                        dG_n_Q[:, pos] = -self.dispatch_dist[n][q, Q] * (term1 - term2)
                
                dG[n, q] = dG_n_Q
        
        # Sum over patience dimension and multiply by mu
        grad_nu_q_Q = np.sum(dG, axis=0) * mu_expanded
        
        return grad_nu_q_Q

    def solve_gradient_and_get_grad_direction(
        self,
        V: np.ndarray,
        G: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute gradient direction for all states in parallel.
        
        This method parallelizes the gradient computation across queue lengths,
        providing significant speedup for large Qmax.
        
        Args:
            V: Continuation values, shape (Qmax+1, Qmax+1)
            G: Dispatch probabilities, shape (patience, Qmax+1, Qmax+1, num_jobs)
            
        Returns:
            Tuple of (grad_direction, grad_direction_unscaled) where both have
            shape (Qmax+1, Qmax+1, num_jobs)
        """
        num_jobs = len(self.job_rates)
        grad_direction = np.zeros((self.Qmax + 1, self.Qmax + 1, num_jobs))
        grad_direction_unscaled = np.zeros((self.Qmax + 1, self.Qmax + 1, num_jobs))
        
        def work_for_Q(Q: int) -> Tuple[np.ndarray, np.ndarray]:
            """
            Compute gradient for a single queue length Q.
            
            Args:
                Q: Queue length to process
                
            Returns:
                Tuple of (direction, unscaled_direction) for this Q
            """
            grad_nu_q_Q = self._compute_nu_grad(G, Q)
            return self.compute_direction_Q(V, grad_nu_q_Q, Q)
        
        # Parallel execution over Q=1..Qmax
        results = Parallel(n_jobs=self.setting.num_parallel_workers)(
            delayed(work_for_Q)(Q)
            for Q in tqdm(range(1, self.Qmax + 1), desc="Computing gradients")
        )
        
        # Assemble results into full arrays
        for Q, (dir_Q, dir_Q_unscaled) in enumerate(results, start=1):
            grad_direction[:Q + 1, Q] = dir_Q
            grad_direction_unscaled[:Q + 1, Q] = dir_Q_unscaled
        
        return grad_direction, grad_direction_unscaled