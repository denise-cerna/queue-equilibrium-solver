import numpy as np
import scipy.sparse.linalg as spla
import scipy.sparse as sp
from scipy.linalg import null_space


class Variance:
    """
    Computes the variance (and mean) of a driver's earnings 
    """

    def __init__(self,
                 V: np.ndarray,
                 alpha: np.ndarray,
                 phi: np.ndarray,
                 inspect: np.ndarray,
                 nu_q_Q: np.ndarray,
                 ss,
                 setting,
                 nu_l=None,
                 nu_u=None
                 ):
        """
        Parameters
        ----------
        V        : ndarray, shape (Qmax+1, Qmax+1)
            Continuation values V[q, Q].
        alpha    : ndarray, shape (Qmax+1, Qmax+1, num_jobs)
            acceptance probabilities alpha[q, Q, l] for job type l.
        phi      : ndarray, shape (Qmax+1,)
            Queue-joining probabilities for arriving drivers.
            phi[Q] = P(driver joins | queue length is Q).
        inspect  : ndarray, shape (Qmax+1, Qmax+1)
            Inspection policy: whether a driver at position q in a queue of
            length Q inspects jobs.
        nu_q_Q   : ndarray or None, shape (Qmax+1, Qmax+1, num_jobs)
            Effective dispatch rates. Computed internally if None.
        ss       : array-like, shape (Qmax+1,)
            Steady-state distribution over queue lengths.
        setting  : object
            Problem parameters (see attributes assigned below).
        nu_l, nu_u : optional arrays
        """

        # ── Problem parameters pulled from `setting` ──────────────────────
        self.earnings             = setting.earnings             # shape (num_jobs,): per-job earnings
        self.job_rates            = setting.job_rates            # shape (num_jobs,): Poisson arrival rates of jobs
        self.driver_arrival_rate  = setting.driver_arrival_rate  # scalar: rate at which new drivers arrive
        self.waiting_cost         = setting.waiting_cost         # scalar: cost per unit time spent waiting
        self.patience             = setting.patience             # int: max number of dispatch attempts before a driver leaves
        self.Qmax                 = setting.Qmax                 # int: maximum queue capacity
        self.dispatching_rule     = setting.dispatching_rule     # str: name of the dispatching policy in use
        self.reneging             = setting.reneging             # scalar: exogenous rate at which drivers abandon the queue
        self.inspection_cost      = setting.inspection_cost      # scalar: cost incurred each time a job is inspected
        self.join_at_front        = setting.join_at_front        # bool: whether arriving drivers join at the front
        self.dispatch_dist        = setting.dispatch_dist        # list of matrices: dispatch probability distributions
        self.gamma                = setting.gamma_cnst           # scalar: stepsize

        # ── Core model objects ────────────────────────────────────────────
        self.V       = V        # continuation values
        self.ss      = ss       # steady-state distribution
        self.alpha   = alpha    # acceptance probabilities
        self.inspect = inspect  # inspection policy
        self.phi     = phi      # joining probabilities

        # Compute or store effective dispatch rates nu_q_Q
        if nu_q_Q is None:
            nu_q_Q, _ = self.get_nu_and_G()
        self.nu_q_Q = nu_q_Q

        # Optional rate bounds (stored for external use)
        self.nu_l = nu_l
        self.nu_u = nu_u

        # Compute variance and mean earnings
        self.variance = self.get_variance()

    def get_nu_and_G(self):
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

    
    def get_rate_matrices(self, sanity_check = False):

        """
        This function computes the rate matrices r_plus, r_minus, total_r, and omega
        using vectorized operations instead of loops.
        It uses the precomputed alpha, nu_q_Q, earnings, phi, and reneging values.
        
        The resulting matrices are stored in self.r_plus_matrix, self.r_minus_matrix,
        self.total_r_matrix, and self.omega_matrix.
        The shapes of the matrices are as follows:
        - r_plus: shape (Qmax+1, Qmax+1)
        - r_minus: shape (Qmax+1, Qmax+1)
        - total_r: shape (Qmax+1, Qmax+1)
        - omega: shape (Qmax+1, Qmax+1)

        In comparison to calling the get rates function in a loop, this method leads 
        to a 10x speed-up for the Qmax = 800 simulations.

        Created: 05/30/2025
        Updated: 05/31/2025 

        """

        Qmax = self.Qmax
        # alpha:    shape (Qmax+1, Qmax+1, num_jobs)
        # nu_q_Q:   shape (Qmax+1, Qmax+1, num_jobs)
        # earnings: shape (num_jobs,)
        # phi:      length (Qmax+1,)
        # reneging, driver_arrival_rate: scalars

        # 1) Build index grids and masks:
        q_idx = np.arange(Qmax + 1)[:, None]  # shape (Qmax+1, 1)
        Q_idx = np.arange(Qmax + 1)[None, :]  # shape (1, Qmax+1)

        # mask_q_leq_Q is True whenever q <= Q (for intermediate sums of M)
        mask_q_leq_Q = (q_idx <= Q_idx)

        # final “valid” mask: q >= 1 and q <= Q
        valid = (q_idx >= 1) & mask_q_leq_Q

        # 2) Precompute
        #    M[i,Q] = sum_{ℓ} [ α[i,Q,ℓ] * ν[i,Q,ℓ] ], but only for i ≤ Q:
        M_raw = np.sum(self.alpha * self.inspect[:, :, None] * self.nu_q_Q, axis=2)          # shape (Qmax+1, Qmax+1)
        M = M_raw * mask_q_leq_Q                                 # zero out i > Q

        # 3) Cumulative‐sum of M along the “i”‐axis:
        #    cs[k,Q] = ∑_{i=0}^k M[i,Q].
        cs = np.cumsum(M, axis=0)                                # shape (Qmax+1, Qmax+1)
        cs_diag = np.diagonal(cs)                                # shape (Qmax+1,)

        # 4) Build r_plus and r_minus in closed form:

        # 4a) r_plus[q,Q] = ∑_{i=0}^{\,q-1} M[i,Q]  +  reneging*(q-1),  for q ≥ 1.  Else 0.
        r_plus = np.zeros((Qmax + 1, Qmax + 1))
        if Qmax > 0:
            # For q=1..Qmax, partial sum ∑_{i=0}^{q-1} M[i,Q] = cs[q-1, Q].
            part_sum = cs[:-1, :]                               # shape (Qmax, Qmax+1), corresponds to q=1..Qmax
            q_vals = np.arange(1, Qmax + 1)                      # [1, 2, …, Qmax]
            # reneging*(q-1) = reneging*(q_vals - 1)
            r_plus[1:, :] = part_sum + self.reneging * ((q_vals - 1)[:, None])
        # Zero out any entry with q=0 or q>Q:
        r_plus = r_plus * valid

        # 4b) r_minus[q,Q] = ∑_{i=q+1}^{Q} M[i,Q]  +  reneging*(Q - q),  for q ≥ 1.  Else 0.
        #    Note: ∑_{i=q+1}^{Q} M[i,Q] = cs[Q, Q] - cs[q, Q]  (if q ≤ Q).
        term1 = cs_diag[None, :] - cs                           # shape (Qmax+1, Qmax+1)
        term2 = self.reneging * (Q_idx - q_idx)                  # shape (Qmax+1, Qmax+1)
        r_minus_all = term1 + term2
        # Force r_minus[0, :] = 0 (original loop never touches q=0)
        r_minus_all[0, :] = 0.0
        # Zero out q>Q, and q=0 remains zero
        r_minus = r_minus_all * valid

        # 5) Build ω[q,Q] = ∑_{ℓ} [ ν[q,Q,ℓ] * α[q,Q,ℓ] * earnings[ℓ] ], only for q ≥ 1 & q ≤ Q.
        W_raw = np.sum(self.alpha * self.inspect[:, :, None] * self.nu_q_Q, axis=2)  # shape (Qmax+1, Qmax+1)
        omega = W_raw * valid

        cost_raw = np.sum(self.nu_q_Q[:, :, 1:] * self.inspect[:, :, None] * self.inspection_cost, axis=2)  # shape (Qmax+1, Qmax+1)
        insp_cost = cost_raw * valid

        # 6) Build the “driver_arrival_rate * φ[Q+1]” term:
        #    Extend φ by one extra zero so that φ_ext[Qmax+1] = 0.
        φ_ext = np.zeros(Qmax + 2)
        φ_ext[: self.phi.shape[0]] = self.phi
        # Now, for each Q=0..Qmax, D_vec[Q] = driver_arrival_rate * φ_ext[Q+1]
        D_vec = self.driver_arrival_rate * φ_ext[1 : Qmax + 2]   # length = Qmax+1
        # Broadcast into a full (Qmax+1)×(Qmax+1) array along the q‐axis:
        D_full = D_vec[None, :]                                  # shape (1, Qmax+1), broadcast→(Qmax+1, Qmax+1)

        # 7) Finally assemble total_r:
        #    total_r[q,Q] = r_plus[q,Q]
        #                  + r_minus[q,Q]
        #                  + M[q,Q]
        #                  + reneging
        #                  + [driver_arrival_rate * φ[Q+1]  if Q < Qmax else 0]
        total_r_all = r_plus + r_minus + M + self.reneging + D_full
        
        # Zero out any q=0 or q>Q
        total_r = total_r_all * valid

        # 8) Store into self
        self.r_plus_matrix  = r_plus
        self.r_minus_matrix = r_minus
        self.total_r_matrix = total_r
        self.r_not_matrix   = omega
        self.insp_cost_matrix = insp_cost

        if sanity_check:
            for q in range(1, Qmax+1):
                for Q in range(q, Qmax+1):
                    total_r, r_plus, r_minus = self.get_rates(q, Q)
                    assert np.isclose(self.r_plus_matrix[q, Q], r_plus), f"r_plus mismatch at (q={q}, Q={Q})"
                    assert np.isclose(self.total_r_matrix[q, Q], total_r), f"total_r mismatch at (q={q}, Q={Q})"
                    assert np.isclose(self.r_not_matrix[q, Q], self.earnings @ (self.alpha[q, Q] * self.nu_q_Q[q, Q])), f"r_not mismatch at (q={q}, Q={Q})"
                    assert np.isclose(self.r_minus_matrix[q, Q], r_minus), f"r_minus mismatch at (q={q}, Q={Q})"

    # Solve closed form system for W
    def get_W(self):
        #self.get_rate_matrices()

        W = np.zeros((self.Qmax+1, self.Qmax+1))
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

        absorbing_edge = 0

        if self.r_not_matrix[q, Q] != 0:
            b_coeff = (self.nu_q_Q[q, Q, :] * self.alpha[q, Q, :]) / self.r_not_matrix[q, Q]
            diff = self.earnings - self.V[q, Q]
            absorbing_edge = np.sum(
                b_coeff * (
                    diff**2
                    - 2 * diff * (self.waiting_cost / self.total_r_matrix[q, Q])
                    + 2 * self.waiting_cost**2 / self.total_r_matrix[q, Q]**2
                )
            )

        join_edge = (self.V[q, Q+1] - self.V[q, Q])**2 - 2*(self.V[q, Q+1] - self.V[q, Q])*(self.waiting_cost / self.total_r_matrix[q, Q]) + 2 * self.waiting_cost**2 / self.total_r_matrix[q, Q]**2

        driver_joined = self.driver_arrival_rate * phi_augmented[Q + 1]
        reward = self.r_not_matrix[q, Q] * absorbing_edge + driver_joined*join_edge

        denominator = self.total_r_matrix[q, Q]
        constants[q, Q] = reward / denominator
        coeff[q, Q] = driver_joined / denominator

        ### Solve the V for Q \geq 2, i.e. rest of the columns 
        for Q in range(2, self.Qmax+1):
            driver_joined = self.driver_arrival_rate * phi_augmented[Q + 1]

            absorbing_edge = 0

            if self.r_not_matrix[q, Q] != 0:
                b_coeff = (self.nu_q_Q[q, Q, :] * self.alpha[q, Q, :]) / self.r_not_matrix[q, Q]
                diff = self.earnings - self.V[q, Q]
                absorbing_edge = np.sum(
                    b_coeff * (
                        diff**2
                        - 2 * diff * (self.waiting_cost / self.total_r_matrix[q, Q])
                        + 2 * self.waiting_cost**2 / self.total_r_matrix[q, Q]**2
                    )
                )

            edgeB = (self.V[q, Q-1] - self.V[q, Q])**2 - 2*(self.V[q, Q-1] - self.V[q, Q])*(self.waiting_cost / self.total_r_matrix[q, Q]) + 2 * self.waiting_cost**2 / self.total_r_matrix[q, Q]**2

            if Q == self.Qmax:
                reward = self.r_not_matrix[q, Q] * absorbing_edge + self.r_minus_matrix[q, Q] * edgeB
            
            else:
                join_edge = (self.V[q, Q+1] - self.V[q, Q])**2 - 2*(self.V[q, Q+1] - self.V[q, Q])*(self.waiting_cost / self.total_r_matrix[q, Q]) + 2 * self.waiting_cost**2 / self.total_r_matrix[q, Q]**2
                reward = self.r_not_matrix[q, Q] * absorbing_edge + driver_joined*join_edge + self.r_minus_matrix[q, Q] * edgeB

            denominator = self.total_r_matrix[q, Q] - self.r_minus_matrix[q, Q] * coeff[q, Q-1]

            constants[q, Q] = (reward + constants[q, Q-1] * self.r_minus_matrix[q, Q]) / denominator
            coeff[q, Q] = driver_joined/denominator
        
        W[q, self.Qmax] = constants[q, self.Qmax]

        for Q in range(self.Qmax-1, 0, -1):
            W[q, Q] = constants[q, Q] + coeff[q, Q] * W[q, Q + 1]

        ###### Focusing on q > 1, i.e. the remaining rows ######

        for q in range(2, self.Qmax+1):
            
            Q = q
            driver_joined = self.driver_arrival_rate * phi_augmented[Q + 1]

            absorbing_edge = 0

            if self.r_not_matrix[q, Q] != 0:
                b_coeff = (self.nu_q_Q[q, Q, :] * self.alpha[q, Q, :]) / self.r_not_matrix[q, Q]
                diff = self.earnings - self.V[q, Q]
                absorbing_edge = np.sum(
                    b_coeff * (
                        diff**2
                        - 2 * diff * (self.waiting_cost / self.total_r_matrix[q, Q])
                        + 2 * self.waiting_cost**2 / self.total_r_matrix[q, Q]**2
                    )
                )

            edgeF = (self.V[q-1, Q-1] - self.V[q, Q])**2 - 2*(self.V[q-1, Q-1] - self.V[q, Q])*(self.waiting_cost / self.total_r_matrix[q, Q]) + 2 * self.waiting_cost**2 / self.total_r_matrix[q, Q]**2

            if Q == self.Qmax:
                reward = self.r_not_matrix[q, Q]*absorbing_edge + self.r_plus_matrix[q, Q] *(edgeF + W[q-1, Q-1])

            else:
                join_edge = (self.V[q, Q+1] - self.V[q, Q])**2 - 2*(self.V[q, Q+1] - self.V[q, Q])*(self.waiting_cost / self.total_r_matrix[q, Q]) + 2 * self.waiting_cost**2 / self.total_r_matrix[q, Q]**2
                reward = self.r_not_matrix[q, Q]*absorbing_edge + driver_joined*join_edge + self.r_plus_matrix[q, Q] *(edgeF + W[q-1, Q-1])
            constants[q, Q] = reward / self.total_r_matrix[q, Q]
            coeff[q, Q] = driver_joined / self.total_r_matrix[q, Q]

            for Q in range(q+1, self.Qmax+1):
                driver_joined = self.driver_arrival_rate * phi_augmented[Q + 1]
                denominator = self.total_r_matrix[q, Q] - self.r_minus_matrix[q, Q] * coeff[q, Q-1]

                absorbing_edge = 0

                if self.r_not_matrix[q, Q] != 0:
                    b_coeff = (self.nu_q_Q[q, Q, :] * self.alpha[q, Q, :]) / self.r_not_matrix[q, Q]
                    diff = self.earnings - self.V[q, Q]
                    absorbing_edge = np.sum(
                        b_coeff * (
                            diff**2
                            - 2 * diff * (self.waiting_cost / self.total_r_matrix[q, Q])
                            + 2 * self.waiting_cost**2 / self.total_r_matrix[q, Q]**2
                        )
                    )

                edgeF = (self.V[q-1, Q-1] - self.V[q, Q])**2 - 2*(self.V[q-1, Q-1] - self.V[q, Q])*(self.waiting_cost / self.total_r_matrix[q, Q]) + 2 * self.waiting_cost**2 / self.total_r_matrix[q, Q]**2
                edgeB = (self.V[q, Q-1] - self.V[q, Q])**2 - 2*(self.V[q, Q-1] - self.V[q, Q])*(self.waiting_cost / self.total_r_matrix[q, Q]) + 2 * self.waiting_cost**2 / self.total_r_matrix[q, Q]**2

                if Q == self.Qmax:
                    reward = self.r_not_matrix[q, Q]*absorbing_edge + self.r_minus_matrix[q, Q] * edgeB + self.r_plus_matrix[q, Q] * (edgeF+ W[q-1, Q-1])
                
                else:
                    join_edge = (self.V[q, Q+1] - self.V[q, Q])**2 - 2*(self.V[q, Q+1] - self.V[q, Q])*(self.waiting_cost / self.total_r_matrix[q, Q]) + 2 * self.waiting_cost**2 / self.total_r_matrix[q, Q]**2
                    reward = self.r_not_matrix[q, Q]*absorbing_edge + driver_joined*join_edge + self.r_minus_matrix[q, Q] * edgeB + self.r_plus_matrix[q, Q] * (edgeF+ W[q-1, Q-1])
                constants[q, Q] = (reward + constants[q, Q-1] * self.r_minus_matrix[q, Q]) / denominator
                coeff[q, Q] = driver_joined / denominator

            W[q, self.Qmax] = constants[q, self.Qmax]
            for Q in range(self.Qmax-1, q-1, -1):
                W[q, Q] = constants[q, Q] + coeff[q, Q] * W[q, Q+1]  

        return W

        
    def get_variance(self):
        self.get_rate_matrices()
        W = self.get_W()
        Wdiag = np.diagonal(W)
        Wdiag = Wdiag[:-1]
        Vdiag = np.diagonal(self.V)
        Vdiag = Vdiag[:-1]

        p = np.sum(self.ss[:-1]*self.phi[1:])
        q = (self.ss[:-1]*self.phi[1:])/p
        
        expW = np.sum(q * (Wdiag))

        VarV = (np.sum(q* ((Vdiag) ** 2))
                            - (np.sum(q * (Vdiag)) ** 2))
        
        variance = p*(VarV +  expW) + p*(1-p)*(np.sum(q* (Vdiag))) ** 2

        ave_earnings = p*np.sum(q * (Vdiag))
        return variance, ave_earnings
    