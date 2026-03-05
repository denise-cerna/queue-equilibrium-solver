import numpy as np
from src.variance_class import Variance
from scipy.linalg import null_space
from typing import Optional, Tuple


class Outcome:
    """
    Represents the outcome of a queueing system simulation with driver-rider matching.
    
    Tracks system metrics including steady-state distribution, throughput, and net revenue
    for a dispatching system with driver arrivals, job queues, and inspection policies.
    """
    
    def __init__(
        self, 
        V: np.ndarray, 
        alpha: np.ndarray, 
        norm: float, 
        phi: np.ndarray, 
        inspect: np.ndarray,
        setting: object, 
        nu_q_Q: np.ndarray, 
        iter: int, 
        total_loss_V: float = 0, 
        total_loss_phi: float = 0, 
        max_loss_V: float = 0,
        max_loss_phi: float = 0,
        max_loss_inspect: float = 0,
        nu_1: Optional[np.ndarray] = None,
        nu_2: Optional[np.ndarray] = None,
        converged: Optional[bool] = None
    ):
        """
        Initialize the Outcome object with system parameters and computed values.
        
        Args:
            V: Continuation values
            alpha: Strategy/policy parameters
            norm: Iteration norm
            phi: Joining decision probabilities
            inspect: Inspection policy
            setting: Configuration object containing system parameters
            nu_q_Q: Queue-dependent rates
            iter: Iteration number
            total_loss_V: Current loss function value for V
            total_loss_phi: Current loss function value for phi
            max_loss_V: Maximum loss function value for V
            max_loss_phi: Maximum loss function value for phi
            max_loss_inspect: Maximum loss function value for inspect
            nu_1: Lower bound rates (for parallel dispatch)
            nu_2: Upper bound rates (for parallel dispatch)
            converged: Whether the algorithm has converged (for logging)
        """
        # System parameters from settings
        self.earnings = setting.earnings
        self.job_rates = setting.job_rates
        self.driver_arrival_rate = setting.driver_arrival_rate
        self.waiting_cost = setting.waiting_cost
        self.patience = setting.patience
        self.Qmax = setting.Qmax
        self.dispatching_rule = setting.dispatching_rule
        self.reneging = setting.reneging
        self.inspection_cost = setting.inspection_cost

        # Computed values
        self.V = V
        self.alpha = alpha
        self.inspect = inspect
        self.norm = norm
        self.phi = phi
        self.iter = iter

        # Loss metrics
        self.loss_V = total_loss_V
        self.loss_phi = total_loss_phi
        self.max_loss_V = max_loss_V
        self.max_loss_phi = max_loss_phi
        self.max_loss_inspect = max_loss_inspect

        # Rate parameters
        self.nu_q_Q = nu_q_Q
        self.nu_1 = nu_1
        self.nu_2 = nu_2

        # CPU metrics (initialized to zero)
        self.avg_cpu = 0
        self.cpu_time = 0
        self.max_cpu = 0

        # Calculate metrics if this isn't the initial iteration
        if self.iter > 0:
            self._calculate_and_store_metrics()
        else:
            self._initialize_zero_metrics()

        if converged:
            print("Calculating variance and average earnings for converged solution...")
            v = Variance(self.V, 
                 self.alpha, 
                 self.phi, 
                 self.inspect, 
                 self.nu_q_Q,
                 self.steady_state,
                 setting
                 )
            self.variance, self.ave_earnings = v.get_variance()

        self._cleanup_temporary_data()


    def _initialize_zero_metrics(self) -> None:
        """Initialize metrics to zero for the initial iteration."""
        self.steady_state = np.zeros(self.Qmax + 1)
        self.throughput = 0
        self.net_revenue = 0
        

    def _calculate_and_store_metrics(self) -> None:
        """Calculate and store all system metrics."""
        self.steady_state = self._compute_steady_state()
        self.queue_length = self._compute_queue_length(self.steady_state)
        self.throughput = self._compute_throughput(self.steady_state, self.nu_q_Q)
        self.net_revenue = self._compute_net_revenue(self.steady_state, self.nu_q_Q)

        # Calculate metrics for parallel dispatch if applicable
        if self.nu_1 is not None and self.nu_2 is not None:
            self._calculate_parallel_metrics()

    def _calculate_parallel_metrics(self) -> None:
        """Calculate metrics for parallel dispatch scenarios."""
        self.steady_state_1 = self._compute_steady_state(self.nu_1)
        self.throughput_1 = self._compute_throughput(self.steady_state_1, self.nu_1)
        self.net_revenue_1 = self._compute_net_revenue(self.steady_state_1, self.nu_1, waiting_cost_multiplier=0.5)
        self.queue_length_1 = self._compute_queue_length(self.steady_state_1)

        self.steady_state_2 = self._compute_steady_state(self.nu_2)
        self.throughput_2 = self._compute_throughput(self.steady_state_2, self.nu_2)
        self.net_revenue_2 = self._compute_net_revenue(self.steady_state_2, self.nu_2, waiting_cost_multiplier=0.5)
        self.queue_length_2 = self._compute_queue_length(self.steady_state_2)

    def _cleanup_temporary_data(self) -> None:
        """Remove temporary data structures to free memory."""
        del self.nu_q_Q
        del self.nu_1
        del self.nu_2

    @staticmethod
    def _compute_queue_length(steady_state: np.ndarray) -> float:
        """
        Compute expected queue length from steady-state distribution.
        
        Args:
            steady_state: Steady-state probability distribution
            
        Returns:
            Expected queue length
        """
        queue_indices = np.arange(len(steady_state))
        return np.sum(steady_state * queue_indices)

    def _build_rate_matrix(self, nu: np.ndarray) -> np.ndarray:
        """
        Build the rate matrix for the continuous-time Markov chain.
        
        Args:
            nu: Queue-dependent service rates
            
        Returns:
            Rate matrix of size (Qmax+1) x (Qmax+1)
        """
        rate_matrix = np.zeros((self.Qmax + 1, self.Qmax + 1))

        for Q in range(self.Qmax + 1):
            if Q == 0:
                # Empty queue: only driver arrivals
                arrival_rate = self.driver_arrival_rate * self.phi[Q + 1]
                rate_matrix[Q, Q + 1] = arrival_rate
                rate_matrix[Q, Q] = -arrival_rate
                
            else:
                # Calculate service rates
                service_col = self.alpha[:, Q] * nu[:, Q] * self.inspect[:, Q, None]
                service_rate = np.sum(np.cumsum(service_col, axis=0)[Q])
                reneging_rate = Q * self.reneging
                total_departure_rate = service_rate + reneging_rate
                
                # Departures (service + reneging)
                rate_matrix[Q, Q - 1] = total_departure_rate
                
                if Q < self.Qmax:
                    # Arrivals (not at max capacity)
                    arrival_rate = self.driver_arrival_rate * self.phi[Q + 1]
                    rate_matrix[Q, Q + 1] = arrival_rate
                    rate_matrix[Q, Q] = -(arrival_rate + total_departure_rate)
                else:
                    # At max capacity: no arrivals
                    rate_matrix[Q, Q] = -total_departure_rate

        return rate_matrix

    def _compute_steady_state(self, nu: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate steady-state distribution by solving the balance equations.
        
        Args:
            nu: Queue-dependent service rates (uses self.nu_q_Q if None)
            
        Returns:
            Steady-state probability distribution
        """
        if nu is None:
            nu = self.nu_q_Q
            
        rate_matrix = self._build_rate_matrix(nu)
        
        # Solve π^T Q = 0 with constraint Σπ = 1
        dimension = rate_matrix.shape[0]
        one_vector = np.ones(dimension)
        
        # Replace last equation with normalization constraint
        A = np.vstack((rate_matrix.T[:-1], one_vector))
        b = np.zeros((dimension, 1))
        b[-1] = 1
        
        pi, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return pi.flatten()

    def _compute_throughput(self, steady_state: np.ndarray, nu: np.ndarray) -> float:
        """
        Calculate system throughput (completed jobs per time unit).
        
        Args:
            steady_state: Steady-state probability distribution
            nu: Queue-dependent service rates
            
        Returns:
            System throughput
        """
        throughput = 0

        for Q in range(1, self.Qmax + 1):
            # Service rates for queue length Q
            service_col = self.alpha[:, Q, 1:] * nu[:, Q, 1:] * self.inspect[:, Q, None]
            service_rate = np.sum(np.cumsum(service_col, axis=0)[Q])
            throughput += steady_state[Q] * service_rate

        return throughput

    def _compute_net_revenue(
        self, 
        steady_state: np.ndarray, 
        nu: np.ndarray,
        waiting_cost_multiplier: float = 1.0
    ) -> float:
        """
        Calculate net revenue (earnings - waiting costs - inspection costs).
        
        Args:
            steady_state: Steady-state probability distribution
            nu: Queue-dependent service rates
            waiting_cost_multiplier: Multiplier for waiting cost (0.5 for parallel metrics, 1.0 for main)
            
        Returns:
            Net revenue per time unit
        """
        net_revenue = 0

        for Q in range(1, self.Qmax + 1):
            # Earnings from completed jobs
            earnings_col = self.alpha[:, Q] * nu[:, Q] * self.earnings * self.inspect[:, Q, None]
            earnings_rate = np.sum(np.cumsum(earnings_col, axis=0)[Q])
            
            # Inspection costs
            inspection_col = nu[:, Q, 1:] * self.inspect[:, Q, None]
            inspection_rate = np.sum(np.cumsum(inspection_col, axis=0)[Q])
            
            # Net revenue contribution at queue length Q
            revenue_contribution = (
                earnings_rate 
                - waiting_cost_multiplier * Q * self.waiting_cost 
                - inspection_rate * self.inspection_cost
            )
            
            net_revenue += steady_state[Q] * revenue_contribution

        return net_revenue

