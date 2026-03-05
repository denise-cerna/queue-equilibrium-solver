import numpy as np
from src.dispatching_class import Dispatching
from typing import List, Optional, Literal


class Setting:
    """
    Configuration settings for the queueing simulation.
    
    Manages system parameters including job earnings, arrival rates, dispatching rules,
    and optimization hyperparameters.
    """
    
    def __init__(
        self,
        earnings: List[float],
        job_rates: List[float],
        driver_arrival_rate: float,
        waiting_cost: float,
        patience: float,
        Qmax: int,
        dispatch_rule: str,
        inspection_cost: float = 0,
        gamma_cnst: float = 1,
        gamma_cnst_grad: float = 1,
        gamma_exp: float = 0,
        gamma_phi_divide_by: float = 1,
        gamma_inspect_divide_by: float = 1,
        beta: float = 0,
        momentum_start_iter: int = 0,
        reneging: float = 0,
        initial_alpha: Literal['RAND', '1'] = 'RAND',
        num_parallel_workers: int = 1,
        join_at_front: float = 0,
        flag_verbose: bool = False,
        endo_reneging: bool = True,
        update_inspect: bool = True,
        firm1_share: float = 0.5,
        firm2_share: float = 0.5,
        total_share: float = 0,
        firm1_dispatch_rule: str = 'STRICT_FIFO',
        firm2_dispatch_rule: str = 'DYNAMIC_RAND_FIFO',
        tie_breaking: str = 'FIFO'
    ):
        """
        Initialize system settings.
        
        Args:
            earnings: List of job earnings by type
            job_rates: List of job arrival rates by type
            driver_arrival_rate: Rate at which drivers arrive
            waiting_cost: Cost per unit time for waiting
            patience: Rider patience level
            Qmax: Maximum queue length
            dispatch_rule: Dispatching policy to use
            inspection_cost: Cost to inspect a job
            gamma_cnst: Constant step size parameter
            gamma_cnst_grad: Constant step size for gradient
            gamma_exp: Exponent for step size decay (0 = constant)
            gamma_phi_divide_by: Divisor for phi step size
            gamma_inspect_divide_by: Divisor for inspection step size
            beta: Momentum parameter (0 = no momentum)
            momentum_start_iter: Iteration to start using momentum
            reneging: Exogenous reneging rate
            initial_alpha: Initial policy ('RAND' or '1')
            num_parallel_workers: Number of parallel workers
            join_at_front: Percentage of drivers joining at front
            flag_verbose: Whether to print verbose output
            endo_reneging: Whether to use endogenous reneging
            update_inspect: Whether to update inspection policy
            firm1_share: Market share for firm 1 (competition mode)
            firm2_share: Market share for firm 2 (competition mode)
            total_share: Total market share
            firm1_dispatch_rule: Dispatching rule for firm 1
            firm2_dispatch_rule: Dispatching rule for firm 2
            tie_breaking: Tie-breaking rule for parallel dispatch
        """
        # Process earnings (add 0 at front for endogenous reneging)
        self.earnings = self._initialize_earnings(earnings, endo_reneging)
        
        # Process job rates (add high rate at front for endogenous reneging)
        self.job_rates = self._initialize_job_rates(job_rates, endo_reneging)
        
        # System parameters
        self.driver_arrival_rate = driver_arrival_rate
        self.waiting_cost = waiting_cost
        self.inspection_cost = inspection_cost
        self.patience = patience
        self.reneging = reneging
        self.Qmax = Qmax
        self.join_at_front = join_at_front
        self.dispatching_rule = dispatch_rule
        self.tie_breaking = tie_breaking
        
        # Initialize dispatching distributions
        self._initialize_dispatching(
            dispatch_rule, 
            firm1_dispatch_rule, 
            firm2_dispatch_rule
        )
        
        # Define array shape
        self.shape = (Qmax + 1, Qmax + 1, len(self.job_rates))
        
        # Initialize policy (alpha)
        self.alpha = self._initialize_alpha(initial_alpha)
        
        # Initialize continuation values
        self.V = np.zeros((Qmax + 1, Qmax + 1))
        
        # Initialize joining policy (always join initially)
        self.phi = (np.diag(self.V) >= 0).astype(int)
        
        # Initialize inspection policy (always inspect)
        self.inspect = np.ones((Qmax + 1, Qmax + 1))
        
        # Optimization hyperparameters
        self.gamma_cnst = gamma_cnst
        self.gamma_cnst_grad = gamma_cnst_grad
        self.gamma_exp = gamma_exp
        self.gamma_phi_divide_by = gamma_phi_divide_by
        self.gamma_inspect_divide_by = gamma_inspect_divide_by
        self.momentum_start_iter = momentum_start_iter
        self.gamma = gamma_cnst
        self.beta = beta
        
        # Parallel processing
        self.num_parallel_workers = num_parallel_workers
        
        # Flags
        self.flag_verbose = flag_verbose
        self.update_inspect = update_inspect
        
        # Competition parameters (unused but kept for compatibility)
        self.firm1_share = firm1_share
        self.firm2_share = firm2_share
        self.total_share = total_share

    def _initialize_earnings(self, earnings: List[float], endo_reneging: bool) -> np.ndarray:
        """
        Initialize earnings array with optional endogenous reneging.
        
        Args:
            earnings: Base earnings list
            endo_reneging: Whether to add 0 earning for reneging option
            
        Returns:
            Numpy array of earnings
        """
        earnings_copy = earnings.copy()
        if endo_reneging:
            earnings_copy.insert(0, 0)
        return np.array(earnings_copy)

    def _initialize_job_rates(self, job_rates: List[float], endo_reneging: bool) -> np.ndarray:
        """
        Initialize job rates array with optional endogenous reneging.
        
        Args:
            job_rates: Base job rates list
            endo_reneging: Whether to add high rate for reneging option
            
        Returns:
            Numpy array of job rates
        """
        job_rates_copy = job_rates.copy()
        if endo_reneging:
            job_rates_copy.insert(0, 100)
        return np.array(job_rates_copy)

    def _initialize_dispatching(
        self,
        dispatch_rule: str,
        firm1_dispatch_rule: str,
        firm2_dispatch_rule: str
    ) -> None:
        """
        Initialize dispatching distributions based on the selected rule.
        
        Args:
            dispatch_rule: Main dispatching rule
            firm1_dispatch_rule: Dispatching rule for firm 1 (competition mode)
            firm2_dispatch_rule: Dispatching rule for firm 2 (competition mode)
        """
        if dispatch_rule == 'COMP':
            # Competition mode: initialize both firms
            self.firm1_dispatch_obj = Dispatching(
                self.earnings,
                self.job_rates,
                self.driver_arrival_rate,
                self.waiting_cost,
                self.patience,
                self.Qmax,
                firm1_dispatch_rule
            )
            self.firm1_dispatch = self.firm1_dispatch_obj.get_dispatch_prob()
            
            self.firm2_dispatch_obj = Dispatching(
                self.earnings,
                self.job_rates,
                self.driver_arrival_rate,
                self.waiting_cost,
                self.patience,
                self.Qmax,
                firm2_dispatch_rule
            )
            self.firm2_dispatch = self.firm2_dispatch_obj.get_dispatch_prob()
            self.dispatch_dist = None
        else:
            # Single firm mode
            self.dispatch_obj = Dispatching(
                self.earnings,
                self.job_rates,
                self.driver_arrival_rate,
                self.waiting_cost,
                self.patience,
                self.Qmax,
                dispatch_rule
            )
            self.dispatch_dist = self.dispatch_obj.get_dispatch_prob()

    def _initialize_alpha(self, initial_alpha: str) -> np.ndarray:
        """
        Initialize the alpha policy matrix.
        
        Args:
            initial_alpha: Initialization strategy ('1' for all ones, 'RAND' for random)
            
        Returns:
            Initialized alpha array
        """
        if initial_alpha == '1':
            alpha = np.ones(self.shape, dtype=np.int32)
        else:
            # Random initialization
            alpha = self._create_random_alpha()
        
        # Set boundary conditions
        alpha[0, :, :] = 0  # No jobs when no drivers in queue
        alpha[:, 0, :] = 0  # No jobs when queue is empty
        alpha[:, :, 0] = 0  # First job type (reneging) set to 0
        
        return alpha

    def _create_random_alpha(self) -> np.ndarray:
        """
        Create a randomly initialized alpha matrix with constraints.
        
        Ensures at least one job type is selected for each state and
        the first job type (reneging) is always included.
        
        Returns:
            Random alpha array satisfying constraints
        """
        alpha = np.random.randint(2, size=self.shape)
        
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                # Always set first job type (reneging)
                alpha[i, j, 0] = 1
                
                # Ensure at least one job type is selected
                if np.sum(alpha[i, j]) == 0:
                    idx = np.random.randint(self.shape[2])
                    alpha[i, j, idx] = 1
        
        return alpha

    def __repr__(self) -> str:
        """Return string representation of Setting object."""
        return (
            f"Setting(\n"
            f"  dispatching_rule={self.dispatching_rule},\n"
            f"  Qmax={self.Qmax},\n"
            f"  driver_arrival_rate={self.driver_arrival_rate},\n"
            f"  waiting_cost={self.waiting_cost},\n"
            f"  num_job_types={len(self.job_rates)}\n"
            f")"
        )