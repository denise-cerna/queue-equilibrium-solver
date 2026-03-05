# The economy class for randomized FIFO heterogeneous driver simulations
# Created: 07/22/2023

import numpy as np
import matplotlib.pyplot as plt
#from numpy import float_
from numpy.typing import NDArray


############################ 2. An Economy ################################

class Economy:

    # Initialization
    def __init__(self, 
                 L: int,
                 J: int, 
                 w: NDArray[np.float64], 
                 W: NDArray[np.float64], 
                 c, 
                 C, 
                 lambda_drivers, 
                 mu_jobs, 
                 patience):

        # number of job types
        self.L = L

        # number of driver types
        self.J = J

        # vector of net revenue of platform
        self.w = w

        # matrix of net revenue of each driver type
        # each row of corresponds to the net revenue of a driver type
        self.W = W

        # cost per unit of time per driver for the paltform
        self.c = c

        # vector of waiting cost of all driver types
        self.C = C

        # the arrival rate of drivers
        self.lambda_drivers = lambda_drivers

        # the arrival rate of each type of jobs
        self.mu_jobs = mu_jobs
        self.mu = np.sum(mu_jobs)
        
        # Riders' patience level
        self.patience = patience

        # Intitialize the preferred locations of drivers: all drivers like everyhthing. 
        # The function "economy_add_location_preference" in constants.py overwrites this
        self.preferred_locs = []
        for j in range(J):
            self.preferred_locs.append(np.arange(L).tolist())

        self.update_object()


    # This function checks that two economies are exactly the same
    def __eq__(self, other):
        if isinstance(other, Economy):
            is_identical = True
            for attr in vars(self):

                # if attr is an array
                if isinstance(getattr(self, attr), np.ndarray):
                    if np.array_equal(getattr(self, attr), getattr(other, attr)):
                        continue
                    else:
                        identical = False
                        print (f"{attr} is not the same!")
                        break

                # if attr is a list    
                elif isinstance(getattr(self, attr), list):
                    if getattr(self, attr) == getattr(other, attr):
                        continue
                    else:
                        identical = False
                        print (f"{attr} is not the same!")
                        break

                else:         
                    if getattr(self, attr) != getattr(other, attr):
                        identical = False
                        print (f"{attr} is not the same!")
                        break
                    
            return is_identical
        
        return False
    

    # We call this when we initialize an economy
    # We also need to call this every time we update any attribute
    def update_object(self):
        
        self.mu = np.sum(self.mu_jobs)

        # Sanity checks on the input variables
        self.sanity_checks()

        # Find the marginal trip
        self.find_marginal_trip()

        # Compute the 1st best outcome
        self.compute_first_best()
        
        # Compute the 2nd best if we have one type of driver
        if self.J == 1:
            self.compute_second_best()

        # Compute the thresholds (i.e. the milestones) under the direct FIFO mechanism
        # self.direct_FIFO_position is an L by 1 array
        self.direct_FIFO_thresholds()

        # Compute the equilibrium under strict FIFO
        if self.J == 1:
            self.compute_strict_FIFO()        
        
        # Find the bins for randomized FIFO
        # self.rand_FIFO_lb (Patience by 1 vector)
        # self.rand_FIFO_ub (Patience by 1 vector)
        self.rand_FIFO_bins()

        # Compute the acceptable jobs for each driver type
        self.compute_acceptable_jobs()


    # ------------------------------------------------------------------------------ # 

    # Some sanity checks
    def sanity_checks(self):

        if self.L <= 0:
            raise Exception("The number of locatiosn is non-positive!") 

        if self.J <= 0:
            raise Exception("The number of driver types is non-positive!") 
    
        if np.any(self.w < 0):
            raise Exception("Negative net revenue for the paltform!")
        
        if np.sum(self.lambda_drivers) <= 0 or np.sum(self.mu_jobs) <= 0:
            raise Exception("Zero driver or rider arrival rates!")
        
        if self.c < 0:
            raise Exception("Negative opportunity cost for the platform!")
        
        # Check if the w of the economy is descending
        if np.any(np.diff(self.w) > 0):
            raise Exception("The net revenue of the platform is not descending!")
        
        # check that the number of the list of preferred locations is 
        # the same as the number of driver types
        # if not we set all locations to be preferred
        if len(self.preferred_locs) != self.J:
            self.preferred_locs = []
            for j in range(self.J):
                self.preferred_locs.append(np.arange(self.L).tolist())

        # If homogeneous drivers, add a flag that indicate whether
        #  w and W are the roughly same
        self.same_driver_platform_payoff = 0
        if self.J == 1:
            if np.allclose(self.w, self.W[0, :]):
                self.same_driver_platform_payoff = 1
        

    # Compute the acceptable jobs for each driver type
    def compute_acceptable_jobs(self):
        
        # initialize the acceptable jobs for each driver type as 
        # a list of lists
        self.acceptable_jobs = [[] for _ in range(self.J)]

        # For each driver type
        for j in range(self.J):
            # For each location
            for i in range(self.L):
                if self.W[j, i] >= 0:
                    self.acceptable_jobs[j].append(i)


    # Function for finding the "marginal trip"
    # This is the lowest-earning trip that's still completed under the first-best
    def find_marginal_trip(self):

        # if self.J > 1:
        #     raise Exception("Marginal trip not implemented for more than one driver type!") 

        # If supply exceeds demand: 
        # the lowest earning trip that's picked up is trip L
        # All riders to location L are picked up
        if np.sum(self.lambda_drivers) > self.mu:
            self.over_supplied = 1
            self.marginal_trip = self.L
            self.marginal_trip_TP = self.mu_jobs[self.L - 1]
            
        else:
            self.over_supplied = 0

            # The under-supplied setting
            
            # We start from all drivers being available
            leftover_drivers = np.sum(self.lambda_drivers)
            
            # For each location
            for i in range(self.L):

                # If we take all riders to this location and still have leftover drivers
                if leftover_drivers > self.mu_jobs[i]:
                    leftover_drivers = leftover_drivers - self.mu_jobs[i]

                else:
                    # This is the last trip that we would be able to do
                    self.marginal_trip = i + 1
                    self.marginal_trip_TP = leftover_drivers
                    del leftover_drivers
                    break
        

    # This function computes the first best outcome of the economy    
    def compute_first_best(self):

        # The first best throughput
        self.fb_TP = min(self.mu, np.sum(self.lambda_drivers))

        # The first best net revenue
        # This is somewhat tricky: we need to do a bi-partite matching when drivers have differen types!
        # Can't believe we need to code up an LP
        self.fb_net_revenue = 0

        # We pick up all riders to destinatiosn before the marginal trip
        for i in range(self.marginal_trip - 1):
            self.fb_net_revenue += self.w[i] * self.mu_jobs[i]
        
        # The marginal driver
        self.fb_net_revenue += (self.w[self.marginal_trip - 1] 
                                * self.marginal_trip_TP)


    # This function computes the second best outcome of the economy
    def compute_second_best(self):

        # When drivers are homogeneous:
        if self.J == 1:

            # SB TP is the same as FB TP
            self.sb_TP = self.fb_TP   

            # The second best total payoff for drivers
            if self.over_supplied:
                self.sb_driver_total_payoff = 0
            else:
                self.sb_driver_total_payoff = (self.sb_TP * 
                                       self.w[self.marginal_trip - 1])

            # The second best queue length
            self.sb_queue_length = (
                (self.fb_net_revenue - self.sb_driver_total_payoff) / self.C[0])

            # The second best net revenue for the platform
            self.sb_net_revenue = (
                self.fb_net_revenue - self.sb_queue_length * self.c)

        else:
            raise Exception("Second best not yet implemented for heterogeneous drivers!")
        


    # Function for computing the cutoff thresholds for direct FIFO
    # This also (approximately) computes the continuation payoff under direct FIFO 
    # for all positions in the queue but we're too lazy to change the name
    def direct_FIFO_thresholds(self):

        self.direct_FIFO_position = np.zeros(self.L)

        cumulative_TP = None
        willingness_to_wait = None

        for i in range(self.L - 1):

            cumulative_TP = sum(self.mu_jobs[0 : i + 1])
            willingness_to_wait = (self.w[i] - self.w[i + 1]) / self.c

            self.direct_FIFO_position[i + 1] = (
                self.direct_FIFO_position[i] + willingness_to_wait * cumulative_TP
            )
        
        self.direct_FIFO_position = np.floor(self.direct_FIFO_position)

        ### Compute the direct FIFO payoff

        # If we have more than one driver type, we don't know what to do!
        if self.J > 1:
            return
        
        # If we're here, we have only one driver type!
        self.cont_payoff_queue_positions = (
            np.arange(0, np.floor(self.sb_queue_length) + 1, 1))

        # Initialization
        self.direct_FIFO_cont_payoff = self.cont_payoff_queue_positions.copy()

        # At the head of the queue, the driver should get the payoff from the best trip
        self.direct_FIFO_cont_payoff[0] = self.W[0, 0]
        
        for q in range(1, int(self.sb_queue_length + 1), 1):
            
            for i in range(self.L):
                
                # Find the first location where the direct FIFO threshold is 
                # at least q
                if self.direct_FIFO_position[i] < q:
                    continue
                else:
                    if self.direct_FIFO_position[i] == q:
                        self.direct_FIFO_cont_payoff[q] = self.W[0, i]
                        break
                    else:
                        cumulative_TP = sum(self.mu_jobs[0 : i])
                        self.direct_FIFO_cont_payoff[q] = (
                            self.W[0, i-1] - self.C[0] *
                            (q - self.direct_FIFO_position[i-1]) / cumulative_TP
                        )
                    break

            # If the sysetem is over-supplied
            if self.over_supplied:
                # If this position is beyond the very last milestone
                if q >= self.direct_FIFO_position[self.L - 1]:
                        self.direct_FIFO_cont_payoff[q] = (
                            self.W[0, self.L-1] - self.C[0] *
                            (q - self.direct_FIFO_position[self.L-1]) / self.mu
                        )

        del cumulative_TP, willingness_to_wait
        

    # This function computes the equilibrium under strict FIFO
    def compute_strict_FIFO(self):

        # When drivers are homogeneous:
        if self.J == 1:
            # If the patience is at least where we send the marginal trip
            # The outcome under strict FIFO is the same as the second best outcome
            if self.direct_FIFO_position[self.marginal_trip - 1] <= self.patience:
                self.strict_FIFO_queue_length = self.sb_queue_length
                self.strict_FIFO_net_revenue = self.sb_net_revenue
                self.strict_FIFO_driver_total_payoff = self.sb_driver_total_payoff
                self.strict_FIFO_TP = self.sb_TP

            else:
                # Otherwise, the system is effectively oversupplied

                last_job_to_be_picked_up = 0

                # Find the last job that's can reach a driver who's 
                # willing to take it
                for i in range(self.marginal_trip):

                    # If the ith job will not reach any driver
                    if self.direct_FIFO_position[i] > self.patience:
                        last_job_to_be_picked_up = i
                        break
                
                # The throughput
                self.strict_FIFO_TP = sum(
                    self.mu_jobs[0 : last_job_to_be_picked_up]
                    )
                
                # The total payoff of drivers is zero
                self.strict_FIFO_driver_total_payoff = 0

                # The net revenue of the platform is zero
                self.strict_FIFO_net_revenue = 0

                # The queue length is equal to the total net revenue from completed trips 
                # divided by the waiting cost
                self.strict_FIFO_queue_length = sum(
                    self.mu_jobs[0 : last_job_to_be_picked_up ] * 
                    self.w[0 : last_job_to_be_picked_up ]
                    ) / self.C[0]
                
        else:
            raise Exception("Strict FIFO not yet implemented for heterogeneous drivers!")        


    # Function for computing the bins for randomized FIFO
    def rand_FIFO_bins(self, 
                       partition_method = 'equal_locations', 
                       ub_option = 0.0, 
                       the_partitions = [], 
                       min_width = 1, 
                       one_fewer_bins = 0):

        # Throw an error if the patience is 1 and we're using one fewer bins
        if one_fewer_bins == 1 and self.patience == 1:
            raise Exception("Can't have one fewer bin with patience is 1")

        # This parrameters controls how wide the bins are
        # If ub_option = 0, the bins are as narrow as possible
        # If ub_option = 1, the bins are as wide as possible
        if ub_option < 0 or ub_option > 1:
            raise Exception("Illegal bin upper bound option!")

   
        if partition_method == 'fixed':
            # Sanity check for illegal cases
            if len(the_partitions) == 0:
                raise Exception("Fixed partition not given!")
            
            if len(the_partitions) > self.patience:
                raise Exception("Number of partitions larger than the patience level!")

        elif partition_method == 'equal_locations':

            if one_fewer_bins:
                num_bins = self.patience - 1
            else:
                num_bins = self.patience

            # Easy case: when the number of bins is at least the index of the 
            # marginal trip, we're going to put a single trip in each bin
            if num_bins >= self.marginal_trip:
                the_partitions = np.arange(1, self.marginal_trip + 1, 1)
            else:
                # In this case we would chop the total number of completed trips into a few partitions
                # And the calculation for the upper and lower bounds would be done later
                residual = np.mod(self.marginal_trip, num_bins)
                loc_per_bin = np.floor(self.marginal_trip / num_bins)

                the_partitions = []
                fist_trip_in_bin = 1

                for k in range(num_bins):
                    the_partitions = np.append(the_partitions, int(np.floor(fist_trip_in_bin)))
                    fist_trip_in_bin = fist_trip_in_bin + loc_per_bin
                    if k < residual:
                        fist_trip_in_bin += 1

        else:
            raise Exception("Unknown partition construction policy!")
            
        # We now compute the ub and lb of bins given the partition    
                
        # Record the number of bins and the partition
        self.num_bins = len(the_partitions)
        self.rand_fifo_partitions = the_partitions

        self.rand_FIFO_lb = np.zeros(self.num_bins)
        self.rand_FIFO_ub = np.zeros(self.num_bins)

        # Add an extra trip to the partitions for convenience of notation
        the_partitions = np.append(the_partitions, self.marginal_trip + 1)

        # For each bin
        for k in range(self.num_bins):

            # The lowest earning trip in this bin
            lowest_w_in_bin = self.w[int(np.floor(the_partitions[k + 1] - 2))]

            # The highest earning trip in the next bin

            # If we're at the last bin and the lowest earning trip is the marginal trip
            if k == self.num_bins-1 and self.marginal_trip == self.L:

                # Average earning drop between locations
                ave_earning_drop = (self.w[0] - self.w[self.L - 1]) / (self.L-1)

                # Drop by this much, lower bounded by zero
                highest_w_next_bin = max(self.w[self.L - 1] - ave_earning_drop,
                                          0)

            else:
                # Otherwise, the highest earning trip in the next bin is the trip at the 
                # beginning of the next bin
                highest_w_next_bin = self.w[int(np.floor(the_partitions[k + 1] - 1))]

            ###  The bin lb ### 
            bin_lb = 0
            for i in range(int(np.floor(the_partitions[k] - 1))):
                bin_lb += (self.w[i] - lowest_w_in_bin) * self.mu_jobs[i] / self.c
            self.rand_FIFO_lb[k] = bin_lb

            ### The bin ub ###
            bin_ub_min = 0
            bin_ub_max = 0

            for i in range(int(np.floor(the_partitions[k + 1] - 1))):
                bin_ub_min += ((self.w[i] - lowest_w_in_bin) 
                               * self.mu_jobs[i] / self.c)            
                bin_ub_max += ((self.w[i] - highest_w_next_bin) 
                               * self.mu_jobs[i] / self.c)    
                 
            self.rand_FIFO_ub[k] = (
                bin_ub_min + (bin_ub_max - bin_ub_min) * ub_option
            )

            # If we're at the last bin  
            if k == self.num_bins - 1:
                # We will not widen the last bin beyond losing net revenue of 1 per unit of time
                self.rand_FIFO_ub[k] = min(self.rand_FIFO_ub[k], 
                                           bin_ub_min + 1.0 / self.c)

        # Round down to the closest integer
        self.rand_FIFO_lb = np.floor(self.rand_FIFO_lb)
        self.rand_FIFO_ub = np.floor(self.rand_FIFO_ub)
        
        # If the sb queue length is smaller than P, just set the bin lb and bin ub both as [1, 2, …, P]
        if self.sb_queue_length <= self.patience:
            self.rand_FIFO_lb = np.arange(self.patience)
            self.rand_FIFO_ub = np.arange(self.patience)

        # Build the actual bins using the lb and ub values computed above
        self.bins = []
        for i in range(self.num_bins):
            self.bins.append([q for q in range(int(self.rand_FIFO_lb[i]), int(self.rand_FIFO_ub[i]+1))])
        

        ### Compute the net payoff under randomized FIFO ###

        # Initialization
        self.rand_FIFO_cont_payoff = self.cont_payoff_queue_positions.copy()

        # If we are not using the most narrow bins
        # The calculation for the continuation payoff is rather complicated
        # So this is not yet implemented
        if ub_option > 0:
            self.rand_FIFO_cont_payoff[:] = np.nan
            return

        # If we're here, we're using the most narrow bins
        for q in range(0, int(self.sb_queue_length + 1), 1):

            for i in range(self.num_bins):

                # Find the first bin where the bin lb is at least q
                if self.rand_FIFO_lb[i] < q:
                    continue
                else:
                    # If the bin lb is exactly q
                    if self.rand_FIFO_lb[i] == q:
                        # The continuation payoff is precisely that of the worst
                        # trip in the bin 
                        self.rand_FIFO_cont_payoff[q] = (
                            self.W[0, int(the_partitions[i+1]-2)])
                        break

                    else:
                        # In this case, the bin lb of the i'th bin is larger than q

                        # Either q is inside the i-1th bin, in which case the cont payoff is the net 
                        # revenue of the worst trip in the i-1th bin
                        
                        if q <= self.rand_FIFO_ub[i - 1]:
                            self.rand_FIFO_cont_payoff[q] = (
                                self.W[0, int(the_partitions[i]-2)]
                            )
                        else:
                            cumulative_TP = np.sum(self.mu_jobs[0 : int(the_partitions[i]-1) ])

                            self.rand_FIFO_cont_payoff[q] = (
                                self.W[0, int(the_partitions[i]-2)] - self.C[0] *
                                (q - self.rand_FIFO_ub[i-1]) / cumulative_TP
                            )
                    break

            # If we're inside the very last bin    
            if (q > self.rand_FIFO_lb[self.num_bins - 1] and 
                q <= self.rand_FIFO_ub[self.num_bins - 1]):
                self.rand_FIFO_cont_payoff[q] = self.W[0, self.marginal_trip - 1]
                
            # If the sysetem is over-supplied
            if self.over_supplied:
                # If this position is beyond the very last milestone
                if q >= self.direct_FIFO_position[self.L-1]:
                        self.rand_FIFO_cont_payoff[q] = (
                            self.W[0, self.L-1] - self.C[0] *
                            (q - self.direct_FIFO_position[self.L-1]) / self.mu
                        )        


    # This function plots the continuation payoff under direct FIFO
    def plot_cont_payoffs(self):

        if self.J == 1:
            plt.figure(figsize=(6, 3))
            plt.grid(True)
            plt.plot(self.cont_payoff_queue_positions, 
                     self.direct_FIFO_cont_payoff, label = 'DirectFIFO')
            plt.plot(self.cont_payoff_queue_positions, 
                     self.rand_FIFO_cont_payoff, label = 'RandFIFO',
                     linestyle = 'dotted')
            
            plt.legend()
            plt.xlabel('Queue Position')
            plt.ylabel('Cont. Payoff')
            plt.show()

        else:
            print("Don't have analytical solution for direct FIFO cont payoff for more than one driver type!")

    # This function prints the members of the economy class
    def display_economy(self):
        attr_to_not_print = [
            "cont_payoff_queue_positions", 
            "direct_FIFO_cont_payoff",
            "rand_FIFO_cont_payoff", 
            "bins",
            ]
        
        for attr, value in vars(self).items():
            if attr not in attr_to_not_print:
                print(f"{attr}: {value}")

        self.plot_cont_payoffs()