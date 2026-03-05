# %%
import numpy as np
import copy
from os.path import dirname, abspath, join
import sys
sys.path.append(abspath(join(dirname(__file__), "../..")))

from src.setting_class import Setting
from src.system_class import System
from src.dispatching_class import Dispatching
from src.simulator_class import *
from src.reporting_class import *
from src.utilities import *
from src.get_economies import *
import numpy as np

def extract_policy(V, earnings):
    """
    Given the converged value function V (shape (Q_max+1,))
    and earnings vector (shape (L,)), return two arrays:

      accept[q, ℓ] = True if accepting job type ℓ at queue‐length q is 
                     (weakly) optimal;
      join[q]      = True if joining (rather than balking) at queue‐length q 
                     is (weakly) optimal.
    """
    Q_max = V.size - 1
    L     = earnings.size

    # 1) Acceptance policy: only defined for q >= 1
    accept = np.zeros((Q_max+1, L), dtype=bool)
    # compare earnings + V[q-1] against V[q] for q=1,...,Q_max
    # shape broadcasting: earnings[None,:] + V[:-1,None]  vs. V[1:,None]
    accept[1:] = (earnings[None, :] + V[:-1, None]) >= V[1:, None]

    # 2) Join policy: compare V[q+1] vs V[q] for q=0,...,Q_max-1
    # If queue is empty, join[0] corresponds to admitting someone to position 1
    join = np.zeros(Q_max+1, dtype=bool)
    join[:-1] = V[1:] >= V[:-1]
    # by convention at Q_max we set join[Q_max] = 0 (no further state)
    return accept.astype(int), join.astype(int)


# %%
def value_iteration(job_rates,
                    earnings,
                    driver_arrival_rate,
                    reneging,
                    waiting_cost,
                    Q_max,
                    tolerance,
                    max_iters,
                    M):

    L      = job_rates.size
    
    # Initial V
    V      = np.zeros(Q_max+1)
    sum_mu = np.sum(job_rates)
    delta_list      = [V]

    for it in range(max_iters):
        V_new = np.empty_like(V)

        # Q = 0
        R0 =  driver_arrival_rate
        V_new[0] = (driver_arrival_rate * max(V[0], V[1]) + (M-R0)*V[0]) / M

        # 1 ≤ Q < Q_max
        for Q in range(1, Q_max+1):
            R  = sum_mu + driver_arrival_rate + reneging * Q
            maxim = np.maximum(earnings + V[Q-1], V[Q])
            service   = np.sum(job_rates * maxim)
            reneg_term = reneging * Q * V[Q-1]
            if Q < Q_max:
                arrival = driver_arrival_rate * max(V[Q], V[Q+1])
            # arrivals blocked, state stays in Q_max
            else:
                arrival = driver_arrival_rate * V[Q]  
            
            no_event = (M-R)*V[Q]

            V_new[Q] = (
                - waiting_cost * Q
                + service
                + arrival
                + reneg_term
                + no_event
            ) / M

        # delta convergence check
        delta  = V_new - V
        delta_list.append(delta)

        # Check against previous delta
        if (np.max(delta_list[-1]) - np.min(delta_list[-1])) < tolerance:
            print(f"Converged in {it} iterations.")
            #print(f"delta: {delta}")
            break

        # print every multiple of 5000
        if it % 5000 == 0:
            print(f"iter {it:4d}, delta_span = {np.max(delta_list[-1]) - np.min(delta_list[-1])}, delta = {np.max(delta)*M}, v[0]= {V[0]:.2f}")
        #print(f"iter {it:4d}, delta_span = {np.max(delta_list[-1]) - np.min(delta_list[-1])}, delta = {np.max(delta)*M}, v[0]= {V[0]:.2f}")
        
        delta_list.append(delta)
        V = copy.deepcopy(V_new)

    accept_policy, join_policy = extract_policy(V_new, earnings)


    return V, np.max(delta), accept_policy, join_policy


# %%
def build_transition_matrix(job_rates,
                            driver_arrival_rate,
                            reneging,
                            accept,   # shape (Q_max+1, L), bool
                            join,     # shape (Q_max+1,),   bool
                            M):
    """
    Returns P of shape ((Q_max+1),(Q_max+1)) where
      P[q,q'] = Prob(state=q -> state=q') per uniformized step.
    """
    Q_max = accept.shape[0] - 1
    L     = job_rates.size
    
    # transition matrix
    P     = np.zeros((Q_max+1, Q_max+1)) 

    sum_mu = np.sum(job_rates)

    # uniformization rate
    M = driver_arrival_rate + sum_mu + reneging*Q_max

    for q in range(Q_max+1):
        # 1) driver arrival
        pa = driver_arrival_rate / M
        if join[q] and q < Q_max:
            P[q, q+1] += pa #increase queue length by 1
        else:
            P[q, q]   += pa # stay in the same state

        # 2) job arrivals (type ℓ)
        for ell in range(L):
            pj = job_rates[ell] / M
            # if q>=1 and accept[q,ell], we transition to q-1
            if q >= 1 and accept[q, ell]:
                P[q, q-1] += pj
            else:
                P[q, q]  += pj

        # 3) reneging
        pr = (reneging * q) / M
        if q >= 1:
            # there is some chance of exgeneous reneging
            P[q, q-1] += pr
        else:
            P[q, q]  += pr

        # 4) null‐event to fill up to 1
        total_events = driver_arrival_rate + sum_mu + reneging * q
        P[q, q] += 1 - (total_events / M)

    return P


def steady_state(P, tol=1e-12, max_iters=10000):
    """
    Power‐iteration to find π: π = π P.
    """
    n = P.shape[0]
    pi = np.ones(n) / n
    for _ in range(max_iters):
        pi_next = pi.dot(P)
        if np.max(np.abs(pi_next - pi)) < tol:
            return pi_next
        pi = pi_next
    return pi  # if no convergence, return last iterate

def compute_throughput(pi, job_rates, accept):
    """
    Compute the long‐run service rate (throughput) in jobs/time.

    Parameters
    ----------
    pi : array_like, shape (Q_max+1,)
        steady‐state probability of being in each queue‐length state q.
    job_rates : array_like, shape (L,)
        continuous‐time arrival rates of each job class ℓ.
    accept : array_like, shape (Q_max+1, L), bool
        accept[q,ℓ] = True if in state q you accept class‐ℓ arrival.

    Returns
    -------
    throughput : float
        expected number of services (matches) completed per unit time.
    """
    # in each state q, class-ℓ arrivals that get accepted occur at rate job_rates[ℓ]
    # so overall throughput = sum_q π[q] * sum_ℓ job_rates[ℓ] * 1{accept[q,ℓ]}
    return float(np.sum(pi[:, None] * accept * job_rates[None, :]))


# %%
economy = get_ohare_economy()
earnings = economy.w.tolist()
job_rates = economy.mu_jobs.tolist()

earnings = earnings.copy()
earnings.insert(0, 0)
earnings = np.array(earnings)

# arrival rate of jobs
# Add 100 to front for endogeneous reneging
job_rates = job_rates.copy()
job_rates.insert(0, 100)
job_rates = np.array(job_rates)

waiting_cost = economy.c
#patience = economy.patience
patience = 800
reneging = 0
driver_arrival_rate = 12
Qmax = 800
tolerance = 1e-6
max_iters = 100000
M = 200
fb_net_rev = []
fb_ss = []
fb_waiting_time = []
fb_throughput = []
accept_policy_list = []
join_policy_list = []

for driver_arrival_rate in range(1, 16):
    V, delta, accept_policy, join_policy = value_iteration( job_rates, earnings, driver_arrival_rate, reneging, waiting_cost, Qmax, tolerance, max_iters, M)
    print("Delta:", delta*M)

    # compure metrics
    P = build_transition_matrix(job_rates, driver_arrival_rate, reneging, accept_policy, join_policy, M)
    pi = steady_state(P)
    ss = np.sum(pi *  list(range(0, len(pi))))
    throughput = compute_throughput(pi, job_rates, accept_policy)

    # save metrics
    fb_ss.append(ss)
    fb_throughput.append(throughput)
    fb_waiting_time.append(ss/throughput)
    fb_net_rev.append(delta * M)
    accept_policy_list.append(accept_policy)
    join_policy_list.append(join_policy)

#save metrics
# Folder where you want to save the plot

save_dir = Path("completed_runs") / "fb_data"
save_dir.mkdir(parents=True, exist_ok=True)  # optional: create folder if missing

np.save(save_dir / 'fb_net_rev.npy', fb_net_rev)
np.save(save_dir / 'fb_accept_policy_list.npy', accept_policy_list)
np.save(save_dir / 'fb_join_policy_list.npy', join_policy_list)
np.save(save_dir / 'fb_queue_length.npy', fb_ss)
np.save(save_dir / 'fb_throughput.npy', fb_throughput)
np.save(save_dir / 'fb_waiting_time.npy', fb_waiting_time)



