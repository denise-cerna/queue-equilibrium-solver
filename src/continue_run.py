from src.system_class import System
from src.outcome_class import Outcome
from src.utilities import *
from src.setting_class import Setting
from src.simulator_class import Simulator

# This Function lets you restart the simulator from the last saved output

def continue_run(max_iterations, tolerance, version, filename, saving_multiple = 50, setting = None):
    # Extract data from last outcome
    outcome_list = read_dill(filename)
    final_outcome = outcome_list[-1]
    driver_arrival_rate = final_outcome.driver_arrival_rate
    waiting_cost = final_outcome.waiting_cost
    patience = final_outcome.patience
    Qmax = final_outcome.Qmax
    dispatching_rule = final_outcome.dispatching_rule
    earnings = final_outcome.earnings
    # change earnings into list and remove the first entry
    earnings = earnings[1:].tolist()
    # change job rates into list and remove the first entry
    # job rates are the same as the earnings
    job_rates = final_outcome.job_rates[1:].tolist()
    reneging = 0

    # Create Corresponding setting
    if setting == None:
        setting = Setting(earnings, job_rates, driver_arrival_rate, waiting_cost, patience, Qmax, dispatching_rule, reneging, initial_alpha = '1')
        setting.alpha = final_outcome.alpha
        setting.V = final_outcome.V
        setting.phi = final_outcome.phi
        setting.inspect = final_outcome.inspect
        # setting.beta = 0.9

    system = System(setting)

    # Adjust Simulator
    simulator = Simulator(system, max_iterations, tolerance, version, saving_multiple)
    simulator.outcome_history = outcome_list
    simulator.iter = final_outcome.iter

    #simulator.direction_history = outcome_list[-1].direction_history
    print("Starting at Iteration: ", final_outcome.iter)
    print("Starting Max V Loss ", final_outcome.max_loss_V)
    print("Starting Max Phi Loss ", final_outcome.max_loss_phi)
    #print("Starting Max Inspect Loss ", final_outcome.max_loss_inspect)
    outcome_list, V, a, join, num_iter, inspect = simulator.run_iterations()
    return outcome_list, V, a, join, num_iter, inspect, simulator
