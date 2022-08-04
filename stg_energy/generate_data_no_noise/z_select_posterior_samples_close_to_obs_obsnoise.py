import numpy as np
from copy import deepcopy
import pandas as pd
from stg_energy import check_if_close_to_obs


def select_ss_close_to_obs(params, stats, seeds):

    """
    Returns those summstats that are within allowed_deviation from x_o.

    Also, the summstats must have at least 8 bursts and no plateaus.
    """

    conditions = check_if_close_to_obs(stats)

    good_params = params[conditions]
    good_data = stats[conditions]
    good_seeds = seeds[conditions]

    return good_params, good_data, good_seeds


path1 = "../../results/simulation_data_Tube_MLslurm_cluster_no_noise/"
path2 = "simulate_11deg_R3_predictives_at_11deg_obsnoise"

theta = pd.read_pickle(path1 + path2 + "/data/valid_circuit_parameters.pkl")
x = pd.read_pickle(path1 + path2 + "/data/valid_simulation_outputs.pkl")
seeds = np.load(path1 + path2 + "/data/valid_seeds.npy")

theta_np = theta.to_numpy()
x_np = x.to_numpy()

theta_close, x_close, seeds_close = select_ss_close_to_obs(theta_np, x_np, seeds)

columns_theta = theta.columns
columns_x = x.columns
theta_close_pd = pd.DataFrame(theta_close, columns=columns_theta)
x_close_pd = pd.DataFrame(x_close, columns=columns_x)

theta_close_pd.to_pickle(path1 + "/close_to_xo_circuit_parameters_obsnoise.pkl")
x_close_pd.to_pickle(path1 + "/close_to_xo_simulation_outputs_obsnoise.pkl")
np.save(path1 + "/close_to_xo_seeds_obsnoise.npy", seeds_close)
