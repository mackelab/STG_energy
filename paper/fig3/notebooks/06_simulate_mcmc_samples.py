from multiprocessing import Pool
import math
import time
from multiprocessing import Pool
from copy import deepcopy
import dill as pickle
import IPython.display as IPd
import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import torch
from matplotlib import ticker
from pyloric import create_prior, simulate, summary_stats
from pyloric.utils import show_traces
from pyloric.utils import energy_of_membrane, energy_of_synapse
from stg_energy import check_if_close_to_obs 
from sbi.analysis import ActiveSubspace
from sbi.utils import BoxUniform
from stg_energy.common import get_labels_8pt
from sbi.analysis import pairplot
from pyloric import create_prior

import stg_energy.fig3_active_subspaces.viz as viz
from stg_energy.fig3_active_subspaces.helper_functions import nth_argmax, nth_argmin

import sys


def simulator(p_with_s):
    p1 = create_prior()
    pars = p1.sample((1,))
    column_names = pars.columns
    circuit_params = np.asarray([p_with_s[:-1]])
    theta_pd = pd.DataFrame(circuit_params, columns=column_names)
    out_target = simulate(
        theta_pd.loc[0], seed=int(p_with_s[-1]), track_energy=True, track_currents=True
    )
    return out_target
    

def run():
    prior_11 = create_prior()
    sys.path.append("home/michael/Documents/sbi/sbi/utils/user_input_checks_utils")
    from sbi.utils import user_input_checks_utils
    sys.modules["sbi.user_input.user_input_checks_utils"] = user_input_checks_utils

    with open(
        "../../../results/trained_neural_nets/inference/posterior_11deg.pickle", "rb"
    ) as handle:
        posterior = pickle.load(handle)
        posterior._device = 'cpu'

    posterior._prior = BoxUniform(posterior._prior.support.base_constraint.lower_bound, posterior._prior.support.base_constraint.upper_bound)
    xo = np.load("../../../results/experimental_data/xo_11deg.npy")

    theta = pd.read_pickle(
        "../../../results/simulation_data_Tube_MLslurm_cluster/close_to_xo_circuit_parameters.pkl"
    )
    x = pd.read_pickle(
        "../../../results/simulation_data_Tube_MLslurm_cluster/close_to_xo_simulation_outputs.pkl"
    )
    seeds = np.load(
        "../../../results/simulation_data_Tube_MLslurm_cluster/close_to_xo_seeds.npy"
    )

    theta_np = theta.to_numpy()
    x_np = x.to_numpy()
    energies = x["energies"]

    quantile = 0.0005
    energies_tt = torch.as_tensor(energies.to_numpy())
    x_tt = torch.as_tensor(x_np, dtype=torch.float32)
    summed_energies = x['energies'].to_numpy()[:, 0] / 10 / 1000
    inds = np.argsort(summed_energies)
    sorted_energies = summed_energies[inds]
    num_vals = sorted_energies.shape[0]
    one_percent_quantile = int(num_vals * quantile)
    ninenine_percent_quantile = int(num_vals * (1-quantile))
    one_percent_energy = sorted_energies[one_percent_quantile]
    ninenine_percent_energy = sorted_energies[ninenine_percent_quantile]
    min_energy_condition = summed_energies < one_percent_energy
    min_energy_theta_abpd = theta_np[min_energy_condition]
    min_energy_seed_abpd = seeds[min_energy_condition]
    min_energy_energies = summed_energies[min_energy_condition]

    energies_tt = torch.as_tensor(energies.to_numpy())
    x_tt = torch.as_tensor(x_np, dtype=torch.float32)
    summed_energies = x['energies'].to_numpy()[:, 1] / 10 / 1000
    inds = np.argsort(summed_energies)
    sorted_energies = summed_energies[inds]
    num_vals = sorted_energies.shape[0]
    one_percent_quantile = int(num_vals * quantile)
    ninenine_percent_quantile = int(num_vals * (1-quantile))
    one_percent_energy = sorted_energies[one_percent_quantile]
    ninenine_percent_energy = sorted_energies[ninenine_percent_quantile]
    min_energy_condition = summed_energies < one_percent_energy
    min_energy_theta_lp = theta_np[min_energy_condition]
    min_energy_seed_lp = seeds[min_energy_condition]
    min_energy_energies = summed_energies[min_energy_condition]

    energies_tt = torch.as_tensor(energies.to_numpy())
    x_tt = torch.as_tensor(x_np, dtype=torch.float32)
    summed_energies = x['energies'].to_numpy()[:, 2] / 10 / 1000
    inds = np.argsort(summed_energies)
    sorted_energies = summed_energies[inds]
    num_vals = sorted_energies.shape[0]
    one_percent_quantile = int(num_vals * quantile)
    ninenine_percent_quantile = int(num_vals * (1-quantile))
    one_percent_energy = sorted_energies[one_percent_quantile]
    ninenine_percent_energy = sorted_energies[ninenine_percent_quantile]
    min_energy_condition = summed_energies < one_percent_energy
    min_energy_theta_py = theta_np[min_energy_condition]
    min_energy_seed_py = seeds[min_energy_condition]
    min_energy_energies = summed_energies[min_energy_condition]


    selected_inds_ab = [1, 3, 7, 8, 16]
    selected_inds_lp = [0, 3, 4, 7, 13]
    selected_inds_py = [0, 1, 5, 12]

    num_sims = 50

    for pair_ab in selected_inds_ab:
        for pair_lp in selected_inds_lp:
            for pair_py in selected_inds_py:
                samples = np.load(f"mcmc_samples_{pair_ab}_{pair_lp}_{pair_py}.npy")[num_sims:num_sims*2]
                np.random.seed(0)
                num_cores = 8
                seeds_sim = np.random.randint(0, 10000, num_sims)

                optimal_1 = min_energy_theta_abpd[0]
                dims_to_sample = [24,25,26,27,28,29,30]
                optimal_1[8:16] = min_energy_theta_lp[pair_lp, 8:16]
                optimal_1[16:24] = min_energy_theta_py[pair_py, 16:24]
                optimal_1 = torch.as_tensor(optimal_1, dtype=torch.float32)

                extended_samples = optimal_1.unsqueeze(0).repeat(num_sims, 1)
                extended_samples[:, dims_to_sample] = torch.as_tensor(samples, dtype=torch.float32)

                params_with_seeds = np.concatenate(
                    (
                        extended_samples.numpy()[:num_sims],
                        seeds_sim[None,].T,
                    ),
                    axis=1,
                )

                with Pool(num_cores) as pool:
                    data = pool.map(simulator, params_with_seeds)

                custom_stats = {
                        "plateau_durations": True,
                        "num_bursts": True,
                        "num_spikes": True,
                        "energies": True,
                        "energies_per_burst": True,
                        "energies_per_spike": True,
                        "pyloric_like": True,
                    }
                stats = [summary_stats(d, stats_customization=custom_stats, t_burn_in=1000) for d in data]
                stats = pd.concat(stats)

                stats.to_pickle(f"simulated_samples_{pair_ab}_{pair_lp}_{pair_py}_2.pkl")

                close_sim = check_if_close_to_obs(stats.to_numpy())
                print("Number of close sims:  ", np.sum(close_sim))

                close_sim = check_if_close_to_obs(stats.to_numpy(), sloppiness_durations=2.0, sloppiness_phases=2.0)
                print("Sloppi 2.0 number of close sims:  ", np.sum(close_sim))

                close_sim = check_if_close_to_obs(stats.to_numpy(), sloppiness_durations=3.0, sloppiness_phases=3.0)
                print("Sloppi 3.0 number of close sims:  ", np.sum(close_sim))

                close_sim = check_if_close_to_obs(stats.to_numpy(), sloppiness_durations=10.0, sloppiness_phases=10.0)
                print("Sloppi 10.0 number of close sims:  ", np.sum(close_sim))

if __name__ == "__main__":
    run()