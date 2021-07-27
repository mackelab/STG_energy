import dill as pickle
import numpy as np
import pandas as pd
import torch
from pyloric import create_prior, simulate, summary_stats
from sbi.utils import BoxUniform
from sbi.analysis import pairplot
from pyloric import create_prior

import sys
from sbi.utils import user_input_checks_utils
from stg_energy.common import check_if_close_to_obs
import argparse


def find_optimal_theta_and_seed(summed_energies, theta_np):
    inds = np.argsort(summed_energies)
    sorted_energies = summed_energies[inds]
    print("sorted_energies first", sorted_energies[:5])
    print("sorted_energies last ", sorted_energies[-5:])
    sorted_theta = theta_np[inds]

    return sorted_theta


def load_theta_x_xo(preparation):
    if preparation == "082":
        theta = pd.read_pickle(
            "../../../results/simulation_data_Tube_MLslurm_cluster/close_to_xo_circuit_parameters.pkl"
        )
        x = pd.read_pickle(
            "../../../results/simulation_data_Tube_MLslurm_cluster/close_to_xo_simulation_outputs.pkl"
        )
        xo = np.load("../../../results/experimental_data/xo_11deg.npy")
    elif preparation == "016":
        theta = pd.read_pickle(
            "../../../results/simulation_data_Tube_MLslurm_cluster/close_to_xo_circuit_parameters_min_burst_condition_016.pkl"
        )
        x = pd.read_pickle(
            "../../../results/simulation_data_Tube_MLslurm_cluster/close_to_xo_simulation_outputs_min_burst_condition_016.pkl"
        )
        xo = np.load("../../../results/experimental_data/xo_11deg_016.npy")
    elif preparation == "078":
        theta = pd.read_pickle(
            "../../../results/simulation_data_Tube_MLslurm_cluster/close_to_xo_circuit_parameters_min_burst_condition_078.pkl"
        )
        x = pd.read_pickle(
            "../../../results/simulation_data_Tube_MLslurm_cluster/close_to_xo_simulation_outputs_min_burst_condition_078.pkl"
        )
        xo = np.load("../../../results/experimental_data/xo_11deg_078.npy")
    else:
        raise NameError

    return theta, x, xo


def sample_mcmc(theta, x, xo, preparation=""):
    theta_np = theta.to_numpy()

    sys.path.append("home/michael/Documents/sbi/sbi/utils/user_input_checks_utils")
    sys.modules["sbi.user_input.user_input_checks_utils"] = user_input_checks_utils

    with open(
        "../../../results/trained_neural_nets/inference/posterior_11deg.pickle", "rb"
    ) as handle:
        posterior = pickle.load(handle)
        posterior._device = "cpu"

    posterior._prior = BoxUniform(
        posterior._prior.support.base_constraint.lower_bound,
        posterior._prior.support.base_constraint.upper_bound,
    )

    abpd_energies = x["energies"].to_numpy()[:, 0] / 10 / 1000
    lp_energies = x["energies"].to_numpy()[:, 1] / 10 / 1000
    py_energies = x["energies"].to_numpy()[:, 2] / 10 / 1000

    min_energy_theta_abpd = find_optimal_theta_and_seed(abpd_energies, theta_np)
    min_energy_theta_lp = find_optimal_theta_and_seed(lp_energies, theta_np)
    min_energy_theta_py = find_optimal_theta_and_seed(py_energies, theta_np)

    print("min_energy_theta_abpd", min_energy_theta_abpd.shape)
    print("min_energy_theta_lp", min_energy_theta_lp.shape)
    print("min_energy_theta_py", min_energy_theta_py.shape)

    dims_to_sample = [24, 25, 26, 27, 28, 29, 30]
    num_sims = 10_000
    for pair_ab in [0]:
        for pair_lp in [3, 4]:
            for pair_py in [2]:
                _ = torch.manual_seed(1)
                np.random.seed(0)
                print("=== Running pair", pair_ab, pair_lp, pair_py, "===")
                optimal_1 = min_energy_theta_abpd[pair_ab]
                optimal_1[8:16] = min_energy_theta_lp[pair_lp, 8:16]
                optimal_1[16:24] = min_energy_theta_py[pair_py, 16:24]
                optimal_1 = torch.as_tensor(optimal_1, dtype=torch.float32)
                samples = posterior.sample_conditional(
                    (num_sims,),
                    condition=optimal_1,
                    dims_to_sample=dims_to_sample,
                    x=xo,
                    mcmc_method="slice_np_vectorized",
                    mcmc_parameters={"init_strategy": "sir", "num_chains": 20},
                )

                repeated_condition = optimal_1.unsqueeze(0).repeat(num_sims, 1)
                repeated_condition[:, dims_to_sample] = samples
                np.save(
                    f"../../../results/mcmc_7d/mcmc_samples_correction_{preparation}_{pair_ab}_{pair_lp}_{pair_py}.npy",
                    repeated_condition,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Take in the preparation")
    parser.add_argument("preparation", type=str)
    args = parser.parse_args()
    preparation = args.preparation

    theta, x, xo = load_theta_x_xo(preparation)
    sample_mcmc(theta, x, xo, preparation)
