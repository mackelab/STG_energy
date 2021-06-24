import dill as pickle
import numpy as np
import pandas as pd
import torch
from pyloric import create_prior
from sbi.utils import BoxUniform
from sbi.analysis import pairplot
from pyloric import create_prior

import sys
from sbi.utils import user_input_checks_utils


prior_11 = create_prior()
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

prior = create_prior(as_torch_dist=True)
lower_b = prior.support.base_constraint.lower_bound.unsqueeze(0)
upper_b = prior.support.base_constraint.upper_bound.unsqueeze(0)
limits = torch.cat((lower_b, upper_b), dim=0).T

xo = np.load("../../../results/experimental_data/xo_11deg.npy")
energies = x["energies"]
energies_tt = torch.as_tensor(energies.to_numpy())
x_tt = torch.as_tensor(x_np, dtype=torch.float32)
summed_energies = np.sum(x["energies"].to_numpy(), axis=1) / 10 / 1000

quantile = 0.00015
energies_tt = torch.as_tensor(energies.to_numpy())
x_tt = torch.as_tensor(x_np, dtype=torch.float32)

abpd_energies = x["energies"].to_numpy()[:, 0] / 10 / 1000
lp_energies = x["energies"].to_numpy()[:, 1] / 10 / 1000
py_energies = x["energies"].to_numpy()[:, 2] / 10 / 1000


def find_optimal_theta_and_seed(summed_energies, theta_np, seeds):
    inds = np.argsort(summed_energies)
    sorted_energies = summed_energies[inds]
    num_vals = sorted_energies.shape[0]
    one_percent_quantile = int(num_vals * quantile)
    one_percent_energy = sorted_energies[one_percent_quantile]

    min_energy_condition = summed_energies < one_percent_energy
    min_energy_theta_abpd = theta_np[min_energy_condition]
    min_energy_seed_abpd = seeds[min_energy_condition]

    return min_energy_theta_abpd, min_energy_seed_abpd


min_energy_theta_abpd, min_energy_seed_abpd = find_optimal_theta_and_seed(
    abpd_energies, theta_np, seeds
)
min_energy_theta_lp, min_energy_seed_lp = find_optimal_theta_and_seed(
    lp_energies, theta_np, seeds
)
min_energy_theta_py, min_energy_seed_py = find_optimal_theta_and_seed(
    py_energies, theta_np, seeds
)

print("min_energy_theta_abpd", min_energy_theta_abpd.shape)
print("min_energy_theta_lp", min_energy_theta_lp.shape)
print("min_energy_theta_py", min_energy_theta_py.shape)

# selected_inds_ab = [0, 1, 3, 7, 8, 16]
# selected_inds_lp = [0, 3, 4, 7, 13]
# selected_inds_py = [0, 1, 5, 12]

# selected_inds_ab = [0, 1]
# selected_inds_lp = [0, 3]
# selected_inds_py = [0, 12]

dims_to_sample = [24, 25, 26, 27, 28, 29, 30]
num_sims = 200
for pair_ab in range(5):
    for pair_lp in range(5):
        for pair_py in range(5):
            _ = torch.manual_seed(0)
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
                mcmc_parameters={"init_strategy": "sir", "num_chains": 4},
            )

            repeated_condition = optimal_1.unsqueeze(0).repeat(num_sims, 1)
            repeated_condition[:, dims_to_sample] = samples
            np.save(
                f"../../../results/mcmc_7d/mcmc_samples_{pair_ab}_{pair_lp}_{pair_py}.npy",
                repeated_condition,
            )
