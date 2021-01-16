from pyloric import simulate, create_prior, summary_stats
import numpy as np
import time

import multiprocessing
from multiprocessing import Pool
import torch
import pandas as pd
import dill as pickle

# File is running on `pyloric` commit:
# 46a527673dae6a25cf6d4d6bdbf14f6f0282796e "stats is now called summary_stats"

# Transmit data:
# scp -r results/trained_neural_nets/inference/posterior_11deg.pickle mdeistler57@134.2.168.52:~/Documents/STG_energy/results/trained_neural_nets/inference/
# scp -r stg_energy/generate_data/* mdeistler57@134.2.168.52:~/Documents/STG_energy/stg_energy/generate_data
# scp -r results/experimental_data/xo_11deg_078.npy mdeistler57@134.2.168.52:~/Documents/STG_energy/results/experimental_data/

# Get data back:
# scp -r mdeistler57@134.2.168.52:~/Documents/STG_energy/results/simulation_data_Tube_MLslurm_cluster/simulate_11deg_R3_predictives_at_27deg_notau_078/data/* results/simulation_data_Tube_MLslurm_cluster/simulate_11deg_R3_predictives_at_27deg_notau_078/data


def my_simulator(params_with_seeds):
    p1 = create_prior(
        customization={
            "Q10_gbar_mem": [True, True, True, True, True, True, True, True],
            "Q10_gbar_syn": [True, True],
            "Q10_tau_m": [False],
            "Q10_tau_h": [False],
            "Q10_tau_CaBuff": [False],
            "Q10_tau_syn": [False, False],
        }
    )
    pars = p1.sample((1,))
    column_names = pars.columns

    parameter_set_pd = pd.DataFrame(
        np.asarray([params_with_seeds[:-1]]), columns=column_names
    )
    out_target = simulate(
        parameter_set_pd.loc[0],
        seed=int(params_with_seeds[-1]),
        dt=0.025,
        t_max=11000,
        temperature=299,
        noise_std=0.001,
        track_energy=True,
    )
    custom_stats = {
        "plateau_durations": True,
        "num_bursts": True,
        "num_spikes": True,
        "energies": True,
        "energies_per_burst": True,
        "energies_per_spike": True,
        "pyloric_like": True,
    }
    return summary_stats(out_target, stats_customization=custom_stats, t_burn_in=1000)


num_repeats = 50

for _ in range(num_repeats):

    num_sims = 10000
    num_cores = 32

    generic_prior = create_prior()
    generic_sample = generic_prior.sample((1,))
    column_names = generic_sample.columns

    global_seed = int((time.time() % 1) * 1e7)
    np.random.seed(global_seed)  # Seeding the seeds for the simulator.
    torch.manual_seed(global_seed)  # Seeding the prior.
    seeds = np.random.randint(0, 10000, (num_sims, 1))

    q10_prior = create_prior(
        customization={
            "Q10_gbar_mem": [True, True, True, True, True, True, True, True],
            "Q10_gbar_syn": [True, True],
            "Q10_tau_m": [False],
            "Q10_tau_h": [False],
            "Q10_tau_CaBuff": [False],
            "Q10_tau_syn": [False, False],
        }
    )
    prior_parameter_sets_pd = q10_prior.sample((num_sims,))
    path = "../../../results/trained_neural_nets/inference/"
    with open(path + "posterior_11deg.pickle", "rb") as handle:
        posterior = pickle.load(handle)
    posterior._device = "cpu"
    x_o = np.load(
        "../../../results/experimental_data/xo_11deg_078.npy",
        allow_pickle=True,
    )
    posterior_parameter_sets = posterior.sample(
        (num_sims,), x=torch.as_tensor([x_o], dtype=torch.float32)
    )
    posterior_data_np = posterior_parameter_sets.detach().numpy()
    params_with_q10s = prior_parameter_sets_pd.to_numpy()
    params_with_q10s[:, :31] = posterior_data_np
    params_with_seeds = np.concatenate((params_with_q10s, seeds), axis=1)

    posterior_parameter_sets_pd = pd.DataFrame(posterior_data_np, columns=column_names)
    prior_parameter_sets_pd["AB/PD"] = posterior_parameter_sets_pd["AB/PD"]
    prior_parameter_sets_pd["LP"] = posterior_parameter_sets_pd["LP"]
    prior_parameter_sets_pd["PY"] = posterior_parameter_sets_pd["PY"]
    prior_parameter_sets_pd["Synapses"] = posterior_parameter_sets_pd["Synapses"]

    print("params_with_seeds", params_with_seeds)
    print("params_with_seeds[0]", params_with_seeds[0])
    print("parameter_sets_pd", prior_parameter_sets_pd)
    print("parameter_sets_pd.loc[0]", prior_parameter_sets_pd.loc[0])

    with Pool(num_cores) as pool:
        start_time = time.time()
        data = pool.map(my_simulator, params_with_seeds)
        print("Simulation time", time.time() - start_time)

    sim_outs = pd.concat(data)
    sim_outs_np = sim_outs.to_numpy()
    not_nan = np.invert(np.any(np.isnan(sim_outs_np), axis=1))
    print("number of good sims:  ", np.sum(not_nan))

    general_path = "/home/macke/mdeistler57/Documents/STG_energy/results/"
    path_to_data = "simulation_data_Tube_MLslurm_cluster/simulate_11deg_R3_predictives_at_27deg_notau_078/data/"
    filename = f"sim_{global_seed}"
    sim_outs.to_pickle(
        general_path + path_to_data + "simulation_outputs/" + filename + ".pkl"
    )
    prior_parameter_sets_pd.to_pickle(
        general_path + path_to_data + "circuit_parameters/" + filename + ".pkl"
    )
    np.save(general_path + path_to_data + "seeds/" + filename, seeds)

    print("============ Finished ============")
