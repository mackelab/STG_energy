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
# scp -r results/trained_neural_nets/inference/posterior_27deg.pickle mdeistler57@134.2.168.52:~/Documents/STG_energy/results/trained_neural_nets/inference/
# scp -r stg_energy/generate_data/* mdeistler57@134.2.168.52:~/Documents/STG_energy/stg_energy/generate_data

# Get data back:
# scp -r mdeistler57@134.2.168.52:~/Documents/STG_energy/results/simulation_data_Tube_MLslurm_cluster/simulate_27deg_R4_predictives_at_27deg/data/* results/simulation_data_Tube_MLslurm_cluster/simulate_27deg_R4_predictives_at_27deg/data


def my_simulator(params_with_seeds):
    p1 = create_prior(
        customization={
            "Q10_gbar_mem": [True, True, True, True, True, True, True, True],
            "Q10_gbar_syn": [True, True],
            "Q10_tau_m": [True],
            "Q10_tau_h": [True],
            "Q10_tau_CaBuff": [True],
            "Q10_tau_syn": [True, True],
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


num_repeats = 17  # 17

for kkkk in range(num_repeats):

    num_sims = 10000
    num_cores = 32

    generic_prior = create_prior()
    generic_sample = generic_prior.sample((1,))
    column_names = generic_sample.columns

    global_seed = kkkk
    np.random.seed(global_seed)  # Seeding the seeds for the simulator.
    torch.manual_seed(global_seed)  # Seeding the prior.
    seeds = np.random.randint(0, 10000, (num_sims, 1))

    q10_prior = create_prior(
        customization={
            "Q10_gbar_mem": [True, True, True, True, True, True, True, True],
            "Q10_gbar_syn": [True, True],
            "Q10_tau_m": [True],
            "Q10_tau_h": [True],
            "Q10_tau_CaBuff": [True],
            "Q10_tau_syn": [True, True],
        }
    )
    path = "../../../results/trained_neural_nets/inference/"
    with open(path + "posterior_27deg.pickle", "rb") as handle:
        posterior = pickle.load(handle)
    x_o = np.load(
        "../../../results/experimental_data/xo_27deg.npy",
        allow_pickle=True,
    )
    posterior_parameter_sets = posterior.sample(
        (num_sims,), x=torch.as_tensor([x_o], dtype=torch.float32)
    )
    posterior_parameter_sets_np = posterior_parameter_sets.numpy()
    save_sets_pd = pd.DataFrame(posterior_parameter_sets_np, columns=q10_prior.sample((1,)).columns)
    params_with_seeds = np.concatenate((posterior_parameter_sets_np, seeds), axis=1)

    print("params_with_seeds", params_with_seeds)
    print("params_with_seeds[0]", params_with_seeds[0])
    print("save_sets_pd", save_sets_pd.loc[0])
    
    with Pool(num_cores) as pool:
        start_time = time.time()
        data = pool.map(my_simulator, params_with_seeds)
        print("Simulation time", time.time() - start_time)

    sim_outs = pd.concat(data)

    general_path = "/home/macke/mdeistler57/Documents/STG_energy/results/"
    path_to_data = "simulation_data_Tube_MLslurm_cluster/simulate_27deg_R4_predictives_at_27deg/data/"
    filename = f"sim_{global_seed}"
    sim_outs.to_pickle(
        general_path + path_to_data + "simulation_outputs/" + filename + ".pkl"
    )
    save_sets_pd.to_pickle(
        general_path + path_to_data + "circuit_parameters/" + filename + ".pkl"
    )
    np.save(general_path + path_to_data + "seeds/" + filename, seeds)

    print("============ Finished ============")
