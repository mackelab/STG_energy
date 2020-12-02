from pyloric import simulate, create_prior, summary_stats
import numpy as np
import time

import multiprocessing
from multiprocessing import Pool
import torch
import pandas as pd

# File is running on `pyloric` commit:
# 46a527673dae6a25cf6d4d6bdbf14f6f0282796e "stats is now called summary_stats"

# Transmit data:
# scp -r stg_energy/generate_data/* mdeistler57@134.2.168.52:~/Documents/STG_energy/stg_energy/generate_data
# scp -r results/simulation_data_Tube_MLslurm_cluster/01_simulate_11deg/* mdeistler57@134.2.168.52:~/Documents/STG_energy/results/simulation_data_Tube_MLslurm_cluster/01_simulate_11deg

# Get data back:
# scp -r mdeistler57@134.2.168.52:~/Documents/STG_energy/results/simulation_data_Tube_MLslurm_cluster/01_simulate_11deg/data/* results/simulation_data_Tube_MLslurm_cluster/01_simulate_11deg/data


def my_simulator(params_with_seeds):
    p1 = create_prior()
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
        temperature=283,
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


num_repeats = 40

for _ in range(num_repeats):

    num_sims = 10000
    num_cores = 32

    global_seed = int((time.time() % 1) * 1e7)
    np.random.seed(global_seed)  # Seeding the seeds for the simulator.
    torch.manual_seed(global_seed)  # Seeding the prior.
    seeds = np.random.randint(0, 10000, (num_sims, 1))

    prior = create_prior()
    parameter_sets = prior.sample((num_sims,))
    data_np = parameter_sets.to_numpy()
    params_with_seeds = np.concatenate((data_np, seeds), axis=1)

    with Pool(num_cores) as pool:
        start_time = time.time()
        data = pool.map(my_simulator, params_with_seeds)
        print("Simulation time", time.time() - start_time)

    sim_outs = pd.concat(data)

    general_path = "/home/macke/mdeistler57/Documents/STG_energy/results/"
    path_to_data = "simulation_data_Tube_MLslurm_cluster/01_simulate_11deg/data/"
    filename = f"sim_{global_seed}"
    sim_outs.to_pickle(
        general_path + path_to_data + "simulation_outputs/" + filename + ".pkl"
    )
    parameter_sets.to_pickle(
        general_path + path_to_data + "circuit_parameters/" + filename + ".pkl"
    )
    np.save(general_path + path_to_data + "seeds/" + filename, seeds)

    print("============ Finished ============")
