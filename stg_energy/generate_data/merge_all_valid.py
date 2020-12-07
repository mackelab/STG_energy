import numpy as np
import pandas as pd
import dill as pickle
import torch

torch.manual_seed(0)

general_path = "../../results/"
path_to_data = "simulation_data_Tube_MLslurm_cluster/01_simulate_11deg/data/all_"
theta_r1 = pd.read_pickle(general_path + path_to_data + "circuit_parameters.pkl")
x_r1 = pd.read_pickle(general_path + path_to_data + "simulation_outputs.pkl")
seeds_r1 = np.load(general_path + path_to_data + "seeds.npy")

path_to_data = "simulation_data_Tube_MLslurm_cluster/01_simulate_11deg_R2/data/all_"
theta_r2 = pd.read_pickle(general_path + path_to_data + "circuit_parameters.pkl")
x_r2 = pd.read_pickle(general_path + path_to_data + "simulation_outputs.pkl")
seeds_r2 = np.load(general_path + path_to_data + "seeds.npy")

path_to_data = "simulation_data_Tube_MLslurm_cluster/01_simulate_11deg_R3/data/all_"
theta_r3 = pd.read_pickle(general_path + path_to_data + "circuit_parameters.pkl")
x = pd.read_pickle(general_path + path_to_data + "simulation_outputs.pkl")
seeds_r3 = np.load(general_path + path_to_data + "seeds.npy")

theta = pd.concat((theta_r1, theta_r2, theta_r3))
x = pd.concat((x_r1, x_r2, x_r3))
seeds = np.concatenate((seeds_r1, seeds_r2, seeds_r3))

path_to_save = "simulation_data_Tube_MLslurm_cluster/all_"
theta.to_pickle(general_path + path_to_save + "circuit_parameters_train.pkl")
x.to_pickle(general_path + path_to_save + "simulation_outputs_train.pkl")
np.save(general_path + path_to_save + "seeds_train.npy", seeds)
