import numpy as np
import pandas as pd


def load_valid_sims_11deg():
    """
    Return the circuit parameters, simulations, and seeds of all `valid` simulations.

    Returns: Parameters, simulation outputs, seeds.
    """
    general_path = "../results/"
    path_to_data = "simulation_data_Tube_MLslurm_cluster/01_simulate_11deg/data/valid_"
    params = pd.read_pickle(general_path + path_to_data + "circuit_parameters.pkl")
    sims = pd.read_pickle(general_path + path_to_data + "simulation_outputs.pkl")
    seeds = np.load(general_path + path_to_data + "seeds.npy")

    return params, sims, seeds


def load_bad_sims_11deg():
    """
    Return the circuit parameters, simulations, and seeds of all `bad` simulations.

    Returns: Parameters, simulation outputs, seeds.
    """
    general_path = "../results/"
    path_to_data = "simulation_data_Tube_MLslurm_cluster/01_simulate_11deg/data/bad_"
    params = pd.read_pickle(general_path + path_to_data + "circuit_parameters.pkl")
    sims = pd.read_pickle(general_path + path_to_data + "simulation_outputs.pkl")
    seeds = np.load(general_path + path_to_data + "seeds.npy")

    return params, sims, seeds


def load_all_sims_11deg():
    """
    Return the circuit parameters, simulations, and seeds of all `bad` simulations.

    Returns: Parameters, simulation outputs, seeds.
    """
    general_path = "../results/"
    path_to_data = "simulation_data_Tube_MLslurm_cluster/01_simulate_11deg/data/all_"
    params = pd.read_pickle(general_path + path_to_data + "circuit_parameters.pkl")
    sims = pd.read_pickle(general_path + path_to_data + "simulation_outputs.pkl")
    seeds = np.load(general_path + path_to_data + "seeds.npy")

    return params, sims, seeds
