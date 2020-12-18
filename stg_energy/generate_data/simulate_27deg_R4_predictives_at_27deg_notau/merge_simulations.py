import pandas as pd
import numpy as np
import os


def merge_dataframes(file_dir: str) -> None:
    """
    Save all files that were simulated on the cluster in a single file.

    Overall, six files are created: for circuit parameters, simulations, seeds. For
    each of them, two files: one with `valid` simulations and one with `bad`
    simulations.

    Args:
        file_dir: Directory in which the files lie.
    """
    # checking for hidden files. If the file starts with '.', we discard it. Also
    # discard readme.txt

    files = os.listdir(file_dir + "simulation_outputs/")
    filenames_sims = []
    for file in files:
        if file[0] != "." and file != "readme.txt":
            filenames_sims.append(file)

    valid_params = pd.DataFrame({})
    valid_sims = pd.DataFrame({})
    valid_seeds = np.asarray([])

    bad_params = pd.DataFrame({})
    bad_sims = pd.DataFrame({})
    bad_seeds = np.asarray([])

    all_params = pd.DataFrame({})
    all_sims = pd.DataFrame({})
    all_seeds = np.asarray([])

    for fname_sims in filenames_sims:
        params = pd.read_pickle(file_dir + "circuit_parameters/" + fname_sims)
        stats = pd.read_pickle(file_dir + "simulation_outputs/" + fname_sims)
        seeds = np.load(file_dir + "seeds/" + fname_sims[:-3] + "npy")

        stats_np = stats.to_numpy()
        condition = np.any(pd.isnull(stats_np), axis=1)

        valid_params = pd.concat(
            [valid_params, params[np.invert(condition)]], ignore_index=True
        )
        valid_sims = pd.concat(
            [valid_sims, stats[np.invert(condition)]], ignore_index=True
        )
        valid_seeds = np.concatenate(
            [valid_seeds, np.squeeze(seeds[np.invert(condition)])]
        )

        bad_params = pd.concat([bad_params, params[condition]], ignore_index=True)
        bad_sims = pd.concat([bad_sims, stats[condition]], ignore_index=True)
        bad_seeds = np.concatenate([bad_seeds, np.squeeze(seeds[condition])])

        all_params = pd.concat([all_params, params], ignore_index=True)
        all_sims = pd.concat([all_sims, stats], ignore_index=True)
        all_seeds = np.concatenate([all_seeds, np.squeeze(seeds)])

    # Save data.
    general_path = "../../../results/"
    path_to_data = "simulation_data_Tube_MLslurm_cluster/simulate_27deg_R4_predictives_at_27deg_notau/data/"
    valid_params.to_pickle(general_path + path_to_data + "valid_circuit_parameters.pkl")
    valid_sims.to_pickle(general_path + path_to_data + "valid_simulation_outputs.pkl")
    np.save(general_path + path_to_data + "valid_seeds", valid_seeds)

    bad_params.to_pickle(general_path + path_to_data + "bad_circuit_parameters.pkl")
    bad_sims.to_pickle(general_path + path_to_data + "bad_simulation_outputs.pkl")
    np.save(general_path + path_to_data + "bad_seeds", bad_seeds)

    all_params.to_pickle(general_path + path_to_data + "all_circuit_parameters.pkl")
    all_sims.to_pickle(general_path + path_to_data + "all_simulation_outputs.pkl")
    np.save(general_path + path_to_data + "all_seeds", all_seeds)
