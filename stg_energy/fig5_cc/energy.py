import numpy as np
from copy import deepcopy
import torch


def extract_min_prob(
    posterior_MAF_11,
    condition1_norm,
    grid_bins,
    dim1,
    dim2,
    lims,
    mode="posterior_prob",
):
    """
    Used for plotting the energy in the plane of the conditional probability

    :param posterior_MAF_11: posterior
    :param condition1_norm: the value (usually posterior sample) we are conditioning on
    :param grid_bins: number of bins / resolution
    :param dim1: int
    :param dim2: int
    :param mode: string. Distinguishes two computations in which the min_prob is computed
    :return: minimum required probability to be simulated
    """
    if mode == "posterior_prob":
        prob = np.exp(posterior_MAF_11.log_prob(condition1_norm).detach().item() - 100)
        min_prob = 0.35 * prob
    else:
        all_probs = []

        vec_dim1 = np.linspace(lims[dim1, 0], lims[dim1, 1], grid_bins)
        vec_dim2 = np.linspace(lims[dim2, 0], lims[dim2, 1], grid_bins)

        for i1, v1 in enumerate(vec_dim1):
            for i2, v2 in enumerate(vec_dim2):
                parameter_set = deepcopy(condition1_norm)

                parameter_set[0, dim1] = v1
                parameter_set[0, dim2] = v2
                prob = np.exp(
                    posterior_MAF_11.log_prob(parameter_set).detach().numpy() - 100
                )
                all_probs.append(prob[0])
        all_probs = np.asarray(all_probs)

        diff_of_min_max = np.max(all_probs) - np.min(all_probs)
        min_prob = np.min(all_probs) + 0.8 * diff_of_min_max

    return min_prob


from pyloric import simulate, summary_stats


def check_if_close_to_obs(stats, observation, num_std, stats_std=None):
    """
    Returns those summstats that are within num_std standard deviations
        from the observation and no bursts and no plateaus.
    :param data: summstats
    :param observation:
    :param num_std:
    :return:
    """
    if stats_std is None:
        # setting to experimental stds from prinz paper

        stats_std = np.asarray(
            [
                279,
                133,
                113,
                150,
                109,
                60,
                169,
                216,
                0.040,
                0.059,
                0.054,
                0.065,
                0.034,
                0.054,
                0.060,
            ]
        )

    data_trunc = np.asarray(stats[:15])
    observation = observation[:15]

    good_sim = True

    # check distance to observation
    diff_to_obs = np.abs(data_trunc - observation) / stats_std[:15]
    if not np.all(diff_to_obs < num_std):
        good_sim = False

    # check if more than 7 bursts
    backup_stats = deepcopy(stats)
    if not np.all(backup_stats[22:24] > 7.5):
        good_sim = False

    # check if no plateaus
    backup_stats = deepcopy(stats)
    if not np.all(backup_stats[15:18] == 2.5):
        good_sim = False

    return good_sim


def energy_of_conditional(
    posterior_MAF_11,
    condition1_norm,
    grid_bins,
    min_prob,
    dim1,
    dim2,
    lims,
    stats_std,
    neuron_to_observe,
    patience=1,
    regression_net=None,
    theta_mean=None,
    theta_std=None,
    x_mean=None,
    x_std=None,
):
    """
    Return image that contains the energy of each parameter value in conditional plane.

    Will return -1.0 if the parameter set has to low of a posterior probability to be
    simulated.

    Will return 0.0 if the parameter set has been simulated, but had summary stats that
    were outside of the permissible distance from the observation.

    Will return the energy if simulated and close enough to the simulation.

    :param posterior_MAF_11: posterior
    :param obs: true observation
    :param condition1_norm: the value (usually posterior sample) we are conditioning on
    :param grid_bins: number of bins / resolution
    :param min_prob: float, minimum required probability to be simulated
    :param dim1: int
    :param dim2: int
    :param neuron_to_observe: string, What neuron should we compute the energy per
        spike of? Either of the following: 'PM', 'LP', 'PY'
    :param patience: how often we simulate when outside of allowed error from
        observation.
    :param regression_net: if None, we simulate to obtain the energy of every point on
        the grid. If provided, we just run the parameter set through the net and let
        it predict the energy.
    :param neural_net_zscore_mean, neural_net_zscore_std: mean and std that are used to
        standardize the parameters before feeding them into the regression_net. Ignored
        if `regression_net=None`.
    :param neural_net_zscore_mean_energy, neural_net_zscore_std_energy: mean and std
        used to un-normalize the energy estimate from the regression net.

    :return: image
    """
    lowest_allowed = min_prob

    npz = np.load(
        "/home/michael/Documents/STG_energy/results/experimental_data/190807_summstats_prep845_082_0044.npz"
    )
    observation = npz["summ_stats"]

    # off-diagonals
    if dim1 != dim2:
        vec_dim1 = np.linspace(lims[dim1, 0], lims[dim1, 1], grid_bins)
        vec_dim2 = np.linspace(lims[dim2, 0], lims[dim2, 1], grid_bins)

        energy_image = -np.ones((grid_bins, grid_bins))
        energy_image_specific_neuron = -np.ones((grid_bins, grid_bins))
        energy_per_spike = -np.ones((grid_bins, grid_bins))
        number_of_spikes_per_burst = -np.ones((grid_bins, grid_bins))

        if regression_net is None:
            for i1, v1 in enumerate(vec_dim1):
                for i2, v2 in enumerate(vec_dim2):
                    parameter_set = deepcopy(condition1_norm)

                    parameter_set[0, dim1] = v1
                    parameter_set[0, dim2] = v2
                    prob = np.exp(
                        posterior_MAF_11.log_prob(parameter_set).detach().item() - 100
                    )
                    if prob > lowest_allowed:
                        remaining_patience = patience
                        successful_trace = False
                        seeds = 8607175
                        iter = 0
                        while (not successful_trace) and (remaining_patience > 0):
                            energy_image[i1, i2] = 0.0
                            energy_image_specific_neuron[i1, i2] = 0.0
                            out_target = simulate(
                                deepcopy(parameter_set[0]),
                                seed=seeds + iter,
                            )
                            ss = summary_stats(out_target)
                            if np.invert(np.any(np.isnan(ss))):
                                num_std = np.asarray(
                                    [
                                        0.02,
                                        0.02,
                                        0.02,
                                        0.02,
                                        0.2,
                                        0.2,
                                        0.2,
                                        0.2,
                                        0.2,
                                        0.2,
                                        0.2,
                                        0.2,
                                        0.2,
                                        0.2,
                                        0.2,
                                    ]
                                )

                                if check_if_close_to_obs(
                                    ss,
                                    observation,
                                    num_std=num_std,
                                    stats_std=stats_std,
                                ):
                                    successful_trace = True
                                    total_energy = np.sum(
                                        out_target["energy"][:, 40000:]
                                    )
                                    energy_image[i1, i2] = total_energy

                                    # E / spike", ss[19:22]
                                    # neuron_to_observe = 'PM' or so...
                                    str_to_ind = {"PM": 0, "LP": 1, "PY": 2}

                                    energy_image_specific_neuron[i1, i2] = np.sum(
                                        out_target["energy"][
                                            str_to_ind[neuron_to_observe], 40000:
                                        ]
                                    )

                                    energy_per_spike[i1, i2] = ss[19:22][
                                        str_to_ind[neuron_to_observe]
                                    ]

                                    # number of spikes", ss[31:34]
                                    # number of bursts", ss[22:25]
                                    number_of_spikes_per_burst[i1, i2] = (
                                        ss[31:34][str_to_ind[neuron_to_observe]]
                                        / ss[22:25][str_to_ind[neuron_to_observe]]
                                    )
                            remaining_patience -= 1
                            iter += 1
        else:
            all_parameter_sets = []
            for i1, v1 in enumerate(vec_dim1):
                for i2, v2 in enumerate(vec_dim2):
                    parameter_set = deepcopy(condition1_norm)
                    parameter_set[0, dim1] = v1
                    parameter_set[0, dim2] = v2
                    all_parameter_sets.append(parameter_set)
            sets_tt = torch.cat(all_parameter_sets)
            probs = np.exp(posterior_MAF_11.log_prob(sets_tt).detach())
            valid_sets = sets_tt[probs > lowest_allowed]
            norm_sets = (valid_sets - theta_mean) / theta_std
            net_preds = regression_net.predict(norm_sets)  # .detach()
            # print("net_preds", net_preds)
            net_E = net_preds[:, 0] * x_std + x_mean
            # print(net_E)
            all_preds = -torch.ones(vec_dim1.shape[0] ** 2)
            all_preds[probs > lowest_allowed] = torch.as_tensor(net_E)
            counter = 0
            for i1, v1 in enumerate(vec_dim1):
                for i2, v2 in enumerate(vec_dim2):
                    energy_image_specific_neuron[i1, i2] = all_preds[counter]
                    counter += 1

    # diagonals
    else:
        vec_dim1 = np.linspace(lims[dim1, 0], lims[dim1, 1], grid_bins)

        energy_image = -np.ones(grid_bins)
        energy_image_specific_neuron = -np.ones(grid_bins)
        energy_per_spike = -np.ones(grid_bins)
        number_of_spikes_per_burst = -np.ones(grid_bins)

        if regression_net is None:
            for i1, v1 in enumerate(vec_dim1):
                parameter_set = deepcopy(condition1_norm)

                parameter_set[0, dim1] = v1
                prob = np.exp(posterior_MAF_11.log_prob(parameter_set).detach().item())
                if prob > lowest_allowed:
                    energy_image[i1] = 0.0
                    energy_image_specific_neuron[i1] = 0.0
                    out_target = simulate(deepcopy(parameter_set[0]), seed=8607175)
                    ss = summary_stats(out_target)
                    if np.invert(np.any(np.isnan(ss))):
                        num_std = np.asarray(
                            [
                                0.02,
                                0.02,
                                0.02,
                                0.02,
                                0.2,
                                0.2,
                                0.2,
                                0.2,
                                0.2,
                                0.2,
                                0.2,
                                0.2,
                                0.2,
                                0.2,
                                0.2,
                            ]
                        )

                        if check_if_close_to_obs(
                            ss, observation, num_std=num_std, stats_std=stats_std
                        ):

                            total_energy = np.sum(out_target["energy"][:, 40000:])
                            energy_image[i1] = total_energy

                            str_to_ind = {"PM": 0, "LP": 1, "PY": 2}

                            energy_image_specific_neuron[i1] = np.sum(
                                out_target["energy"][
                                    str_to_ind[neuron_to_observe], 40000:
                                ]
                            )

                            energy_per_spike[i1] = ss[19:22][
                                str_to_ind[neuron_to_observe]
                            ]

                            number_of_spikes_per_burst[i1] = (
                                ss[31:34][str_to_ind[neuron_to_observe]]
                                / ss[22:25][str_to_ind[neuron_to_observe]]
                            )
        else:
            all_parameter_sets = []
            for i1, v1 in enumerate(vec_dim1):
                parameter_set = deepcopy(condition1_norm)
                parameter_set[0, dim1] = v1
                all_parameter_sets.append(parameter_set)
            sets_tt = torch.cat(all_parameter_sets)
            probs = np.exp(posterior_MAF_11.log_prob(sets_tt).detach())
            valid_sets = sets_tt[probs > lowest_allowed]
            norm_sets = (valid_sets - theta_mean) / theta_std
            net_preds = regression_net.predict(norm_sets)  # .detach()
            net_E = net_preds[:, 0] * x_std + x_mean
            all_preds = -torch.ones(vec_dim1.shape[0])
            all_preds[probs > lowest_allowed] = torch.as_tensor(net_E)
            counter = 0
            for i1, v1 in enumerate(vec_dim1):
                energy_image_specific_neuron[i1] = all_preds[counter]
                counter += 1

    return (
        energy_image,
        energy_image_specific_neuron,
        energy_per_spike,
        number_of_spikes_per_burst,
    )
