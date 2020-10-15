import numpy as np
from copy import deepcopy

# from find_pyloric import check_ss_to_obs_diff_stds


def select_ss_close_to_obs(
    params,
    stats,
    seeds,
    observation,
    num_std,
    stats_std=None,
    new_burst_position_in_ss=False,
):
    """
    Returns those summstats that are within num_std standard deviations
        from the observation and no bursts and no plateaus.
    :param data: summstats
    :param observation:
    :param num_std:
    :param new_burst_position_in_ss: if False, then the number of bursts is assumed to
        the last ss. This was before I added e.g. voltage moments. Later, the number of
        bursts is in position 22:25
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

    data_trunc = np.asarray(stats[:, :15])
    observation = observation[:15]

    # check distance to observation
    diff_to_obs = np.abs(data_trunc - observation) / stats_std
    good_data = stats[np.all(diff_to_obs < num_std, axis=1)]
    good_params = params[np.all(diff_to_obs < num_std, axis=1)]
    good_seeds = seeds[np.all(diff_to_obs < num_std, axis=1)]

    # TODO: UNCOMMENT
    # # check if more than 7 bursts
    # if new_burst_position_in_ss:
    #     backup_stats = deepcopy(good_data)
    #     good_data = good_data[np.all(backup_stats[:, 22:25] > 7.5, axis=1)]
    #     good_params = good_params[np.all(backup_stats[:, 22:25] > 7.5, axis=1)]
    #     good_seeds = good_seeds[np.all(backup_stats[:, 22:25] > 7.5, axis=1)]
    # else:
    #     backup_stats = deepcopy(good_data)
    #     good_data = good_data[np.all(backup_stats[:, -3:] > 7.5, axis=1)]
    #     good_params = good_params[np.all(backup_stats[:, -3:] > 7.5, axis=1)]
    #     good_seeds = good_seeds[np.all(backup_stats[:, -3:] > 7.5, axis=1)]

    # check if no plateaus
    backup_stats = deepcopy(good_data)
    good_data = good_data[np.all(backup_stats[:, 15:18] == 2.5, axis=1)]
    good_params = good_params[np.all(backup_stats[:, 15:18] == 2.5, axis=1)]
    good_seeds = good_seeds[np.all(backup_stats[:, 15:18] == 2.5, axis=1)]

    # check if no NaN
    backup_stats = deepcopy(good_data)
    good_data = good_data[np.invert(np.any(np.isnan(backup_stats), axis=1))]
    good_params = good_params[np.invert(np.any(np.isnan(backup_stats), axis=1))]
    good_seeds = good_seeds[np.invert(np.any(np.isnan(backup_stats), axis=1))]

    return good_params, good_data, good_seeds


# def check_if_close_to_obs(stats, observation, num_std, stats_std=None):
#     """
#     Returns those summstats that are within num_std standard deviations
#         from the observation and no bursts and no plateaus.
#     :param data: summstats
#     :param observation:
#     :param num_std:
#     :return:
#     """
#     if stats_std is None:
#         #setting to experimental stds from prinz paper
#
#         stats_std = np.asarray(
#             [279, 133, 113, 150, 109, 60, 169, 216, 0.040, 0.059, 0.054, 0.065, 0.034,
#              0.054, 0.060])
#
#     data_trunc = np.asarray(stats[:15])
#     observation = observation[:15]
#
#     good_sim = True
#
#     # check distance to observation
#     diff_to_obs = np.abs(data_trunc - observation) / stats_std[:15]
#     if not np.all(diff_to_obs < num_std):
#         good_sim = False
#
#     # check if more than 7 bursts
#     backup_stats = deepcopy(stats)
#     if not np.all(backup_stats[22:24] > 7.5):
#         good_sim = False
#
#     # check if no plateaus
#     backup_stats = deepcopy(stats)
#     if not np.all(backup_stats[15:18] == 2.5):
#         good_sim = False
#
#     return good_sim
#
#
# def extract_min_prob(posterior_MAF_11, condition1_norm, grid_bins, dim1, dim2, lims, mode='posterior_prob'):
#     """
#     Used for plotting the energy in the plane of the conditional probability
#
#     :param posterior_MAF_11: posterior
#     :param condition1_norm: the value (usually posterior sample) we are conditioning on
#     :param grid_bins: number of bins / resolution
#     :param dim1: int
#     :param dim2: int
#     :param mode: string. Distinguishes two computations in which the min_prob is computed
#     :return: minimum required probability to be simulated
#     """
#     if mode == "posterior_prob":
#         prob = np.exp(posterior_MAF_11.log_prob(condition1_norm).detach().item())
#         min_prob = 0.35 * prob
#     else:
#         all_probs = []
#
#         vec_dim1 = np.linspace(lims[dim1, 0], lims[dim1, 1], grid_bins)
#         vec_dim2 = np.linspace(lims[dim2, 0], lims[dim2, 1], grid_bins)
#
#         for i1, v1 in enumerate(vec_dim1):
#             for i2, v2 in enumerate(vec_dim2):
#                 parameter_set = deepcopy(condition1_norm)
#
#                 parameter_set[0, dim1] = v1
#                 parameter_set[0, dim2] = v2
#                 prob = np.exp(posterior_MAF_11.log_prob(parameter_set).detach().numpy())
#                 all_probs.append(prob[0])
#         all_probs = np.asarray(all_probs)
#
#         diff_of_min_max = np.max(all_probs) - np.min(all_probs)
#         min_prob = np.min(all_probs) + 0.8 * diff_of_min_max
#
#     return min_prob
#
#
# def energy_of_conditional(posterior_MAF_11, pyloric_sim, energy_calc, condition1_norm,
#                           grid_bins, min_prob, dim1, dim2, lims, stats_std,
#                           neuron_to_observe='AB/PD'):
#     """
#     Builds the image that contains the energy of each parameter value in conditional plane.
#
#     :param posterior_MAF_11: posterior
#     :param pyloric_sim: simulator object
#     :param pyloric_sim: summstats object
#     :param obs: true observation
#     :param condition1_norm: the value (usually posterior sample) we are conditioning on
#     :param grid_bins: number of bins / resolution
#     :param min_prob: float, minimum required probability to be simulated
#     :param dim1: int
#     :param dim2: int
#     :param neuron_to_observe: string, If mode == 'spike', then what neuron should we compute the energy per spike of?
#             either of the following: 'PM', 'LP', 'PY'
#     :param mode: string: 'spike' computes energy per spike. 'total' computes total energy
#
#     :return: image
#     """
#     lowest_allowed = min_prob
#
#     npz = np.load(
#         '../../../results/experimental/summstats/845_082/190807_summstats_prep845_082_0044.npz')
#     observation = npz['summ_stats']
#
#     # off-diagonals
#     if dim1 != dim2:
#         vec_dim1 = np.linspace(lims[dim1, 0], lims[dim1, 1], grid_bins)
#         vec_dim2 = np.linspace(lims[dim2, 0], lims[dim2, 1], grid_bins)
#
#         energy_image = -np.ones((grid_bins, grid_bins))
#         energy_per_spike = -np.ones((grid_bins, grid_bins))
#         number_of_spikes_per_burst = -np.ones((grid_bins, grid_bins))
#         spike_width = np.zeros((grid_bins, grid_bins))
#
#         for i1, v1 in enumerate(vec_dim1):
#             for i2, v2 in enumerate(vec_dim2):
#                 parameter_set = deepcopy(condition1_norm)
#
#                 parameter_set[0,dim1] = v1
#                 parameter_set[0,dim2] = v2
#                 prob = np.exp(posterior_MAF_11.log_prob(parameter_set).detach().item())
#                 if prob > lowest_allowed:
#                     energy_image[i1, i2] = 0.0
#                     out_target = pyloric_sim[0].gen_single(deepcopy(parameter_set[0]), seed_sim=True, to_seed=8607175)
#                     ss = energy_calc.calc([out_target])[0]
#                     ss_dict = energy_calc.calc_dict([out_target])[0]
#                     if np.invert(np.any(np.isnan(ss))):
#                         num_std = np.asarray(
#                             [0.02, 0.02, 0.02, 0.02, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
#                              0.2, 0.2, 0.2]) * 2
#
#                         if check_if_close_to_obs(ss, observation, num_std=num_std, stats_std=stats_std):
#                                 total_energy = np.sum(out_target['energy'][:, 40000:])
#                                 energy_image[i1, i2] = total_energy
#
#                                 energy_per_spike[i1, i2] = ss_dict['energy'][
#                                     neuron_to_observe]
#
#                                 spike_width[i1, i2] = np.mean(
#                                     ss_dict['rebound_times'][neuron_to_observe])
#
#                                 number_of_spikes_per_burst[i1, i2] = \
#                                     ss_dict['num_spikes'][neuron_to_observe] / \
#                                     ss_dict['num_bursts'][neuron_to_observe]
#     # diagonals
#     else:
#         vec_dim1 = np.linspace(lims[dim1, 0], lims[dim1, 1], grid_bins)
#
#         energy_image = -np.ones(grid_bins)
#         energy_per_spike = -np.ones(grid_bins)
#         number_of_spikes_per_burst = -np.ones(grid_bins)
#         spike_width = np.zeros(grid_bins)
#         for i1, v1 in enumerate(vec_dim1):
#             parameter_set = deepcopy(condition1_norm)
#
#             parameter_set[0, dim1] = v1
#             prob = np.exp(posterior_MAF_11.log_prob(parameter_set).detach().item())
#             if prob > lowest_allowed:
#                 energy_image[i1] = 0.0
#                 out_target = pyloric_sim[0].gen_single(deepcopy(parameter_set[0]),
#                                                        seed_sim=True,
#                                                        to_seed=8607175)
#                 ss = energy_calc.calc([out_target])[0]
#                 ss_dict = energy_calc.calc_dict([out_target])[0]
#                 if np.invert(np.any(np.isnan(ss))):
#                     num_std = np.asarray(
#                         [0.02, 0.02, 0.02, 0.02, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
#                          0.2,
#                          0.2, 0.2, 0.2]) * 2
#
#                     if check_if_close_to_obs(ss, observation, num_std=num_std,
#                                              stats_std=stats_std):
#
#                         total_energy = np.sum(out_target['energy'][:, 40000:])
#                         energy_image[i1] = total_energy
#
#                         energy_per_spike[i1] = ss_dict['energy'][
#                             neuron_to_observe]
#
#                         spike_width[i1] = np.mean(
#                             ss_dict['rebound_times'][neuron_to_observe])
#
#                         number_of_spikes_per_burst[i1] = \
#                             ss_dict['num_spikes'][neuron_to_observe] / \
#                             ss_dict['num_bursts'][neuron_to_observe]
#
#     return energy_image, energy_per_spike, number_of_spikes_per_burst, spike_width
