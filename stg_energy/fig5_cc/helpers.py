import numpy as np
from stg_energy.fig5_cc.conditional_density import eval_conditional_density
import sys

import stg_energy.fig5_cc.energy as ue
import pickle
from copy import deepcopy


def extract_the_data(
    dim1,
    dim2,
    posterior,
    condition1_norm,
    lims_unnorm,
    grid_bins,
    stats_std,
    neuron_to_observe,
    all_conditional_correlations,
    all_energy_images,
    all_energy_specific,
    all_energy_per_spike,
    all_num_spikes_per_burst,
    regression_net=None,
    neural_net_zscore_mean=None,
    neural_net_zscore_std=None,
    neural_net_zscore_mean_energy=None,
    neural_net_zscore_std_energy=None,
    min_prob=None,
):
    if dim1 >= dim2:
        p_vector = eval_conditional_density(
            posterior,
            condition1_norm,
            lims_unnorm,
            dim1,
            dim2,
            resolution=grid_bins,
            log=False,
        )
        p_vector = p_vector / np.max(p_vector)  # just to scale it to 1

        if min_prob is None:
            # get the minimum requried probability to be simulated
            threshold_for_simulating = (
                ue.extract_min_prob(
                    posterior,
                    condition1_norm,
                    grid_bins,
                    dim1,
                    dim2,
                    lims_unnorm,
                    mode="posterior_prob",
                )
                / 1.5
            )

        # get the energies in the conditional plane
        (
            energy_image,
            energy_specific,
            energy_per_spike,
            num_spikes,
        ) = ue.energy_of_conditional(
            posterior,
            condition1_norm,
            grid_bins,
            min_prob,
            dim1,
            dim2,
            lims_unnorm,
            stats_std=stats_std,
            neuron_to_observe=neuron_to_observe,
            regression_net=regression_net,
            neural_net_zscore_mean=neural_net_zscore_mean,
            neural_net_zscore_std=neural_net_zscore_std,
            neural_net_zscore_mean_energy=neural_net_zscore_mean_energy,
            neural_net_zscore_std_energy=neural_net_zscore_std_energy,
        )

        all_conditional_correlations.append(p_vector)
        all_energy_images.append(energy_image)
        all_energy_specific.append(energy_specific)
        all_energy_per_spike.append(energy_per_spike)
        all_num_spikes_per_burst.append(num_spikes)
    return (
        all_conditional_correlations,
        all_energy_images,
        all_energy_specific,
        all_energy_per_spike,
        all_num_spikes_per_burst,
    )


def generate_and_store_data(
    neuron1,
    neuron2,
    neuron_to_observe,
    grid_bins,
    posterior,
    condition1_norm,
    lims_unnorm,
    stats_std,
    pairs=None,
    store_as=None,
    regression_net=None,
    neural_net_zscore_mean=None,
    neural_net_zscore_std=None,
    neural_net_zscore_mean_energy=None,
    neural_net_zscore_std_energy=None,
    min_prob=None,
    net1=None,
    mean1=None,
    std1=None,
    net2=None,
    mean2=None,
    std2=None,
    net3=None,
    mean3=None,
    std3=None,
):
    all_conditional_correlations = []
    all_energy_images = []
    all_energy_specific = []
    all_energy_per_spike = []
    all_num_spikes_per_burst = []

    if store_as is None:
        store_as = neuron_to_observe

    if pairs is None:
        for dim1 in neuron1:
            for dim2 in neuron2:
                (
                    all_conditional_correlations,
                    all_energy_images,
                    all_energy_specific,
                    all_energy_per_spike,
                    all_num_spikes_per_burst,
                ) = extract_the_data(
                    dim1,
                    dim2,
                    posterior,
                    condition1_norm,
                    lims_unnorm,
                    grid_bins,
                    stats_std,
                    neuron_to_observe,
                    all_conditional_correlations,
                    all_energy_images,
                    all_energy_specific,
                    all_energy_per_spike,
                    all_num_spikes_per_burst,
                    regression_net=regression_net,
                    neural_net_zscore_mean=neural_net_zscore_mean,
                    neural_net_zscore_std=neural_net_zscore_std,
                    neural_net_zscore_mean_energy=neural_net_zscore_mean_energy,
                    neural_net_zscore_std_energy=neural_net_zscore_std_energy,
                    min_prob=min_prob,
                )
    else:
        for p, n in zip(pairs, neuron_to_observe):
            dim1 = p[0]
            dim2 = p[1]
            nets = {"PM": net1, "LP": net2, "PY": net3}
            means = {"PM": mean1, "LP": mean2, "PY": mean3}
            stds = {"PM": std1, "LP": std2, "PY": std3}
            net_ = nets[n]
            mean_ = means[n]
            std_ = stds[n]
            (
                all_conditional_correlations,
                all_energy_images,
                all_energy_specific,
                all_energy_per_spike,
                all_num_spikes_per_burst,
            ) = extract_the_data(
                dim1,
                dim2,
                posterior,
                condition1_norm,
                lims_unnorm,
                grid_bins,
                stats_std,
                n,
                all_conditional_correlations,
                all_energy_images,
                all_energy_specific,
                all_energy_per_spike,
                all_num_spikes_per_burst,
                regression_net=net_,
                neural_net_zscore_mean=neural_net_zscore_mean,
                neural_net_zscore_std=neural_net_zscore_std,
                neural_net_zscore_mean_energy=mean_,
                neural_net_zscore_std_energy=std_,
                min_prob=min_prob,
            )

    with open(
        f"../../results/conditional_correlation_energy/201007_{store_as}_all_stored_data_from_energy_all_conditional_correlations_nn.pickle",
        "wb",
    ) as handle:
        pickle.dump(
            all_conditional_correlations, handle, protocol=pickle.HIGHEST_PROTOCOL
        )

    with open(
        f"../../results/conditional_correlation_energy/201007_{store_as}_all_stored_data_from_energy_all_energy_images_nn.pickle",
        "wb",
    ) as handle:
        pickle.dump(all_energy_images, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(
        f"../../results/conditional_correlation_energy/201007_{store_as}_all_stored_data_from_all_energy_specific_nn.pickle",
        "wb",
    ) as handle:
        pickle.dump(all_energy_specific, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(
        f"../../results/conditional_correlation_energy/201007_{store_as}_all_stored_data_from_energy_all_energy_per_spike_nn.pickle",
        "wb",
    ) as handle:
        pickle.dump(all_energy_per_spike, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(
        f"../../results/conditional_correlation_energy/201007_{store_as}_all_stored_data_from_energy_all_num_spikes_per_burst_nn.pickle",
        "wb",
    ) as handle:
        pickle.dump(all_num_spikes_per_burst, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return (
        all_conditional_correlations,
        all_energy_images,
        all_energy_specific,
        all_energy_per_spike,
        all_num_spikes_per_burst,
    )
