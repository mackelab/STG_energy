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
    theta_mean=None,
    theta_std=None,
    x_mean=None,
    x_std=None,
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
            theta_mean=theta_mean,
            theta_std=theta_std,
            x_mean=x_mean,
            x_std=x_std,
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
    theta_mean=None,
    theta_std=None,
    x_mean=None,
    x_std=None,
    min_prob=None,
    net1=None,
    net2=None,
    net3=None,
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
                    theta_mean=theta_mean,
                    theta_std=theta_std,
                    x_mean=x_mean,
                    x_std=x_std,
                    min_prob=min_prob,
                )
    else:
        for p, n in zip(pairs, neuron_to_observe):
            dim1 = p[0]
            dim2 = p[1]
            nets = {"PM": net1, "LP": net2, "PY": net3}
            net_ = nets[n]
            if regression_net is None:
                used_net = net_
            else:
                used_net = regression_net
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
                regression_net=used_net,  # net_
                theta_mean=theta_mean,
                theta_std=theta_std,
                x_mean=x_mean,
                x_std=x_std,
                min_prob=min_prob,
            )

    with open(
        f"../../../results/conditional_correlation_energy/211007_{store_as}_all_stored_data_from_energy_all_conditional_correlations_nn.pickle",
        "wb",
    ) as handle:
        pickle.dump(
            all_conditional_correlations, handle, protocol=pickle.HIGHEST_PROTOCOL
        )

    with open(
        f"../../../results/conditional_correlation_energy/211007_{store_as}_all_stored_data_from_energy_all_energy_images_nn.pickle",
        "wb",
    ) as handle:
        pickle.dump(all_energy_images, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(
        f"../../../results/conditional_correlation_energy/211007_{store_as}_all_stored_data_from_all_energy_specific_nn.pickle",
        "wb",
    ) as handle:
        pickle.dump(all_energy_specific, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(
        f"../../../results/conditional_correlation_energy/211007_{store_as}_all_stored_data_from_energy_all_energy_per_spike_nn.pickle",
        "wb",
    ) as handle:
        pickle.dump(all_energy_per_spike, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(
        f"../../../results/conditional_correlation_energy/211007_{store_as}_all_stored_data_from_energy_all_num_spikes_per_burst_nn.pickle",
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
