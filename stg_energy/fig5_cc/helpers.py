import numpy as np
from conditional_density import eval_conditional_density
import sys; sys.path.append('../../../utils')
import energy as ue
import pickle

def extract_the_data(dim1, dim2, posterior, condition1_norm, lims_unnorm, grid_bins,
                     pyloric_sim, energy_calc, stats_std, neuron_to_observe,
                     all_conditional_correlations, all_energy_images,
                     all_energy_per_spike, all_num_spikes_per_burst, all_spike_width):
    if dim1 >= dim2:
        print("===== New pair =====")
        p_vector = eval_conditional_density(posterior, condition1_norm, lims_unnorm,
                                            dim1, dim2, resolution=grid_bins,
                                            log=False)
        p_vector = p_vector / np.max(p_vector)  # just to scale it to 1

        # get the minimum requried probability to be simulated
        min_prob = ue.extract_min_prob(posterior, condition1_norm, grid_bins, dim1,
                                       dim2, lims_unnorm, mode='posterior_prob')

        # get the energies in the conditional plane
        energy_image, energy_per_spike, num_spikes_per_burst, spike_width = ue.energy_of_conditional(
            posterior, pyloric_sim, energy_calc, condition1_norm, grid_bins,
            min_prob,
            dim1, dim2, lims_unnorm, stats_std=stats_std,
            neuron_to_observe=neuron_to_observe)

        # small adjustments for plotting
        energy_image[energy_image == -1.0] = -np.max(energy_image)

        all_conditional_correlations.append(p_vector)
        all_energy_images.append(energy_image)
        all_energy_per_spike.append(energy_per_spike)
        all_num_spikes_per_burst.append(num_spikes_per_burst)
        all_spike_width.append(spike_width)
    return all_conditional_correlations, all_energy_images, all_energy_per_spike, all_num_spikes_per_burst, all_spike_width


def generate_and_store_data(
        neuron1,
        neuron2,
        neuron_to_observe,
        grid_bins,
        posterior,
        condition1_norm,
        lims_unnorm,
        pyloric_sim,
        energy_calc,
        stats_std,
        pairs=None,
        store_as=None):
    all_energy_images = []
    all_conditional_correlations = []
    all_energy_per_spike = []
    all_num_spikes_per_burst = []
    all_spike_width = []

    if store_as is None:
        store_as = neuron_to_observe

    if pairs is None:
        for dim1 in neuron1:
            for dim2 in neuron2:
                all_conditional_correlations,\
                all_energy_images, \
                all_energy_per_spike, \
                all_num_spikes_per_burst, \
                all_spike_width = extract_the_data(dim1, dim2, posterior, condition1_norm, lims_unnorm,
                                 grid_bins,
                                 pyloric_sim, energy_calc, stats_std, neuron_to_observe,
                                 all_conditional_correlations, all_energy_images,
                                 all_energy_per_spike, all_num_spikes_per_burst,
                                 all_spike_width)
    else:
        for p, n in zip(pairs, neuron_to_observe):
            dim1 = p[0]
            dim2 = p[1]
            all_conditional_correlations, \
            all_energy_images, \
            all_energy_per_spike, \
            all_num_spikes_per_burst, \
            all_spike_width = extract_the_data(dim1, dim2, posterior, condition1_norm,
                                               lims_unnorm,
                                               grid_bins,
                                               pyloric_sim, energy_calc, stats_std,
                                               n,
                                               all_conditional_correlations,
                                               all_energy_images,
                                               all_energy_per_spike,
                                               all_num_spikes_per_burst,
                                               all_spike_width)

    with open(
            f'../../results/conditional_correlation_energy/200703_{store_as}_all_stored_data_from_energy_all_conditional_correlations.pickle',
            'wb') as handle:
        pickle.dump(all_conditional_correlations, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)
    with open(
            f'../../results/conditional_correlation_energy/200703_{store_as}_all_stored_data_from_energy_all_energy_images.pickle',
            'wb') as handle:
        pickle.dump(all_energy_images, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(
            f'../../results/conditional_correlation_energy/200703_{store_as}_all_stored_data_from_energy_all_energy_per_spike.pickle',
            'wb') as handle:
        pickle.dump(all_energy_per_spike, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(
            f'../../results/conditional_correlation_energy/200703_{store_as}_all_stored_data_from_energy_all_num_spikes_per_burst.pickle',
            'wb') as handle:
        pickle.dump(all_num_spikes_per_burst, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(
            f'../../results/conditional_correlation_energy/200703_{store_as}_all_stored_data_from_energy_all_spike_width.pickle',
            'wb') as handle:
        pickle.dump(all_spike_width, handle, protocol=pickle.HIGHEST_PROTOCOL)