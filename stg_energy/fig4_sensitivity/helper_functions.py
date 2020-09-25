import numpy as np
import torch
from sklearn.linear_model import LinearRegression


def prepare_data(all_energies_per_spike, numSpikesPerBurst, num_test):
    """
    Normalizes data for neural net and cuts away a test set.
    """
    test_energies_per_spike = all_energies_per_spike[:num_test]
    train_energies_per_spike = all_energies_per_spike[num_test:]

    mean_energies_per_spike = np.mean(train_energies_per_spike, axis=0)
    std_energies_per_spike = np.std(train_energies_per_spike, axis=0)

    train_energies_per_spike_norm = (
        train_energies_per_spike - mean_energies_per_spike
    ) / std_energies_per_spike
    train_energies_per_spike_norm = train_energies_per_spike_norm[None, :].T

    test_energies_per_spike_norm = (
        test_energies_per_spike - mean_energies_per_spike
    ) / std_energies_per_spike
    test_energies_per_spike_norm = test_energies_per_spike_norm[None, :].T

    test_num_spikes = numSpikesPerBurst[:num_test]
    train_num_spikes = numSpikesPerBurst[num_test:]
    mean_num_spikes = np.mean(train_num_spikes, axis=0)
    std_num_spikes = np.std(train_num_spikes, axis=0)

    train_num_spikes_norm = (train_num_spikes - mean_num_spikes) / std_num_spikes
    train_num_spikes_norm = train_num_spikes_norm[None, :].T

    test_num_spikes_norm = (test_num_spikes - mean_num_spikes) / std_num_spikes
    test_num_spikes_norm = test_num_spikes_norm[None, :].T

    return (
        mean_energies_per_spike,
        std_energies_per_spike,
        train_energies_per_spike_norm,
        test_energies_per_spike_norm,
        mean_num_spikes,
        std_num_spikes,
        train_num_spikes_norm,
        test_num_spikes_norm,
    )


def regression_plot(
    mean_energies_per_spike,
    std_energies_per_spike,
    train_energies_per_spike_norm,
    test_energies_per_spike_norm,
    mean_num_spikes,
    std_num_spikes,
    train_num_spikes_norm,
    test_num_spikes_norm,
):
    """
    Linear regression from number of spikes onto energy per spike.
    """

    lin_reg = LinearRegression()
    lin_reg.fit(train_num_spikes_norm, train_energies_per_spike_norm)

    x_vals = np.linspace(-3, 3, 100)[None,].T
    y_regs = lin_reg.predict(x_vals)

    unnorm_x_vals = x_vals * std_num_spikes + mean_num_spikes
    unnorm_y_regs = y_regs * std_energies_per_spike + mean_energies_per_spike

    unnorm_x = test_num_spikes_norm[:500] * std_num_spikes + mean_num_spikes
    unnorm_y = (
        test_energies_per_spike_norm[:500] * std_energies_per_spike
        + mean_energies_per_spike
    )

    return unnorm_x_vals, unnorm_y_regs, unnorm_x, unnorm_y


def get_gradient(converged_nn, test_params):
    """
    Get the average gradient of converged_nn at the parameter test_params.
    """

    num_samples = 100

    cum_grad = torch.zeros((1, 31))
    for test_param in test_params[:num_samples]:
        input_theta = torch.tensor([test_param])
        input_theta.requires_grad = True
        predictions = converged_nn.forward(input_theta)
        loss = predictions.mean()
        loss.backward()
        gradient_input = input_theta.grad
        cum_grad += gradient_input
    cum_grad /= num_samples
    cum_grad = cum_grad.numpy()

    return cum_grad
