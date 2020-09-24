import numpy as np
import torch
from sklearn.linear_model import LinearRegression


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


def active_subspace(converged_nn, test_params):
    """
    Returns eigenvalues and eigenvectors of active subspace in ascending order.
    """

    num_samples = 1000

    cum_grad_matrix = torch.zeros((31, 31))
    for test_param in test_params[:num_samples]:
        input_theta = torch.tensor([test_param])
        input_theta.requires_grad = True
        predictions = converged_nn.forward(input_theta)
        loss = predictions.mean()
        loss.backward()
        gradient_input = torch.squeeze(input_theta.grad)
        cum_grad_matrix += torch.ger(gradient_input, gradient_input)
    cum_grad_matrix /= num_samples

    e_vals, e_vecs = torch.symeig(cum_grad_matrix, eigenvectors=True)

    return e_vals, e_vecs
