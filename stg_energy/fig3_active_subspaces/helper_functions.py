import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from copy import deepcopy


def nth_argmin(all_projected, n: int):
    """
    Returns the index of the n-th smallest element in an array.

    :param all_projected: The array to be searched.
    :param n: n -> The how many-eth smallest to pick.
    """
    return _index_of_nth_min_or_max(all_projected, n, "min")


def nth_argmax(all_projected, n: int):
    """
    Returns the index of the n-th largest element in an array.

    :param all_projected: The array to be searched.
    :param n: n -> The how many-eth largest to pick.
    """
    return _index_of_nth_min_or_max(all_projected, n, "max")


def _index_of_nth_min_or_max(all_projected, num_to_ignore: int, min_or_max: str) -> int:
    """
    Returns the index of the n-th largest or smallest element in an array.

    :param all_projected: The array to be searched.
    :param num_to_ignore: n -> The how many-eth largest or smallest to pick.
    :param min_or_max: Whether to search for the smallest ('min') or largest ('max').
    """
    fn = torch.argmin if min_or_max == "min" else torch.argmax

    projected = deepcopy(all_projected)
    bounced = []
    for _ in range(num_to_ignore):
        argmin_energy_per_spike = fn(projected)
        projected = np.delete(projected, argmin_energy_per_spike)
        bounced.append(argmin_energy_per_spike)

    index_of_minimal = fn(projected)

    num_lower = np.asarray(bounced) < index_of_minimal.numpy()
    num_correction_indices = np.sum(num_lower)

    return index_of_minimal + num_correction_indices
