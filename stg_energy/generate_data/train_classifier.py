import numpy as np
import pandas as pd
import pickle
import torch

from pyloric import create_prior
from sbi.utils import RejectionEstimator
from stg_energy.utils import load_all_sims_11deg

torch.manual_seed(0)

theta, x, _ = load_all_sims_11deg()
theta = torch.as_tensor(theta.to_numpy(), dtype=torch.float32)
x = torch.as_tensor(x.to_numpy(), dtype=torch.float32)

# Load prior.
prior = create_prior()

# Set up restriction estimator.
rejection_estimator = RejectionEstimator(prior=prior)
rejection_estimator.append_simulations(theta, x).train(subsample_bad_sims="auto")

# Generate `restricted_prior`. Will be the proposal for the next round of simulations.
restricted_prior = rejection_estimator.restrict_prior()
restricted_prior.print_false_positive_rate()

with open(
    "../../results/trained_neural_nets/inference/restricted_prior.pickle", "wb"
) as handle:
    pickle.dump(restricted_prior, handle, protocal=pickle.HIGHEST_PROTOCOL)
