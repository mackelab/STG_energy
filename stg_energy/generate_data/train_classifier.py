import numpy as np
import pandas as pd
import dill as pickle
import torch
from multiprocessing import Pool

from pyloric import create_prior
from sbi.utils import RestrictionEstimator
from sbi.inference import prepare_for_sbi
from stg_energy.utils import load_all_sims_11deg
from bayes_opt import BayesianOptimization, UtilityFunction
from sbi.user_input.user_input_checks_utils import PytorchReturnTypeWrapper

torch.manual_seed(0)

general_path = "../../results/"
path_to_data = "simulation_data_Tube_MLslurm_cluster/01_simulate_11deg/data/all_"
theta = pd.read_pickle(general_path + path_to_data + "circuit_parameters.pkl")
x = pd.read_pickle(general_path + path_to_data + "simulation_outputs.pkl")

theta = torch.as_tensor(theta.to_numpy(), dtype=torch.float32)
x = torch.as_tensor(x.to_numpy(), dtype=torch.float32)

# Load prior.
prior = create_prior(as_torch_dist=True)
prior = PytorchReturnTypeWrapper(prior)


def compute_false_positives(batch_size, lr, dropout, hidden, blocks):
    torch.manual_seed(0)
    # def compute_false_positives(args):
    # Set up restriction estimator.
    # batch_size, lr, dropout, hidden, blocks = *args
    print("batch_size", batch_size)
    print("lr", lr)
    print("dropout", dropout)
    print("hidden", hidden)
    print("blocks", blocks)

    restriction_estimator = RestrictionEstimator(
        prior=prior,
        hidden_features=int(hidden),
        num_blocks=int(blocks),
        dropout_probability=dropout,
    )
    restriction_estimator.append_simulations(theta, x).train(
        max_num_epochs=250,
        subsample_invalid_sims="auto",
        learning_rate=lr,
        training_batch_size=int(batch_size),
    )

    # Generate `restricted_prior`. Will be the proposal for the next round.
    restricted_prior = restriction_estimator.restrict_prior()

    restricted_prior.tune_rejection_threshold(0.01)
    fp = restricted_prior.print_false_positive_rate()

    print("fraction_false_positives", fp)

    file_dir = "../../results/trained_neural_nets/inference/"
    with open(file_dir + f"optimized_network.pickle", "wb",) as handle:
        pickle.dump(restricted_prior, handle, protocol=4)
    with open(file_dir + f"optimized_network_estimator.pickle", "wb",) as handle:
        pickle.dump(restriction_estimator, handle, protocol=4)
    # with open(file_dir + f"restricted_prior_3million_fp_{fp}.pickle", "wb",) as handle:
    #     pickle.dump(restricted_prior, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return -fp


trained_net = compute_false_positives(
    199, 0.000758956039499681, 0.437553839592111, 80, 5
)

# # Bounded region of parameter space
# hyperparameter_bounds = {
#     "batch_size": (100, 200),
#     "lr": (1e-4, 1e-3),
#     "dropout": (0.0, 0.5),
#     "hidden": (80, 150),
#     "blocks": (5, 10),
# }
#
# # Perform Bayesian optimization on the hyperparameters.
# optimizer = BayesianOptimization(
#     f=compute_false_positives, pbounds=hyperparameter_bounds, random_state=0,
# )
# optimizer.maximize(
#     init_points=15, n_iter=8,
# )
# print("Optimal parameters:  ", optimizer.max)
