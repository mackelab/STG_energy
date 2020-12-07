import numpy as np
import pandas as pd
import dill as pickle
import torch

from pyloric import create_prior
from sbi.inference import SNPE
from sbi.utils import posterior_nn
from sbi.user_input.user_input_checks_utils import PytorchReturnTypeWrapper
from bayes_opt import BayesianOptimization, UtilityFunction


torch.manual_seed(0)

general_path = "../../results/"
path_to_data = "simulation_data_Tube_MLslurm_cluster/all_"
theta = pd.read_pickle(general_path + path_to_data + "circuit_parameters_train.pkl")
x = pd.read_pickle(general_path + path_to_data + "simulation_outputs_train.pkl")

theta = torch.as_tensor(theta.to_numpy(), dtype=torch.float32)
x = torch.as_tensor(x.to_numpy(), dtype=torch.float32)

# Load prior.
prior = create_prior(as_torch_dist=True)
prior = PytorchReturnTypeWrapper(prior)


def train_flow(batch_size, lr, hidden, layers):
    # Run SNPE.
    estimator = posterior_nn("nsf", hidden_features=hidden, num_transforms=layers)
    inference = SNPE(prior, density_estimator=estimator)
    density_estimator = inference.append_simulations(theta, x).train(
        training_batch_size=batch_size, learning_rate=lr
    )
    posterior = inference.build_posterior()
    best_log_prob = inference._best_val_log_prob
    print("log_prob:  ", best_log_prob)

    # Save data.
    with open(
        f"../../results/trained_neural_nets/inference/flow_{best_log_prob}.pickle", "wb"
    ) as handle:
        pickle.dump(handle, posterior, protocol=4)
    with open(
        f"../../results/trained_neural_nets/inference/inference_{best_log_prob}.pickle",
        "wb",
    ) as handle:
        pickle.dump(handle, inference_object, protocol=4)


# Bounded region of parameter space
hyperparameter_bounds = {
    "batch_size": (100, 250),
    "lr": (1e-4, 1e-3),
    "hidden": (50, 150),
    "layers": (5, 10),
}

# Perform Bayesian optimization on the hyperparameters.
optimizer = BayesianOptimization(
    f=compute_false_positives, pbounds=hyperparameter_bounds, random_state=0,
)
optimizer.maximize(
    init_points=15, n_iter=8,
)
print("Optimal parameters:  ", optimizer.max)
