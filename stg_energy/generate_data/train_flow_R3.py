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

print("theta.shape", theta.shape)
print("x.shape", x.shape)

# Load prior.
prior = create_prior(as_torch_dist=True)
prior = PytorchReturnTypeWrapper(prior)


def train_flow(batch_size, lr, hidden, layers):
    # Run SNPE.
    print("batch_size, lr, hidden, layers: ", batch_size, lr, hidden, layers)
    estimator = posterior_nn(
        "nsf", hidden_features=int(hidden), num_transforms=int(layers)
    )
    inference = SNPE(prior, density_estimator=estimator)
    density_estimator = inference.append_simulations(theta, x).train(
        training_batch_size=int(batch_size), learning_rate=lr, max_num_epochs=250
    )
    posterior = inference.build_posterior()
    best_log_prob = inference._best_val_log_prob
    print("log_prob:  ", best_log_prob)

    # Save data.
    with open(
        f"../../results/trained_neural_nets/inference/flow_{best_log_prob}.pickle", "wb"
    ) as handle:
        pickle.dump(posterior, handle, protocol=4)
    with open(
        f"../../results/trained_neural_nets/inference/inference_snpe_{best_log_prob}.pickle",
        "wb",
    ) as handle:
        pickle.dump(inference, handle, protocol=4)

    return best_log_prob


# Bounded region of parameter space
hyperparameter_bounds = {
    "batch_size": (100, 250),
    "lr": (1e-4, 1e-3),
    "hidden": (50, 150),
    "layers": (5, 10),
}

# Perform Bayesian optimization on the hyperparameters.
optimizer = BayesianOptimization(
    f=train_flow, pbounds=hyperparameter_bounds, random_state=0,
)
optimizer.maximize(
    init_points=5, n_iter=1,
)
print("Optimal parameters:  ", optimizer.max)
