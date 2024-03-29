import numpy as np
import pandas as pd
import dill as pickle
import torch

from pyloric import create_prior
from sbi.inference import SNPE
from sbi.utils import posterior_nn, BoxUniform
from sbi.user_input.user_input_checks_utils import PytorchReturnTypeWrapper
import time


def train_flow(hyperparams):

    batch_size = hyperparams[0].item()
    lr = hyperparams[1].item()
    hidden = hyperparams[2].item()
    layers = hyperparams[3].item()
    seed = torch.randint(1000000, (1,))
    torch.manual_seed(seed)

    general_path = "../../results/"
    path_to_data = "simulation_data_Tube_MLslurm_cluster/simulate_11deg_R3_predictives_at_27deg_notau_078/data/valid_"
    theta = pd.read_pickle(general_path + path_to_data + "circuit_parameters.pkl")
    x = pd.read_pickle(general_path + path_to_data + "simulation_outputs.pkl")

    theta = torch.as_tensor(theta.to_numpy(), dtype=torch.float32)
    x = torch.as_tensor(x.to_numpy(), dtype=torch.float32)

    x = x[:, :18]
    print("x[0]", x[0])

    print("theta.shape", theta.shape)
    print("x.shape", x.shape)

    # Load prior.
    prior = create_prior(
        customization={
            "Q10_gbar_mem": [True, True, True, True, True, True, True, True],
            "Q10_gbar_syn": [True, True],
            "Q10_tau_m": [False],
            "Q10_tau_h": [False],
            "Q10_tau_CaBuff": [False],
            "Q10_tau_syn": [False, False],
        },
        as_torch_dist=True,
    )
    prior = PytorchReturnTypeWrapper(prior)

    print("1")

    # Run SNPE.
    print("batch_size, lr, hidden, layers: ", batch_size, lr, hidden, layers)
    estimator = posterior_nn(
        "nsf", hidden_features=int(hidden), num_transforms=int(layers)
    )

    print("2")

    inference = SNPE(prior, density_estimator=estimator)
    print("3")
    start_time = time.time()
    density_estimator = inference.append_simulations(theta, x).train(
        training_batch_size=int(batch_size), learning_rate=lr, max_num_epochs=250
    )
    print("training time", time.time() - start_time)
    posterior = inference.build_posterior()
    best_log_prob = inference._best_val_log_prob
    print(
        "batch_size, lr, hidden, layers,  log_prob:  ",
        batch_size,
        lr,
        hidden,
        layers,
        best_log_prob,
    )

    print("6")

    # Save data.
    with open(
        f"../../results/trained_neural_nets/inference/temp27_flow_notau_078_{best_log_prob}_{seed}.pickle",
        "wb",
    ) as handle:
        pickle.dump(posterior, handle, protocol=4)
    with open(
        f"../../results/trained_neural_nets/inference/temp27_inference_snpe_notau_078_{best_log_prob}_{seed}.pickle",
        "wb",
    ) as handle:
        pickle.dump(inference, handle, protocol=4)
    with open(
        f"../../results/trained_neural_nets/inference/temp27_hyperparams_notau_078_{best_log_prob}_{seed}.pickle",
        "wb",
    ) as handle:
        pickle.dump(hyperparams, handle, protocol=4)

    print("7")

    return best_log_prob


print("starting script")

global_seed = int((time.time() % 1) * 1e7)
torch.manual_seed(global_seed)

print("starting script2")

hyperparams = torch.tensor([[200, 1e-4, 200, 10]])
st = time.time()
for h in hyperparams:
    print("entering loop")
    result = train_flow(h)

print("============== Finished ================", time.time() - st)
