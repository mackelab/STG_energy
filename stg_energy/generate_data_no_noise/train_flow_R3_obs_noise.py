import numpy as np
import pandas as pd
import dill as pickle
import torch

from pyloric import create_prior
from sbi.inference import SNPE
from sbi.utils import posterior_nn, BoxUniform
from sbi.utils.user_input_checks_utils import PytorchReturnTypeWrapper
import time

# upload data:
# scp -r results/simulation_data_Tube_MLslurm_cluster/all_circuit_parameters_train.pkl mdeistler57@134.2.168.52:~/Documents/STG_energy/results/simulation_data_Tube_MLslurm_cluster/
# scp -r results/simulation_data_Tube_MLslurm_cluster/all_simulation_outputs_train.pkl mdeistler57@134.2.168.52:~/Documents/STG_energy/results/simulation_data_Tube_MLslurm_cluster/


def train_flow(hyperparams):

    batch_size = hyperparams[0].item()
    lr = hyperparams[1].item()
    hidden = hyperparams[2].item()
    layers = hyperparams[3].item()
    seed = torch.randint(1000000, (1,))
    torch.manual_seed(seed)

    general_path = "../../results/"
    path_to_data = "simulation_data_Tube_MLslurm_cluster_no_noise/all_"
    theta = pd.read_pickle(general_path + path_to_data + "circuit_parameters_train.pkl")
    x = pd.read_pickle(general_path + path_to_data + "simulation_outputs_train.pkl")

    theta = torch.as_tensor(theta.to_numpy(), dtype=torch.float32)
    x = torch.as_tensor(x.to_numpy(), dtype=torch.float32)

    x = x[:, :18]
    print("x[0]", x[0])

    noise_std = torch.as_tensor(
        [
            10,
            5,
            5,
            5,
            0.005,
            0.005,
            0.005,
            0.005,
            0.005,
            5,
            5,
            5,
            5,
            0.005,
            0.005,
            0.1,
            0.1,
            0.1,
        ]
    )
    noise = torch.randn(x.shape) * noise_std
    x = x + noise

    print("theta.shape", theta.shape)
    print("x.shape", x.shape)

    # Load prior.
    prior = create_prior(as_torch_dist=True)
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
        f"../../results/trained_neural_nets/inference/flow_{best_log_prob}_{seed}_no_noise_obsnoise.pickle",
        "wb",
    ) as handle:
        pickle.dump(posterior, handle, protocol=4)

    with open(
        f"../../results/trained_neural_nets/inference/posterior_11deg_no_noise_obsnoise.pickle",
        "wb",
    ) as handle:
        pickle.dump(posterior, handle, protocol=4)

    with open(
        f"../../results/trained_neural_nets/inference/inference_snpe_{best_log_prob}_{seed}_no_noise_obsnoise.pickle",
        "wb",
    ) as handle:
        pickle.dump(inference, handle, protocol=4)
    with open(
        f"../../results/trained_neural_nets/inference/hyperparams_{best_log_prob}_{seed}_no_noise_obsnoise.pickle",
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
