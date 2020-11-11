from pyloric import simulate, create_prior, stats
from joblib import Parallel, delayed
import numpy as np
import time
import multiprocessing
from multiprocessing import Pool

print("Number of cores: ", multiprocessing.cpu_count())


def simulator(params_set):
    out_target = simulate(params_set[:-1], seed=int(params_set[-1]),)
    return stats(out_target)


num_sims = 512
num_cores = 32

prior = create_prior()
parameter_sets = prior.sample((num_sims,))
seeds = np.random.randint(0, 10000, (num_sims, 1))
params_with_seeds = np.concatenate((parameter_sets, seeds), axis=1)

pool = Pool(processes=num_cores)
data = []

start_time = time.time()
data.append(pool.map(simulator, params_with_seeds))
print("Simulation time", time.time() - start_time)

sim_outs = np.asarray(simulation_outputs)
print("np.shape", np.shape(sim_outs))
print("one of them:  ", sim_outs[9])
#
#
# def simulator(params_set):
#     time.sleep(1.0)
#     return 0.0
#
#
# pool = Pool(processes=num_cores)
# data = []
# params = np.zeros((num_sims, 1))
# start_time = time.time()
# # simulation_outputs = Parallel(n_jobs=num_cores, prefer="threads")(
# #     delayed(simulator)(batch) for batch in params
# # )
# data.append(pool.map(simulator, params))
# print("Simulation time", time.time() - start_time)
