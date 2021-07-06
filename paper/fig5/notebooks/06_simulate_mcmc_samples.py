from multiprocessing import Pool
import numpy as np
import pandas as pd
from pyloric import create_prior, simulate, summary_stats
from stg_energy import check_if_close_to_obs
from pyloric import create_prior
import argparse


def simulator(p_with_s):
    p1 = create_prior()
    pars = p1.sample((1,))
    column_names = pars.columns
    circuit_params = np.asarray([p_with_s[:-1]])
    theta_pd = pd.DataFrame(circuit_params, columns=column_names)
    out_target = simulate(
        theta_pd.loc[0], seed=int(p_with_s[-1]), track_energy=True, track_currents=True
    )
    return out_target


def run(preparation):

    num_sims = 50
    np.random.seed(0)

    if preparation == "016":
        xo = np.load("../../../results/experimental_data/xo_11deg_016.npy")
        min_num_bursts = 6.5
    elif preparation == "078":
        xo = np.load("../../../results/experimental_data/xo_11deg_078.npy")
        min_num_bursts = 6.5
    elif preparation == "082":
        xo = np.load("../../../results/experimental_data/xo_11deg.npy")
        min_num_bursts = 7.5
    else:
        raise NameError

    for k in range(20):

        for pair_ab in range(5):
            for pair_lp in range(5):
                for pair_py in range(5):
                    print("pair", pair_ab, pair_lp, pair_py)
                    samples = np.load(
                        f"../../../results/mcmc_7d/mcmc_samples_{preparation}_{pair_ab}_{pair_lp}_{pair_py}.npy"
                    )[num_sims * k : num_sims * (k + 1)]
                    print("samples", samples.shape)
                    num_cores = 8
                    seeds_sim = np.random.randint(0, 10000, num_sims)

                    params_with_seeds = np.concatenate(
                        (
                            samples[:num_sims],
                            seeds_sim[
                                None,
                            ].T,
                        ),
                        axis=1,
                    )

                    with Pool(num_cores) as pool:
                        data = pool.map(simulator, params_with_seeds)

                    custom_stats = {
                        "plateau_durations": True,
                        "num_bursts": True,
                        "num_spikes": True,
                        "energies": True,
                        "energies_per_burst": True,
                        "energies_per_spike": True,
                        "pyloric_like": True,
                    }
                    stats = [
                        summary_stats(
                            d, stats_customization=custom_stats, t_burn_in=1000
                        )
                        for d in data
                    ]
                    stats = pd.concat(stats)

                    stats.to_pickle(
                        f"../../../results/mcmc_7d/simulated_samples_{preparation}_{pair_ab}_{pair_lp}_{pair_py}_{k}.pkl"
                    )
                    np.save(
                        f"../../../results/mcmc_7d/seeds_for_simulating_mcmc_{preparation}_{k}.npy",
                        seeds_sim,
                    )

                    close_sim = check_if_close_to_obs(
                        stats.to_numpy(), xo=xo[:15], min_num_bursts=min_num_bursts
                    )
                    print("Number of close sims:  ", np.sum(close_sim))

                    close_sim = check_if_close_to_obs(
                        stats.to_numpy(),
                        xo=xo[:15],
                        min_num_bursts=min_num_bursts,
                        sloppiness_durations=2.0,
                        sloppiness_phases=2.0,
                    )
                    print("Sloppi 2.0 number of close sims:  ", np.sum(close_sim))

                    close_sim = check_if_close_to_obs(
                        stats.to_numpy(),
                        xo=xo[:15],
                        min_num_bursts=min_num_bursts,
                        sloppiness_durations=3.0,
                        sloppiness_phases=3.0,
                    )
                    print("Sloppi 3.0 number of close sims:  ", np.sum(close_sim))

                    close_sim = check_if_close_to_obs(
                        stats.to_numpy(),
                        xo=xo[:15],
                        min_num_bursts=min_num_bursts,
                        sloppiness_durations=10.0,
                        sloppiness_phases=10.0,
                    )
                    print("Sloppi 10.0 number of close sims:  ", np.sum(close_sim))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Take in the preparation")
    parser.add_argument("preparation", type=str)
    args = parser.parse_args()
    preparation = args.preparation
    print("mcmc name", preparation)

    run(preparation)
