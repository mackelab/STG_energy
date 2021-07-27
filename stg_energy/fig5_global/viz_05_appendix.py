import stg_energy.fig2_inference.viz as viz

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pyloric import simulate, create_prior, summary_stats
import pandas as pd

from stg_energy.common import check_if_close_to_obs, get_labels_8pt


custom_stats = {
    "plateau_durations": True,
    "num_bursts": True,
    "num_spikes": True,
    "energies": True,
    "energies_per_burst": True,
    "energies_per_spike": True,
    "pyloric_like": True,
}


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


def plot_energy_of_theta(
    index,
    min_energy_theta,
    min_energy_seed,
    time_vec,
    time_len,
    offset=60000,
    figsize=(2.2, 1.2),
    labelpad=0
):
    successful_samples = min_energy_theta[index]
    trace = simulator(
        np.concatenate((successful_samples, np.asarray([min_energy_seed[index]])))
    )
    stats = summary_stats(trace, stats_customization=custom_stats, t_burn_in=1000)
    energy = np.sum(stats["energies"].to_numpy() / 1000 / 10)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.bar([0], energy, color="k", width=0.2)
    ax.set_xlim([-0.2, 0.2])
    ax.set_ylabel(r"Energy ($\mu$J/s)", labelpad=labelpad)
    ax.set_ylim([0.0, 30.0])
    ax.set_yticks([0, 30])
    ax.set_xticks([])
    ax.set_xticklabels([])
    print(energy)


def plot_overall_efficient(
    index,
    min_energy_theta,
    min_energy_seed,
    time_vec,
    time_len,
    offset=60000,
    figsize=(2.2, 1.2),
):

    successful_samples = min_energy_theta[index]
    trace = simulator(
        np.concatenate((successful_samples, np.asarray([min_energy_seed[index]])))
    )

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    viz.vis_sample_plain(
        voltage_trace=trace,
        t=time_vec,
        axV=ax,
        time_len=int(time_len),
        offset=offset,
        col="k",
        scale_bar=True,
        print_label=False,
    )


multipliers = [1.0, 100.0, 10.00, 10.0, 100.0, 1.0, 10000.0, 10000.0]
all_mult = multipliers * 3
all_mult += [1_000_000] * 7

labels_ = get_labels_8pt()


def plot_params(
    successful_samples,
    params_to_plot,
    labels=True,
    width=1.3,
    height=1.0,
    labelpad=-3,
    ylim=[0, 150],
):
    params_to_plot = np.asarray(params_to_plot)
    fig, ax0 = plt.subplots(1, 1, figsize=(width, height))

    successful_samples[-7:] = np.exp(successful_samples[-7:])
    plotted_params = successful_samples[params_to_plot]
    plotted_params = plotted_params * np.asarray(all_mult)[params_to_plot]
    cond = params_to_plot < 24
    ax0.bar(np.arange(len(params_to_plot[cond])), plotted_params[cond], color="k")
    if labels:
        ax0.set_xticks(range(len(params_to_plot[cond])))
        ax0.set_xticklabels(
            (np.asarray(labels_)[np.asarray(params_to_plot[cond])]).tolist(),
            rotation=90,
        )
    else:
        ax0.set_xticks(range(len(params_to_plot[cond])))
        ax0.set_xticklabels([])
    ax0.set_ylabel(
        r"$\overline{g} \;\; \mathrm{(mS / }\mathrm{cm}^2)$", labelpad=labelpad
    )
    ax0.set_ylim(ylim)
    ax0.set_yticks(ylim)


def plot_synapses(
    successful_samples,
    params_to_plot,
    labels=True,
    width=0.5,
    ylim=[0, 150],
    height=1.0,
    labelpad=-10,
):
    params_to_plot = np.asarray(params_to_plot)

    fig, ax1 = plt.subplots(1, 1, figsize=(width, height))

    successful_samples[-7:] = np.exp(successful_samples[-7:])
    plotted_params = successful_samples[params_to_plot]
    plotted_params = plotted_params * np.asarray(all_mult)[params_to_plot]

    cond = params_to_plot >= 24
    ax1.bar(np.arange(len(params_to_plot[cond])), plotted_params[cond], color="k")
    if labels:
        ax1.set_xticks(range(len(params_to_plot[cond])))
        ax1.set_xticklabels(
            (np.asarray(labels_)[np.asarray(params_to_plot[cond])]).tolist(),
            rotation=90,
        )
    else:
        ax1.set_xticks(range(len(params_to_plot[cond])))
        ax1.set_xticklabels([])
    ax1.set_ylabel(r"$\overline{g} \;\; \mathrm{(nS)}$", labelpad=labelpad)
    ax1.set_yticks(ylim)
    ax1.set_ylim(ylim)
    ax1.set_xlim(-0.6, 1.6)


def load_theta_x_seeds(pair_ab, pair_lp, pair_py, preparation):
    all_stats_loaded = []
    all_seeds_loaded = []
    for k in range(20):
        stats_loaded = pd.read_pickle(
            f"../../../results/mcmc_7d/simulated_samples_{preparation}_{pair_ab}_{pair_lp}_{pair_py}_{k}.pkl"
        )
        seeds_loaded = np.load(
            f"../../../results/mcmc_7d/seeds_for_simulating_mcmc_{preparation}_{k}.npy"
        )
        all_stats_loaded.append(stats_loaded)
        all_seeds_loaded.append(seeds_loaded)
    all_stats_loaded = np.concatenate(all_stats_loaded)
    all_seeds_loaded = np.concatenate(all_seeds_loaded)
    samples = np.load(
        f"../../../results/mcmc_7d/mcmc_samples_{preparation}_{pair_ab}_{pair_lp}_{pair_py}.npy"
    )

    return samples, all_stats_loaded, all_seeds_loaded

def load_theta_x_seeds_correction(pair_ab, pair_lp, pair_py, preparation):
    all_stats_loaded = []
    all_seeds_loaded = []
    for k in range(200):
        stats_loaded = pd.read_pickle(
            f"../../../results/mcmc_7d/simulated_samples_correction_{preparation}_{pair_ab}_{pair_lp}_{pair_py}_{k}.pkl"
        )
        seeds_loaded = np.load(
            f"../../../results/mcmc_7d/seeds_for_simulating_mcmc_correction_{preparation}_{k}.npy"
        )
        all_stats_loaded.append(stats_loaded)
        all_seeds_loaded.append(seeds_loaded)
    all_stats_loaded = np.concatenate(all_stats_loaded)
    all_seeds_loaded = np.concatenate(all_seeds_loaded)
    samples = np.load(
        f"../../../results/mcmc_7d/mcmc_samples_correction_{preparation}_{pair_ab}_{pair_lp}_{pair_py}.npy"
    )

    return samples, all_stats_loaded, all_seeds_loaded


def load_x_pd(pair_ab, pair_lp, pair_py, preparation):
    all_stats_loaded = []
    for k in range(20):
        stats_loaded = pd.read_pickle(
            f"../../../results/mcmc_7d/simulated_samples_{preparation}_{pair_ab}_{pair_lp}_{pair_py}_{k}.pkl"
        )
        all_stats_loaded.append(stats_loaded)
    all_stats_loaded = pd.concat(all_stats_loaded)

    return all_stats_loaded


def load_x_pd_correction(pair_ab, pair_lp, pair_py, preparation):
    all_stats_loaded = []
    for k in range(200):
        stats_loaded = pd.read_pickle(
            f"../../../results/mcmc_7d/simulated_samples_correction_{preparation}_{pair_ab}_{pair_lp}_{pair_py}_{k}.pkl"
        )
        all_stats_loaded.append(stats_loaded)
    all_stats_loaded = pd.concat(all_stats_loaded)

    return all_stats_loaded

def plot_stuff(
    preparation,
    selected_ones,
    time_vec,
    time_len,
    allowed_std,
    width=2.2,
    height=1.2,
    offset=100000,
    index=0,
):
    samples, stats, all_seeds_loaded = load_theta_x_seeds(
        *selected_ones, preparation=preparation
    )
    xo, min_num_bursts = load_xo_minnumbursts_of_preparation(preparation=preparation)
    close_sim = check_if_close_to_obs(
        stats,
        xo=xo,
        min_num_bursts=min_num_bursts,
        sloppiness_durations=allowed_std,
        sloppiness_phases=allowed_std,
    )

    successful_samples = samples[close_sim]
    successful_seeds = all_seeds_loaded[close_sim]
    print("successful_seeds", successful_seeds.shape)
    trace = simulator(
        np.concatenate(
            (successful_samples[index], np.asarray([successful_seeds[index]]))
        )
    )
    stats = summary_stats(trace, stats_customization=custom_stats, t_burn_in=1000)

    fig, ax = plt.subplots(1, 1, figsize=(width, height))
    viz.vis_sample_plain(
        voltage_trace=trace,
        t=time_vec,
        axV=ax,
        time_len=int(time_len),
        offset=offset,
        col="k",
        scale_bar=True,
        print_label=False,
    )


def plot_energies(preparation, selected_ones_1, allowed_std, labels=True, index=0):
    stats = load_x_pd(*selected_ones_1, preparation=preparation)
    xo, min_num_bursts = load_xo_minnumbursts_of_preparation(preparation=preparation)
    close_sim = check_if_close_to_obs(
        stats.to_numpy(),
        xo=xo,
        min_num_bursts=min_num_bursts,
        sloppiness_durations=allowed_std,
        sloppiness_phases=allowed_std,
    )
    successful_stats1 = stats[close_sim].iloc[index]

    energy1 = successful_stats1["energies"].to_numpy() / 10 / 1000

    summed_energies_total1 = np.sum(np.asarray(energy1))

    fig, ax = plt.subplots(1, 1, figsize=(0.7, 0.8))
    ax.bar(
        range(4),
        [*energy1, summed_energies_total1],
        width=0.75,
        color=["#3182bd", "#fc8d59", "#2ca25f", "k"],
    )
    ax.set_xticks(range(4))
    if labels:
        ax.set_xticklabels(["AB/PD", "LP", "PY", "Sum"], rotation=90)
    else:
        ax.set_xticks([])
    ax.set_ylim([0, 4.0])
    ax.set_ylabel("Energy ($\mu$J/s)")

    return summed_energies_total1


def load_xo_minnumbursts_of_preparation(preparation):
    if preparation == "082":
        return np.load("../../../results/experimental_data/xo_11deg.npy")[:15], 7.5
    if preparation == "016":
        return np.load("../../../results/experimental_data/xo_11deg_016.npy")[:15], 6.5
    if preparation == "078":
        return np.load("../../../results/experimental_data/xo_11deg_078.npy")[:15], 6.5
