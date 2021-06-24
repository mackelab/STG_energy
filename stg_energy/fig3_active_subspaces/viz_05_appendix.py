import stg_energy.fig2_inference.viz as viz

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pyloric import simulate, create_prior, summary_stats
import pandas as pd
from stg_energy.common import get_labels_8pt, check_if_close_to_obs


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


def plot_overall_efficient(
    index, min_energy_theta, min_energy_seed, time_vec, time_len, offset=60000
):

    successful_samples = min_energy_theta[index]
    trace = simulator(
        np.concatenate((successful_samples, np.asarray([min_energy_seed[index]])))
    )
    stats = summary_stats(trace, stats_customization=custom_stats, t_burn_in=1000)

    with mpl.rc_context(fname="../../../.matplotlibrc"):
        fig, ax = plt.subplots(1, 1, figsize=(2.2, 1.2))
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


multipliers = [1.0, 100.0, 10.00, 10.0, 10.0, 1.0, 10000.0, 10000.0]
all_mult = multipliers * 3
all_mult += [1_000_000] * 7

labels_ = get_labels_8pt()


def plot_params(successful_samples, params_to_plot, labels=True, width=1.3):
    params_to_plot = np.asarray(params_to_plot)

    with mpl.rc_context(fname="../../../.matplotlibrc"):
        fig, ax0 = plt.subplots(1, 1, figsize=(width, 1.0))

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
            ax0.set_xticks([])
        ax0.set_ylabel(
            r"$\overline{g} \;\; \mathrm{(mS / }\mathrm{cm}^2)$", labelpad=-3
        )
        ax0.set_ylim([0, 700.0])
        ax0.set_yticks([0, 700])


def plot_synapses(
    successful_samples, params_to_plot, labels=True, width=0.5, ylim=[0, 150]
):
    params_to_plot = np.asarray(params_to_plot)

    with mpl.rc_context(fname="../../../.matplotlibrc"):
        fig, ax1 = plt.subplots(1, 1, figsize=(width, 1.0))

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
            ax1.set_xticks([])
        ax1.set_ylabel(r"$\overline{g} \;\;\; \mathrm{(nS)}$", labelpad=-10)
        ax1.set_yticks(ylim)
        ax1.set_ylim(ylim)
        ax1.set_xlim(-0.6, 1.6)


def load_theta_x_seeds(pair_ab, pair_lp, pair_py):
    all_stats_loaded = []
    all_seeds_loaded = []
    for k in range(4):
        stats_loaded = pd.read_pickle(
            f"../../../results/mcmc_7d/simulated_samples_{pair_ab}_{pair_lp}_{pair_py}_{k}.pkl"
        )
        seeds_loaded = np.load(
            f"../../../results/mcmc_7d/seeds_for_simulating_mcmc.npy"
        )
        all_stats_loaded.append(stats_loaded)
        all_seeds_loaded.append(seeds_loaded)
    all_stats_loaded = np.concatenate(all_stats_loaded)
    all_seeds_loaded = np.concatenate(all_seeds_loaded)
    samples = np.load(
        f"../../../results/mcmc_7d/mcmc_samples_{pair_ab}_{pair_lp}_{pair_py}.npy"
    )

    return samples, all_stats_loaded, all_seeds_loaded


def load_x_pd(pair_ab, pair_lp, pair_py):
    all_stats_loaded = []
    for k in range(4):
        stats_loaded = pd.read_pickle(
            f"../../../results/mcmc_7d/simulated_samples_{pair_ab}_{pair_lp}_{pair_py}_{k}.pkl"
        )
        all_stats_loaded.append(stats_loaded)
    all_stats_loaded = pd.concat(all_stats_loaded)

    return all_stats_loaded


def plot_stuff(selected_ones, time_vec, time_len, allowed_std):
    samples, stats, all_seeds_loaded = load_theta_x_seeds(*selected_ones)
    close_sim = check_if_close_to_obs(
        stats, sloppiness_durations=allowed_std, sloppiness_phases=allowed_std
    )

    successful_samples = samples[close_sim]
    trace = simulator(np.concatenate((successful_samples[0], np.asarray([0]))))
    stats = summary_stats(trace, stats_customization=custom_stats, t_burn_in=1000)

    with mpl.rc_context(fname="../../../.matplotlibrc"):
        fig, ax = plt.subplots(1, 1, figsize=(2.2, 1.2))
        viz.vis_sample_plain(
            voltage_trace=trace,
            t=time_vec,
            axV=ax,
            time_len=int(time_len),
            offset=100000,
            col="k",
            scale_bar=True,
            print_label=False,
        )


def plot_energies(selected_ones_1, allowed_std, labels=True):
    stats = load_x_pd(*selected_ones_1)
    close_sim = check_if_close_to_obs(
        stats.to_numpy(),
        sloppiness_durations=allowed_std,
        sloppiness_phases=allowed_std,
    )
    successful_stats1 = stats[close_sim].iloc[0]

    energy1 = successful_stats1["energies"].to_numpy() / 10 / 1000

    summed_energies_total1 = np.sum(np.asarray(energy1))

    with mpl.rc_context(fname="../../../.matplotlibrc"):
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
