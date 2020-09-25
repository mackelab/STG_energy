import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("../../../")


def all_sensitivity_bars(cum_grad, ylim, figsize):
    fig, ax = plt.subplots(1, figsize=figsize)

    barlist = ax.bar(np.arange(len(cum_grad[0])), cum_grad[0], width=0.5)
    for i in range(0, 8):
        barlist[i].set_color("#3182bd")
    for i in range(8, 16):
        barlist[i].set_color("#fc8d59")
    for i in range(16, 24):
        barlist[i].set_color("#2ca25f")
    for i in range(24, 31):
        barlist[i].set_color("k")
    ax.set_ylim(ylim)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.get_xaxis().set_ticks([])

    ax.set_xticks([-0.5, 7.5, 15.5, 23.5, 30.5])
    ax.set_xticklabels(
        [
            r"$\;\;\;\;\;\;\;\;\;\;\;\;\;\;\mathdefault{AB/PD}$",
            r"$\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\mathdefault{LP}$",
            r"$\;\;\;\;\;\;\;\;\;\;\;\;\;\mathdefault{PY}$",
            r"$\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\mathdefault{Synapses}$",
            "",
        ]
    )


def other_sensitivity_bars(cum_grad, ylim, figsize):
    fig, ax = plt.subplots(1, figsize=figsize)

    barlist = ax.bar(np.arange(len(cum_grad[0])), cum_grad[0], width=0.5)
    for i in range(0, 8):
        barlist[i].set_color("#fc8d59")
    for i in range(8, 16):
        barlist[i].set_color("#2ca25f")
    for i in range(16, 23):
        barlist[i].set_color("k")
    ax.set_ylim(ylim)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.spines["left"].set_visible(False)
    ax.set_yticks([])
    ax.set_xlim([-0.5, 22.5])

    ax.set_xticks([-0.5, 7.5, 15.5, 22.5])
    ax.set_xticklabels(
        [
            r"$\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\mathdefault{LP}$",
            r"$\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\mathdefault{PY}$",
            r"$\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\mathdefault{Synapses}$",
            "",
        ]
    )


def single_neuron_sensitivity_bar(
    cum_grad, ylim, figsize, start=0, end=8, color="#3182bd"
):
    fig, ax = plt.subplots(1, figsize=figsize)

    barlist = ax.bar(
        np.arange(len(cum_grad[0][start:end])), cum_grad[0][start:end], width=0.5
    )
    for i in range(8):
        barlist[i].set_color(color)
    ax.set_ylim(ylim)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.get_xaxis().set_ticks([])

    ax.set_xticks(np.arange(8))
    ax.set_xticklabels(["Na", "CaT", "CaS", "A", "KCa", "Kd", "H", "Leak"])


def plot_energy_scape(
    out_target, neuron_to_plot, t, t_min, t_max, figsize, xlabel, ylabel,
):

    cols_hex = [
        "#1b9e77",
        "#d95f02",
        "#7570b3",
        "#e7298a",
        "#66a61e",
        "#e6ab02",
        "#a6761d",
        "#666666",
    ]

    # build energyscape
    all_energies = out_target["all_energies"]
    all_currents_PD = all_energies[:, neuron_to_plot, :]
    t = t[0 : t_max - t_min]

    fig, ax = plt.subplots(2, 1, figsize=figsize, gridspec_kw={"height_ratios": [3, 1]})
    for i in range(8):
        # ax[0].plot(t, all_currents_PD[i,t_min:t_max])
        summed_currents_until = np.sum(all_currents_PD[:i, t_min:t_max], axis=0)
        summed_currents_include = np.sum(all_currents_PD[: i + 1, t_min:t_max], axis=0)
        ax[0].fill_between(
            t, summed_currents_until, summed_currents_include, color=cols_hex[i]
        )

    ax[0].set_ylim([0, 200])
    ax[1].plot(t, out_target["data"][neuron_to_plot, t_min:t_max])
    ax[1].set_ylim([-80, 70])

    for a in ax:
        a.tick_params(axis="both", which="major")
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)

    ax[0].axes.get_xaxis().set_ticks([])

    if not xlabel:
        ax[1].axes.get_xaxis().set_ticks([])
    else:
        ax[1].set_xlabel("Time (ms)")
    if not ylabel:
        ax[0].axes.get_yaxis().set_ticks([])
        ax[1].axes.get_yaxis().set_ticks([])
    else:
        ax[0].set_ylabel("Energy")
        ax[1].set_ylabel("Voltage")
