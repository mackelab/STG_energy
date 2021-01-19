import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
from mpl_toolkits.axisartist.grid_helper_curvelinear import GridHelperCurveLinear
from mpl_toolkits.axisartist import Subplot, ParasiteAxesAuxTrans
import mpl_toolkits.axisartist.angle_helper as angle_helper
from mpl_toolkits.axisartist.grid_finder import FixedLocator
import math
import matplotlib.cm
from matplotlib.colors import Normalize
from typing import Optional
import torch

from stg_energy.common import col, _update, _format_axis
from scipy.stats import gaussian_kde
import os
import matplotlib as mpl
import seaborn as sns
from pyloric.utils import energy_of_membrane


def vis_sample_plain(
    voltage_trace,
    t,
    axV,
    t_on=None,
    t_off=None,
    col="k",
    print_label=False,
    time_len=None,
    offset=0,
    scale_bar=True,
):
    """
    Function of Kaan, modified by Michael. Used for plotting fig 5b Prinz.

    :param sample: membrane/synaptic conductances
    :param t_on:
    :param t_off:
    :param with_ss: bool, True if bars for summary stats are wanted
    :param with_params: bool, True if bars for parameters are wanted
    :return: figure object
    """

    font_size = 15.0
    current_counter = 0

    dt = t[1] - t[0]
    scale_bar_breadth = 500
    scale_bar_voltage_breadth = 50

    offscale = 100
    offvolt = -50

    if scale_bar:
        scale_col = "k"
    else:
        scale_col = "w"

    data = voltage_trace

    Vx = data["data"]

    current_col = 0
    for j in range(3):
        if time_len is not None:
            axV.plot(
                t[10000 + offset : 10000 + offset + time_len : 5],
                Vx[j, 10000 + offset : 10000 + offset + time_len : 5] + 140.0 * (2 - j),
                lw=0.3,
                c=col,
            )
        else:
            axV.plot(
                t,
                Vx[j] + 120.0 * (2 - j),
                lw=0.3,
                c=col[current_col],
            )
        current_col += 1

    if print_label:
        axV.plot(
            [1100.0 + (offset - 26500) * (t[1] - t[0])],
            [300],
            color=col,
            marker="o",
            markeredgecolor="w",
            ms=8,
            markeredgewidth=1.0,
            path_effects=[pe.Stroke(linewidth=1.3, foreground="k"), pe.Normal()],
        )

    if scale_bar:

        # time bar
        axV.plot(
            (offset + 5500) * dt
            + offscale
            + np.arange(scale_bar_breadth)[:: scale_bar_breadth - 1],
            (-40 + offvolt)
            * np.ones_like(np.arange(scale_bar_breadth))[:: scale_bar_breadth - 1],
            lw=1.0,
            color="w",
        )

        # voltage bar
        axV.plot(
            (2850 + offset * dt + offscale)
            * np.ones_like(np.arange(scale_bar_voltage_breadth))[
                :: scale_bar_voltage_breadth - 1
            ],
            275
            + np.arange(scale_bar_voltage_breadth)[:: scale_bar_voltage_breadth - 1],
            lw=1.0,
            color=scale_col,
            zorder=10,
        )

    box = axV.get_position()

    if t_on is not None:
        axV.axvline(t_on, c="r", ls="--")

    if t_on is not None:
        axV.axvline(t_off, c="r", ls="--")

    axV.set_position([box.x0, box.y0, box.width, box.height])
    axV.axes.get_yaxis().set_ticks([])
    axV.axes.get_xaxis().set_ticks([])

    axV.spines["right"].set_visible(False)
    axV.spines["top"].set_visible(False)
    axV.spines["bottom"].set_visible(False)
    axV.spines["left"].set_visible(False)

    current_counter += 1


def active_subspace_sketch():
    figureratio = 1.5
    figuresize = 1.3

    vec1 = np.linspace(-3, 3, int(figureratio * 100))
    vec2 = np.linspace(-3, 3, 100)

    X, Y = np.meshgrid(vec1, vec2)
    dist = np.abs(X + 1.5 * Y)

    m_e = [0.0, -0.1]
    s_e = [6.0, 0.7]
    m_e_2 = [1.1, 1.7]
    s_e_2 = [1.5, 6.0]
    dists_e_1 = s_e[0] * (X - m_e[0]) ** 2 + s_e[1] * (Y - m_e[1]) ** 2
    dists_e_2 = s_e_2[0] * (X - m_e_2[0]) ** 2 + s_e_2[1] * (Y - m_e_2[1]) ** 2
    dists_e = np.minimum(dists_e_1, dists_e_2)
    allowed_dist = 5.0
    thr_dists_e = dists_e < allowed_dist
    inds = np.where(np.abs(dists_e - allowed_dist) < 0.5)
    inds = np.asarray(inds).T

    fig, ax = plt.subplots(1, 1, figsize=(figureratio * figuresize, figuresize))
    image_to_plot = -np.sqrt(dist)

    im = ax.imshow(image_to_plot, cmap="autumn_r")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Parameter 1")
    ax.set_ylabel("Parameter 2")

    cbar = plt.colorbar(im, aspect=15, fraction=0.04, pad=0.04)
    cbar.set_ticks([])
    cbar.set_label("Energy", labelpad=5)
    ax.arrow(75, 50, 10, -5, head_width=7, head_length=9, facecolor="k")
    ax.arrow(75, 50, 10, 20, head_width=7, head_length=9, facecolor="k")


def scatter_sensitivity_consumption(all_fractions, eigenvector, arrows: bool = True):
    fig, ax = plt.subplots(1, 1, figsize=(1.75, 1.75))

    markers = ["o", "^", "s"]
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
    labels = ["Na", "CaT", "CaS", "A", "KCa", "Kd", "H", "leak"]
    labels2 = ["AB/PD", "LP", "PY"]
    ims = []
    #     for synapse in range(7):
    #         im = ax.scatter(
    #             all_fractions[24+synapse],
    #             np.abs(eigenvector)[24+synapse],
    #             c='k',
    #             marker='o',
    #             s=10
    #         )
    #     ims.append(im)
    for neuron in range(3):
        for channel, c in enumerate(cols_hex):
            im = ax.scatter(
                all_fractions[channel + neuron * 8],
                np.abs(eigenvector)[channel + neuron * 8],
                c=c,
                marker=markers[neuron],
                s=14,
            )
            if neuron == 0:
                ims.append(im)

    # Dummy scatter outside of limits for the legend:
    ims2 = []
    ims2.append(ax.scatter([10.0], [10.0], c="k", marker="o", s=14))
    ims2.append(ax.scatter([10.0], [10.0], c="k", marker="^", s=14))
    ims2.append(ax.scatter([10.0], [10.0], c="k", marker="s", s=14))

    leg1 = ax.legend(
        ims,
        labels,
        bbox_to_anchor=(1.05, 0.84, 0.1, 0.5),
        labelspacing=0.3,
        columnspacing=0.5,
        markerfirst=False,
        handletextpad=-0.4,
        ncol=4,
    )
    leg2 = ax.legend(
        ims2,
        labels2,
        bbox_to_anchor=(1.05, 0.62, 0.1, 0.5),
        labelspacing=0.3,
        columnspacing=0.5,
        markerfirst=False,
        handletextpad=-0.4,
        ncol=3,
    )
    ax.add_artist(leg1)
    #     import itertools
    #     def flip(items, ncol):
    #         return itertools.chain(*[items[i::ncol] for i in range(ncol)])
    #     handles, labels = ax.get_legend_handles_labels()
    #     ax.legend(flip(handles, 2), flip(labels, 2), bbox_to_anchor=(1.1, -0.9, 0.1, 0.5), labelspacing=0.3, columnspacing=0.5, markerfirst=False, handletextpad=-0.4, ncol=4)

    if arrows:
        # right
        ax.arrow(
            0.59, 0.19, 0.08, -0.065, head_width=0.04, head_length=0.04, facecolor="k"
        )
        # middle
        ax.arrow(
            0.185, 0.649, 0.08, -0.065, head_width=0.04, head_length=0.04, facecolor="k"
        )
        # left
        ax.arrow(
            0.0695, 0.49, 0.0, -0.1, head_width=0.04, head_length=0.04, facecolor="k"
        )

    ax.plot([0.0, 0.7], [0.0, 0.7], color="grey", alpha=0.5)
    ax.set_xlim([0.0, 0.78])
    ax.set_ylim([0.0, 0.78])

    ax.set_xlabel("Consumed Energy")
    ax.set_ylabel("|Active dim.|")


def curvelinear_test1(
    fig,
    angle_within_90deg,
    projected_num_spikes,
    projected_e_per_spike,
    energy_PM_train,
    parameter_set1_dim1,
    parameter_set1_dim2,
    parameter_set2_dim1,
    parameter_set2_dim2,
    parameter_set3_dim1,
    parameter_set3_dim2,
):
    xlims = [-3.4, 2.2]
    ylims = [-2.62, 2.8]

    # We need this ratio because the scale on the y-axis and on the x-axis is
    # different. The ration is approximately 2.7 / 1.8, which is the ratio of
    # the mins of the data in the respective dimensions. Correctness of the
    # ration 1.55 was checked by manually rotating the ylabel with:
    # `ax1.axis["y"].label.set_rotation(-19)`. 19 degree is the true angle from
    # a right angle.
    ratio_yx = 1.55

    def tr(x, y):
        x, y = np.asarray(x), np.asarray(y)
        return x - y / math.tan(angle_within_90deg) / ratio_yx, y

    def inv_tr(x, y):
        x, y = np.asarray(x), np.asarray(y)
        return x + y * math.tan(angle_within_90deg) * ratio_yx, y

    grid_locator1 = FixedLocator([-1.8, 0, 1.8])
    grid_locator2 = FixedLocator([-2.1, 0])

    grid_helper = GridHelperCurveLinear(
        aux_trans=(tr, inv_tr), grid_locator1=grid_locator1, grid_locator2=grid_locator2
    )

    ax1 = Subplot(fig, 1, 1, 1, grid_helper=grid_helper)
    fig.add_subplot(ax1)

    ax1.grid(True, which="minor", zorder=-100)

    projected_data_x, projected_data_y = tr(
        projected_num_spikes[:4000].T, projected_e_per_spike[:4000].T
    )
    im = ax1.scatter(
        projected_data_x,
        projected_data_y,
        s=3,
        c=energy_PM_train[:4000] / 10 / 1000,
        cmap="autumn_r",
    )
    ax1.scatter(*tr(parameter_set1_dim2, parameter_set1_dim1), color="#0570b0")
    ax1.scatter(*tr(parameter_set2_dim2, parameter_set2_dim1), color="#0570b0")
    ax1.scatter(*tr(parameter_set3_dim2, parameter_set3_dim1), color="#0570b0")

    ax1.annotate(
        "",
        xy=(-1.1, -1.6),
        xytext=(-0.5, -0.4),
        arrowprops=dict(
            facecolor="#0570b0",
            edgecolor="#0570b0",
            headwidth=5.4,
            headlength=5.7,
            width=0.1,
        ),
    )

    ax1.axis["right"].set_visible(False)
    ax1.axis["top"].set_visible(False)
    ax1.axis["bottom"].set_visible(False)
    ax1.axis["left"].set_visible(False)

    xx, yy = tr([3, 6], [5.0, 10.0])
    ax1.plot(xx, yy)

    ax1.set_xlim(xlims)
    ax1.set_ylim(ylims)

    ax1.set_ylabel("Proj. to 1st E-vec: E / spike")
    ax1.set_xlabel("Proj. to 1st E-vec: E / spike")

    ax1.axis["y"] = ax1.new_floating_axis(0, -2.4)
    ax1.axis["x"] = ax1.new_floating_axis(1, -2.6)
    ax1.axis["y"].set_axis_direction("top")
    ax1.axis["y"].label.set_axis_direction("bottom")
    ax1.axis["y"].set_ticklabel_direction("+")
    ax1.axis["y"].invert_ticklabel_direction()

    cbar = plt.colorbar(im, aspect=25, fraction=0.04, pad=0.04)
    cbar.set_ticks(
        [
            np.min(energy_PM_train[:4000] / 10 / 1000),
            np.max(energy_PM_train[:4000] / 10 / 1000),
        ]
    )
    cbar.set_ticklabels(["0.4", "10"])
    cbar.set_label("Energy (AB/PD)", labelpad=-7)


def bars_for_energy(
    num_spikes_ABPD,
    min_num_spikes,
    max_num_spikes,
    energyperspike_ABPD_sim,
    min_energy_per_spike,
    max_energy_per_spike,
    energies_ABPD_sim,
    min_energy,
    max_energy,
):
    fig, ax = plt.subplots(1, 1, figsize=(0.7, 1.0))
    height_num_spikes = (num_spikes_ABPD - min_num_spikes) / (
        max_num_spikes - min_num_spikes
    )
    height_energy_per_spike = (energyperspike_ABPD_sim - min_energy_per_spike) / (
        max_energy_per_spike - min_energy_per_spike
    )
    height_energy = (energies_ABPD_sim - min_energy) / (max_energy - min_energy)
    ax.bar(
        np.arange(0, 3),
        [height_num_spikes, height_energy_per_spike, height_energy],
        width=0.3,
        facecolor="k",
    )
    ax.set_ylim([0, 1])
    ax.set_xlim([-0.4, 2.4])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["#spikes", "E/spike", "Energy"], rotation=45)
    ax.set_yticklabels(["min", "max"])


def energy_scape(
    out_target,
    t,
    figsize,
    cols,
    time_len,
    offset,
    ylimE=[0, 300],
    v_labelpad=4.8,
    neuron=2,
    ylabels=True,
):
    fig, ax = plt.subplots(2, 1, figsize=figsize)
    iii = 0

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

    names = ["AB", "LP", "PY"]

    current_col = 0
    Vx = out_target["voltage"]
    axV = ax[0]
    for j in range(neuron, neuron + 1):
        if time_len is not None:
            axV.plot(
                t[:time_len:5],
                Vx[j, 10000 + offset : 10000 + offset + time_len : 5],
                lw=0.6,
                c=cols[iii],
            )
        else:
            axV.plot(t / 1000, Vx[j] + 120.0 * (2 - j), lw=0.6, c=cols[iii])
        current_col += 1

    box = axV.get_position()

    axV.set_position([box.x0, box.y0, box.width, box.height])
    if ylabels:
        axV.set_ylabel("V (" + names[neuron] + ")\n (mV)", labelpad=v_labelpad)
    else:
        axV.set_yticks([])
    axV.tick_params(axis="both", which="major")

    axV.spines["right"].set_visible(False)
    axV.spines["top"].set_visible(False)

    axV.set_xticks([])
    axV.set_ylim([-90, 60])

    plt.subplots_adjust(wspace=0.05)

    axS = ax[1]
    all_energies = np.asarray(energy_of_membrane(out_target))

    for current_current in range(neuron, neuron + 1):
        all_currents_PD = all_energies[:, current_current, :]

        for i in range(8):
            # times 10 because: times 10000 for cm**2, but /1000 for micro from nano J
            summed_currents_until = (
                np.sum(
                    all_currents_PD[:i, 10000 + offset : 10000 + offset + time_len : 5],
                    axis=0,
                )
                / 1000
            )
            summed_currents_include = np.sum(
                all_currents_PD[
                    : i + 1,
                    10000 + offset : 10000 + offset + time_len : 5,
                ]
                / 1000,
                axis=0,
            )
            axS.fill_between(
                t[:time_len:5],
                summed_currents_until,
                summed_currents_include,
                color=cols_hex[i],
            )
    axS.spines["right"].set_visible(False)
    axS.spines["top"].set_visible(False)
    axS.set_xlabel("Time (ms)")
    if ylabels:
        axS.set_ylabel("E (" + names[neuron] + ")\n $(\mu$J/s)", labelpad=6.0)
    else:
        axS.set_yticks([])
    axS.set_ylim(ylimE)
    axS.tick_params(axis="both", which="major")

    plt.subplots_adjust(wspace=0.3, hspace=0.3)


def all_sensitivity_bars(cum_grad, ylim, figsize, ylabel=None):
    fig, ax = plt.subplots(1, figsize=figsize)

    barlist = ax.bar(np.arange(len(cum_grad[0])), cum_grad[0], width=0.3)
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

    if ylabel is not None:
        ax.set_ylabel(ylabel)


def synapse_sensitivity_bars(
    cum_grad, ylim, figsize, ylabel=None, plot_labels=True, color="#2ca25f", title=None
):
    fig, ax = plt.subplots(1, figsize=figsize)

    min_height = 0.01
    cum_grad[torch.logical_and(cum_grad > -min_height, cum_grad < 0.0)] = -min_height
    cum_grad[torch.logical_and(cum_grad < min_height, cum_grad > 0.0)] = min_height

    _ = ax.bar(
        np.arange(1, 1 + len(cum_grad[0])),
        cum_grad[0],
        width=0.9 / figsize[0],
        color=color,
    )

    ax.set_ylim(ylim)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.get_xaxis().set_ticks([])

    ax.set_xticks(range(1, 8))
    if plot_labels:
        ax.set_xticklabels(
            ["AB-LP", "PD-LP", "AB-PY", "PD-PY", "LP-PD", "LP-PY", "PY-LP"], rotation=45
        )
        ax.set_xlim(0.7, 7.3)
    else:
        ax.set_xlim(0.7, 7.3)
        ax.set_xticks([])

    if ylabel is not None:
        ax.set_ylabel(ylabel)
        ax.set_yticks([-1, 0, 1])
        ax.set_xlim(0.5, 7.3)
    else:
        ax.spines["left"].set_visible(False)
        ax.set_yticks([])

    if title is not None:
        ax.set_title(title)


def py_sensitivity_bars_cosyne(
    vec,
    ylim,
    figsize,
    ylabel=None,
    plot_labels=True,
    color="#2ca25f",
    legend: bool = True,
    legend_y_offset: float = 0.0,
    title: Optional[str] = None,
    title_x_offset: float = 0.0,
):
    # Very small bars are not visible, which is ugly.
    min_height = 0.03
    vec[torch.logical_and(vec > -min_height, vec < 0.0)] = -min_height
    vec[torch.logical_and(vec < min_height, vec > 0.0)] = min_height

    fig, ax = plt.subplots(1, figsize=figsize)
    _ = ax.bar(
        np.arange(1, 9) - 0.2,
        vec[:8],
        width=0.4 / figsize[0],
        color="#3182bd",
    )
    _ = ax.bar(
        np.arange(1, 9),
        vec[8:16],
        width=0.4 / figsize[0],
        color="#fc8d59",
    )
    _ = ax.bar(
        np.arange(1, 9) + 0.2,
        vec[16:24],
        width=0.4 / figsize[0],
        color="#2ca25f",
    )

    ax.set_ylim(ylim)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.get_xaxis().set_ticks([])

    ax.set_xticks(range(1, 9))
    ax.set_xticklabels(["Na", "CaT", "CaS", "A", "KCa", "Kd", "H", "leak"])
    ax.set_xlim(0.5, 8.5)
    if not plot_labels:
        ax.set_xticks([])

    if legend:
        ax.legend(
            ["AB/PD", "LP", "PY"],
            bbox_to_anchor=(1.3, 0.8 + legend_y_offset),
            handlelength=0.3,
            handletextpad=0.35,
        )

    ax.text(3.3 + title_x_offset, 0.8, title)

    if ylabel is not None:
        ax.set_ylabel(ylabel)


def plot_eigenvalues(
    cum_grad,
    figsize,
    ylabel="log(Eigenvalue)",
    color="#045a8d",
    title=None,
    xlabel=True,
):
    fig, ax = plt.subplots(1, figsize=figsize)

    _ = ax.bar(np.arange(len(cum_grad)), cum_grad, width=0.5, color=color)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_ylabel(ylabel)

    if xlabel:
        ax.set_xlabel("Dimension")
    else:
        ax.set_xticks([])

    if title is not None:
        ax.set_title(title)


def sensitivity_hist(shift_in_mean_normalized, figsize):
    fig, ax = plt.subplots(1, 4, figsize=figsize)
    ax[0].bar(np.arange(8), shift_in_mean_normalized[:8])
    ax[1].bar(np.arange(8), shift_in_mean_normalized[8:16], color="orange")
    ax[2].bar(np.arange(8), shift_in_mean_normalized[16:24], color="g")
    ax[3].bar(np.arange(7), shift_in_mean_normalized[24:], color="k")

    for i, a in enumerate(ax):
        a.set_ylim(-1, 1)
        a.set_xticks(np.arange(8))
        if i < 3:
            a.set_xticklabels(["Na", "CaT", "CaS", "A", "KCa", "Kd", "H", "Leak"])
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    ax[0].set_ylabel("Influence on energy")


def oneDmarginal(samples, points=[], **kwargs):
    opts = {
        # what to plot on triagonal and diagonal subplots
        "upper": "hist",  # hist/scatter/None/cond
        "diag": "hist",  # hist/None/cond
        # 'lower': None,     # hist/scatter/None  # TODO: implement
        # title and legend
        "title": None,
        "legend": False,
        # labels
        "labels": [],  # for dimensions
        "labels_points": [],  # for points
        "labels_samples": [],  # for samples
        "labelpad": None,
        # colors
        "samples_colors": plt.rcParams["axes.prop_cycle"].by_key()["color"],
        "points_colors": plt.rcParams["axes.prop_cycle"].by_key()["color"],
        # subset
        "subset": None,
        # conditional posterior requires condition and pdf1
        "pdfs": None,
        "condition": None,
        # axes limits
        "limits": [],
        # ticks
        "ticks": [],
        "tickformatter": mpl.ticker.FormatStrFormatter("%g"),
        "tick_labels": None,
        "tick_labelpad": None,
        # options for hist
        "hist_diag": {"alpha": 1.0, "bins": 25, "density": False, "histtype": "step"},
        # options for kde
        "kde_diag": {"bw_method": "scott", "bins": 100, "color": "black"},
        # options for contour
        "contour_offdiag": {"levels": [0.68]},
        # options for scatter
        "scatter_offdiag": {
            "alpha": 0.5,
            "edgecolor": "none",
            "rasterized": False,
        },
        # options for plot
        "plot_offdiag": {},
        # formatting points (scale, markers)
        "points_diag": {},
        "points_offdiag": {
            "marker": ".",
            "markersize": 20,
        },
        # matplotlib style
        "style": "../../.matplotlibrc",
        # other options
        "fig_size": (10, 10),
        "fig_bg_colors": {"upper": None, "diag": None, "lower": None},
        "fig_subplots_adjust": {
            "top": 0.9,
        },
        "subplots": {},
        "despine": {
            "offset": 5,
        },
        "title_format": {"fontsize": 16},
    }

    oneDmarginal.defaults = opts.copy()
    opts = _update(opts, kwargs)

    # Prepare samples
    if type(samples) != list:
        samples = [samples]

    # Prepare points
    if type(points) != list:
        points = [points]
    points = [np.atleast_2d(p) for p in points]

    # Dimensions
    dim = samples[0].shape[1]
    num_samples = samples[0].shape[0]

    # TODO: add asserts checking compatiblity of dimensions

    # Prepare labels
    if opts["labels"] == [] or opts["labels"] is None:
        labels_dim = ["dim {}".format(i + 1) for i in range(dim)]
    else:
        labels_dim = opts["labels"]

    # Prepare limits
    if opts["limits"] == [] or opts["limits"] is None:
        limits = []
        for d in range(dim):
            min = +np.inf
            max = -np.inf
            for sample in samples:
                min_ = sample[:, d].min()
                min = min_ if min_ < min else min
                max_ = sample[:, d].max()
                max = max_ if max_ > max else max
            limits.append([min, max])
    else:
        if len(opts["limits"]) == 1:
            limits = [opts["limits"][0] for _ in range(dim)]
        else:
            limits = opts["limits"]

    # Prepare ticks
    if opts["ticks"] == [] or opts["ticks"] is None:
        ticks = None
    else:
        if len(opts["ticks"]) == 1:
            ticks = [opts["ticks"][0] for _ in range(dim)]
        else:
            ticks = opts["ticks"]

    # Prepare diag/upper/lower
    if type(opts["diag"]) is not list:
        opts["diag"] = [opts["diag"] for _ in range(len(samples))]
    if type(opts["upper"]) is not list:
        opts["upper"] = [opts["upper"] for _ in range(len(samples))]
    # if type(opts['lower']) is not list:
    #    opts['lower'] = [opts['lower'] for _ in range(len(samples))]
    opts["lower"] = None

    # Style
    if opts["style"] in ["dark", "light"]:
        style = os.path.join(
            os.path.dirname(__file__), "matplotlib_{}.style".format(opts["style"])
        )
    else:
        style = opts["style"]

    # Apply custom style as context
    with mpl.rc_context(fname=style):

        # Figure out if we subset the plot
        subset = opts["subset"]
        if subset is None:
            rows = cols = dim
            subset = [i for i in range(dim)]
        else:
            if type(subset) == int:
                subset = [subset]
            elif type(subset) == list:
                pass
            else:
                raise NotImplementedError
            rows = cols = len(subset)

        fig, axes = plt.subplots(1, cols, figsize=opts["fig_size"], **opts["subplots"])

        # Style figure
        fig.subplots_adjust(**opts["fig_subplots_adjust"])
        fig.suptitle(opts["title"], **opts["title_format"])

        col_idx = -1
        for col in range(dim):
            if col not in subset:
                continue
            else:
                col_idx += 1

            current = "diag"

            ax = axes[col_idx]
            plt.sca(ax)

            # Background color
            if (
                current in opts["fig_bg_colors"]
                and opts["fig_bg_colors"][current] is not None
            ):
                ax.set_facecolor(opts["fig_bg_colors"][current])

            # Axes
            if opts[current] is None:
                ax.axis("off")
                continue

            # Limits
            if limits is not None:
                ax.set_xlim((limits[col][0], limits[col][1]))
                if current != "diag":
                    ax.set_ylim((limits[row][0], limits[row][1]))
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()

            # Ticks
            if ticks is not None:
                ax.set_xticks((ticks[col][0], ticks[col][1]))
                if current != "diag":
                    ax.set_yticks((ticks[row][0], ticks[row][1]))

            # Despine
            sns.despine(ax=ax, **opts["despine"])

            # Formatting axes
            if opts["lower"] is None or col == dim - 1:
                _format_axis(
                    ax,
                    xhide=False,
                    xlabel=labels_dim[col],
                    yhide=True,
                    tickformatter=opts["tickformatter"],
                )
                if opts["labelpad"] is not None:
                    ax.xaxis.labelpad = opts["labelpad"]
            else:
                _format_axis(ax, xhide=True, yhide=True)

            if opts["tick_labels"] is not None:
                ax.set_xticklabels(
                    (str(opts["tick_labels"][col][0]), str(opts["tick_labels"][col][1]))
                )
                if opts["tick_labelpad"] is not None:
                    ax.tick_params(axis="x", which="major", pad=opts["tick_labelpad"])

            # Diagonals
            if len(samples) > 0:
                for n, v in enumerate(samples):
                    if opts["diag"][n] == "hist":
                        h = plt.hist(
                            v[:, col],
                            color=opts["samples_colors"][n],
                            **opts["hist_diag"]
                        )
                    elif opts["diag"][n] == "kde":
                        density = gaussian_kde(
                            v[:, col], bw_method=opts["kde_diag"]["bw_method"]
                        )
                        xs = np.linspace(xmin, xmax, opts["kde_diag"]["bins"])
                        ys = density(xs)
                        h = plt.plot(
                            xs,
                            ys,
                            color=opts["samples_colors"][n],
                        )
                    else:
                        pass

            if len(points) > 0:
                extent = ax.get_ylim()
                for n, v in enumerate(points):
                    h = plt.plot(
                        [v[:, col], v[:, col]],
                        extent,
                        color=opts["points_colors"][n],
                        **opts["points_diag"]
                    )

        if len(subset) < dim:
            ax = axes[-1]
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            text_kwargs = {"fontsize": plt.rcParams["font.size"] * 2.0}
            ax.text(x1 + (x1 - x0) / 8.0, (y0 + y1) / 2.0, "...", **text_kwargs)

    return fig, axes
