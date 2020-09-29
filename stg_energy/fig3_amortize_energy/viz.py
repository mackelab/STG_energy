import numpy as np
import matplotlib.pyplot as plt
import sys

from stg_energy.common import col, _update, _format_axis
from scipy.stats import gaussian_kde
import os
import matplotlib as mpl
import seaborn as sns


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
                t, Vx[j] + 120.0 * (2 - j), lw=0.3, c=col[current_col],
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


def energy_scape(
    out_target,
    t,
    figsize,
    cols,
    time_len,
    offset,
    ylimE=[0, 2000],
    v_labelpad=4.8,
    neuron=2,
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

    names = ["PM", "LP", "PY"]

    current_col = 0
    Vx = out_target["data"]
    axV = ax[0]
    for j in range(neuron, neuron + 1):
        if time_len is not None:
            axV.plot(
                t[:time_len:5],
                Vx[j, 10000 + offset : 10000 + offset + time_len : 5] + 130.0 * (2 - j),
                lw=0.6,
                c=cols[iii],
            )
        else:
            axV.plot(t / 1000, Vx[j] + 120.0 * (2 - j), lw=0.6, c=cols[iii])
        current_col += 1

    box = axV.get_position()

    axV.set_position([box.x0, box.y0, box.width, box.height])
    axV.set_ylabel("V (" + names[neuron] + ")\n (mV)", labelpad=v_labelpad)
    axV.tick_params(axis="both", which="major")

    axV.spines["right"].set_visible(False)
    axV.spines["top"].set_visible(False)

    axV.set_xticks([])

    plt.subplots_adjust(wspace=0.05)

    axS = ax[1]
    all_energies = out_target["all_energies"]

    for current_current in range(neuron, neuron + 1):
        all_currents_PD = all_energies[:, current_current, :]

        for i in range(8):
            # times 10 because: times 10000 for cm**2, but /1000 for micro from nano J
            summed_currents_until = (
                np.sum(
                    all_currents_PD[:i, 10000 + offset : 10000 + offset + time_len : 5],
                    axis=0,
                )
                * 10
            )
            summed_currents_include = np.sum(
                all_currents_PD[
                    : i + 1, 10000 + offset : 10000 + offset + time_len : 5,
                ]
                * 10,
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
    axS.set_ylabel("E (" + names[neuron] + ")\n $(\mu$J/s)", labelpad=1)
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


def py_sensitivity_bars(
    cum_grad, ylim, figsize, ylabel=None, plot_labels=True, color="#2ca25f"
):
    fig, ax = plt.subplots(1, figsize=figsize)

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

    ax.set_xticks(range(1, 9))
    ax.set_xticklabels(["Na", "CaT", "CaS", "A", "KCa", "Kd", "H", "leak"], rotation=45)
    ax.set_xlim(0.5, 8.5)
    if not plot_labels:
        ax.spines["left"].set_visible(False)
        ax.set_yticks([])
        ax.set_xlim(0.7, 8.3)

    if ylabel is not None:
        ax.set_ylabel(ylabel)


def synapse_sensitivity_bars(
    cum_grad, ylim, figsize, ylabel=None, plot_labels=True, color="#2ca25f"
):
    fig, ax = plt.subplots(1, figsize=figsize)

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
        ax.spines["left"].set_visible(False)
        ax.set_yticks([])
        ax.set_xlim(0.7, 7.3)

    if ylabel is not None:
        ax.set_ylabel(ylabel)


def plot_eigenvalues(cum_grad, figsize, ylabel="log(Eigenvalue)", color="#045a8d"):
    fig, ax = plt.subplots(1, figsize=figsize)

    _ = ax.bar(np.arange(len(cum_grad)), cum_grad, width=0.5, color=color)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xlabel("Dimension")
    ax.set_ylabel(ylabel)


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
        "scatter_offdiag": {"alpha": 0.5, "edgecolor": "none", "rasterized": False,},
        # options for plot
        "plot_offdiag": {},
        # formatting points (scale, markers)
        "points_diag": {},
        "points_offdiag": {"marker": ".", "markersize": 20,},
        # matplotlib style
        "style": "../../.matplotlibrc",
        # other options
        "fig_size": (10, 10),
        "fig_bg_colors": {"upper": None, "diag": None, "lower": None},
        "fig_subplots_adjust": {"top": 0.9,},
        "subplots": {},
        "despine": {"offset": 5,},
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
                        h = plt.plot(xs, ys, color=opts["samples_colors"][n],)
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
