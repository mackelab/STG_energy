import numpy as np
from stg_energy.common import pick_synapse
import matplotlib.pyplot as plt
from typing import Optional


def get_labels_8pt_supp(mathmode=False, include_q10=True):
    membrane_names = [
        [
            r"$\mathdefault{AB-Na}\;$  ",
            r"$\mathdefault{AB-CaT}\;$  ",
            r"$\mathdefault{AB-CaS}\;$  ",
            r"$\mathdefault{AB-A}\;$  ",
            r"$\mathdefault{AB-KCa}\;$  ",
            r"$\mathdefault{AB-Kd}\;$  ",
            r"$\mathdefault{AB-H}\;$  ",
            r"$\mathdefault{AB-leak}\;$  ",
        ],
        [
            r"$\mathdefault{LP-Na}\;$  ",
            r"$\mathdefault{LP-CaT}\;$  ",
            r"$\mathdefault{LP-CaS}\;$  ",
            r"$\mathdefault{LP-A}\;$  ",
            r"$\mathdefault{LP-KCa}\;$  ",
            r"$\mathdefault{LP-Kd}\;$  ",
            r"$\mathdefault{LP-H}\;$  ",
            r"$\mathdefault{LP-leak}\;$  ",
        ],
        [
            r"$\mathdefault{PY-Na}\;$  ",
            r"$\mathdefault{PY-CaT}\;$  ",
            r"$\mathdefault{PY-CaS}\;$  ",
            r"$\mathdefault{PY-A}\;$  ",
            r"$\mathdefault{PY-KCa}\;$  ",
            r"$\mathdefault{PY-Kd}\;$  ",
            r"$\mathdefault{PY-H}\;$  ",
            r"$\mathdefault{PY-leak}\;$  ",
        ],
    ]
    if mathmode:
        membrane_names = [
            [
                r"$\mathrm{AB}_\mathrm{Na}$",
                r"$\mathrm{AB}_\mathrm{CaT}$",
                r"$\mathrm{AB}_\mathrm{CaS}$",
                r"$\mathrm{AB}_\mathrm{A}$",
                r"$\mathrm{AB}_\mathrm{KCa}$",
                r"$\mathrm{AB}_\mathrm{Kd}$",
                r"$\mathrm{AB}_\mathrm{H}$",
                r"$\mathrm{AB}_\mathrm{leak}$",
            ],
            [
                r"$\mathrm{LP}_\mathrm{Na}$",
                r"$\mathrm{LP}_\mathrm{CaT}$",
                r"$\mathrm{LP}_\mathrm{CaS}$",
                r"$\mathrm{LP}_\mathrm{A}$",
                r"$\mathrm{LP}_\mathrm{KCa}$",
                r"$\mathrm{LP}_\mathrm{Kd}$",
                r"$\mathrm{LP}_\mathrm{H}$",
                r"$\mathrm{LP}_\mathrm{leak}$",
            ],
            [
                r"$\mathrm{PY}_\mathrm{Na}$",
                r"$\mathrm{PY}_\mathrm{CaT}$",
                r"$\mathrm{PY}_\mathrm{CaS}$",
                r"$\mathrm{PY}_\mathrm{A}$",
                r"$\mathrm{PY}_\mathrm{KCa}$",
                r"$\mathrm{PY}_\mathrm{Kd}$",
                r"$\mathrm{PY}_\mathrm{H}$",
                r"$\mathrm{PY}_\mathrm{leak}$",
            ],
        ]
    membrane_names = np.asarray(membrane_names)
    relevant_membrane_names = membrane_names
    synapse_names = np.asarray([pick_synapse(num, True) for num in range(7)])
    relevant_labels = np.concatenate((relevant_membrane_names.flatten(), synapse_names))
    # q10_names = [u'Q_{10} g\u0305_{glut}', u'Q_{10} g\u0305_{chol}', r'Q_{10} \tau_{glut}', r'Q_{10} \tau_{chol}']
    if include_q10:
        q10_names = [
            "Q_{10} Na",
            "Q_{10} CaT",
            "Q_{10} CaS",
            "Q_{10} CaA",
            "Q_{10} KCa",
            "Q_{10} Kd",
            "Q_{10} H",
            "Q_{10} leak",
            u"Q_{10} g\u0305_{glut}",
            u"Q_{10} g\u0305_{chol}",
        ]
        relevant_labels = np.concatenate((relevant_labels, q10_names))

    return relevant_labels


neutypes = ["PM", "LP", "PY"]


def py_sensitivity_bars_q10(
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
    fig, ax = plt.subplots(1, figsize=figsize)
    _ = ax.bar(
        np.arange(1, 9) - 0.27,
        vec[:8],
        width=0.4 / figsize[0],
        color="#3182bd",
    )
    _ = ax.bar(
        np.arange(1, 9) - 0.09,
        vec[8:16],
        width=0.4 / figsize[0],
        color="#fc8d59",
    )
    _ = ax.bar(
        np.arange(1, 9) + 0.09,
        vec[16:24],
        width=0.4 / figsize[0],
        color="#2ca25f",
    )
    _ = ax.bar(
        np.arange(1, 9) + 0.27,
        vec[31:39],
        width=0.4 / figsize[0],
        color="k",
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
            ["AB/PD", "LP", "PY", "$Q_{10}$"],
            bbox_to_anchor=(1.3, 0.9 + legend_y_offset),
            handlelength=0.3,
            handletextpad=0.35,
        )

    ax.text(3.3 + title_x_offset, 0.8, title)

    if ylabel is not None:
        ax.set_ylabel(ylabel)


neutypes = ["PM", "LP", "PY"]


def vis_sample_plain(
    voltage_trace,
    t,
    axV,
    t_on=None,
    t_off=None,
    col=["k", "k", "k"],
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

    Vx = data["voltage"]

    current_col = 0
    for j in range(len(neutypes)):
        if time_len is not None:
            axV.plot(
                t[10000 + offset : 10000 + offset + time_len : 5],
                Vx[j, 10000 + offset : 10000 + offset + time_len : 5] + 140.0 * (2 - j),
                label=neutypes[j],
                lw=0.3,
                c=col,
            )
        else:
            axV.plot(
                t,
                Vx[j] + 120.0 * (2 - j),
                label=neutypes[j],
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

    axV.set_ylim([-85, 340])

    current_counter += 1
