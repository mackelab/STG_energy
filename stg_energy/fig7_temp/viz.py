import numpy as np
from stg_energy.common import pick_synapse
import matplotlib.pyplot as plt


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


def vis_sample_plain(
    voltage_trace,
    t,
    axV,
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

    current_counter = 0

    data = voltage_trace

    Vx = data["data"]

    current_col = 0
    for j in range(len(neutypes)):
        if time_len is not None:
            axV.plot(
                t[:time_len:5] / 1000,
                Vx[j, 10000 + offset : 10000 + offset + time_len : 5] + 130.0 * (2 - j),
                label="",
                lw=0.6,
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
    #
    # if print_label:
    #     axV.plot([1100.0 + (offset - 26500) * (t[1] - t[0])], [300], color=col, marker='o',
    #              markeredgecolor='w', ms=8,
    #              markeredgewidth=1.0, path_effects=[pe.Stroke(linewidth=1.3, foreground='k'), pe.Normal()])

    box = axV.get_position()
    axV.set_position([box.x0, box.y0, box.width, box.height])
    axV.axes.get_yaxis().set_ticks([])
    axV.axes.get_xaxis().set_ticks([])

    axV.spines["right"].set_visible(False)
    axV.spines["top"].set_visible(False)
    axV.spines["bottom"].set_visible(False)
    axV.spines["left"].set_visible(False)
