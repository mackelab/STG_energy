import numpy as np
import matplotlib.pyplot as plt
from stg_energy.common import col
from pyloric.utils import energy_of_membrane


def energy_scape_both(
    all_out_targets,
    t,
    figsize,
    offset=None,
    neuron_to_inspect=2,
    set_xlim=True,
    cols=None,
    max_dur=40.0,
):

    fig, ax = plt.subplots(2, 2, figsize=figsize)
    iii = 0
    time_len = int(3 * 1000 / 0.025 * 0.015 / 40.0 * max_dur)  # 45 ms
    if offset is None:
        offset = [277500, 225100]

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

    for out_target in all_out_targets:

        current_col = 0
        Vx = out_target["voltage"]
        axV = ax[0, iii]
        for j in range(2, 3):
            if time_len is not None:
                axV.plot(
                    t[:time_len:5],
                    Vx[j, 10000 + offset[iii] : 10000 + offset[iii] + time_len : 5]
                    + 130.0 * (2 - j),
                    lw=1.6,
                    c=cols[iii],
                )
            else:
                axV.plot(t / 1000, Vx[j] + 120.0 * (2 - j), lw=0.6, c=cols[iii])
            current_col += 1

        axV.set_ylabel("Voltage", labelpad=5)
        axV.set_ylim([-65, 50])
        axV.set_xlim([0, 40])
        axV.set_xticks([])

        axV.spines["right"].set_visible(False)
        axV.spines["top"].set_visible(False)

        if iii == 1:
            axV.set_yticks([])
            axV.set_ylabel("")
            axV.spines["left"].set_visible(False)
        axV.set_yticks([])

        iii += 1

    iii = 0
    if offset is None:
        offset = [277500, 225100]

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

    for out_target in all_out_targets:

        axS = ax[1, iii]
        all_energies = np.asarray(energy_of_membrane(out_target))

        for current_current in range(neuron_to_inspect, neuron_to_inspect + 1):
            all_currents_PD = all_energies[:, current_current, :]

            summed_currents_include = (
                np.sum(
                    all_currents_PD[
                        :, 10000 + offset[iii] : 10000 + offset[iii] + time_len : 5
                    ],
                    axis=0,
                )
                / 1000
                + 68
                - 34 * current_current
            )
            for i in range(8):
                summed_currents_until = (
                    np.sum(
                        all_currents_PD[
                            :i, 10000 + offset[iii] : 10000 + offset[iii] + time_len : 5
                        ],
                        axis=0,
                    )
                    / 1000
                    + 68
                    - 34 * current_current
                )
                summed_currents_include = (
                    np.sum(
                        all_currents_PD[
                            : i + 1,
                            10000 + offset[iii] : 10000 + offset[iii] + time_len : 5,
                        ],
                        axis=0,
                    )
                    / 1000
                    + 68
                    - 34 * current_current
                )
                axS.fill_between(
                    t[:time_len:5],
                    summed_currents_until,
                    summed_currents_include,
                    color=cols_hex[i],
                )
        axS.set_xlabel("Time")
        axS.set_ylabel("Energy", labelpad=5.7)

        if set_xlim:
            axS.set_ylim([0, 200])
        axS.set_xlim([0, 40])
        if iii == 1:
            axS.set_yticks([])
            axS.set_ylabel("")
            axS.spines["left"].set_visible(False)
        axS.set_xticks([])
        axS.set_yticks([])
        iii += 1

    newax = fig.add_axes([0.6, 0.1, 0.8, 0.8], anchor="NE", zorder=-1)
    from matplotlib.cbook import get_sample_data

    image_file = "/home/michael/Documents/STG_energy/paper/tweet/fig/lobster.png"

    im = plt.imread(get_sample_data(image_file))

    newax.imshow(im)
    newax.axis("off")

    newax.text(
        0,
        760,
        "Drawing: Marder & Bucher, 2007",
        fontdict={"fontsize": 5, "c": "grey"},
    )

    axS.text(-46.5, 54.5, "Na", fontdict={"fontsize": 8, "c": "#1b9e77"}, zorder=-1)
    axS.text(-46.5, 109.5, "Ca", fontdict={"fontsize": 8, "c": "#d95f02"}, zorder=-1)
    axS.text(-46.5, 164.5, "K", fontdict={"fontsize": 8, "c": "#e6ab02"}, zorder=-1)

    plt.subplots_adjust(wspace=0.2)
