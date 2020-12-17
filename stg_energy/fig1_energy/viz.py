import numpy as np
import matplotlib.pyplot as plt
from stg_energy.common import col
from pyloric.utils import energy_of_membrane


def compare_voltage_low_and_high_energy_trace(all_out_targets, t, figsize, offset=None):
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    iii = 0
    time_len = int(3 * 1000 / 0.025)  # 3 seconds
    if offset is None:
        offset = [160000, 0]
    print("Showing :  ", time_len / 40000, "seconds")
    print("Scalebar indicates:  50mV")

    for out_target in all_out_targets:

        current_col = 0
        Vx = out_target["voltage"]
        axV = ax[iii]
        for j in range(3):
            if time_len is not None:
                axV.plot(
                    t[:time_len:5] / 1000,
                    Vx[j, 10000 + offset[iii] : 10000 + offset[iii] + time_len : 5]
                    + 130.0 * (2 - j),
                    linewidth=0.6,
                    c="k",
                )
            else:
                axV.plot(t / 1000, Vx[j] + 120.0 * (2 - j), lw=0.6, c="k")
            current_col += 1

        box = axV.get_position()

        axV.set_position([box.x0, box.y0, box.width, box.height])

        axV.spines["right"].set_visible(False)
        axV.spines["top"].set_visible(False)
        axV.set_yticks([])
        if iii == 0:
            axV.set_ylabel("Voltage")
        axV.set_xlabel("Time (seconds)")
        # axV.set_ylabel("Voltage")
        axV.set_ylim([-90, 320])

        # scale bar
        end_val_x = (t[:time_len:5] / 1000)[-1] + 0.1
        axV.plot([end_val_x, end_val_x], [-20, 30], c="k")

        iii += 1

    plt.subplots_adjust(wspace=0.1)


def compare_energy_low_and_high_energy_trace(all_out_targets, t, figsize, offset=None):
    """

    Args:
        all_out_targets: [description]
        t: [description]
        figsize: [description]
        offset: [description]
    """
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    iii = 0
    time_len = int(3 * 1000 / 0.025)  # 3 seconds
    print("Showing :  ", time_len / 40000, "seconds")
    print("Scalebar indicates:  100 micro Joule / second")
    if offset is None:
        offset = [160000, 0]

    for out_target in all_out_targets:
        # start only at 40,000 because of burn-in.
        # divide by 4000 because of stepsize (divison by 40000 = 0.025ms stepsize).
        axS = ax[iii]
        all_energies = np.asarray(energy_of_membrane(out_target))

        for current_current in range(3):
            all_currents_PD = all_energies[:, current_current, :]
            summed_currents_include = (
                np.sum(
                    all_currents_PD[
                        :, 10000 + offset[iii] : 10000 + offset[iii] + time_len : 5
                    ]
                    / 1000,
                    axis=0,
                )
                + 660
                - 330 * current_current
            )
            axS.plot(
                t[:time_len:5] / 1000,
                summed_currents_include,
                color="#666666",
                linewidth=0.6,
            )

        axS.spines["right"].set_visible(False)
        axS.spines["top"].set_visible(False)
        axS.set_ylim([0, 990])

        end_val_x = (t[:time_len:5] / 1000)[-1] + 0.1
        axS.plot([end_val_x, end_val_x], [40, 140], c="k")
        axS.set_yticks([])
        if iii == 0:
            axS.set_ylabel("Energy")
        iii += 1
        axS.set_xlabel("Time (seconds)")

    plt.subplots_adjust(wspace=0.1)


def energy_scape_voltage(all_out_targets, t, figsize, cols, offset=None):
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    iii = 0
    time_len = int(3 * 1000 / 0.025 * 0.015)  # 45 ms
    if offset is None:
        offset = [370000, 189010]

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
        axV = ax[iii]
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

        box = axV.get_position()

        axV.set_position([box.x0, box.y0, box.width, box.height])
        axV.set_ylabel("Voltage PY (mV)")
        axV.tick_params(axis="both", which="major")
        axV.set_ylim([-65, 50])
        axV.set_xticks([])

        axV.spines["right"].set_visible(False)
        axV.spines["top"].set_visible(False)

        plt.subplots_adjust(wspace=0.05)

        if iii == 1:
            axV.set_yticks([])
            axV.set_ylabel("")
            axV.spines["left"].set_visible(False)

        iii += 1

    plt.subplots_adjust(wspace=0.3, hspace=0.3)


def energy_scape_energy(
    all_out_targets,
    t,
    figsize,
    offset=None,
    neuron_to_inspect=2,
    set_xlim=True,
):
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    iii = 0
    time_len = int(3 * 1000 / 0.025 * 0.015)  # 45 ms
    print("time_len", time_len)
    if offset is None:
        offset = [370000, 189010]

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

        axS = ax[iii]
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
        axS.spines["right"].set_visible(False)
        axS.spines["top"].set_visible(False)
        axS.set_xlabel("Time (ms)")
        axS.set_ylabel("Energy PY ($\mu$J/s)")

        if set_xlim:
            axS.set_ylim([0, 270])
            axS.set_xlim([0, 40])
        axS.tick_params(axis="both", which="major")
        if iii == 1:
            axS.legend(
                ("Na", "CaT", "CaS", "A", "KCa", "Kd", "H", "Leak"),
                bbox_to_anchor=(1.17, 1.6),
                ncol=4,
            )
            axS.set_yticks([])
            axS.set_ylabel("")
            axS.spines["left"].set_visible(False)
        iii += 1

    plt.subplots_adjust(wspace=0.2)


def energy_scape_energy_cosyne(
    all_out_targets,
    t_experimental,
    t,
    figsize,
    cols,
    offset=None,
    neuron_to_inspect=2,
    time_len_multiplier=1.0,
    set_xlim=True,
):
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    iii = 0
    time_len = int(
        400
        * (t_experimental[1] - t_experimental[0])
        / 0.025
        / 1e-3
        * time_len_multiplier
    )
    print("time_len", time_len)
    if offset is None:
        offset = [170000, 189010]

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

        axS = ax[iii]
        all_energies = out_target["all_energies"]
        points = []
        for c in cols_hex:
            p = axS.scatter(1e6, 1e6, c=c)
            points.append(p)

        for current_current in range(neuron_to_inspect, neuron_to_inspect + 1):
            # times 10 because: times 10000 for cm**2, but /1000 for micro from nano J
            all_currents_PD = all_energies[:, current_current, :] * 10 / 1000

            summed_currents_include = (
                np.sum(
                    all_currents_PD[
                        :, 10000 + offset[iii] : 10000 + offset[iii] + time_len : 5
                    ],
                    axis=0,
                )
                + 460
                - 230 * current_current
            )
            for i in range(8):
                summed_currents_until = (
                    np.sum(
                        all_currents_PD[
                            :i, 10000 + offset[iii] : 10000 + offset[iii] + time_len : 5
                        ],
                        axis=0,
                    )
                    + 460
                    - 230 * current_current
                )
                summed_currents_include = (
                    np.sum(
                        all_currents_PD[
                            : i + 1,
                            10000 + offset[iii] : 10000 + offset[iii] + time_len : 5,
                        ],
                        axis=0,
                    )
                    + 460
                    - 230 * current_current
                )
                axS.fill_between(
                    t[:time_len:5],
                    summed_currents_until,
                    summed_currents_include,
                    color=cols_hex[i],
                )

        print("ok")
        Vx = (out_target["data"] + 63) / 60
        for j in range(2, 3):
            if time_len is not None:
                axS.plot(
                    t[:time_len:5],
                    Vx[j, 10000 + offset[iii] : 10000 + offset[iii] + time_len : 5]
                    + 130.0 * (2 - j),
                    lw=0.6,
                    linestyle="--",
                    c="k",
                )
            else:
                axS.plot(t / 1000, Vx[j] + 120.0 * (2 - j), lw=0.6, c="k")

        axS.spines["right"].set_visible(False)
        axS.spines["top"].set_visible(False)
        axS.set_xlabel("Time (ms)")
        axS.set_ylabel("Energy PY (mJ/s)")

        if set_xlim:
            axS.set_ylim([0, 1.850])
            axS.set_xlim([0, 40])
        axS.tick_params(axis="both", which="major")
        if iii == 1:
            axS.legend(
                ("Voltage", "Na", "CaT", "CaS", "A", "KCa", "Kd", "H", "Leak"),
                bbox_to_anchor=(1.0, 1.46),
                ncol=9,
                columnspacing=0.55,
                handletextpad=0.15,
                handlelength=0.7,
            )

            axS.set_yticks([])
            axS.set_ylabel("")
            axS.spines["left"].set_visible(False)
        iii += 1

    plt.subplots_adjust(wspace=0.2)
