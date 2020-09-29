import numpy as np
import matplotlib.pyplot as plt
from stg_energy.common import col


def compare_voltage_low_and_high_energy_trace(
    all_out_targets, t_experimental, t, cols, figsize
):
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    iii = 0
    time_len = int(30000 * (t_experimental[1] - t_experimental[0]) / 0.025 / 1e-3)
    offset = [160000, 0]
    print("Showing :  ", time_len / 40000, "seconds")
    print("Scalebar indicates:  50mV")

    for out_target in all_out_targets:

        current_col = 0
        Vx = out_target["data"]
        axV = ax[iii]
        for j in range(3):
            if time_len is not None:
                axV.plot(
                    t[:time_len:5] / 1000,
                    Vx[j, 10000 + offset[iii] : 10000 + offset[iii] + time_len : 5]
                    + 130.0 * (2 - j),
                    linewidth=0.6,
                    c=cols[iii],
                )
            else:
                axV.plot(t / 1000, Vx[j] + 120.0 * (2 - j), lw=0.6, c=cols[iii])
            current_col += 1

        box = axV.get_position()

        axV.set_position([box.x0, box.y0, box.width, box.height])
        axV.axes.get_yaxis().set_ticks([])

        axV.spines["right"].set_visible(False)
        axV.spines["top"].set_visible(False)
        axV.spines["left"].set_visible(False)
        axV.spines["bottom"].set_visible(False)
        # axV.set_ylabel("Voltage")
        axV.set_xticks([])
        axV.set_ylim([-80, 320])

        end_val_x = (t[:time_len:5] / 1000)[-1] + 0.1
        axV.plot([end_val_x, end_val_x], [-20, 30], c="k")

        iii += 1

    plt.subplots_adjust(wspace=0.1)


def compare_energy_low_and_high_energy_trace(
    all_out_targets, t_experimental, t, cols, figsize
):
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    iii = 0
    time_len = int(30000 * (t_experimental[1] - t_experimental[0]) / 0.025 / 1e-3)
    print("Showing :  ", time_len / 40000, "seconds")
    print("Scalebar indicates:  1000 micro Joule / second")
    offset = [160000, 0]

    for out_target in all_out_targets:
        # start only at 40,000 because of burn-in.
        # divide by 4000 because of stepsize (divison by 40000 = 0.025ms stepsize).
        axS = ax[iii]
        all_energies = out_target["all_energies"]

        for current_current in range(3):
            all_currents_PD = all_energies[:, current_current, :]
            summed_currents_include = (
                np.sum(
                    all_currents_PD[
                        :, 10000 + offset[iii] : 10000 + offset[iii] + time_len : 5
                    ],
                    axis=0,
                )
                + 400
                - 200 * current_current
            )
            axS.plot(
                t[:time_len:5] / 1000, summed_currents_include, color="k", linewidth=0.6
            )

        axS.spines["right"].set_visible(False)
        axS.spines["top"].set_visible(False)
        axS.spines["bottom"].set_visible(False)
        axS.spines["left"].set_visible(False)
        axS.set_ylim([0, 600])

        end_val_x = (t[:time_len:5] / 1000)[-1] + 0.1
        axS.plot([end_val_x, end_val_x], [20, 120], c="k")
        # axS.set_ylabel("Instant. Energy")
        axS.set_xticks([])
        axS.set_yticks([])
        iii += 1

        # start only at 40,000 because of burn-in.
        # divide by 4000 because of stepsize (divison by 40000 = 0.025ms stepsize).
        total_energy = np.sum(out_target["energy"][:, 40000:]) / 40000
        axS.set_title("Average Energy: %.1f $\mu$J/s" % total_energy)

    plt.subplots_adjust(wspace=0.1)


def energy_scape_voltage(
    all_out_targets, t_experimental, t, figsize, cols,
):
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    iii = 0
    time_len = int(400 * (t_experimental[1] - t_experimental[0]) / 0.025 / 1e-3)
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

        current_col = 0
        Vx = out_target["data"]
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
    all_out_targets, t_experimental, t, figsize, cols,
):
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    iii = 0
    time_len = int(400 * (t_experimental[1] - t_experimental[0]) / 0.025 / 1e-3)
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

        for current_current in range(2, 3):
            # times 10 because: times 10000 for cm**2, but /1000 for micro from nano J
            all_currents_PD = all_energies[:, current_current, :] * 10

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
        axS.spines["right"].set_visible(False)
        axS.spines["top"].set_visible(False)
        axS.set_xlabel("Time (ms)")
        axS.set_ylabel("Energy PY ($\mu$J)")
        axS.set_ylim([0, 1500])
        axS.set_xlim([0, 40])
        axS.tick_params(axis="both", which="major")
        if iii == 1:
            axS.legend(
                ("Na", "CaT", "CaS", "A", "KCa", "Kd", "H", "Leak"),
                bbox_to_anchor=(0.95, 1.6),
                ncol=4,
            )
            axS.set_yticks([])
            axS.set_ylabel("")
            axS.spines["left"].set_visible(False)
        iii += 1

    plt.subplots_adjust(wspace=0.2)
