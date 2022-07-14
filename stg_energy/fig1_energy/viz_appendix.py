import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np


def generate_figure_small(
    data1,
    time_vec,
    cond_ind,
    reversal_potential,
    start=36000,
    end=38000,
    ind1=0,
    ind2=1,
    ymax=80000,
    name="1",
):
    time_vector = time_vec[: end - start]

    voltage0, power_power0, current_power0 = energy_of_current(
        data1, ind1, cond_ind, reversal_potential, start, end
    )
    ratio0 = np.sum(power_power0) / np.sum(current_power0)

    voltage1, power_power1, current_power1 = energy_of_current(
        data1, ind2, cond_ind, reversal_potential, start, end
    )
    ratio1 = np.sum(power_power1) / np.sum(current_power1)

    with mpl.rc_context(fname="../../../.matplotlibrc"):

        col_current = "g"
        col_power = "#fc8d59"

        fig, ax = plt.subplots(2, 2, figsize=(6, 4.5))

        ax[0, 0].plot(time_vector, voltage0, c="k")
        ax[0, 0].scatter([7], [43], c="#d7301f")
        ax[0, 0].set_title("Configuration 1")
        ax[0, 0].set_ylabel("Voltage")

        ax[0, 1].plot(time_vector, voltage1, c="k")
        ax[0, 1].scatter([7], [43], c="#0570b0")
        ax[0, 1].set_title("Configuration 2")

        for ind_ in range(2):
            ax[0, ind_].set_ylim([-70, 50])
            ax[0, ind_].set_xticks([0, 30])
            ax[0, ind_].set_xticklabels([])

        ax[1, 0].plot(time_vector, power_power0, c=col_current)
        ax[1, 0].plot(time_vector, current_power0 * 50, c=col_power)

        ax[1, 1].plot(time_vector, power_power1, c=col_current)
        ax[1, 1].plot(time_vector, current_power1 * 50, c=col_power)

        for ind_ in range(2):
            ax[1, ind_].set_xlabel("Time (ms)")
            ax[1, ind_].set_ylim([0, ymax])
            ax[1, ind_].set_yticks([0, ymax])
            ax[1, ind_].set_xticks([0, 30])

        for x_ in range(2):
            for y_ in range(2):
                ax[x_, y_].set_xlim([0, 30])

        plt.savefig(f"../svg/panel_explanation_of_proportionality_b{name}.svg")

        return ratio0, ratio1


def energy_of_current(
    data, index, conductance_ind, reversal_potential, start, end, neuron_ind=0
):

    voltage = data[index]["voltage"][neuron_ind, start:end]
    conductance = data[index]["membrane_conds"][neuron_ind, conductance_ind, start:end]

    current_square = (voltage - reversal_potential) ** 2
    current = np.abs(voltage - reversal_potential)
    power_ = current_square * conductance
    current_ = current * conductance
    return voltage, power_, current_


# def energy_extracted_from_sim(cond_ind, reversal_potential, ind1=0, ind2=1):
#     def energy1_of_ind(ind):
#         sod1 = data1[ind]["membrane_conds"][0, cond_ind, 30000:300000]
#         v1 = data1[ind]["voltage"][0, 30000:300000]
#         sod_energy1 = np.sum(sod1 * (v1 - reversal_potential) ** 2)
#         return sod_energy1

#     def energy2_of_ind(ind):
#         sod1 = data2[ind]["membrane_conds"][0, cond_ind, 30000:300000]
#         v1 = data1[ind]["voltage"][0, 30000:300000]
#         sod_energy2 = np.sum(sod1 * np.abs(v1 - reversal_potential) * 50)
#         return sod_energy2

#     e1 = np.asarray([energy1_of_ind(i) for i in range(n_samp)])
#     e2 = np.asarray([energy2_of_ind(i) for i in range(n_samp)])

#     ratio = e1 / e2
#     print("Ratio of ratios", ratio[ind1] / ratio[ind2])
#     return e1, e2


# def generate_figure(
#     cond_ind, reversal_potential, start=36000, end=38000, ind1=0, ind2=1
# ):

#     v1 = data1[ind1]["voltage"][0, start:end]
#     g1 = data1[ind1]["membrane_conds"][0, cond_ind, start:end]

#     v2 = data1[ind2]["voltage"][0, start:end]
#     g2 = data1[ind2]["membrane_conds"][0, cond_ind, start:end]

#     time_vector = time_vec[: end - start]

#     with mpl.rc_context(fname="../../../.matplotlibrc"):

#         col_current = "g"
#         col_power = "#fc8d59"

#         fig, ax = plt.subplots(4, 2, figsize=(6, 4.5))
#         ax[0, 0].plot(time_vector, v1, c="k")
#         ax[0, 0].scatter([7], [43], c="#d7301f")
#         ax[0, 1].scatter([7], [43], c="#0570b0")
#         ax[0, 0].set_title("Configuration 1")
#         ax[0, 0].set_ylabel("Voltage")
#         ax[0, 0].set_ylim([-80, 50])
#         ax[0, 0].set_xticks([0, 30])
#         ax[0, 0].set_xticklabels([])

#         ax[1, 0].plot(time_vector, g1, c="k")
#         ax[1, 0].set_ylabel(r"$\overline{g}_{\mathrm{Na}}$")
#         ax[1, 0].set_ylim([0, 55])
#         ax[1, 0].set_yticks([0, 55])
#         ax[1, 0].set_xticks([0, 30])
#         ax[1, 0].set_xticklabels([])

#         print("reversal_potential", reversal_potential)

#         current_square = (v1 - reversal_potential) ** 2
#         current = np.abs(v1 - reversal_potential) * 50
#         ax[2, 0].plot(time_vector, current_square, c=col_current)
#         ax[2, 0].plot(time_vector, current, c=col_power)
#         #         ax[2, 0].legend([r"$(V-E_{\mathrm{Na}})^2$", r"$100 \cdot (V-E_{\mathrm{Na}})$"], handlelength=0.7, handletextpad=0.2)
#         ax[2, 0].set_ylim([-1000, 15000])
#         ax[2, 0].set_yticks([-1000, 15000])
#         ax[2, 0].set_xticks([0, 30])
#         ax[2, 0].set_xticklabels([])
#         power_power = current_square * g1
#         current_power = current * g1
#         ax[3, 0].plot(time_vector, power_power, c=col_current)
#         ax[3, 0].plot(time_vector, current_power, c=col_power)
#         #         ax[3, 0].legend([r"$\overline{g}_{\mathrm{Na}} (V-E_{\mathrm{Na}})^2$", r"$100 \cdot \overline{g}_{\mathrm{Na}} (V-E_{\mathrm{Na}})$"], handlelength=0.7, handletextpad=0.2)
#         ratio0 = np.sum(power_power) / np.sum(current_power)
#         ax[3, 0].set_xlabel("Time (ms)")
#         ax[3, 0].set_ylim([0, 500000])
#         ax[3, 0].set_yticks([0, 500000])
#         ax[3, 0].set_xticks([0, 30])

#         ax[0, 1].plot(time_vector, v2, c="k")
#         ax[0, 1].set_title("Configuration 2")
#         ax[0, 1].set_ylim([-80, 50])
#         ax[0, 1].set_xticks([0, 30])
#         ax[0, 1].set_xticklabels([])
#         ax[1, 1].plot(time_vector, g2, c="k")
#         ax[1, 1].set_ylim([0, 55])
#         ax[1, 1].set_yticks([0, 55])
#         ax[1, 1].set_xticks([0, 30])
#         ax[1, 1].set_xticklabels([])
#         current_square = (v2 - reversal_potential) ** 2
#         current = np.abs(v2 - reversal_potential) * 50
#         ax[2, 1].plot(time_vector, current_square, c=col_current)
#         ax[2, 1].plot(time_vector, current, c=col_power)
#         ax[2, 1].set_ylim([-1000, 15000])
#         ax[2, 1].set_yticks([-1000, 15000])
#         ax[2, 1].set_xticks([0, 30])
#         ax[2, 1].set_xticklabels([])
#         power_power = current_square * g2
#         current_power = current * g2
#         ax[3, 1].plot(time_vector, power_power, c=col_current)
#         ax[3, 1].plot(time_vector, current_power, c=col_power)
#         ax[3, 1].set_xlabel("Time (ms)")
#         ax[3, 1].set_ylim([0, 500000])
#         ax[3, 1].set_yticks([0, 500000])
#         ax[3, 1].set_xticks([0, 30])

#         for x_ in range(4):
#             for y_ in range(2):
#                 ax[x_, y_].set_xlim([0, 30])

#         plt.savefig("../svg/panel_explanation_of_proportionality_b.svg")

#         ratio1 = np.sum(power_power) / np.sum(current_power)
#         print("Individual ratios", ratio0, ratio1)
#         print("Ratio of ratios", ratio0 / ratio1)
