import numpy as np
import matplotlib.pyplot as plt
import sys; sys.path.append('../../../')
from common import col

def compare_low_and_high_energy_trace(
        all_out_targets,
        t_experimental,
        t,
        cols,
        figsize
):
    fig, ax = plt.subplots(2, 2, figsize=figsize)
    iii = 0
    time_len = int(30000 * (t_experimental[1] - t_experimental[0]) / 0.025 / 1e-3)
    offset = [160000, 0]

    for out_target in all_out_targets:
        total_energy = np.sum(out_target['energy'])

        current_col = 0
        Vx = out_target['data']
        axV = ax[0, iii]
        for j in range(3):
            if time_len is not None:
                axV.plot(t[:time_len:5] / 1000,
                         Vx[j, 10000+offset[iii] : 10000+offset[iii]+time_len : 5] + 130.0 * (2 - j),
                         lw=0.6, c=cols[iii])
            else:
                axV.plot(t / 1000, Vx[j] + 120.0 * (2 - j), lw=0.6, c=cols[iii])
            current_col += 1

        box = axV.get_position()

        axV.set_position([box.x0, box.y0, box.width, box.height])
        axV.axes.get_yaxis().set_ticks([])
        axV.set_xlabel('time [seconds]')

        axV.spines['right'].set_visible(False)
        axV.spines['top'].set_visible(False)
        axV.set_xlabel('Time (s)')
        axV.set_ylabel('Voltage (mV)')
        axV.tick_params(axis='both', which='major')

        plt.subplots_adjust(wspace=0.05)

        axV.set_title('Total energy:  {}.0'.format(int(total_energy / 10000)))

        axS = ax[1, iii]
        all_energies = out_target['all_energies']

        for current_current in range(3):
            all_currents_PD = all_energies[:, current_current, :]
            summed_currents_include = np.sum(all_currents_PD[:,
                                             10000 + offset[iii]:10000 + offset[
                                                 iii] + time_len:5],
                                             axis=0) + 400 - 200 * current_current
            zero_val = np.zeros_like(
                summed_currents_include) + 400 - 200 * current_current
            axS.fill_between(t[:time_len:5] / 1000, zero_val, summed_currents_include,
                             color=col['GT'])

        axS.spines['right'].set_visible(False)
        axS.spines['top'].set_visible(False)
        axS.set_xlabel('Time (s)')
        axS.set_ylabel('Energy')
        axS.set_ylim([0, 600])
        axS.axes.get_yaxis().set_ticks([])
        axS.tick_params(axis='both', which='major')
        iii += 1

    plt.subplots_adjust(wspace=0.3, hspace=0.3)


def energy_scape(
        all_out_targets,
        t_experimental,
        t,
        figsize,
        cols,
):
    fig, ax = plt.subplots(2, 2, figsize=figsize)
    iii = 0
    time_len = int(400 * (t_experimental[1] - t_experimental[0]) / 0.025 / 1e-3)
    offset = [170700, 190000]

    cols_hex = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02',
                '#a6761d', '#666666']

    for out_target in all_out_targets:

        current_col = 0
        Vx = out_target['data']
        axV = ax[0, iii]
        for j in range(2, 3):
            if time_len is not None:
                axV.plot(t[:time_len:5], Vx[j, 10000 + offset[iii]:10000 + offset[
                    iii] + time_len:5] + 130.0 * (2 - j),
                         lw=1.6, c=cols[iii])
            else:
                axV.plot(t / 1000, Vx[j] + 120.0 * (2 - j), lw=0.6, c=cols[iii])
            current_col += 1

        box = axV.get_position()

        axV.set_position([box.x0, box.y0, box.width, box.height])
        axV.set_xlabel('Time (ms)')
        axV.set_ylabel('Voltage (mV)')
        axV.tick_params(axis='both', which='major')
        axV.set_ylim([-65, 50])

        axV.spines['right'].set_visible(False)
        axV.spines['top'].set_visible(False)

        plt.subplots_adjust(wspace=0.05)

        axS = ax[1, iii]
        all_energies = out_target['all_energies']

        for current_current in range(2, 3):
            all_currents_PD = all_energies[:, current_current, :]

            summed_currents_include = np.sum(all_currents_PD[:,
                                             10000 + offset[iii]:10000 + offset[
                                                 iii] + time_len:5],
                                             axis=0) + 460 - 230 * current_current
            for i in range(8):
                summed_currents_until = np.sum(all_currents_PD[:i,
                                               10000 + offset[iii]:10000 + offset[
                                                   iii] + time_len:5],
                                               axis=0) + 460 - 230 * current_current
                summed_currents_include = np.sum(all_currents_PD[:i + 1,
                                                 10000 + offset[iii]:10000 + offset[
                                                     iii] + time_len:5],
                                                 axis=0) + 460 - 230 * current_current
                axS.fill_between(t[:time_len:5], summed_currents_until,
                                 summed_currents_include, color=cols_hex[i])
        axS.spines['right'].set_visible(False)
        axS.spines['top'].set_visible(False)
        axS.set_xlabel('Time (ms)')
        axS.set_ylabel('Energy')
        axS.set_ylim([0, 150])
        axS.set_xlim([0, 40])
        axS.tick_params(axis='both', which='major')
        if iii == 1:
            axS.legend(('Na', 'CaT', 'CaS', 'A', 'KCa', 'Kd', 'H', 'Leak'),
                       bbox_to_anchor=(1.6, 1.05))
        iii += 1

    plt.subplots_adjust(wspace=0.3, hspace=0.3)