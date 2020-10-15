import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe


def viz_path_and_samples_abstract_twoRows(
    traces, t, figsize, mycols=None, offsets=0, time_len=165000
):

    counter = 0

    scalebar = [False, False, True, False, False, False]

    color_mixture1 = 0.33 * np.asarray(list(mycols["CONSISTENT1"])) + 0.67 * np.asarray(
        list(mycols["CONSISTENT2"])
    )
    color_mixture2 = 0.67 * np.asarray(list(mycols["CONSISTENT1"])) + 0.33 * np.asarray(
        list(mycols["CONSISTENT2"])
    )

    cols = [
        mycols["CONSISTENT1"],
        mycols["CONSISTENT2"],
        color_mixture1,
        color_mixture2,
    ]

    gridspec = dict(
        hspace=0.16, wspace=0.16, width_ratios=[1, 1, 1], height_ratios=[1, 1]
    )
    fig, ax = plt.subplots(
        2, 3, facecolor="white", figsize=figsize, gridspec_kw=gridspec
    )

    row_access = 0
    print_label = True

    for col in range(6):

        out_target = traces[col]
        col_access = col % 3

        print(col_access, row_access)
        vis_sample_plain(
            voltage_trace=out_target,
            t=t,
            axV=ax[row_access, col_access],
            time_len=time_len,
            offset=offsets[counter],
            col="k",
            scale_bar=scalebar[counter],
            scale_bar_voltage=scalebar[counter],
            print_label=False,
        )
        if counter == 2:
            row_access += 1

        counter += 1

    for aa in ax:
        for a in aa:
            a.spines["right"].set_visible(False)
            a.spines["top"].set_visible(False)
            a.spines["bottom"].set_visible(False)
            a.spines["left"].set_visible(False)

    return fig


neutypes = ["PM", "LP", "PY"]


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
    scale_bar_voltage=False,
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

        # # time bar bottom left
        # axV.plot(
        #     (offset + 5500) * dt
        #     + offscale
        #     + np.arange(scale_bar_breadth)[:: scale_bar_breadth - 1],
        #     (-40 + offvolt)
        #     * np.ones_like(np.arange(scale_bar_breadth))[:: scale_bar_breadth - 1],
        #     lw=1.0,
        #     color="r",
        # )
        # time bar
        axV.plot(
            (offset + 5500) * dt
            + offscale
            + 2200
            + np.arange(scale_bar_breadth)[:: scale_bar_breadth - 1],
            (-40 + offvolt + 415)
            * np.ones_like(np.arange(scale_bar_breadth))[:: scale_bar_breadth - 1],
            lw=1.0,
            color="k",
        )

        print("hi")
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


def get_summ_stat_name(num):
    if num == 0:
        return r"$T$"  # 'Cycle period'
    if num == 1:
        return r"$d^b_{PD}$"  # 'Burst length PD'
    if num == 2:
        return r"$d^b_{LP}$"  # 'Burst length LP'
    if num == 3:
        return r"$d^b_{PY}$"  # 'Burst length PY'
    if num == 4:
        return r"$\Delta t^{es}_{PD\mathrm{-}LP}$"  # 'End to start PD-LP'
    if num == 5:
        return r"$\Delta t^{es}_{LP\mathrm{-}PY}$"  # 'End to start LP-PY'
    if num == 6:
        return r"$\Delta t^{ss}_{PD\mathrm{-}LP}$"  # 'Start to start PD-LP'
    if num == 7:
        return r"$\Delta t^{ss}_{LP\mathrm{-}PY}$"  # 'Start to start LP-PY'
    if num == 8:
        return r"$d_{PD}$"  # 'Duty cycle PD'
    if num == 9:
        return r"$d_{LP}$"  # 'Duty cycle LP'
    if num == 10:
        return r"$d_{PY}$"  # 'Duty cycle PY'
    if num == 11:
        return r"$\Delta\phi_{PD\mathrm{-}LP}$"  # 'Phase gap PD-LP'
    if num == 12:
        return r"$\Delta\phi_{LP\mathrm{-}PY}$"  # 'Phase gap LP-PY'
    if num == 13:
        return r"$\phi_{LP}$"  # 'Phase LP'
    if num == 14:
        return r"$\phi_{PY}$"  # 'Phase PY'


def get_summ_stat_name_text(num):
    if num == 0:
        return r"$T$"  # 'Cycle period'
    if num == 1:
        return r"$d^{\mathdefault{b}}_{\mathdefault{AB}}$"  # 'Burst length PD'
    if num == 2:
        return r"$d^{\mathdefault{b}}_{\mathdefault{LP}}$"  # 'Burst length LP'
    if num == 3:
        return r"$d^{\mathdefault{b}}_{\mathdefault{PY}}$"  # 'Burst length PY'
    if num == 4:
        return r"$\Delta t^{\mathdefault{es}}_{\mathdefault{AB-LP}}$"  # 'End to start PD-LP'
    if num == 5:
        return r"$\Delta t^{\mathdefault{es}}_{\mathdefault{LP-PY}}$"  # 'End to start LP-PY'
    if num == 6:
        return r"$\Delta t^{\mathdefault{ss}}_{\mathdefault{AB-LP}}$"  # 'Start to start PD-LP'
    if num == 7:
        return r"$\Delta t^{\mathdefault{ss}}_{\mathdefault{LP-PY}}$"  # 'Start to start LP-PY'
    if num == 8:
        return r"$d_{\mathdefault{AB}}$"  # 'Duty cycle PD'
    if num == 9:
        return r"$d_{\mathdefault{LP}}$"  # 'Duty cycle LP'
    if num == 10:
        return r"$d_{\mathdefault{PY}}$"  # 'Duty cycle PY'
    if num == 11:
        return r"$\Delta\phi_{\mathdefault{AB-LP}}$"  # 'Phase gap PD-LP'
    if num == 12:
        return r"$\Delta\phi_{\mathdefault{LP-PY}}$"  # 'Phase gap LP-PY'
    if num == 13:
        return r"$\phi_{\mathdefault{LP}}$"  # 'Phase LP'
    if num == 14:
        return r"$\phi_{\mathdefault{PY}}$"  # 'Phase PY'


def get_summ_stat_name_asterisk(num):
    if num == 0:
        return r"$T$"  # 'Cycle period'
    if num == 1:
        return r"$d^b_{PD}$"  # 'Burst length PD'
    if num == 2:
        return r"$d^b_{LP}$"  # 'Burst length LP'
    if num == 3:
        return r"$d^b_{PY}$"  # 'Burst length PY'
    if num == 4:
        return r"$\Delta t^{es}_{PD\mathrm{-}LP}$"  # 'End to start PD-LP'
    if num == 5:
        return r"$\Delta t^{es}_{LP\mathrm{-}PY}$"  # 'End to start LP-PY'
    if num == 6:
        return r"$\Delta t^{ss}_{PD\mathrm{-}LP}$"  # 'Start to start PD-LP'
    if num == 7:
        return r"$\Delta t^{ss}_{LP\mathrm{-}PY}$"  # 'Start to start LP-PY'
    if num == 8:
        return r"$d_{PD}^{\#}$"  # 'Duty cycle PD'
    if num == 9:
        return r"$d_{LP}^{\#}$"  # 'Duty cycle LP'
    if num == 10:
        return r"$d_{PY}^{\#}$"  # 'Duty cycle PY'
    if num == 11:
        return r"$\Delta\theta_{PD\mathrm{-}LP}^{\#}$"  # 'Phase gap PD-LP'
    if num == 12:
        return r"$\Delta\theta_{LP\mathrm{-}PY}^{\#}$"  # 'Phase gap LP-PY'
    if num == 13:
        return r"$\theta_{LP}^{\#}$"  # 'Phase LP'
    if num == 14:
        return r"$\theta_{PY}^{\#}$"  # 'Phase PY'


def get_synapse_name(num):
    return r"$g_{%s}$" % (pick_synapse(num))


# get the title of the synapses
def pick_synapse(num, mathmode=False):
    if mathmode:
        if num == 0:
            return r"$\mathdefault{AB-LP}$"
        if num == 1:
            return r"$\mathdefault{PD-LP}$"
        if num == 2:
            return r"$\mathdefault{AB-PY}$"
        if num == 3:
            return r"$\mathdefault{PD-PY}$"
        if num == 4:
            return r"$\mathdefault{LP-PD}$"
        if num == 5:
            return r"$\mathdefault{LP-PY}$"
        if num == 6:
            return r"$\mathdefault{PY-LP}$"
    else:
        if num == 0:
            return "AB-LP"
        if num == 1:
            return "PD-LP"
        if num == 2:
            return "AB-PY"
        if num == 3:
            return "PD-PY"
        if num == 4:
            return "LP-PD"
        if num == 5:
            return "LP-PY"
        if num == 6:
            return "PY-LP"
