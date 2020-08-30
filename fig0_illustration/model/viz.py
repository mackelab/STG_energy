import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

def viz_path_and_samples_abstract_twoRows(
        traces,
        t,
        figsize,
        mycols=None,
        offsets=0,
        time_len=165000
):

    counter = 0

    scalebar = [False, True, False, False]

    color_mixture1 = 0.33 * np.asarray(list(mycols['CONSISTENT1'])) + 0.67 * np.asarray(
        list(mycols['CONSISTENT2']))
    color_mixture2 = 0.67 * np.asarray(list(mycols['CONSISTENT1'])) + 0.33 * np.asarray(
        list(mycols['CONSISTENT2']))

    cols = [mycols['CONSISTENT1'], mycols['CONSISTENT2'], color_mixture1, color_mixture2]

    gridspec = dict(hspace=0.03, wspace=0.03, width_ratios=[1, 1], height_ratios=[1, 1])
    fig, ax = plt.subplots(2, 2, facecolor='white', figsize=figsize, gridspec_kw=gridspec)

    row_access = 0
    print_label = True

    for col in range(4):

        out_target = traces[col]
        col_access = col % 2

        if col > 1: print_label=False

        vis_sample_plain(
            voltage_trace=out_target, t=t, axV=ax[row_access, col_access], time_len=time_len,
            offset=offsets[counter], col=cols[col], scale_bar=scalebar[counter],
            print_label=print_label)
        if counter == 1:
            row_access += 1

        counter+=1

    for aa in ax:
        for a in aa:
            a.spines['right'].set_visible(False)
            a.spines['top'].set_visible(False)
            a.spines['bottom'].set_visible(False)
            a.spines['left'].set_visible(False)

    return fig

neutypes = [ 'PM', 'LP', 'PY' ]

def vis_sample_plain(
        voltage_trace,
        t,
        axV,
        t_on=None,
        t_off=None,
        col='k',
        print_label=False,
        time_len=None,
        offset=0,
        scale_bar=True
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

    if scale_bar: scale_col = 'k'
    else: scale_col = 'w'

    data = voltage_trace

    Vx = data['data']
    params = data['params']

    current_col = 0
    for j in range(len(neutypes)):
        if time_len is not None:
            axV.plot(t[10000+offset:10000+offset+time_len:5], Vx[j, 10000+offset:10000+offset+time_len:5] + 140.0 * (2 - j),
                     label=neutypes[j], lw=0.3, c=col)
        else:
            axV.plot(t, Vx[j] + 120.0 * (2 - j), label=neutypes[j], lw=0.3, c=col[current_col])
        current_col += 1

    if print_label:
        axV.plot([1100.0 + (offset - 26500) * (t[1] - t[0])], [300], color=col, marker='o',
                 markeredgecolor='w', ms=8,
                 markeredgewidth=1.0, path_effects=[pe.Stroke(linewidth=1.3, foreground='k'), pe.Normal()])

    if scale_bar:

        # time bar
        axV.plot((offset+5500)*dt+offscale + np.arange(scale_bar_breadth)[::scale_bar_breadth - 1],
                 (-40+offvolt) * np.ones_like(np.arange(scale_bar_breadth))[::scale_bar_breadth - 1],
                 lw=1.0, color='w')

        # voltage bar
        axV.plot(
            (2850 + offset*dt + offscale) * np.ones_like(np.arange(scale_bar_voltage_breadth))[::scale_bar_voltage_breadth - 1],
            275 + np.arange(scale_bar_voltage_breadth)[::scale_bar_voltage_breadth - 1],
            lw=1.0, color=scale_col, zorder=10)


    box = axV.get_position()

    if t_on is not None:
        axV.axvline(t_on, c='r', ls='--')

    if t_on is not None:
        axV.axvline(t_off, c='r', ls='--')

    axV.set_position([box.x0, box.y0, box.width, box.height])
    axV.axes.get_yaxis().set_ticks([])
    axV.axes.get_xaxis().set_ticks([])

    axV.spines['right'].set_visible(False)
    axV.spines['top'].set_visible(False)
    axV.spines['bottom'].set_visible(False)
    axV.spines['left'].set_visible(False)

    current_counter += 1


def vis_ABPD_plain(
        voltage_trace,
        t,
        axV,
        t_on=None,
        t_off=None,
        col='k',
        print_label=False,
        time_len=None,
        offset=0,
        scale_bar=True
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

    Vx = data['data']

    current_col = 0
    j = 0  # neuron 0 -> AB/PD
    if time_len is not None:
        axV.plot(t[10000+offset:10000+offset+time_len:5], Vx[j, 10000+offset:10000+offset+time_len:5],
                 label=neutypes[j], lw=1.0, c=col)
    current_col += 1

    box = axV.get_position()

    if t_on is not None:
        axV.axvline(t_on, c='r', ls='--')

    if t_on is not None:
        axV.axvline(t_off, c='r', ls='--')

    axV.set_position([box.x0, box.y0, box.width, box.height])
    axV.axes.get_yaxis().set_ticks([])
    axV.axes.get_xaxis().set_ticks([])

    axV.spines['right'].set_visible(False)
    axV.spines['top'].set_visible(False)
    axV.spines['bottom'].set_visible(False)
    axV.spines['left'].set_visible(False)

    current_counter += 1

