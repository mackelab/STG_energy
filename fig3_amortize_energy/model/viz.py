import numpy as np
import matplotlib.pyplot as plt
import sys; sys.path.append('../../../')
from common import col, _update, _format_axis
from scipy.stats import gaussian_kde
import os
import matplotlib as mpl
import seaborn as sns

def sensitivity_hist(shift_in_mean_normalized, figsize):
    fig, ax = plt.subplots(1, 4, figsize=figsize)
    ax[0].bar(np.arange(8), shift_in_mean_normalized[:8])
    ax[1].bar(np.arange(8), shift_in_mean_normalized[8:16], color='orange')
    ax[2].bar(np.arange(8), shift_in_mean_normalized[16:24], color='g')
    ax[3].bar(np.arange(7), shift_in_mean_normalized[24:], color='k')

    for i, a in enumerate(ax):
        a.set_ylim(-1, 1)
        a.set_xticks(np.arange(8))
        if i < 3: a.set_xticklabels(['Na', 'CaT', 'CaS', 'A', 'KCa', 'Kd', 'H', 'Leak'])
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
    ax[0].set_ylabel("Influence on energy")


def oneDmarginal(samples, points=[], **kwargs):
    opts = {
        # what to plot on triagonal and diagonal subplots
        'upper': 'hist',  # hist/scatter/None/cond
        'diag': 'hist',  # hist/None/cond
        # 'lower': None,     # hist/scatter/None  # TODO: implement

        # title and legend
        'title': None,
        'legend': False,

        # labels
        'labels': [],  # for dimensions
        'labels_points': [],  # for points
        'labels_samples': [],  # for samples
        'labelpad': None,

        # colors
        'samples_colors': plt.rcParams['axes.prop_cycle'].by_key()['color'],
        'points_colors': plt.rcParams['axes.prop_cycle'].by_key()['color'],

        # subset
        'subset': None,

        # conditional posterior requires condition and pdf1
        'pdfs': None,
        'condition': None,

        # axes limits
        'limits': [],

        # ticks
        'ticks': [],
        'tickformatter': mpl.ticker.FormatStrFormatter('%g'),
        'tick_labels': None,
        'tick_labelpad': None,

        # options for hist
        'hist_diag': {
            'alpha': 1.,
            'bins': 25,
            'density': False,
            'histtype': 'step'
        },

        # options for kde
        'kde_diag': {
            'bw_method': 'scott',
            'bins': 100,
            'color': 'black'
        },

        # options for contour
        'contour_offdiag': {
            'levels': [0.68]
        },

        # options for scatter
        'scatter_offdiag': {
            'alpha': 0.5,
            'edgecolor': 'none',
            'rasterized': False,
        },

        # options for plot
        'plot_offdiag': {},

        # formatting points (scale, markers)
        'points_diag': {
        },
        'points_offdiag': {
            'marker': '.',
            'markersize': 20,
        },

        # matplotlib style
        'style': '../../.matplotlibrc',

        # other options
        'fig_size': (10, 10),
        'fig_bg_colors':
            {'upper': None,
             'diag': None,
             'lower': None},
        'fig_subplots_adjust': {
            'top': 0.9,
        },
        'subplots': {
        },
        'despine': {
            'offset': 5,
        },
        'title_format': {
            'fontsize': 16
        },
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
    if opts['labels'] == [] or opts['labels'] is None:
        labels_dim = ['dim {}'.format(i+1) for i in range(dim)]
    else:
        labels_dim = opts['labels']

    # Prepare limits
    if opts['limits'] == [] or opts['limits'] is None:
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
        if len(opts['limits']) == 1:
            limits = [opts['limits'][0] for _ in range(dim)]
        else:
            limits = opts['limits']

    # Prepare ticks
    if opts['ticks'] == [] or opts['ticks'] is None:
        ticks = None
    else:
        if len(opts['ticks']) == 1:
            ticks = [opts['ticks'][0] for _ in range(dim)]
        else:
            ticks = opts['ticks']

    # Prepare diag/upper/lower
    if type(opts['diag']) is not list:
        opts['diag'] = [opts['diag'] for _ in range(len(samples))]
    if type(opts['upper']) is not list:
        opts['upper'] = [opts['upper'] for _ in range(len(samples))]
    #if type(opts['lower']) is not list:
    #    opts['lower'] = [opts['lower'] for _ in range(len(samples))]
    opts['lower'] = None

    # Style
    if opts['style'] in ['dark', 'light']:
        style = os.path.join(
            os.path.dirname(__file__),
            'matplotlib_{}.style'.format(opts['style']))
    else:
        style = opts['style'];

    # Apply custom style as context
    with mpl.rc_context(fname=style):

        # Figure out if we subset the plot
        subset = opts['subset']
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

        fig, axes = plt.subplots(1, cols, figsize=opts['fig_size'], **opts['subplots'])

        # Style figure
        fig.subplots_adjust(**opts['fig_subplots_adjust'])
        fig.suptitle(opts['title'], **opts['title_format'])

        col_idx = -1
        for col in range(dim):
            if col not in subset:
                continue
            else:
                col_idx += 1

            current = 'diag'

            ax = axes[col_idx]
            plt.sca(ax)

            # Background color
            if current in opts['fig_bg_colors'] and \
                opts['fig_bg_colors'][current] is not None:
                ax.set_facecolor(
                    opts['fig_bg_colors'][current])

            # Axes
            if opts[current] is None:
                ax.axis('off')
                continue

            # Limits
            if limits is not None:
                ax.set_xlim(
                    (limits[col][0], limits[col][1]))
                if current != 'diag':
                    ax.set_ylim(
                        (limits[row][0], limits[row][1]))
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()

            # Ticks
            if ticks is not None:
                ax.set_xticks(
                    (ticks[col][0], ticks[col][1]))
                if current != 'diag':
                    ax.set_yticks(
                        (ticks[row][0], ticks[row][1]))

            # Despine
            sns.despine(ax=ax, **opts['despine'])

            # Formatting axes
            if opts['lower'] is None or col == dim-1:
                _format_axis(ax, xhide=False, xlabel=labels_dim[col],
                    yhide=True, tickformatter=opts['tickformatter'])
                if opts['labelpad'] is not None: ax.xaxis.labelpad = opts['labelpad']
            else:
                _format_axis(ax, xhide=True, yhide=True)

            if opts['tick_labels'] is not None:
                ax.set_xticklabels(
                    (str(opts['tick_labels'][col][0]), str(opts['tick_labels'][col][1])))
                if opts['tick_labelpad'] is not None:
                    ax.tick_params(axis='x', which='major', pad=opts['tick_labelpad'])

            # Diagonals
            if len(samples) > 0:
                for n, v in enumerate(samples):
                    if opts['diag'][n] == 'hist':
                        h = plt.hist(
                            v[:, col],
                            color=opts['samples_colors'][n],
                            **opts['hist_diag']
                        )
                    elif opts['diag'][n] == 'kde':
                        density = gaussian_kde(
                            v[:, col],
                            bw_method=opts['kde_diag']['bw_method'])
                        xs = np.linspace(xmin, xmax, opts['kde_diag']['bins'])
                        ys = density(xs)
                        h = plt.plot(xs, ys,
                            color=opts['samples_colors'][n],
                        )
                    else:
                        pass

            if len(points) > 0:
                extent = ax.get_ylim()
                for n, v in enumerate(points):
                    h = plt.plot(
                        [v[:, col], v[:, col]],
                        extent,
                        color=opts['points_colors'][n],
                        **opts['points_diag']
                    )

        if len(subset) < dim:
            ax = axes[-1]
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            text_kwargs = {'fontsize': plt.rcParams['font.size']*2.}
            ax.text(x1 + (x1 - x0) / 8., (y0 + y1) / 2., '...', **text_kwargs)

    return fig, axes
