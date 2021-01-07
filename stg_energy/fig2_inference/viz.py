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

    col = "k" if scale_bar else "w"

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
        color=col,
    )

    # voltage bar
    axV.plot(
        (2850 + offset * dt + offscale)
        * np.ones_like(np.arange(scale_bar_voltage_breadth))[
            :: scale_bar_voltage_breadth - 1
        ],
        275 + np.arange(scale_bar_voltage_breadth)[:: scale_bar_voltage_breadth - 1],
        lw=1.0,
        color=col,
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


# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

import collections
import inspect
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import matplotlib as mpl
import numpy as np
import six
import torch
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde

from sbi.utils.conditional_density import eval_conditional_density

try:
    collectionsAbc = collections.abc
except:
    collectionsAbc = collections


def hex2rgb(hex):
    # Pass 16 to the integer function for change of base
    return [int(hex[i : i + 2], 16) for i in range(1, 6, 2)]


def rgb2hex(RGB):
    # Components need to be integers for hex to make sense
    RGB = [int(x) for x in RGB]
    return "#" + "".join(
        ["0{0:x}".format(v) if v < 16 else "{0:x}".format(v) for v in RGB]
    )


def _update(d, u):
    # https://stackoverflow.com/a/3233356
    for k, v in six.iteritems(u):
        dv = d.get(k, {})
        if not isinstance(dv, collectionsAbc.Mapping):
            d[k] = v
        elif isinstance(v, collectionsAbc.Mapping):
            d[k] = _update(dv, v)
        else:
            d[k] = v
    return d


def _format_axis(ax, xhide=True, yhide=True, xlabel="", ylabel="", tickformatter=None):
    for loc in ["right", "top", "left", "bottom"]:
        ax.spines[loc].set_visible(False)
    if xhide:
        ax.set_xlabel("")
        ax.xaxis.set_ticks_position("none")
        ax.xaxis.set_tick_params(labelbottom=False)
    if yhide:
        ax.set_ylabel("")
        ax.yaxis.set_ticks_position("none")
        ax.yaxis.set_tick_params(labelleft=False)
    if not xhide:
        ax.set_xlabel(xlabel)
        ax.xaxis.set_ticks_position("bottom")
        ax.xaxis.set_tick_params(labelbottom=True)
        if tickformatter is not None:
            ax.xaxis.set_major_formatter(tickformatter)
        ax.spines["bottom"].set_visible(True)
    if not yhide:
        ax.set_ylabel(ylabel)
        ax.yaxis.set_ticks_position("left")
        ax.yaxis.set_tick_params(labelleft=True)
        if tickformatter is not None:
            ax.yaxis.set_major_formatter(tickformatter)
        ax.spines["left"].set_visible(True)
    return ax


def probs2contours(probs, levels):
    """Takes an array of probabilities and produces an array of contours at specified percentile levels
    Parameters
    ----------
    probs : array
        Probability array. doesn't have to sum to 1, but it is assumed it contains all the mass
    levels : list
        Percentile levels, have to be in [0.0, 1.0]
    Return
    ------
    Array of same shape as probs with percentile labels
    """
    # make sure all contour levels are in [0.0, 1.0]
    levels = np.asarray(levels)
    assert np.all(levels <= 1.0) and np.all(levels >= 0.0)

    # flatten probability array
    shape = probs.shape
    probs = probs.flatten()

    # sort probabilities in descending order
    idx_sort = probs.argsort()[::-1]
    idx_unsort = idx_sort.argsort()
    probs = probs[idx_sort]

    # cumulative probabilities
    cum_probs = probs.cumsum()
    cum_probs /= cum_probs[-1]

    # create contours at levels
    contours = np.ones_like(cum_probs)
    levels = np.sort(levels)[::-1]
    for level in levels:
        contours[cum_probs <= level] = level

    # make sure contours have the order and the shape of the original
    # probability array
    contours = np.reshape(contours[idx_unsort], shape)

    return contours


def ensure_numpy(t: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Returns np.ndarray if torch.Tensor was provided.

    Used because samples_nd() can only handle np.ndarray.
    """
    if isinstance(t, torch.Tensor):
        return t.numpy()
    else:
        return t


def pairplot(
    samples: Union[
        List[np.ndarray], List[torch.Tensor], np.ndarray, torch.Tensor
    ] = None,
    points: Optional[
        Union[List[np.ndarray], List[torch.Tensor], np.ndarray, torch.Tensor]
    ] = None,
    limits: Optional[Union[List, torch.Tensor]] = None,
    subset: List[int] = None,
    upper: Optional[str] = "hist",
    diag: Optional[str] = "hist",
    fig_size: Tuple = (10, 10),
    labels: Optional[List[str]] = None,
    points_colors: List[str] = plt.rcParams["axes.prop_cycle"].by_key()["color"],
    **kwargs
):
    """
    Plot samples in a 2D grid showing marginals and pairwise marginals.

    Each of the diagonal plots can be interpreted as a 1D-marginal of the distribution
    that the samples were drawn from. Each upper-diagonal plot can be interpreted as a
    2D-marginal of the distribution.

    Args:
        samples: Samples used to build the histogram.
        points: List of additional points to scatter.
        limits: Array containing the plot xlim for each parameter dimension. If None,
            just use the min and max of the passed samples
        subset: List containing the dimensions to plot. E.g. subset=[1,3] will plot
            plot only the 1st and 3rd dimension but will discard the 0th and 2nd (and,
            if they exist, the 4th, 5th and so on).
        upper: Plotting style for upper diagonal, {hist, scatter, contour, cond, None}.
        diag: Plotting style for diagonal, {hist, cond, None}.
        fig_size: Size of the entire figure.
        labels: List of strings specifying the names of the parameters.
        points_colors: Colors of the `points`.
        **kwargs: Additional arguments to adjust the plot, see the source code in
            `_get_default_opts()` in `sbi.utils.plot` for more details.

    Returns: figure and axis of posterior distribution plot
    """

    # TODO: add color map support
    # TODO: automatically determine good bin sizes for histograms
    # TODO: add legend (if legend is True)

    opts = _get_default_opts()
    # update the defaults dictionary by the current values of the variables (passed by
    # the user)
    opts = _update(opts, locals())
    opts = _update(opts, kwargs)

    # Prepare samples
    if type(samples) != list:
        samples = ensure_numpy(samples)
        samples = [samples]
    else:
        for i, sample_pack in enumerate(samples):
            samples[i] = ensure_numpy(samples[i])

    # Dimensionality of the problem.
    dim = samples[0].shape[1]

    # Prepare limits. Infer them from samples if they had not been passed.
    if limits == [] or limits is None:
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
        if len(limits) == 1:
            limits = [limits[0] for _ in range(dim)]
        else:
            limits = limits
    limits = torch.as_tensor(limits)

    # Prepare diag/upper/lower
    if type(opts["diag"]) is not list:
        opts["diag"] = [opts["diag"] for _ in range(len(samples))]
    if type(opts["upper"]) is not list:
        opts["upper"] = [opts["upper"] for _ in range(len(samples))]
    # if type(opts['lower']) is not list:
    #    opts['lower'] = [opts['lower'] for _ in range(len(samples))]
    opts["lower"] = None

    def diag_func(row, **kwargs):
        if len(samples) > 0:
            for n, v in enumerate(samples):
                if opts["diag"][n] == "hist":
                    h = plt.hist(
                        v[:, row], color=opts["samples_colors"][n], **opts["hist_diag"]
                    )
                elif opts["diag"][n] == "kde":
                    density = gaussian_kde(
                        v[:, row], bw_method=opts["kde_diag"]["bw_method"]
                    )
                    xs = np.linspace(
                        limits[row, 0], limits[row, 1], opts["kde_diag"]["bins"]
                    )
                    ys = density(xs)
                    h = plt.plot(
                        xs,
                        ys,
                        color=opts["samples_colors"][n],
                    )
                else:
                    pass

    def upper_func(row, col, limits, **kwargs):
        if len(samples) > 0:
            for n, v in enumerate(samples):
                if opts["upper"][n] == "hist" or opts["upper"][n] == "hist2d":
                    hist, xedges, yedges = np.histogram2d(
                        v[:, col],
                        v[:, row],
                        range=[
                            [limits[col][0], limits[col][1]],
                            [limits[row][0], limits[row][1]],
                        ],
                        **opts["hist_offdiag"]
                    )
                    h = plt.imshow(
                        hist.T,
                        origin="lower",
                        extent=[
                            xedges[0],
                            xedges[-1],
                            yedges[0],
                            yedges[-1],
                        ],
                        aspect="auto",
                    )

                elif opts["upper"][n] in [
                    "kde",
                    "kde2d",
                    "contour",
                    "contourf",
                ]:
                    density = gaussian_kde(
                        v[:, [col, row]].T,
                        bw_method=opts["kde_offdiag"]["bw_method"],
                    )
                    X, Y = np.meshgrid(
                        np.linspace(
                            limits[col][0],
                            limits[col][1],
                            opts["kde_offdiag"]["bins"],
                        ),
                        np.linspace(
                            limits[row][0],
                            limits[row][1],
                            opts["kde_offdiag"]["bins"],
                        ),
                    )
                    positions = np.vstack([X.ravel(), Y.ravel()])
                    Z = np.reshape(density(positions).T, X.shape)

                    if opts["upper"][n] == "kde" or opts["upper"][n] == "kde2d":
                        h = plt.imshow(
                            Z,
                            extent=[
                                limits[col][0],
                                limits[col][1],
                                limits[row][0],
                                limits[row][1],
                            ],
                            origin="lower",
                            aspect="auto",
                        )
                    elif opts["upper"][n] == "contour":
                        if opts["contour_offdiag"]["percentile"]:
                            Z = probs2contours(Z, opts["contour_offdiag"]["levels"])
                        else:
                            Z = (Z - Z.min()) / (Z.max() - Z.min())
                        h = plt.contour(
                            X,
                            Y,
                            Z,
                            origin="lower",
                            extent=[
                                limits[col][0],
                                limits[col][1],
                                limits[row][0],
                                limits[row][1],
                            ],
                            colors=opts["samples_colors"][n],
                            levels=opts["contour_offdiag"]["levels"],
                        )
                    else:
                        pass
                elif opts["upper"][n] == "scatter":
                    h = plt.scatter(
                        v[:, col],
                        v[:, row],
                        color=opts["samples_colors"][n],
                        **opts["scatter_offdiag"]
                    )
                elif opts["upper"][n] == "plot":
                    h = plt.plot(
                        v[:, col],
                        v[:, row],
                        color=opts["samples_colors"][n],
                        **opts["plot_offdiag"]
                    )
                else:
                    pass

    return _pairplot_scaffold(diag_func, upper_func, dim, limits, points, opts)


def conditional_pairplot(
    density: Any,
    condition: torch.Tensor,
    limits: Union[List, torch.Tensor],
    points: Optional[
        Union[List[np.ndarray], List[torch.Tensor], np.ndarray, torch.Tensor]
    ] = None,
    subset: List[int] = None,
    resolution: int = 50,
    fig_size: Tuple = (10, 10),
    labels: Optional[List[str]] = None,
    points_colors: List[str] = plt.rcParams["axes.prop_cycle"].by_key()["color"],
    **kwargs
):
    r"""
    Plot conditional distribution given all other parameters.

    The conditionals can be interpreted as slices through the `density` at a location
    given by `condition`.

    For example:
    Say we have a 3D density with parameters $\theta_0$, $\theta_1$, $\theta_2$ and
    a condition $c$ passed by the user in the `condition` argument.
    For the plot of $\theta_0$ on the diagonal, this will plot the conditional
    $p(\theta_0 | \theta_1=c[1], \theta_2=c[2])$. For the upper
    diagonal of $\theta_1$ and $\theta_2$, it will plot
    $p(\theta_1, \theta_2 | \theta_0=c[0])$. All other diagonals and upper-diagonals
    are built in the corresponding way.

    Args:
        density: Probability density with a `log_prob()` method.
        condition: Condition that all but the one/two regarded parameters are fixed to.
            The condition should be of shape (1, dim_theta), i.e. it could e.g. be
            a sample from the posterior distribution.
        limits: Limits in between which each parameter will be evaluated.
        points: Additional points to scatter.
        subset: List containing the dimensions to plot. E.g. subset=[1,3] will plot
            plot only the 1st and 3rd dimension but will discard the 0th and 2nd (and,
            if they exist, the 4th, 5th and so on)
        resolution: Resolution of the grid at which we evaluate the `pdf`.
        fig_size: Size of the entire figure.
        labels: List of strings specifying the names of the parameters.
        points_colors: Colors of the `points`.
        **kwargs: Additional arguments to adjust the plot, see the source code in
            `_get_default_opts()` in `sbi.utils.plot` for more details.

    Returns: figure and axis of posterior distribution plot
    """

    # Setting these is required because _pairplot_scaffold will check if opts['diag'] is
    # `None`. This would break if opts has no key 'diag'. Same for 'upper'.
    diag = "cond"
    upper = "cond"

    opts = _get_default_opts()
    # update the defaults dictionary by the current values of the variables (passed by
    # the user)
    opts = _update(opts, locals())
    opts = _update(opts, kwargs)

    # Dimensions
    dim = condition.shape[-1]

    # Prepare limits
    if len(opts["limits"]) == 1:
        limits = [opts["limits"][0] for _ in range(dim)]
    else:
        limits = opts["limits"]
    limits = torch.as_tensor(limits)

    opts["lower"] = None

    # Infer the margin. This is to avoid that we evaluate the posterior **exactly**
    # at the boundary.
    limits_diffs = limits[:, 1] - limits[:, 0]
    eps_margins = limits_diffs / 1e5

    def diag_func(row, **kwargs):
        p_vector = eval_conditional_density(
            opts["density"],
            opts["condition"],
            limits,
            row,
            row,
            resolution=resolution,
            eps_margins1=eps_margins[row],
            eps_margins2=eps_margins[row],
        ).numpy()
        h = plt.plot(
            np.linspace(
                limits[row, 0],
                limits[row, 1],
                resolution,
            ),
            p_vector,
            c=opts["samples_colors"][0],
        )

    def upper_func(row, col, **kwargs):
        p_image = eval_conditional_density(
            opts["density"],
            opts["condition"],
            limits,
            row,
            col,
            resolution=resolution,
            eps_margins1=eps_margins[row],
            eps_margins2=eps_margins[col],
        ).numpy()
        h = plt.imshow(
            p_image.T,
            origin="lower",
            extent=[
                limits[col, 0],
                limits[col, 1],
                limits[row, 0],
                limits[row, 1],
            ],
            aspect="auto",
        )

    return _pairplot_scaffold(diag_func, upper_func, dim, limits, points, opts)


def _pairplot_scaffold(diag_func, upper_func, dim, limits, points, opts):
    """
    Builds the scaffold for any function that plots parameters in a pairplot setting.

    Args:
        diag_func: Plotting function that will be executed for the diagonal elements of
            the plot. It will be passed the current `row` (i.e. which parameter that is
            to be plotted) and the `limits` for all dimensions.
        upper_func: Plotting function that will be executed for the upper-diagonal
            elements of the plot. It will be passed the current `row` and `col` (i.e.
            which parameters are to be plotted and the `limits` for all dimensions.
        dim: The dimensionality of the density.
        limits: Limits for each parameter.
        points: Additional points to be scatter-plotted.
        opts: Dictionary built by the functions that call `pairplot_scaffold`. Must
            contain at least `labels`, `ticks`, `subset`, `fig_size`, `subplots`,
            `fig_subplots_adjust`, `title`, `title_format`, ..

    Returns: figure and axis
    """

    # Prepare points
    if points is None:
        points = []
    if type(points) != list:
        points = ensure_numpy(points)
        points = [points]
    points = [np.atleast_2d(p) for p in points]
    points = [np.atleast_2d(ensure_numpy(p)) for p in points]

    # TODO: add asserts checking compatibility of dimensions

    # Prepare labels
    if opts["labels"] == [] or opts["labels"] is None:
        labels_dim = ["dim {}".format(i + 1) for i in range(dim)]
    else:
        labels_dim = opts["labels"]

    # Prepare ticks
    if opts["ticks"] == [] or opts["ticks"] is None:
        ticks = None
    else:
        if len(opts["ticks"]) == 1:
            ticks = [opts["ticks"][0] for _ in range(dim)]
        else:
            ticks = opts["ticks"]

    # Figure out if we subset the plot
    subset = opts["subset"]
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

    fig, axes = plt.subplots(rows, cols, figsize=opts["fig_size"], **opts["subplots"])
    # Cast to ndarray in case of 1D subplots.
    axes = np.array(axes).reshape(rows, cols)

    # Style figure
    fig.subplots_adjust(**opts["fig_subplots_adjust"])
    fig.suptitle(opts["title"], **opts["title_format"])

    # Style axes
    row_idx = -1
    for row in range(dim):
        if row not in subset:
            continue
        else:
            row_idx += 1

        col_idx = -1
        for col in range(dim):
            if col not in subset:
                continue
            else:
                col_idx += 1

            if row == col:
                current = "diag"
            elif row < col:
                current = "upper"
            else:
                current = "lower"

            ax = axes[row_idx, col_idx]
            plt.sca(ax)

            # Background color
            if (
                current in opts["fig_bg_colors"]
                and opts["fig_bg_colors"][current] is not None
            ):
                ax.set_facecolor(opts["fig_bg_colors"][current])

            # Axes
            if opts[current] is None:
                ax.axis("off")
                continue

            # Limits
            ax.set_xlim((limits[col][0], limits[col][1]))
            if current != "diag":
                ax.set_ylim((limits[row][0], limits[row][1]))

            # Ticks
            if ticks is not None:
                ax.set_xticks((ticks[col][0], ticks[col][1]))
                if current != "diag":
                    ax.set_yticks((ticks[row][0], ticks[row][1]))

            # Despine
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines["bottom"].set_position(("outward", opts["despine"]["offset"]))

            # Formatting axes
            if current == "diag":  # off-diagnoals
                if opts["lower"] is None or col == dim - 1:
                    _format_axis(
                        ax,
                        xhide=False,
                        xlabel=labels_dim[col],
                        yhide=True,
                        tickformatter=opts["tickformatter"],
                    )
                    if opts["labelpad"] is not None:
                        ax.xaxis.labelpad = opts["labelpad"]
                else:
                    _format_axis(ax, xhide=True, yhide=True)
            else:  # off-diagnoals
                if row == dim - 1:
                    _format_axis(
                        ax,
                        xhide=False,
                        xlabel=labels_dim[col],
                        yhide=True,
                        tickformatter=opts["tickformatter"],
                    )
                else:
                    _format_axis(ax, xhide=True, yhide=True)
            if opts["tick_labels"] is not None:
                ax.set_xticklabels(
                    (
                        str(opts["tick_labels"][col][0]),
                        str(opts["tick_labels"][col][1]),
                    )
                )
                if opts["tick_labelpad"] is not None:
                    ax.tick_params(axis="x", which="major", pad=opts["tick_labelpad"])

            # Diagonals
            if current == "diag":
                diag_func(row=row, limits=limits)

                if len(points) > 0:
                    extent = ax.get_ylim()
                    for n, v in enumerate(points):
                        h = plt.plot(
                            [v[:, row], v[:, row]],
                            extent,
                            color=opts["points_colors"][n],
                            **opts["points_diag"]
                        )

            # Off-diagonals
            else:
                upper_func(
                    row=row,
                    col=col,
                    limits=limits,
                )

                if len(points) > 0:

                    for n, v in enumerate(points):
                        h = plt.plot(
                            v[:, col],
                            v[:, row],
                            color=opts["points_colors"][n],
                            **opts["points_offdiag"]
                        )

    if len(subset) < dim:
        for row in range(len(subset)):
            ax = axes[row, len(subset) - 1]
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            text_kwargs = {"fontsize": plt.rcParams["font.size"] * 2.0}
            ax.text(x1 + (x1 - x0) / 8.0, (y0 + y1) / 2.0, "...", **text_kwargs)
            if row == len(subset) - 1:
                ax.text(
                    x1 + (x1 - x0) / 12.0,
                    y0 - (y1 - y0) / 1.5,
                    "...",
                    rotation=-45,
                    **text_kwargs
                )

    return fig, axes


def _get_default_opts():
    """ Return default values for plotting specs."""

    return {
        # 'lower': None,     # hist/scatter/None  # TODO: implement
        # title and legend
        "title": None,
        "legend": False,
        # labels
        "labels_points": [],  # for points
        "labels_samples": [],  # for samples
        # colors
        "samples_colors": plt.rcParams["axes.prop_cycle"].by_key()["color"],
        # ticks
        "ticks": [],
        "tickformatter": mpl.ticker.FormatStrFormatter("%g"),
        "tick_labels": None,
        # options for hist
        "hist_diag": {"alpha": 1.0, "bins": 50, "density": False, "histtype": "step"},
        "hist_offdiag": {
            # 'edgecolor': 'none',
            # 'linewidth': 0.0,
            "bins": 50,
        },
        # options for kde
        "kde_diag": {"bw_method": "scott", "bins": 50, "color": "black"},
        "kde_offdiag": {"bw_method": "scott", "bins": 50},
        # options for contour
        "contour_offdiag": {"levels": [0.68], "percentile": True},
        # options for scatter
        "scatter_offdiag": {
            "alpha": 0.5,
            "edgecolor": "none",
            "rasterized": False,
        },
        # options for plot
        "plot_offdiag": {},
        # formatting points (scale, markers)
        "points_diag": {},
        "points_offdiag": {
            "marker": ".",
            "markersize": 20,
        },
        # other options
        "fig_bg_colors": {"upper": None, "diag": None, "lower": None},
        "fig_subplots_adjust": {
            "top": 0.9,
        },
        "subplots": {},
        "despine": {
            "offset": 5,
        },
        "title_format": {"fontsize": 16},
    }
