import numpy as np
import matplotlib.pyplot as plt
from stg_energy.fig5_cc.conditional_density import eval_conditional_density
from copy import deepcopy
from stg_energy.common import col, _update, _format_axis
from scipy.stats import gaussian_kde
import seaborn as sns
import matplotlib as mpl


def simplest_vis(out_trace, time_len=None):
    Vx = out_trace["data"]

    fig, axV = plt.subplots(1, figsize=(14, 3))

    for j in range(3):
        if time_len is not None:
            axV.plot(Vx[j, 10000 : 10000 + time_len] + 130.0 * (2 - j), lw=0.6)
        else:
            axV.plot(Vx[j] + 130.0 * (2 - j), lw=0.6)


def oneDmarginal(samples, points=[], **kwargs):
    opts = {
        # what to plot on triagonal and diagonal subplots
        "upper": "hist",  # hist/scatter/None/cond
        "diag": "hist",  # hist/None/cond
        # 'lower': None,     # hist/scatter/None  # TODO: implement
        # title and legend
        "title": None,
        "legend": False,
        # labels
        "labels": [],  # for dimensions
        "labels_points": [],  # for points
        "labels_samples": [],  # for samples
        "labelpad": None,
        # colors
        "samples_colors": plt.rcParams["axes.prop_cycle"].by_key()["color"],
        "points_colors": plt.rcParams["axes.prop_cycle"].by_key()["color"],
        # subset
        "subset": None,
        # conditional posterior requires condition and pdf1
        "pdfs": None,
        "condition": None,
        # axes limits
        "limits": [],
        # ticks
        "ticks": [],
        "tickformatter": mpl.ticker.FormatStrFormatter("%g"),
        "tick_labels": None,
        "tick_labelpad": None,
        # options for hist
        "hist_diag": {"alpha": 1.0, "bins": 25, "density": False, "histtype": "step"},
        # options for kde
        "kde_diag": {"bw_method": "scott", "bins": 100, "color": "black"},
        # options for contour
        "contour_offdiag": {"levels": [0.68]},
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
        # matplotlib style
        "style": "../../../.matplotlibrc",
        # other options
        "fig_size": (10, 10),
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
    if opts["labels"] == [] or opts["labels"] is None:
        labels_dim = ["dim {}".format(i + 1) for i in range(dim)]
    else:
        labels_dim = opts["labels"]

    # Prepare limits
    if opts["limits"] == [] or opts["limits"] is None:
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
        if len(opts["limits"]) == 1:
            limits = [opts["limits"][0] for _ in range(dim)]
        else:
            limits = opts["limits"]

    # Prepare ticks
    if opts["ticks"] == [] or opts["ticks"] is None:
        ticks = None
    else:
        if len(opts["ticks"]) == 1:
            ticks = [opts["ticks"][0] for _ in range(dim)]
        else:
            ticks = opts["ticks"]

    # Prepare diag/upper/lower
    if type(opts["diag"]) is not list:
        opts["diag"] = [opts["diag"] for _ in range(len(samples))]
    if type(opts["upper"]) is not list:
        opts["upper"] = [opts["upper"] for _ in range(len(samples))]
    # if type(opts['lower']) is not list:
    #    opts['lower'] = [opts['lower'] for _ in range(len(samples))]
    opts["lower"] = None

    # Style
    if opts["style"] in ["dark", "light"]:
        style = os.path.join(
            os.path.dirname(__file__), "matplotlib_{}.style".format(opts["style"])
        )
    else:
        style = opts["style"]

    # Apply custom style as context
    with mpl.rc_context(fname=style):

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

        fig, axes = plt.subplots(1, cols, figsize=opts["fig_size"], **opts["subplots"])

        # Style figure
        fig.subplots_adjust(**opts["fig_subplots_adjust"])
        fig.suptitle(opts["title"], **opts["title_format"])

        col_idx = -1
        for col in range(dim):
            if col not in subset:
                continue
            else:
                col_idx += 1

            current = "diag"

            ax = axes[col_idx]
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
            if limits is not None:
                ax.set_xlim((limits[col][0], limits[col][1]))
                if current != "diag":
                    ax.set_ylim((limits[row][0], limits[row][1]))
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()

            # Ticks
            if ticks is not None:
                ax.set_xticks((ticks[col][0], ticks[col][1]))
                if current != "diag":
                    ax.set_yticks((ticks[row][0], ticks[row][1]))

            # Despine
            sns.despine(ax=ax, **opts["despine"])

            # Formatting axes
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

            if opts["tick_labels"] is not None:
                ax.set_xticklabels(
                    (str(opts["tick_labels"][col][0]), str(opts["tick_labels"][col][1]))
                )
                if opts["tick_labelpad"] is not None:
                    ax.tick_params(axis="x", which="major", pad=opts["tick_labelpad"])

            # Diagonals
            if len(samples) > 0:
                for n, v in enumerate(samples):
                    if opts["diag"][n] == "hist":
                        h = plt.hist(
                            v[:, col],
                            color=opts["samples_colors"][n],
                            **opts["hist_diag"]
                        )
                    elif opts["diag"][n] == "kde":
                        density = gaussian_kde(
                            v[:, col], bw_method=opts["kde_diag"]["bw_method"]
                        )
                        xs = np.linspace(xmin, xmax, opts["kde_diag"]["bins"])
                        ys = density(xs)
                        h = plt.plot(
                            xs,
                            ys,
                            color=opts["samples_colors"][n],
                        )
                    elif opts["diag"][n] == "cond":
                        p_vector = eval_conditional_density(
                            opts["pdfs"][n],
                            [opts["condition"][n]],
                            opts["limits"],
                            col,
                            col,
                            resolution=opts["hist_diag"]["bins"],
                            log=False,
                        )
                        p_vector = p_vector / np.max(p_vector)  # just to scale it to 1
                        h = plt.plot(
                            np.linspace(
                                opts["limits"][col, 0],
                                opts["limits"][col, 1],
                                opts["hist_diag"]["bins"],
                            ),
                            p_vector,
                            c=opts["samples_colors"][n],
                        )
                    else:
                        pass

            if len(points) > 0:
                extent = ax.get_ylim()
                for n, v in enumerate(points):
                    h = plt.plot(
                        [v[:, col], v[:, col]],
                        extent,
                        color=opts["points_colors"][n],
                        **opts["points_diag"]
                    )

        if len(subset) < dim:
            ax = axes[-1]
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            text_kwargs = {"fontsize": plt.rcParams["font.size"] * 2.0}
            ax.text(x1 + (x1 - x0) / 8.0, (y0 + y1) / 2.0, "...", **text_kwargs)

    return fig, axes


def singleOneDmarginal(samples, points=[], **kwargs):
    opts = {
        # what to plot on triagonal and diagonal subplots
        "upper": "hist",  # hist/scatter/None/cond
        "diag": "hist",  # hist/None/cond
        # 'lower': None,     # hist/scatter/None  # TODO: implement
        # title and legend
        "title": None,
        "legend": False,
        # labels
        "labels": [],  # for dimensions
        "labels_points": [],  # for points
        "labels_samples": [],  # for samples
        "labelpad": None,
        # colors
        "samples_colors": plt.rcParams["axes.prop_cycle"].by_key()["color"],
        "points_colors": plt.rcParams["axes.prop_cycle"].by_key()["color"],
        # subset
        "subset": None,
        # conditional posterior requires condition and pdf1
        "pdfs": None,
        "condition": None,
        # axes limits
        "limits": [],
        # ticks
        "ticks": [],
        "tickformatter": mpl.ticker.FormatStrFormatter("%g"),
        "tick_labels": None,
        "tick_labelpad": None,
        # options for hist
        "hist_diag": {"alpha": 1.0, "bins": 25, "density": False, "histtype": "step"},
        # options for kde
        "kde_diag": {"bw_method": "scott", "bins": 100, "color": "black"},
        # options for contour
        "contour_offdiag": {"levels": [0.68]},
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
        # matplotlib style
        "style": "../../../.matplotlibrc",
        # other options
        "fig_size": (10, 10),
        "fig_bg_colors": {"upper": None, "diag": None, "lower": None},
        "fig_subplots_adjust": {
            "top": 0.9,
        },
        "subplots": {},
        "despine": {
            "offset": 5,
        },
        "title_format": {"fontsize": 16},
        "log": False,
        "thr_at": None,
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
    if opts["labels"] == [] or opts["labels"] is None:
        labels_dim = ["dim {}".format(i + 1) for i in range(dim)]
    else:
        labels_dim = opts["labels"]

    # Prepare limits
    if opts["limits"] == [] or opts["limits"] is None:
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
        if len(opts["limits"]) == 1:
            limits = [opts["limits"][0] for _ in range(dim)]
        else:
            limits = opts["limits"]

    # Prepare ticks
    if opts["ticks"] == [] or opts["ticks"] is None:
        ticks = None
    else:
        if len(opts["ticks"]) == 1:
            ticks = [opts["ticks"][0] for _ in range(dim)]
        else:
            ticks = opts["ticks"]

    # Prepare diag/upper/lower
    if type(opts["diag"]) is not list:
        opts["diag"] = [opts["diag"] for _ in range(len(samples))]
    if type(opts["upper"]) is not list:
        opts["upper"] = [opts["upper"] for _ in range(len(samples))]
    # if type(opts['lower']) is not list:
    #    opts['lower'] = [opts['lower'] for _ in range(len(samples))]
    opts["lower"] = None

    # Style
    if opts["style"] in ["dark", "light"]:
        style = os.path.join(
            os.path.dirname(__file__), "matplotlib_{}.style".format(opts["style"])
        )
    else:
        style = opts["style"]

    # Apply custom style as context
    with mpl.rc_context(fname=style):

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

        fig, axes = plt.subplots(1, 1, figsize=opts["fig_size"], **opts["subplots"])

        # Style figure
        fig.subplots_adjust(**opts["fig_subplots_adjust"])
        fig.suptitle(opts["title"], **opts["title_format"])

        col_idx = -1
        for col in range(dim):
            if col not in subset:
                continue
            else:
                col_idx += 1

            current = "diag"

            ax = axes
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
            if limits is not None:
                ax.set_xlim((limits[col][0], limits[col][1]))
                if current != "diag":
                    ax.set_ylim((limits[row][0], limits[row][1]))
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()

            # Ticks
            if ticks is not None:
                ax.set_xticks((ticks[col][0], ticks[col][1] + 0.005))
                if current != "diag":
                    ax.set_yticks((ticks[row][0], ticks[row][1]))

            # Despine
            sns.despine(ax=ax, **opts["despine"])
            #
            # # Formatting axes
            # if opts["lower"] is None or col == dim - 1:
            #     _format_axis(
            #         ax,
            #         xhide=False,
            #         xlabel=labels_dim[col],
            #         yhide=True,
            #         tickformatter=opts["tickformatter"],
            #     )
            #     if opts["labelpad"] is not None:
            #         ax.xaxis.labelpad = opts["labelpad"]
            # else:
            #     _format_axis(ax, xhide=True, yhide=True)

            if opts["tick_labels"] is not None:
                ax.set_xticklabels(
                    (str(opts["tick_labels"][col][0]), str(opts["tick_labels"][col][1]))
                )
                if opts["tick_labelpad"] is not None:
                    ax.tick_params(axis="x", which="major", pad=opts["tick_labelpad"])

            ax.set_ylabel(
                r"$p(\overline{g}_{\mathrm{Na}} | x_o, \theta_{\backslash \overline{g}_{\mathrm{Na}}})$",
                labelpad=-6,
            )
            ax.spines["left"].set_visible(True)

            # Diagonals
            if len(samples) > 0:
                for n, v in enumerate(samples):
                    if opts["diag"][n] == "hist":
                        h = plt.hist(
                            v[:, col],
                            color=opts["samples_colors"][n],
                            **opts["hist_diag"]
                        )
                    elif opts["diag"][n] == "kde":
                        density = gaussian_kde(
                            v[:, col], bw_method=opts["kde_diag"]["bw_method"]
                        )
                        xs = np.linspace(xmin, xmax, opts["kde_diag"]["bins"])
                        ys = density(xs)
                        h = plt.plot(
                            xs,
                            ys,
                            color=opts["samples_colors"][n],
                        )
                    elif opts["diag"][n] == "cond":
                        p_vector = eval_conditional_density(
                            opts["pdfs"][n],
                            [opts["condition"][n]],
                            opts["limits"],
                            col,
                            col,
                            resolution=opts["hist_diag"]["bins"],
                            log=opts["log"],
                        )
                        if opts["thr_at"] is not None:
                            thr = opts["thr_at"]
                            p_vector[p_vector > thr] = 1.0
                            p_vector[p_vector < thr] = 0.0
                        # print("p_vector")
                        if opts["log"] == False:
                            p_vector = p_vector / np.max(
                                p_vector
                            )  # just to scale it to 1
                        h = plt.plot(
                            np.linspace(
                                opts["limits"][col, 0],
                                opts["limits"][col, 1],
                                opts["hist_diag"]["bins"],
                            ),
                            p_vector,
                            c=opts["samples_colors"][n],
                        )
                    else:
                        pass

            ax.set_yticks([np.min(p_vector) - 0.01])
            ax.set_ylim([np.min(p_vector) - 0.01, np.max(p_vector) + 0.03])
            ax.set_yticklabels([0.0])
            ax.set_xlabel("AB-Na", labelpad=-3)

            if len(points) > 0:
                extent = ax.get_ylim()
                for n, v in enumerate(points):
                    h = plt.plot(
                        [v[:, col], v[:, col]],
                        extent,
                        color=opts["points_colors"][n],
                        **opts["points_diag"]
                    )

    return fig, axes


def single2Dmarginal(samples, points=[], **kwargs):
    """Plot samples and points

    See `opts` below for available keyword arguments.
    """
    opts = {
        # what to plot on triagonal and diagonal subplots
        "upper": "hist",  # hist/scatter/None/cond
        "diag": "hist",  # hist/None/cond
        #'lower': None,     # hist/scatter/None  # TODO: implement
        # title and legend
        "title": None,
        "legend": False,
        # labels
        "labels": [],  # for dimensions
        "labels_points": [],  # for points
        "labels_samples": [],  # for samples
        "labelpad": None,  # (int or None). If not None, the labels will be shifted downwards by labelpad
        # colors
        "samples_colors": plt.rcParams["axes.prop_cycle"].by_key()["color"],
        "points_colors": plt.rcParams["axes.prop_cycle"].by_key()["color"],
        # subset
        "subset": None,
        # conditional posterior requires condition and pdf1
        "pdfs": None,
        "condition": None,
        # axes limits
        "limits": [],
        # ticks
        "ticks": [],
        "tickformatter": mpl.ticker.FormatStrFormatter("%g"),
        "tick_labels": None,
        "tick_labelpad": None,  # (None or int). If not None, the ticklabels will be shifted downwards by tick_labelpad
        # options for hist
        "hist_diag": {"alpha": 1.0, "bins": 25, "density": False, "histtype": "step"},
        "hist_offdiag": {
            #'edgecolor': 'none',
            #'linewidth': 0.0,
            "bins": 25,
        },
        # options for kde
        "kde_diag": {"bw_method": "scott", "bins": 100, "color": "black"},
        "kde_offdiag": {"bw_method": "scott", "bins": 25},
        # options for contour
        "contour_offdiag": {"levels": [0.68]},
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
        # matplotlib style
        "style": "../../../.matplotlibrc",
        # other options
        "fig_size": (10, 10),
        "fig_bg_colors": {"upper": None, "diag": None, "lower": None},
        "fig_subplots_adjust": {
            "top": 0.9,
        },
        "subplots": {},
        "despine": {
            "offset": 5,
        },
        "title_format": {"fontsize": 16},
        "log": False,
        "thr_at": None,
    }
    # TODO: add color map support
    # TODO: automatically determine good bin sizes for histograms
    # TODO: get rid of seaborn dependency for despine
    # TODO: add legend (if legend is True)

    single2Dmarginal.defaults = opts.copy()
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
    if opts["labels"] == [] or opts["labels"] is None:
        labels_dim = ["dim {}".format(i + 1) for i in range(dim)]
    else:
        labels_dim = opts["labels"]

    # Prepare limits
    if opts["limits"] == [] or opts["limits"] is None:
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
        if len(opts["limits"]) == 1:
            limits = [opts["limits"][0] for _ in range(dim)]
        else:
            limits = opts["limits"]

    # Prepare ticks
    if opts["ticks"] == [] or opts["ticks"] is None:
        ticks = None
    else:
        if len(opts["ticks"]) == 1:
            ticks = [opts["ticks"][0] for _ in range(dim)]
        else:
            ticks = opts["ticks"]

    # Prepare diag/upper/lower
    if type(opts["diag"]) is not list:
        opts["diag"] = [opts["diag"] for _ in range(len(samples))]
    if type(opts["upper"]) is not list:
        opts["upper"] = [opts["upper"] for _ in range(len(samples))]
    # if type(opts['lower']) is not list:
    #    opts['lower'] = [opts['lower'] for _ in range(len(samples))]
    opts["lower"] = None

    # Style
    if opts["style"] in ["dark", "light"]:
        style = os.path.join(
            os.path.dirname(__file__), "matplotlib_{}.style".format(opts["style"])
        )
    else:
        style = opts["style"]

    # Apply custom style as context
    with mpl.rc_context(fname=style):

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

        fig, axes = plt.subplots(1, 1, figsize=opts["fig_size"], **opts["subplots"])

        # Style figure
        fig.subplots_adjust(**opts["fig_subplots_adjust"])
        fig.suptitle(opts["title"], **opts["title_format"])

        # Style axes
        row_idx = -1
        for row in subset:
            if row not in subset:
                continue
            else:
                row_idx += 1

            col_idx = -1
            for col in subset:
                if col not in subset:
                    continue
                else:
                    col_idx += 1

                if row == col:
                    current = "diag"
                elif row > col:
                    current = "upper"
                else:
                    current = "lower"

                ax = axes
                plt.sca(ax)

                # Background color
                if (
                    current in opts["fig_bg_colors"]
                    and opts["fig_bg_colors"][current] is not None
                ):
                    ax.set_facecolor(opts["fig_bg_colors"][current])

                # Axes
                if current != "upper":
                    continue
                else:
                    # Limits
                    if limits is not None:
                        ax.set_xlim((limits[col][0], limits[col][1]))
                        if current != "diag":
                            ax.set_ylim((limits[row][0], limits[row][1]))
                    xmin, xmax = ax.get_xlim()
                    ymin, ymax = ax.get_ylim()

                    # Formatting axes
                    ax.set_xticks([ticks[col][0], ticks[col][1]])
                    ax.set_yticks([ticks[row][0], ticks[row][1]])
                    ax.set_xticklabels(
                        (
                            str(opts["tick_labels"][col][0]),
                            str(opts["tick_labels"][col][1]),
                        )
                    )
                    ax.set_yticklabels(
                        (opts["tick_labels"][row][0], opts["tick_labels"][row][1])
                    )
                    ax.set_xlabel(opts["labels"][col], labelpad=-3)
                    ax.set_ylabel(opts["labels"][row], labelpad=-9)

                if col != row:
                    print("running this", current)
                    if len(samples) > 0:
                        for n, v in enumerate(samples):
                            if (
                                opts["upper"][n] == "hist"
                                or opts["upper"][n] == "hist2d"
                            ):
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

                                if (
                                    opts["upper"][n] == "kde"
                                    or opts["upper"][n] == "kde2d"
                                ):
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
                                        **opts["contour_offdiag"]
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
                            elif opts["upper"][n] == "cond":
                                p_image = eval_conditional_density(
                                    opts["pdfs"][n],
                                    [opts["condition"][n]],
                                    opts["limits"],
                                    row,
                                    col,
                                    resolution=opts["hist_offdiag"]["bins"],
                                    log=opts["log"],
                                )
                                # print("min", np.min(p_image))
                                # print("max", np.max(p_image))
                                # mythr = 1e-21
                                # p_image[p_image > mythr] = mythr
                                if opts["thr_at"] is not None:
                                    thr = opts["thr_at"]
                                    p_image[p_image > thr] = 1.0
                                    p_image[p_image < thr] = 0.0
                                h = plt.imshow(
                                    p_image,
                                    origin="lower",
                                    extent=[
                                        opts["limits"][col, 0],
                                        opts["limits"][col, 1],
                                        opts["limits"][row, 0],
                                        opts["limits"][row, 1],
                                    ],
                                    aspect="auto",
                                )
                                # plt.colorbar(h)
                            else:
                                pass

                    if len(points) > 0:

                        for n, v in enumerate(points):
                            h = plt.plot(
                                v[:, col],
                                v[:, row],
                                color=opts["points_colors"][n],
                                **opts["points_offdiag"]
                            )

    return fig, axes, h


neutypes = ["PM", "LP", "PY"]


def vis_sample_plain(
    voltage_trace,
    t,
    axV,
    t_on=None,
    t_off=None,
    col=["k", "k", "k"],
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

    if scale_bar:

        # time bar
        axV.plot(
            (offset + 5500) * dt
            + offscale
            + np.arange(scale_bar_breadth)[:: scale_bar_breadth - 1],
            (-40 + offvolt)
            * np.ones_like(np.arange(scale_bar_breadth))[:: scale_bar_breadth - 1],
            lw=1.0,
            color="w",
        )

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

    axV.set_ylim([-85, 340])

    current_counter += 1


def plot_energy_scape(
    out_target,
    time,
    neuron_to_plot,
    t_min,
    t_max,
    figsize,
):
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

    # build energyscape
    all_energies = out_target["all_energies"]
    all_currents_PD = all_energies[:, neuron_to_plot, :]
    t = time[0 : t_max - t_min]

    fig, ax = plt.subplots(2, 1, figsize=figsize, gridspec_kw={"height_ratios": [3, 1]})
    for i in range(8):
        summed_currents_until = np.sum(all_currents_PD[:i, t_min:t_max], axis=0)
        summed_currents_include = np.sum(all_currents_PD[: i + 1, t_min:t_max], axis=0)
        ax[0].fill_between(
            t, summed_currents_until, summed_currents_include, color=cols_hex[i]
        )

    ax[0].set_ylim([0, 150])
    ax[1].plot(t, out_target["data"][neuron_to_plot, t_min:t_max])
    ax[1].set_ylim([-80, 70])

    for a in ax:
        a.set_ylabel("Energy")
        if a == ax[1]:
            a.set_xlabel("Time (ms)")
        a.tick_params(axis="both", which="major")
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)

    ax[0].axes.get_xaxis().set_ticks([])
    ax[1].set_ylabel("Voltage")


def compute_energy_difference(energy_image):
    """
    For a given energy image or energy vector, compute how much more efficient the most
    efficient solution is than the least efficient solution. Returns this value in
    percent and hence ranges from [0, 100].
    """

    if np.any(energy_image > 0.0):

        energy_of_most_efficient = np.min(energy_image[energy_image > 0.0])
        energy_of_least_efficient = np.max(energy_image)

        energy_diff = energy_of_least_efficient - energy_of_most_efficient
        percentage_diff = energy_diff / energy_of_least_efficient
    else:
        print(
            "Problem: not a single model was good. Shape of image", energy_image.shape
        )
        percentage_diff = 0.0

    return percentage_diff


def plot_a_single_energy_matrix(m, ax):
    m = np.fliplr(m)
    ax.imshow(m.T)


def plot_a_single_energy_vector(v, ax):
    ax.plot(v)


def energy_matrix(list_of_all_energy_images, figsize):
    # K = N(N+1)/2 --> N = ...
    number_of_dimensions = int(
        (-1 + np.sqrt(1 + 8 * len(list_of_all_energy_images))) / 2
    )
    fig, axes = plt.subplots(
        number_of_dimensions, number_of_dimensions, figsize=figsize
    )

    counter = 0

    for col in range(number_of_dimensions):
        for row in range(number_of_dimensions):
            current_matrix = list_of_all_energy_images[counter]
            ax = axes[row, col]
            if col > row:
                plot_a_single_energy_matrix(current_matrix, ax)
                counter += 1
            elif col == row:
                plot_a_single_energy_vector(current_matrix, ax)
                counter += 1
    plt.show()


def fill_matrix(m):
    new_m = deepcopy(np.asarray(m))
    shape1, shape2 = np.shape(m)
    for row in range(shape1):
        for col in range(shape2):
            if row > col:
                new_m[row, col] = m[col, row]
    return new_m


def energy_gain_matrix(
    list_of_all_energy_images, figsize, title="", lims=[0.0, 80], ylabel=True
):

    av_matrix_with_energy_gains = []
    for l in list_of_all_energy_images:
        # K = N(N+1)/2 --> N = ...
        number_of_dimensions = int((-1 + np.sqrt(1 + 8 * len(l))) / 2)

        counter = 0
        matrix_with_energy_gains = -0.1 * np.ones(
            (number_of_dimensions, number_of_dimensions)
        )

        for col in range(number_of_dimensions):
            for row in range(number_of_dimensions):
                if col >= row:
                    current_matrix = l[counter]
                    matrix_with_energy_gains[row, col] = compute_energy_difference(
                        current_matrix
                    )
                    counter += 1

        matrix_with_energy_gains = fill_matrix(matrix_with_energy_gains) * 100
        av_matrix_with_energy_gains.append(matrix_with_energy_gains)
    av_matrix_with_energy_gains = np.mean(av_matrix_with_energy_gains, axis=0)
    print("max", np.max(av_matrix_with_energy_gains))
    # print("max gain", np.max(matrix_with_energy_gains))
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(av_matrix_with_energy_gains, clim=lims, cmap="Blues")
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    ax.set_title(title, pad=10)
    ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7])
    ax.set_xticklabels(["Na", "CaT", "CaS", "A", "KCa", "Kd", "H", "Leak"])
    ax.set_yticks([0, 1, 2, 3, 4, 5, 6, 7])
    if ylabel:
        ax.set_yticklabels(["Na", "CaT", "CaS", "A", "KCa", "Kd", "H", "Leak"])
    else:
        ax.set_yticklabels([])
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
    ax.set_ylim([7.5, -0.5])
    return im


def energy_gain_matrix_syn(
    list_of_all_energy_images, figsize, title="", lims=[0.0, 30], ylabel=True
):
    all_mat = []
    for l in list_of_all_energy_images:
        num_x = 8  # 8 parameters per post-synaptic neuron
        num_y = 7  # 7 synapses

        matrix_with_energy_gains = -0.1 * np.ones((num_x * num_y))

        for counter in range(num_x * num_y):
            matrix_with_energy_gains[counter] = compute_energy_difference(l[counter])

        matrix_with_energy_gains = np.reshape(matrix_with_energy_gains, (num_y, num_x))

        matrix_with_energy_gains *= 100
        all_mat.append(matrix_with_energy_gains)

    all_mat = np.mean(all_mat, axis=0)
    print("max", np.max(all_mat))
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(all_mat, clim=lims, cmap="Blues")
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    ax.set_title(title, pad=10)
    ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7])
    ax.set_xticklabels(["Na", "CaT", "CaS", "A", "KCa", "Kd", "H", "Leak"])
    ax.set_yticks([0, 1, 2, 3, 4, 5, 6])
    if ylabel:
        ax.set_yticklabels(
            ["AB-LP", "PD-LP", "AB-PY", "PD-PY", "LP-PD", "LP-PY", "PY-LP"]
        )
    else:
        ax.set_yticklabels([])
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
    ax.set_ylim([6.5, -0.5])
    return im


def energy_gain_matrix_syn_vec(
    list_of_all_energy_images, figsize, title="", lims=[0.0, 30]
):
    all_mat = []
    for l in list_of_all_energy_images:

        matrix_with_energy_gains = -0.1 * np.ones((7))

        for counter in range(7):
            matrix_with_energy_gains[counter] = compute_energy_difference(l[counter])

        matrix_with_energy_gains *= 100
        all_mat.append(matrix_with_energy_gains)

    all_mat = np.mean(all_mat, axis=0)
    all_mat = np.asarray([all_mat]).T
    print("max", np.max(all_mat))
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(all_mat, clim=lims, cmap="Blues")
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    ax.set_title(title, pad=10)
    ax.set_xticklabels([])
    ax.set_yticks([0, 1, 2, 3, 4, 5, 6])
    ax.set_yticklabels(["AB-LP", "PD-LP", "AB-PY", "PD-PY", "LP-PD", "LP-PY", "PY-LP"])
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
    ax.set_ylim([6.5, -0.5])
    return im
