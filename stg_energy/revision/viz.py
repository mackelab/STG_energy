import numpy as np
import matplotlib.pyplot as plt
from stg_energy.common import col
from pyloric.utils import energy_of_membrane


def vis_trace(out_target, t, figsize, offset=None, ylabel=True):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    time_len = int(3 * 1000 / 0.025)  # 3 seconds
    if offset is None:
        offset = 164000
    print("Showing :  ", time_len / 40000, "seconds")
    print("Scalebar indicates:  50mV")

    current_col = 0
    Vx = out_target["voltage"]
    for j in range(3):
        if time_len is not None:
            ax.plot(
                t[:time_len:5] / 1000,
                Vx[j, 10000 + offset : 10000 + offset + time_len : 5] + 130.0 * (2 - j),
                linewidth=0.6,
                c="k",
            )
        else:
            ax.plot(t / 1000, Vx[j] + 120.0 * (2 - j), lw=0.6, c="k")
        current_col += 1

    box = ax.get_position()

    ax.set_position([box.x0, box.y0, box.width, box.height])

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_yticks([])
    if ylabel:
        ax.set_ylabel("Voltage")
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylim([-90, 320])
    ax.set_xlim([-0.1, (t[:time_len:5] / 1000)[-1] + 0.2])
    ax.set_xlim([0, 3])

    # scale bar
    end_val_x = (t[:time_len:5] / 1000)[-1] + 0.15
    ax.plot([end_val_x, end_val_x], [260, 310], c="k")
