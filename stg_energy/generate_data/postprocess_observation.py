import numpy as np


def build_reordered_stats(stats):
    """
    Reorder the summary statistics because the new version of `pyloric` has a new order.
    """
    r = np.zeros((18))
    r[0:4] = stats[0:4]
    r[4:7] = stats[8:11]
    r[7:9] = stats[13:15]
    r[9:11] = stats[6:8]
    r[11:13] = stats[4:6]
    r[13:15] = stats[11:13]
    r[15:18] = stats[15:18]
    return r


def obtain_lp_py_gap(stats):
    """
    In the new version of `pyloric`, we return the start-to-start delay between LP and PY. Before, 
    it was the start-to-start delay between AB/PD and PY. This function corrects for this.
    """
    stats[10] = stats[10] - stats[9]
    return stats


stats_original = np.load('../../results/experimental_data/190807_summstats_prep845_082_0044.npz')['summ_stats']
new_stats = obtain_lp_py_gap(build_reordered_stats(stats_original))
np.save('../../results/experimental_data/201210_summstats_reordered_prep845_082_0044', new_stats)