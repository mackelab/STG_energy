import numpy as np
import os


def merge_samples(filedir, name='params'):
    """
    Since sampling from MAFs requires rejection sampling, we do this externally.
    This function then merges the externally created files into a single list.

    :param filedir: string to folder, e.g. '../results/samples/samples_13D_new'
    :return: all_conds: list of samples
    """
    files = os.listdir(filedir)
    filenames = []
    for file in files:
        if file[0] != '.' and file != 'readme.txt':
            filenames.append(file)
    all_conds = []
    for fname in filenames:
        data = np.load("{}/{}".format(filedir, fname))  # samples_dir = results/samples/
        conductances = data[name]
        for cond in conductances:
            all_conds.append(cond)
    return all_conds
