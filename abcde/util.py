import os
import random
from datetime import datetime
from pathlib import Path
from typing import Union

import numpy as np
import torch
# noinspection PyProtectedMember
from aim import Session
from scipy.stats._stats import _kendall_dis


def fix_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


class ExperimentSetup:
    def __init__(self, experiment: str, create_latest: bool = False):
        """ Keeps track of the experiment path, model save path, log directory, and sessions """
        experiment_path: Path
        self.experiment_path = Path('experiments') / datetime.now().replace(microsecond=0).isoformat()
        self.model_save_path = self.experiment_path / 'models/'
        self.log_dir = self.experiment_path / 'logs/'
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if create_latest:
            latest = Path('experiments/latest/').absolute()
            latest.unlink(missing_ok=True)
            latest.symlink_to(self.experiment_path.absolute(), target_is_directory=True)

        print(f'Logging experiments at: `{self.experiment_path.absolute()}`')
        self.aim_session = Session(experiment=experiment)


def kendall_tau(x, y):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    if x.size != y.size:
        raise ValueError("All inputs to `kendalltau` must be of the same size, "
                         "found x-size %s and y-size %s" % (x.size, y.size))
    elif not x.size or not y.size:
        return np.nan  # Return NaN if arrays are empty

    def count_rank_tie(ranks):
        cnt = np.bincount(ranks).astype('int64', copy=False)
        cnt = cnt[cnt > 1]
        return ((cnt * (cnt - 1) // 2).sum(),
                (cnt * (cnt - 1.) * (cnt - 2)).sum(),
                (cnt * (cnt - 1.) * (2 * cnt + 5)).sum())

    size = x.size
    perm = np.argsort(y)  # sort on y and convert y to dense ranks
    x, y = x[perm], y[perm]
    y = np.r_[True, y[1:] != y[:-1]].cumsum(dtype=np.intp)

    # stable sort on x and convert x to dense ranks
    perm = np.argsort(x, kind='mergesort')
    x, y = x[perm], y[perm]
    x = np.r_[True, x[1:] != x[:-1]].cumsum(dtype=np.intp)

    dis = _kendall_dis(x, y)  # discordant pairs

    obs = np.r_[True, (x[1:] != x[:-1]) | (y[1:] != y[:-1]), True]
    cnt = np.diff(np.where(obs)[0]).astype('int64', copy=False)

    ntie = (cnt * (cnt - 1) // 2).sum()  # joint ties
    xtie, x0, x1 = count_rank_tie(x)  # ties in x, stats
    ytie, y0, y1 = count_rank_tie(y)  # ties in y, stats

    tot = (size * (size - 1)) // 2

    con_minus_dis = tot - xtie - ytie + ntie - 2 * dis
    tau = con_minus_dis / tot
    # Limit range to fix computational errors
    tau = min(1., max(-1., tau))

    return tau


def top_k_ranking_accuracy(y_true: np.ndarray, y_pred: np.ndarray, k: Union[int, float]):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    assert y_true.shape == y_pred.shape
    if isinstance(k, float):
        k = int(k * len(y_true))
        k = max(k, 1)

    intersection = np.intersect1d(y_true[:k], y_pred[:k])
    return len(intersection) / k
