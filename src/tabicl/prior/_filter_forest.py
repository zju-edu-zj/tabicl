"""Execute the fixed full prior-filter forest without estimator scaffolding.

Every tree is freshly fitted. Only bootstrap plans (independent of X and y)
are cached. This intentionally supports a narrow, tested sklearn ABI; callers
use the ordinary estimator for other versions, missing values, or large plans.
"""
from functools import lru_cache
from typing import NamedTuple
import warnings

import numpy as np
import sklearn


SUPPORTED_SKLEARN = '1.7.2'


class _SplitterSeed(NamedTuple):
    """The single seed draw made by sklearn 1.7.2 Splitter.init.

    The tree uses its own C uint32 random stream after this draw. Supplying
    that identical initial value avoids rebuilding a NumPy MT19937 per tree;
    it neither shares nor freezes any of the tree's subsequent random draws.
    """
    value: int

    def randint(self, low, high):
        if low != 0 or high != np.iinfo(np.int32).max:
            raise RuntimeError('Unexpected sklearn splitter RNG contract')
        return self.value


@lru_cache(maxsize=4)
def _bootstrap_plan(n, n_estimators):
    # Same seed stream as BaseForest._make_estimator/_set_random_states.
    seeds = np.random.RandomState(1).randint(np.iinfo(np.int32).max, size=n_estimators)
    plans = []
    votes = np.zeros(n, dtype=np.int64)
    for seed in seeds:
        indices = np.random.RandomState(int(seed)).randint(0, n, n, dtype=np.int32)
        weights = np.bincount(indices, minlength=n).astype(np.float64)
        oob = np.flatnonzero(weights == 0)
        votes[oob] += 1
        weights.flags.writeable = False
        oob.flags.writeable = False
        splitter_seed = _SplitterSeed(int(np.random.RandomState(int(seed)).randint(np.iinfo(np.int32).max)))
        plans.append((splitter_seed, weights, oob))
    missing = bool((votes == 0).any())
    votes[votes == 0] = 1  # sklearn zero-fills rows with no OOB vote.
    votes.flags.writeable = False
    return tuple(plans), votes, missing


def planned_oob_prediction(X, Y, n_estimators=25):
    """Return full depth-six ExtraTrees OOB predictions, or None to fall back.

    Fixed contract: regression squared_error, bootstrap n samples, seed 1,
    max_features=1.0, no sample weights except bootstrap, all default split
    constraints. Y may be one-hot classification targets or real regression.
    The input is never modified; no fitted tree or data enters the cache.
    """
    if (sklearn.__version__ != SUPPORTED_SKLEARN or not isinstance(X, np.ndarray)
            or not isinstance(Y, np.ndarray) or X.ndim != 2 or Y.ndim not in (1, 2)
            or X.dtype.kind != 'f' or Y.dtype.kind not in 'fiu'
            or len(X) != len(Y) or min(X.shape) < 1 or len(X) < 2
            or n_estimators < 1 or len(X)*n_estimators*16 > 2*1024*1024
            or (Y.ndim == 2 and Y.shape[1] < 1)
            or not np.isfinite(X).all() or not np.isfinite(Y).all()):
        return None
    from sklearn.tree._criterion import MSE
    from sklearn.tree._splitter import RandomSplitter
    from sklearn.tree._tree import Tree, DepthFirstTreeBuilder

    with np.errstate(over='ignore'):
        X = np.asarray(X, dtype=np.float32, order='C')
    if not np.isfinite(X).all():
        return None
    Y = np.ascontiguousarray(Y.reshape(len(X), -1), dtype=np.float64)
    n, d = X.shape
    n_outputs = Y.shape[1]
    plans, votes, missing = _bootstrap_plan(n, n_estimators)
    prediction = np.zeros((n, n_outputs), dtype=np.float64)
    n_classes = np.ones(n_outputs, dtype=np.intp)
    for splitter_seed, weights, oob in plans:
        # These are exactly ExtraTreeRegressor._fit's default constructors.
        # Crucially, criterion, splitter and tree are NEW for every dataset.
        criterion = MSE(n_outputs, n)
        splitter = RandomSplitter(criterion, d, 1, 0.0, splitter_seed, None)
        tree = Tree(d, n_classes, n_outputs)
        builder = DepthFirstTreeBuilder(splitter, 2, 1, 0.0, 6, 0.0)
        builder.build(tree, X, Y, weights, None)
        if len(oob):
            prediction[oob] += tree.predict(X[oob]).reshape(len(oob), n_outputs)
    if missing:
        warnings.warn('Some inputs do not have OOB scores. This probably means '
                      'too few trees were used to compute any reliable OOB estimates.',
                      UserWarning, stacklevel=2)
    prediction /= votes[:, None]
    return prediction
