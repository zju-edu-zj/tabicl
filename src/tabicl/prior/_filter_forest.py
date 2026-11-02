"""Execute the fixed full prior-filter forest without estimator scaffolding.

Every tree is freshly fitted. Only bootstrap plans (independent of X and y)
are cached. The sklearn path supports audited calling conventions; the native
path has no sklearn private-ABI dependency. Unsupported input shapes use the
ordinary estimator fallback in the caller.
"""
from functools import lru_cache
from typing import NamedTuple
import re
import warnings
import os

import numpy as np
import sklearn


def sklearn_tree_api(version=None):
    """Audited Python-to-Cython calling conventions, not a binary ABI promise.

    Load each installed sklearn's own extension. Never share compiled tree
    objects across environments. Unreleased/future minors take the public path.
    1.4 added monotonic_cst and changed classifier node values to probabilities.
    """
    match = re.fullmatch(r'1\.(\d+)\.\d+(?:\.post\d+)?(?:\+[\w.]+)?',
                         sklearn.__version__ if version is None else version)
    if match is None or not 3 <= int(match[1]) <= 9:
        return None
    return 'legacy' if int(match[1]) == 3 else 'monotonic'


class _SplitterSeed(NamedTuple):
    """The single seed draw made by the audited sklearn Splitter.init.

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
    return _planned_prediction(X, Y, n_estimators)


def planned_classification_oob_prediction(X, labels, n_classes, n_estimators=25):
    """Full class-count/Gini forest. Same real-arithmetic objective as one-hot MSE.

    All classes, features, bootstrap samples and trees are retained. Numerical
    split ties need not produce identical trees to the MSE representation.
    Binary filtering deliberately continues to use the faster single-column MSE.
    Validate integer codes before exposing them to the unchecked Cython criterion.
    """
    if (not isinstance(labels, np.ndarray) or labels.ndim != 1
            or labels.dtype.kind not in 'iu' or len(labels) < 2
            or not isinstance(n_classes, (int, np.integer)) or n_classes < 2
            or n_classes * len(labels) * 8 > 16 * 1024**2
            or labels.min() < 0 or labels.max() >= n_classes):
        return None
    return _planned_prediction(X, labels, n_estimators, n_classes)


def _planned_prediction(X, Y, n_estimators, classes=None):
    mode = os.environ.get('TABICL_PRIOR_FOREST', 'sklearn')
    if mode not in ('sklearn', 'native'):
        raise ValueError('TABICL_PRIOR_FOREST must be sklearn or native')
    api = sklearn_tree_api()
    if ((api is None and mode == 'sklearn') or not isinstance(X, np.ndarray)
            or not isinstance(Y, np.ndarray) or X.ndim != 2 or Y.ndim not in (1, 2)
            or X.dtype.kind != 'f' or Y.dtype.kind not in 'fiu'
            or len(X) != len(Y) or min(X.shape) < 1 or len(X) < 2
            or n_estimators < 1 or len(X)*n_estimators*16 > 2*1024*1024
            or (Y.ndim == 2 and Y.shape[1] < 1)
            or not np.isfinite(X).all() or not np.isfinite(Y).all()):
        return None
    if mode == 'native' and (len(X) > 65536 or X.shape[1] > 4096
            or (Y.ndim == 2 and Y.shape[1] > 256)
            or (classes is not None and classes > 256) or n_estimators > 256):
        # The public estimator remains the caller's fallback outside the
        # bounded native contract. Check before allocating a bootstrap plan.
        return None
    with np.errstate(over='ignore'):
        X = np.asarray(X, dtype=np.float32, order='C')
    if not np.isfinite(X).all():
        return None
    Y = np.ascontiguousarray(Y.reshape(len(X), -1), dtype=np.float64)
    if mode == 'native':
        from .graph_lib._distance import _native_available
        import torch
        if not _native_available() or not hasattr(torch.ops.tabicl_prior_exact, 'forest_oob_cpu'):
            raise RuntimeError('native forest requires the current prior runtime; run scripts/prepare_prior_runtime.py')
        weights, seeds = _native_bootstrap_plan(len(X), n_estimators)
        prediction = torch.ops.tabicl_prior_exact.forest_oob_cpu(
            torch.from_numpy(X), torch.from_numpy(Y), weights, seeds, classes or 0, 6).numpy()
        if _bootstrap_plan(len(X), n_estimators)[2]:
            warnings.warn('Some inputs do not have OOB scores. This probably means '
                          'too few trees were used to compute any reliable OOB estimates.', UserWarning, stacklevel=2)
        return prediction
    from sklearn.tree._criterion import MSE, Gini
    from sklearn.tree._splitter import RandomSplitter
    from sklearn.tree._tree import Tree, DepthFirstTreeBuilder
    n, d = X.shape
    n_outputs = Y.shape[1] if classes is None else 1
    plans, votes, missing = _bootstrap_plan(n, n_estimators)
    width = n_outputs if classes is None else classes
    prediction = np.zeros((n, width), dtype=np.float64)
    n_classes = np.ones(n_outputs, dtype=np.intp) if classes is None else np.array([classes], dtype=np.intp)
    for splitter_seed, weights, oob in plans:
        # These are exactly ExtraTreeRegressor._fit's default constructors.
        # Crucially, criterion, splitter and tree are NEW for every dataset.
        criterion = MSE(n_outputs, n) if classes is None else Gini(1, n_classes)
        args = (criterion, d, 1, 0.0, splitter_seed)
        splitter = RandomSplitter(*args, None) if api == 'monotonic' else RandomSplitter(*args)
        tree = Tree(d, n_classes, n_outputs)
        builder = DepthFirstTreeBuilder(splitter, 2, 1, 0.0, 6, 0.0)
        builder.build(tree, X, Y, weights, None)
        if len(oob):
            values = tree.predict(X[oob]).reshape(len(oob), width)
            if classes is not None and api == 'legacy':
                # Before 1.4 Tree.predict returns leaf counts, not probabilities.
                values /= values.sum(axis=1, keepdims=True)
            prediction[oob] += values
    if missing:
        warnings.warn('Some inputs do not have OOB scores. This probably means '
                      'too few trees were used to compute any reliable OOB estimates.',
                      UserWarning, stacklevel=2)
    prediction /= votes[:, None]
    return prediction


@lru_cache(maxsize=4)
def _native_bootstrap_plan(n, n_estimators):
    import torch
    plans, _, _ = _bootstrap_plan(n, n_estimators)
    # Copies own their data; never expose the readonly cached NumPy weights to
    # a writable Tensor view. No data-dependent state enters this cache.
    weights = torch.from_numpy(np.stack([p[1] for p in plans]))
    seeds = torch.tensor([p[0].value for p in plans], dtype=torch.int64)
    return weights, seeds
