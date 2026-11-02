"""Standalone sklearn matrix check; does not import torch or TabICL's package.

Install a wheel in a separate --target directory and prepend it to PYTHONPATH.
The active training environment is never upgraded/downgraded by this script.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    path = Path(__file__).resolve().parents[1] / 'src/tabicl/prior/_filter_forest.py'
    spec = importlib.util.spec_from_file_location('forest_api_probe', path)
    f = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = f
    spec.loader.exec_module(f)
    assert f.sklearn_tree_api() is not None, sklearn.__version__
    records = []
    for n, d in [(16, 1), (133, 8), (257, 100), (2048, 8), (4096, 16)]:
        for c in (1, 3, 10):
            for kind in ('continuous', 'discrete', 'constant_padding'):
                rng = np.random.RandomState(n+d+c)
                x = rng.normal(size=(n, d)).astype(np.float32)
                labels = rng.randint(max(2, c), size=n)
                y = rng.normal(size=(n, 1)) if c == 1 else np.eye(c)[labels]
                if kind == 'discrete':
                    x = np.round(x)
                if kind == 'constant_padding':
                    x[:, d//2:] = 0
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', UserWarning)
                    reg = ExtraTreesRegressor(n_estimators=25, bootstrap=True, oob_score=True,
                        random_state=1, max_depth=6, n_jobs=1).fit(x, y[:, 0] if c == 1 else y)
                    got = f.planned_oob_prediction(x, y)
                    np.testing.assert_array_equal(got, reg.oob_prediction_.reshape(n, -1))
                    if c > 1:
                        clf = ExtraTreesClassifier(n_estimators=25, bootstrap=True, oob_score=True,
                            random_state=1, max_depth=6, max_features=1., n_jobs=1).fit(x, labels)
                        expected = np.zeros((n, c))
                        expected[:, clf.classes_] = clf.oob_decision_function_
                        pred = f.planned_classification_oob_prediction(x, labels, c)
                        np.testing.assert_allclose(pred, expected, rtol=1e-14, atol=1e-14)
                records.append(dict(n=n, d=d, c=c, kind=kind))
    # Exercise missing votes and a globally absent class with the same bootstrap.
    x = np.arange(16, dtype=np.float32)[:, None]
    labels = np.arange(16) % 2 * 2
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        clf = ExtraTreesClassifier(n_estimators=1, bootstrap=True, oob_score=True,
                    random_state=1, max_depth=6, max_features=1.).fit(x, labels)
        expected = np.zeros((16, 3))
        expected[:, clf.classes_] = clf.oob_decision_function_
        np.testing.assert_allclose(f.planned_classification_oob_prediction(x, labels, 3, 1), expected)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(sklearn=sklearn.__version__, numpy=np.__version__, python=sys.version,
                  convention=f.sklearn_tree_api(), regression_cases=45, classification_cases=31,
                  passed=True, records=records)
    a.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}), flush=True)


if __name__ == '__main__':
    main()
