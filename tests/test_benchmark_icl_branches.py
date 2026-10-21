"""Verify cross-device benchmark scheduling without requiring multiple GPUs."""

import json
from pathlib import Path
import subprocess
import sys


def test_benchmark_plan_keeps_dense_comparisons_on_the_same_gpu(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/benchmark_icl_branches.py"
    output = subprocess.check_output([
        sys.executable, str(script), "--output", str(tmp_path),
        "--gpus", "0", "2", "--rows", "1024", "4096",
        "--variants", "dense", "global_elu", "global_sparse_alternating",
        "--plan",
    ], text=True)
    plan = json.loads(output)
    assert set(plan) == {"0", "2"}
    assert len(plan["0"]) == len(plan["2"]) == 3
    assert all("r1024_" in name for name in plan["0"])
    assert all("r4096_" in name for name in plan["2"])
    assert list(tmp_path.iterdir()) == []
