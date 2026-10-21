import importlib.util
import json
from pathlib import Path

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("analyze_checkpoint", SCRIPT)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def test_model_statistics_detect_components_and_nonfinite_values():
    result = analysis.analyze_model_state({
        "col_embedder.weight": torch.tensor([1.0, -1.0]),
        "icl_predictor.blocks.2.attn.fusion_gate_weight": torch.tensor([0.0, 2.0]),
        "icl_predictor.blocks.2.attn.qk_norm.q_weight": torch.tensor([float("nan")]),
    })
    assert result["total_numel"] == 5
    assert result["nonfinite_total"] == 1
    assert set(result["by_component"]) == {"column_encoder", "icl_encoder"}
    assert "icl_predictor.blocks.2.attn.fusion_gate_weight" in result["special_tensors"]
    assert "icl_predictor.blocks.2" in result["by_layer"]


def test_training_and_validation_log_summaries(tmp_path):
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text("\n".join([
        json.dumps({"step": 1, "elapsed_s": 2.0, "ce": 2.0, "train_time": 1.0, "prior_time": 0.5}),
        json.dumps({"step": 2, "elapsed_s": 4.0, "ce": 1.0, "train_time": 1.5, "prior_time": 0.7}),
    ]) + "\n")
    result = analysis.analyze_metrics(metrics)
    assert result["last_step"] == 2
    assert result["numeric_metrics"]["ce"]["last_100_mean"] == 1.5
    assert result["mean_train_time_s"] == 1.25

    validation = tmp_path / "validation.jsonl"
    validation.write_text(json.dumps({"step": 2, "cases": [
        {"id": "a", "support": 4096, "ce": 1.0, "accuracy": 0.7,
         "diag/icl_predictor.blocks.2.attn/retained_mass": 0.8},
        {"id": "b", "support": 4096, "ce": 2.0, "accuracy": 0.5,
         "diag/icl_predictor.blocks.2.attn/retained_mass": 0.6},
    ]}) + "\n")
    val = analysis.analyze_validation(validation)
    assert val["steps"]["2"]["support_4096/ce"] == 1.5
    assert val["steps"]["2"]["support_4096/diag/icl_predictor.blocks.2.attn/retained_mass"] == 0.7
