import pytest
import torch

from tabicl._model.tabicl import TabICL
from tabicl.train._optimizer import build_optimizer
from tabicl.train._train_config import build_parser


def _model(attention):
    return TabICL(
        embed_dim=16,
        col_num_blocks=1,
        col_nhead=2,
        col_num_inds=4,
        col_ssmax="qassmax-mlp-elementwise",
        row_num_blocks=1,
        row_nhead=2,
        row_num_cls=2,
        icl_num_blocks=3,
        icl_nhead=2,
        zero_init=False,
        icl_attention=attention,
        icl_attention_impl_version=8 if attention == "dual_path" else 5,
        icl_partition="content_tree",
        icl_global_type="elu",
        icl_block_size=16,
        icl_num_selected_blocks=1,
        icl_qk_norm=True,
        icl_ssmax=False,
    )


def _config():
    return build_parser().parse_args([
        "--device", "cpu", "--muon", "True", "--muon_route_mode", "all_matrix",
        "--lr", "0.0004", "--adamw_lr", "0.0001", "--weight_decay", "0.01",
    ])


@pytest.mark.parametrize("attention", ["dual_path", "adaptive_frontier"])
def test_new_attention_all_matrix_muon_has_finite_update(attention):
    torch.manual_seed(9)
    model = _model(attention)
    optimizer = build_optimizer(model, _config())
    routed = {name: group for group in optimizer.param_groups for name in group["param_names"]}
    if attention == "dual_path":
        assert routed[
            "icl_predictor.tf_icl.blocks.0.attn.global_in_proj_weight"
        ]["use_muon"]
        assert routed[
            "icl_predictor.tf_icl.blocks.0.attn.gate_controller.global_route_weight"
        ]["use_muon"]
    else:
        assert not routed[
            "icl_predictor.tf_icl.blocks.0.attn.summary_bias"
        ]["use_muon"]
    x = torch.randn(2, 35, 3)
    y = torch.randint(0, 3, (2, 25))
    output = model(x, y)
    output.float().square().mean().backward()
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters() if parameter.grad is not None
    )
    optimizer.step()
