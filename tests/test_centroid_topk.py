"""The fast selector must preserve the existing score kernel and topk policy."""
from copy import deepcopy

import pytest
import torch

from tabicl._model.icl_attention import SetRetrievalAttention

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA routing kernel")


def reference(q, centers, budget, elements):
    from tabicl._model._routing_triton import centroid_scores
    chunk = max(1, elements // (q.shape[0]*q.shape[1]*centers.shape[-2]))
    return torch.cat([centroid_scores(part, centers).topk(budget, -1).indices
                      for part in q.split(chunk, dim=2)], 2).to(torch.int32)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("leaves,budget", [(1, 1), (13, 4), (64, 8), (129, 2), (375, 4), (1024, 1)])
@pytest.mark.parametrize("elements", [8192, 8*1024**2])
def test_exact_selection_with_tails_strided_queries_and_chunk_boundaries(dtype, leaves, budget, elements):
    from tabicl._model._routing_triton import centroid_topk
    torch.manual_seed(151)
    q = torch.randn(2, 131, 3, 64, device="cuda", dtype=dtype).transpose(1, 2)
    centers = torch.randn(2, 3, leaves, 64, device="cuda")
    actual = centroid_topk(q, centers, budget, elements)
    assert actual.is_contiguous() and actual.dtype == torch.int32
    assert torch.equal(actual, reference(q, centers, budget, elements))


@pytest.mark.parametrize("leaves", [17, 64, 129, 375, 1024])
@pytest.mark.parametrize("kind", ["zero", "duplicate", "nan", "inf", "negative_inf", "query_nan"])
def test_framework_tie_and_nonfinite_selection(leaves, kind):
    from tabicl._model._routing_triton import centroid_topk
    torch.manual_seed(729)
    q = torch.randn(1, 137, 2, 64, device="cuda").transpose(1, 2)
    centers = torch.randn(1, 2, leaves, 64, device="cuda")
    if kind == "zero":
        centers.zero_()
    elif kind == "duplicate":
        centers[:, :, 3:] = centers[:, :, :1]
    elif kind == "query_nan":
        q[:, :, 7, 0] = float("nan")
    else:
        centers[:, :, 0, 0] = -float("inf") if kind == "negative_inf" else float(kind)
    assert torch.equal(centroid_topk(q, centers, 4, 8192), reference(q, centers, 4, 8192))


@pytest.mark.parametrize("leaves,budget", [(2, 2), (3, 3), (4, 4), (8, 8),
                                            (17, 1), (17, 3), (64, 5), (129, 7), (375, 8)])
@pytest.mark.parametrize("kind", ["zero", "random", "nan"])
def test_all_supported_budgets_and_selecting_all_leaves(leaves, budget, kind):
    from tabicl._model._routing_triton import centroid_topk
    torch.manual_seed(182)
    q = torch.randn(1, 2, 137, 64, device="cuda")
    centers = torch.randn(1, 2, leaves, 64, device="cuda")
    if kind == "zero":
        centers.zero_()
    elif kind == "nan":
        centers[:, :, 0, 0] = float("nan")
    assert torch.equal(centroid_topk(q, centers, budget, 8192), reference(q, centers, budget, 8192))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("partition", ["projection", "content_tree"])
@pytest.mark.parametrize("mode", ["sparse", "fused_sparse"])
@pytest.mark.parametrize("queries", [137, 513])
def test_attention_outputs_and_gradients_match_reference_selector(dtype, partition, mode, queries):
    torch.manual_seed(72)
    op = SetRetrievalAttention(64, 2, qk_norm=False, block_size=32,
                              num_selected_blocks=2, partition=partition, mode=mode).cuda().to(dtype)
    old = deepcopy(op)
    old.use_fast_centroid_topk = False
    inputs = [torch.randn(1, 2, n, 32, device="cuda", dtype=dtype, requires_grad=True)
              for n in (queries, 291, 291)]
    copies = [x.detach().clone().requires_grad_() for x in inputs]
    method = "_sparse_attention" if mode == "sparse" else "_hybrid_attention"
    expected = getattr(old, method)(*copies)
    actual = getattr(op, method)(*inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for a, b in zip(torch.autograd.grad(actual.float().square().mean(), inputs),
                    torch.autograd.grad(expected.float().square().mean(), copies)):
        # Existing sparse dK/dV reductions can change their summation order
        # between identical launches; route IDs and forward remain bitwise.
        # Grouped backward's atomic reverse-route insertion also changes the
        # final BF16 rounding. Allow two BF16 ULPs near zero in that path.
        tolerance = dict(rtol=0.02, atol=3e-8) if dtype == torch.bfloat16 else dict(rtol=1e-5, atol=1e-9)
        torch.testing.assert_close(a, b, **tolerance)


def test_checkpoint_context_replays_instead_of_reselecting(monkeypatch):
    from tabicl._model import _routing_triton
    op = SetRetrievalAttention(64, 2, qk_norm=False, block_size=32,
                              num_selected_blocks=2, partition="content_tree").cuda()
    q, k = [torch.randn(1, 2, n, 32, device="cuda") for n in (137, 291)]
    record, replay = op.checkpoint_contexts()
    with record:
        expected = op.route(q, k)
    def fail(*args, **kwargs):
        raise AssertionError("checkpoint recomputation must replay discrete routes")
    monkeypatch.setattr(_routing_triton, "centroid_topk", fail)
    with replay:
        actual = op.route(q, k)
    assert all(a is b for a, b in zip(actual, expected))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("leaves,budget,support", [(1, 1, 17), (13, 4, 401), (64, 8, 2041), (80, 4, 2537)])
def test_fused_route_quality_matches_portable_reference(dtype, leaves, budget, support):
    from tabicl._model._routing_triton import centroid_topk_quality

    torch.manual_seed(183)
    q = torch.randn(2, 3, 137, 32, device="cuda", dtype=dtype)
    centers = torch.randn(2, 3, leaves, 32, device="cuda")
    lengths = torch.full((leaves,), 32, device="cuda", dtype=torch.long)
    lengths[-1] = support - 32 * (leaves - 1)
    actual_ids, actual_quality = centroid_topk_quality(
        q, centers, lengths, budget, support, 8192
    )
    module = SetRetrievalAttention(64, 2, block_size=32, num_selected_blocks=budget)
    module.use_triton = False
    expected_ids, expected_quality = module._centroid_route_quality(
        q, centers, lengths, budget, support
    )
    assert torch.equal(actual_ids, expected_ids)
    torch.testing.assert_close(actual_quality, expected_quality, atol=2e-5, rtol=2e-5)
    assert torch.isfinite(actual_quality).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_independent_v1_cuda_migration_preserves_initial_output(dtype):
    torch.manual_seed(184)
    legacy = SetRetrievalAttention(
        64, 2, block_size=32, num_selected_blocks=2, partition="content_tree",
        fusion="legacy",
    ).cuda().to(dtype).eval()
    v1 = SetRetrievalAttention(
        64, 2, block_size=32, num_selected_blocks=2, partition="content_tree",
        fusion="independent_v1",
    ).cuda().to(dtype).eval()
    v1.load_state_dict(legacy.state_dict(), strict=False)
    with torch.no_grad():
        v1.local_out_delta.weight.zero_()
        v1.fusion_route_weight.zero_()
        v1.fusion_route_bias.zero_()
    query = torch.randn(1, 137, 64, device="cuda", dtype=dtype)
    support = torch.randn(1, 291, 64, device="cuda", dtype=dtype)
    expected = legacy(query, support, support)
    actual = v1(query, support, support)
    tolerance = 2e-3 if dtype == torch.bfloat16 else 2e-6
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
