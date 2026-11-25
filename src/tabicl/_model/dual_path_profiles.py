"""Formal one-shot and online-hidden inference profiles for Dual-Path V3."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

from .content_index_v3 import ContentIndexV3, DualPathLayout
from .dual_path_snapshot import SnapshotProfile, SupportSnapshot
from .dual_path_v3 import DualPathRetrievalAttentionV3
from .layerwise_inference import layerwise_dual_path_forward
from .paged_runtime import (
    BoundedPageCache,
    BoundedStagingPool,
    HiddenSnapshotRuntime,
)


@dataclass(frozen=True)
class OneShotLayerwiseProfile:
    projection_chunk_rows: int = 65536
    support_query_tile_rows: int = 256
    query_tile_rows: int = 256
    page_location: str = "device"


@dataclass(frozen=True)
class OnlineHiddenProfile:
    query_tile_rows: int = 256
    hidden_page_rows: int = 4096
    max_cached_pages: int = 64
    max_cache_bytes: int = 1 << 30
    max_staging_buffers: int = 2
    max_staging_bytes: int = 1 << 28
    hidden_location: str = "host"


def _index_state(index: ContentIndexV3) -> dict[str, Tensor]:
    return {
        "order": index.order,
        "leaf_representatives": index.leaf_representatives,
        "leaf_rep_counts": index.leaf_rep_counts,
        "leaf_lengths": index.leaf_lengths,
        "tree_representatives": index.tree_representatives,
        "tree_rep_counts": index.tree_rep_counts,
    }


def _restore_index(
    state: dict[str, Tensor],
    *,
    layout: DualPathLayout,
    support_size: int,
    device: torch.device,
) -> ContentIndexV3:
    tensors = {name: value.to(device) for name, value in state.items()}
    num_leaves = int(tensors["leaf_lengths"].numel())
    nodes = int(tensors["tree_rep_counts"].shape[0])
    leaf_capacity = (nodes + 1) // 2
    return ContentIndexV3(
        order=tensors["order"],
        leaf_representatives=tensors["leaf_representatives"],
        leaf_rep_counts=tensors["leaf_rep_counts"],
        leaf_lengths=tensors["leaf_lengths"],
        tree_representatives=tensors["tree_representatives"],
        tree_rep_counts=tensors["tree_rep_counts"],
        leaf_capacity=leaf_capacity,
        num_leaves=num_leaves,
        support_size=support_size,
        layout=layout,
    )


def one_shot_layerwise_forward(
    encoder,
    support: Tensor,
    query: Tensor,
    profile: OneShotLayerwiseProfile | None = None,
):
    profile = OneShotLayerwiseProfile() if profile is None else profile
    return layerwise_dual_path_forward(
        encoder,
        support,
        query,
        projection_chunk_rows=profile.projection_chunk_rows,
        support_query_tile_rows=profile.support_query_tile_rows,
        query_tile_rows=profile.query_tile_rows,
        page_location=profile.page_location,
    )


@torch.inference_mode()
def build_online_hidden_snapshot(
    encoder,
    support: Tensor,
    directory,
    *,
    model_fingerprint: str,
    projection_chunk_rows: int = 65536,
    support_query_tile_rows: int = 256,
    generation: int = 0,
) -> SupportSnapshot:
    """Persist every layer input while retaining no layer K/V tensors."""

    if support.ndim != 3 or support.shape[0] != 1:
        raise ValueError("online hidden snapshots require support shaped [1,N,D]")
    if min(projection_chunk_rows, support_query_tile_rows) < 1:
        raise ValueError("tile sizes must be positive")
    if encoder.rope is not None:
        raise NotImplementedError("online hidden snapshots need offset-aware RoPE")
    if not encoder.blocks:
        raise ValueError("encoder must contain at least one layer")
    first_attention = encoder.blocks[0].attn
    if not isinstance(first_attention, DualPathRetrievalAttentionV3):
        raise TypeError("online hidden snapshots require Dual-Path V3 layers")
    layout = first_attention.layout
    snapshot = SupportSnapshot.create(
        profile=SnapshotProfile.ONLINE_HIDDEN,
        support_size=support.shape[1],
        model_dim=support.shape[2],
        expected_layers=len(encoder.blocks),
        model_fingerprint=model_fingerprint,
        layout=asdict(layout),
        generation=generation,
        directory=directory,
    )
    current = support
    for layer_index, block in enumerate(encoder.blocks):
        if not isinstance(block.attn, DualPathRetrievalAttentionV3):
            raise TypeError("every snapshot layer must use Dual-Path V3")
        if block.attn.layout != layout:
            raise ValueError("all snapshot layers must share one Dual-Path layout")
        prepared = block.attn.prepare_support_streaming(
            current,
            block.norm1,
            rope=None,
            location="host",
            chunk_rows=projection_chunk_rows,
        )
        elu_state, elu_normalizer = prepared.global_state["elu"]
        snapshot.add_layer(
            layer_index,
            current[0].detach().cpu(),
            global_state={
                "elu_state": elu_state,
                "elu_normalizer": elu_normalizer,
            },
            index_state=_index_state(prepared.index),
        )
        if layer_index + 1 < len(encoder.blocks):
            next_support = torch.empty_like(current)
            for start in range(0, current.shape[1], support_query_tile_rows):
                end = min(start + support_query_tile_rows, current.shape[1])
                rows = current[:, start:end]
                normalized = block.norm1(rows)
                attention = block.attn.apply_prepared(normalized, prepared)
                output = rows + block.dropout1(attention)
                next_support[:, start:end] = output + block._ff_block(block.norm2(output))
            current = next_support
        del prepared
    snapshot.seal()
    return snapshot


def _selected_original_rows(index: ContentIndexV3, selected: Tensor) -> tuple[Tensor, Tensor]:
    leaf_rows = index.layout.logical_leaf_rows
    positions = (
        selected.long()[..., None] * leaf_rows
        + torch.arange(leaf_rows, device=selected.device)
    ).flatten(-2)
    valid = positions < index.support_size
    positions = positions.clamp_max(index.support_size - 1)
    order = index.order[:, :, None].expand(-1, -1, positions.shape[2], -1)
    original = order.gather(3, positions)
    return original[0], valid[0]


def _project_selected_hidden(block, provider, row_ids: Tensor, valid: Tensor):
    attention = block.attn
    target = attention.in_proj_weight.device
    unique, inverse = torch.unique(row_ids.flatten(), sorted=True, return_inverse=True)
    hidden = provider.read_rows(unique, device=target).to(attention.in_proj_weight.dtype)
    normalized = block.norm1(hidden)
    selected = normalized.index_select(0, inverse.to(target)).reshape(
        *row_ids.shape, attention.embed_dim
    )
    heads, queries, rows, _ = selected.shape
    key = torch.empty(
        (1, heads, queries, rows, attention.head_dim),
        device=target,
        dtype=selected.dtype,
    )
    value = torch.empty_like(key)
    for head in range(heads):
        start = head * attention.head_dim
        end = start + attention.head_dim
        key[0, head] = F.linear(
            selected[head],
            attention.in_proj_weight[attention.embed_dim + start : attention.embed_dim + end],
            attention.in_proj_bias[attention.embed_dim + start : attention.embed_dim + end],
        )
        value[0, head] = F.linear(
            selected[head],
            attention.in_proj_weight[2 * attention.embed_dim + start : 2 * attention.embed_dim + end],
            attention.in_proj_bias[2 * attention.embed_dim + start : 2 * attention.embed_dim + end],
        )
    key = attention.qk_norm.normalize_k(key)
    return key, value, valid[None].to(target)


def _online_attention(block, normalized_query: Tensor, layer_runtime, layout: DualPathLayout):
    from ._elu_kernel import apply_elu_state

    attention = block.attn
    device = normalized_query.device
    index = _restore_index(
        layer_runtime.index_state(device),
        layout=layout,
        support_size=layer_runtime.snapshot.manifest.support_size,
        device=device,
    )
    sparse_q, global_q = attention._project_queries(normalized_query, rope=None)
    route = index.search(sparse_q)
    row_ids, valid = _selected_original_rows(index, route.logical_leaf_ids)
    key, value, valid = _project_selected_hidden(
        block, layer_runtime.hidden_provider, row_ids, valid
    )
    visible = min(index.support_size, layout.selected_rows)
    attention_q = attention.qk_norm.scale_query(sparse_q, visible)
    logits = torch.einsum(
        "bhqd,bhqkd->bhqk", attention_q.float(), key.float()
    ) / math.sqrt(attention.head_dim)
    logits = logits.masked_fill(~valid, -float("inf"))
    probability = logits.softmax(-1)
    sparse_heads = torch.einsum(
        "bhqk,bhqkd->bhqd", probability.to(value.dtype), value
    )
    evidence = torch.stack(
        (
            probability.max(-1).values,
            -(probability * probability.clamp_min(1e-20).log()).sum(-1),
            probability.square().sum(-1).clamp_min(1e-20).reciprocal(),
        ),
        -1,
    )
    global_state = layer_runtime.global_state(device)
    global_heads = apply_elu_state(
        global_q,
        global_state["elu_state"],
        global_state["elu_normalizer"],
    )
    return attention._fuse(
        normalized_query,
        sparse_q,
        global_heads,
        sparse_heads,
        evidence,
        route,
        index.support_size,
    )


@torch.inference_mode()
def online_hidden_forward(
    encoder,
    query: Tensor,
    snapshot: SupportSnapshot | str | Path,
    profile: OnlineHiddenProfile | None = None,
) -> Tensor:
    """Execute queries against fixed per-layer hidden snapshots."""

    if query.ndim != 3 or query.shape[0] != 1:
        raise ValueError("online hidden queries require shape [1,Q,D]")
    if encoder.rope is not None:
        raise NotImplementedError("online hidden inference needs offset-aware RoPE")
    profile = OnlineHiddenProfile() if profile is None else profile
    owned_snapshot = not isinstance(snapshot, SupportSnapshot)
    if owned_snapshot:
        snapshot = SupportSnapshot.load(snapshot, lazy=True, mmap=True)
    if snapshot.profile != SnapshotProfile.ONLINE_HIDDEN:
        raise ValueError("snapshot is not an online-hidden profile")
    if snapshot.manifest.expected_layers != len(encoder.blocks):
        raise ValueError("snapshot layer count does not match the encoder")
    layout = DualPathLayout(**snapshot.manifest.layout)
    cache = BoundedPageCache(profile.max_cached_pages, profile.max_cache_bytes)
    staging = BoundedStagingPool(
        profile.max_staging_buffers,
        profile.max_staging_bytes,
        pin_memory=query.device.type == "cuda",
    )
    runtime = HiddenSnapshotRuntime(
        snapshot,
        location=profile.hidden_location,
        device=query.device,
        page_rows=profile.hidden_page_rows,
        max_open_layers=1,
        page_cache=cache,
        staging_pool=staging,
    )
    current = query
    try:
        for layer_index, block in enumerate(encoder.blocks):
            next_query = torch.empty_like(current)
            layer_runtime = runtime.open_layer(layer_index)
            for start in range(0, current.shape[1], profile.query_tile_rows):
                end = min(start + profile.query_tile_rows, current.shape[1])
                rows = current[:, start:end]
                attention = _online_attention(
                    block, block.norm1(rows), layer_runtime, layout
                )
                output = rows + block.dropout1(attention)
                next_query[:, start:end] = output + block._ff_block(block.norm2(output))
            current = next_query
    finally:
        runtime.close()
        if owned_snapshot:
            snapshot.close()
    return current


__all__ = [
    "OneShotLayerwiseProfile",
    "OnlineHiddenProfile",
    "one_shot_layerwise_forward",
    "build_online_hidden_snapshot",
    "online_hidden_forward",
]
