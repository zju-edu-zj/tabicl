"""Layerwise Dual-Path inference with bounded K/V lifetime."""

from __future__ import annotations

import torch

from .dual_path_v3 import DualPathRetrievalAttentionV3


def _apply_block(block, rows, prepared, rope):
    if not block.norm_first:
        raise NotImplementedError("layerwise Dual-Path currently requires pre-norm")
    normalized = block.norm1(rows)
    attention = block.attn.apply_prepared(normalized, prepared, rope=rope)
    output = rows + block.dropout1(attention)
    return output + block._ff_block(block.norm2(output))


@torch.inference_mode()
def layerwise_dual_path_forward(
    encoder,
    support,
    query,
    *,
    projection_chunk_rows=65536,
    support_query_tile_rows=256,
    query_tile_rows=256,
    page_location="device",
    return_support=False,
):
    """Execute each layer with one temporary page store and a strict barrier."""
    if support.ndim != 3 or query.ndim != 3:
        raise ValueError("support and query must have shape [B,N,D]")
    if support.shape[0] != query.shape[0] or support.shape[-1] != query.shape[-1]:
        raise ValueError("support and query batch/model dimensions must match")
    if min(projection_chunk_rows, support_query_tile_rows, query_tile_rows) < 1:
        raise ValueError("tile sizes must be positive")

    current_support, current_query = support, query
    for layer_index, block in enumerate(encoder.blocks):
        if not isinstance(block.attn, DualPathRetrievalAttentionV3):
            raise TypeError("layerwise executor requires every ICL block to use Dual-Path V3")
        prepared = block.attn.prepare_support_streaming(
            current_support,
            block.norm1,
            rope=encoder.rope,
            location=page_location,
            chunk_rows=projection_chunk_rows,
        )

        next_query = torch.empty_like(current_query)
        for start in range(0, current_query.shape[1], query_tile_rows):
            end = min(start + query_tile_rows, current_query.shape[1])
            next_query[:, start:end] = _apply_block(
                block, current_query[:, start:end], prepared, encoder.rope
            )

        last_layer = layer_index + 1 == len(encoder.blocks)
        if not last_layer or return_support:
            next_support = torch.empty_like(current_support)
            for start in range(0, current_support.shape[1], support_query_tile_rows):
                end = min(start + support_query_tile_rows, current_support.shape[1])
                next_support[:, start:end] = _apply_block(
                    block, current_support[:, start:end], prepared, encoder.rope
                )
        else:
            next_support = None

        # This assignment is the layer barrier. All reads from S_l finish before
        # S_l is released or replaced by S_{l+1}.
        del prepared
        current_query = next_query
        if next_support is not None:
            current_support = next_support

    return (current_support, current_query) if return_support else current_query
