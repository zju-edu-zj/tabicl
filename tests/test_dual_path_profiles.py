import torch

from tabicl._model.dual_path_profiles import (
    OnlineHiddenProfile,
    OneShotLayerwiseProfile,
    build_online_hidden_snapshot,
    one_shot_layerwise_forward,
    online_hidden_forward,
)
from tabicl._model.encoders import Encoder


def _encoder():
    encoder = Encoder(
        2,
        32,
        2,
        64,
        attention_type="dual_path",
        sparse_every=1,
        block_size=32,
        num_selected_blocks=32,
        query_chunk_size=4,
        router="hierarchical",
        router_beam=64,
        partition="content_tree",
        dual_controller="none",
        qk_norm=True,
        zero_init=False,
    ).eval()
    for block in encoder.blocks:
        block.attn.use_triton = False
    return encoder


def test_formal_profiles_match_regular_encoder(tmp_path):
    torch.manual_seed(20261125)
    encoder = _encoder()
    support = torch.randn(1, 65, 32)
    query = torch.randn(1, 9, 32)
    with torch.inference_mode():
        expected = encoder(
            torch.cat((support, query), 1), train_size=support.shape[1]
        )[:, support.shape[1] :]
        one_shot = one_shot_layerwise_forward(
            encoder,
            support,
            query,
            OneShotLayerwiseProfile(
                projection_chunk_rows=17,
                support_query_tile_rows=5,
                query_tile_rows=3,
                page_location="host",
            ),
        )
        snapshot = build_online_hidden_snapshot(
            encoder,
            support,
            tmp_path / "snapshot",
            model_fingerprint="unit-test",
            projection_chunk_rows=17,
            support_query_tile_rows=5,
        )
        online = online_hidden_forward(
            encoder,
            query,
            snapshot,
            OnlineHiddenProfile(
                query_tile_rows=3,
                hidden_page_rows=7,
                max_cached_pages=2,
                max_cache_bytes=2 * 7 * 32 * 4,
                max_staging_buffers=1,
                max_staging_bytes=65 * 32 * 4,
            ),
        )
    torch.testing.assert_close(one_shot, expected, atol=6e-6, rtol=6e-5)
    torch.testing.assert_close(online, expected, atol=6e-6, rtol=6e-5)
