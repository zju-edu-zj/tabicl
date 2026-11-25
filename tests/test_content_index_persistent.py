import json

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from tabicl._model._content_tree import content_tree_order
from tabicl._model.content_index_persistent import (
    ALGORITHM_VERSION,
    FORMAT_NAME,
    MANIFEST_NAME,
    SCHEMA_VERSION,
    PersistentContentIndex,
    PersistentContentIndexBuilder,
    PersistentIndexLayout,
)


def _keys(num_heads=2, rows=53, head_dim=8):
    generator = torch.Generator().manual_seed(20261125)
    return torch.randn(num_heads, rows, head_dim, generator=generator)


def _build(path, keys, *, layout, shard_rows, row_base=0):
    builder = PersistentContentIndexBuilder(
        path,
        num_heads=keys.shape[0],
        head_dim=keys.shape[2],
        layout=layout,
    )
    row_ids = torch.arange(
        row_base, row_base + keys.shape[1], dtype=torch.int64
    )
    for start in range(0, keys.shape[1], shard_rows):
        end = min(start + shard_rows, keys.shape[1])
        for head in range(keys.shape[0]):
            builder.ingest(head, row_ids[start:end], keys[head, start:end])
    index = builder.finalize()
    assert builder.finalized
    return index


def _array(index, head, name):
    return np.asarray(index._arrays[head][name].array)


def _leaf_rows(index, head):
    leaf_ids = torch.arange(
        index.manifest["heads"][head]["leaves"], dtype=torch.int64
    )
    rows, valid = index.rows_for_leaves(head, leaf_ids)
    return [rows[leaf, valid[leaf]].tolist() for leaf in range(leaf_ids.numel())]


def _reference_leaf_sets(keys, leaf_rows):
    heads, rows, head_dim = keys.shape
    direction = F.normalize(
        torch.arange(1, head_dim + 1, dtype=torch.float32), dim=0
    )
    initial = torch.einsum("hsd,d->hs", keys.float(), direction).argsort(
        dim=-1, stable=True
    )
    order = content_tree_order(
        keys[None], initial[None], leaf_rows, use_triton=False
    )[0]
    result = []
    for head in range(heads):
        result.append([
            set(order[head, start : start + leaf_rows].tolist())
            for start in range(0, rows, leaf_rows)
        ])
    return result


def test_content_tree_leaf_sets_match_existing_reference_without_ties(tmp_path):
    layout = PersistentIndexLayout(
        logical_leaf_rows=4,
        physical_page_rows=10,
        build_chunk_rows=7,
    )
    keys = _keys(rows=37)
    index = _build(tmp_path / "index", keys, layout=layout, shard_rows=7)
    expected = _reference_leaf_sets(keys, layout.logical_leaf_rows)

    assert index.manifest["partition"] == "content_tree"
    assert "rank-sampled-farthest-pair" in index.manifest["partition_semantics"]
    for head in range(keys.shape[0]):
        actual = [set(rows) for rows in _leaf_rows(index, head)]
        assert actual == expected[head]


def test_different_ingest_shards_produce_identical_content_tree_leaves(tmp_path):
    keys = _keys(rows=61)
    small_layout = PersistentIndexLayout(
        logical_leaf_rows=4,
        physical_page_rows=11,
        build_chunk_rows=7,
    )
    large_layout = PersistentIndexLayout(
        logical_leaf_rows=4,
        physical_page_rows=11,
        build_chunk_rows=128,
    )
    small = _build(tmp_path / "small", keys, layout=small_layout, shard_rows=7)
    large = _build(tmp_path / "large", keys, layout=large_layout, shard_rows=61)

    for head in range(keys.shape[0]):
        assert _leaf_rows(small, head) == _leaf_rows(large, head)
        assert np.allclose(
            _array(small, head, "leaf_representatives"),
            _array(large, head, "leaf_representatives"),
        )


def test_physical_page_size_does_not_change_logical_leaf_sets(tmp_path):
    keys = _keys(rows=47)
    first = _build(
        tmp_path / "pages-10",
        keys,
        layout=PersistentIndexLayout(
            logical_leaf_rows=4,
            physical_page_rows=10,
            build_chunk_rows=9,
        ),
        shard_rows=9,
    )
    second = _build(
        tmp_path / "pages-13",
        keys,
        layout=PersistentIndexLayout(
            logical_leaf_rows=4,
            physical_page_rows=13,
            build_chunk_rows=9,
        ),
        shard_rows=9,
    )

    for head in range(keys.shape[0]):
        assert _leaf_rows(first, head) == _leaf_rows(second, head)
        assert not np.array_equal(
            _array(first, head, "leaf_page_ids"),
            _array(second, head, "leaf_page_ids"),
        )


def test_finalize_manifest_and_explicit_leaf_page_mapping(tmp_path):
    layout = PersistentIndexLayout(
        logical_leaf_rows=4,
        physical_page_rows=10,
        build_chunk_rows=8,
    )
    index = _build(tmp_path / "index", _keys(), layout=layout, shard_rows=8)
    manifest = json.loads(
        (index.path / MANIFEST_NAME).read_text(encoding="utf-8")
    )

    assert manifest["format"] == FORMAT_NAME
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["algorithm_version"] == ALGORITHM_VERSION
    assert manifest["row_id_dtype"] == "int64"
    assert not any("inverse" in path.name for path in index.path.iterdir())
    assert not any(path.name.startswith(".head-") for path in index.path.iterdir())
    for head, metadata in enumerate(manifest["heads"]):
        counts = _array(index, head, "leaf_counts")
        page_ids = _array(index, head, "leaf_page_ids")
        page_offsets = _array(index, head, "leaf_page_offsets")
        page_leaf_offsets = _array(index, head, "page_leaf_offsets")
        page_leaf_ids = _array(index, head, "page_leaf_ids")
        page_row_counts = _array(index, head, "page_row_counts")
        assert counts.sum() == 53
        assert counts.max() <= layout.logical_leaf_rows
        assert np.all(page_offsets + counts <= layout.physical_page_rows)
        for page in range(metadata["pages"]):
            leaves = page_leaf_ids[
                page_leaf_offsets[page] : page_leaf_offsets[page + 1]
            ]
            assert np.all(page_ids[leaves] == page)
            assert counts[leaves].sum() == page_row_counts[page]


def test_search_matches_chunked_reference_for_exhaustive_index(tmp_path):
    layout = PersistentIndexLayout(
        logical_leaf_rows=4,
        physical_page_rows=10,
        selected_logical_leaves=5,
        exhaustive_leaf_limit=100,
        build_chunk_rows=9,
    )
    index = _build(tmp_path / "index", _keys(rows=67), layout=layout, shard_rows=9)
    query = torch.randn(2, 4, 8, generator=torch.Generator().manual_seed(41))
    result = index.search(query)
    reference = index.reference_search(query, chunk_leaves=2)
    large_chunk = index.reference_search(query, chunk_leaves=10_000)

    assert torch.equal(result.logical_leaf_ids, reference.logical_leaf_ids)
    assert torch.equal(reference.logical_leaf_ids, large_chunk.logical_leaf_ids)
    assert torch.equal(result.physical_page_ids, reference.physical_page_ids)
    assert torch.equal(result.candidate_rows, reference.candidate_rows)


def test_save_load_and_int64_row_ids_roundtrip(tmp_path):
    layout = PersistentIndexLayout(
        logical_leaf_rows=4,
        physical_page_rows=9,
        selected_logical_leaves=4,
        build_chunk_rows=6,
    )
    row_base = 10_000_000_000
    index = _build(
        tmp_path / "source",
        _keys(num_heads=1, rows=43, head_dim=6),
        layout=layout,
        shard_rows=6,
        row_base=row_base,
    )
    copied = index.save(tmp_path / "copy")
    loaded = PersistentContentIndex.load(tmp_path / "copy")
    query = torch.randn(1, 2, 6)
    result = loaded.search(query)
    rows, valid = loaded.rows_for_leaves(0, result.logical_leaf_ids[0, 0])
    page_rows, page_valid = loaded.rows_for_pages(
        0, result.physical_page_ids[0, 0].unique()
    )

    assert copied.manifest == loaded.manifest == index.manifest
    assert rows.dtype == page_rows.dtype == torch.int64
    assert rows[valid].min() >= row_base
    assert page_rows[page_valid].min() >= row_base
    assert index.manifest["heads"][0]["min_row_id"] == row_base
    assert index.manifest["heads"][0]["max_row_id"] == row_base + 42

    manifest_path = loaded.path / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = SCHEMA_VERSION + 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported content-index schema"):
        PersistentContentIndex.load(loaded.path)


def test_builder_rejects_non_monotonic_or_oversized_shards(tmp_path):
    layout = PersistentIndexLayout(build_chunk_rows=2)
    builder = PersistentContentIndexBuilder(
        tmp_path / "index", num_heads=1, head_dim=3, layout=layout
    )
    builder.ingest(0, torch.tensor([10, 11]), torch.randn(2, 3))
    with pytest.raises(ValueError, match="strictly increasing"):
        builder.ingest(0, torch.tensor([11, 12]), torch.randn(2, 3))
    with pytest.raises(ValueError, match="exceeds"):
        builder.ingest(0, torch.tensor([12, 13, 14]), torch.randn(3, 3))
    builder.close()
