import json

import pytest
import torch
from torch.nn import functional as F

from tabicl._model.dual_path_snapshot import (
    SNAPSHOT_SCHEMA_VERSION,
    SnapshotProfile,
    SnapshotState,
    SupportSnapshot,
)
from tabicl._model.paged_runtime import (
    BoundedPageCache,
    BoundedStagingPool,
    DevicePageProvider,
    HeadProjectionWeights,
    HostPageProvider,
    HiddenSnapshotRuntime,
    OnDemandHeadProjector,
)


def _layer_tensors(rows=11, model_dim=8, seed=0):
    generator = torch.Generator().manual_seed(seed)
    hidden = torch.randn(rows, model_dim, generator=generator)
    global_state = {
        "elu_state": torch.randn(2, 4, 4, generator=generator),
        "elu_normalizer": torch.randn(2, 4, generator=generator),
    }
    index_state = {
        "order": torch.randperm(rows, generator=generator).to(torch.int32),
        "tree_representatives": torch.randn(2, 7, 2, 4, generator=generator),
    }
    return hidden, global_state, index_state


def _snapshot(profile, *, rows=11, model_dim=8, expected_layers=2):
    return SupportSnapshot.create(
        profile=profile,
        support_size=rows,
        model_dim=model_dim,
        expected_layers=expected_layers,
        model_fingerprint="checkpoint-sha256",
        layout={
            "logical_leaf_rows": 32,
            "storage_page_rows": 128,
            "selected_logical_leaves": 32,
        },
        generation=4,
        snapshot_id="support-v4",
        metadata={"dataset": "tiny"},
    )


def test_online_hidden_snapshot_roundtrip_is_versioned_and_lazy(tmp_path):
    snapshot = SupportSnapshot.create(
        profile=SnapshotProfile.ONLINE_HIDDEN,
        support_size=11,
        model_dim=8,
        expected_layers=2,
        model_fingerprint="checkpoint-sha256",
        layout={
            "logical_leaf_rows": 32,
            "storage_page_rows": 128,
            "selected_logical_leaves": 32,
        },
        generation=4,
        snapshot_id="support-v4",
        metadata={"dataset": "tiny"},
        directory=tmp_path / "snapshot",
    )
    expected = {}
    for layer_index in range(2):
        hidden, global_state, index_state = _layer_tensors(seed=layer_index)
        snapshot.add_layer(
            layer_index,
            hidden,
            global_state=global_state,
            index_state=index_state,
        )
        expected[layer_index] = (hidden, global_state, index_state)
        assert all(
            isinstance(asset, type(tmp_path)) for asset in snapshot._assets.values()
        )
    snapshot.seal()

    path = snapshot.save(tmp_path / "snapshot")
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["schema_version"] == SNAPSHOT_SCHEMA_VERSION
    assert manifest["profile"] == "online_hidden"
    assert manifest["state"] == "sealed"
    assert manifest["batch_size"] == 1
    assert len(manifest["layers"]) == 2
    roles = {reference["role"] for reference in manifest["tensors"].values()}
    assert all(
        role == "hidden" or role.startswith(("global:", "index:"))
        for role in roles
    )
    assert all("kv" not in role.lower() for role in roles)
    assert all(
        reference["byte_size"] > 0 and len(reference["sha256"]) == 64
        for reference in manifest["tensors"].values()
    )

    loaded = SupportSnapshot.load(path, lazy=True, mmap=True, strict=True)
    assert loaded.state == SnapshotState.SEALED
    assert loaded._loaded == {}
    layer = loaded.layer(1)
    torch.testing.assert_close(loaded.tensor(layer.hidden), expected[1][0])
    for name, reference in layer.global_state.items():
        torch.testing.assert_close(loaded.tensor(reference), expected[1][1][name])
    for name, reference in layer.index_state.items():
        torch.testing.assert_close(loaded.tensor(reference), expected[1][2][name])
    assert set(loaded._loaded) == {
        layer.hidden,
        *layer.global_state.values(),
        *layer.index_state.values(),
    }


def test_strict_load_rejects_corrupt_tensor_file(tmp_path):
    snapshot = _snapshot(SnapshotProfile.ONE_SHOT, expected_layers=1)
    hidden, global_state, index_state = _layer_tensors()
    snapshot.add_layer(
        0, hidden, global_state=global_state, index_state=index_state
    )
    snapshot.seal()
    path = snapshot.save(tmp_path / "snapshot")
    manifest = json.loads((path / "manifest.json").read_text())
    reference = next(iter(manifest["tensors"].values()))
    tensor_path = path / reference["file"]
    tensor_path.write_bytes(tensor_path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="byte size mismatch"):
        SupportSnapshot.load(path, strict=True)


def test_snapshot_explicitly_rejects_batched_datasets():
    with pytest.raises(ValueError, match="batch_size=1"):
        SupportSnapshot.create(
            profile=SnapshotProfile.ONLINE_HIDDEN,
            support_size=11,
            model_dim=8,
            batch_size=2,
            expected_layers=2,
            model_fingerprint="checkpoint-sha256",
            layout={"logical_leaf_rows": 32, "storage_page_rows": 128},
        )


def test_snapshot_rejects_future_manifest_schema(tmp_path):
    snapshot = _snapshot(SnapshotProfile.ONE_SHOT, expected_layers=1)
    hidden, global_state, index_state = _layer_tensors()
    snapshot.add_layer(
        0, hidden, global_state=global_state, index_state=index_state
    )
    snapshot.seal()
    path = snapshot.save(tmp_path / "snapshot")
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = SNAPSHOT_SCHEMA_VERSION + 1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="newer than supported"):
        SupportSnapshot.load(path)


def test_incremental_online_build_resumes_from_building_manifest(tmp_path):
    path = tmp_path / "snapshot"
    snapshot = SupportSnapshot.create(
        profile=SnapshotProfile.ONLINE_HIDDEN,
        support_size=11,
        model_dim=8,
        expected_layers=2,
        model_fingerprint="checkpoint-sha256",
        layout={"logical_leaf_rows": 32, "storage_page_rows": 128},
        directory=path,
    )
    hidden0, global0, index0 = _layer_tensors(seed=31)
    snapshot.add_layer(0, hidden0, global_state=global0, index_state=index0)

    resumed = SupportSnapshot.load(path, lazy=True)
    assert resumed.state == SnapshotState.BUILDING
    hidden1, global1, index1 = _layer_tensors(seed=32)
    resumed.add_layer(1, hidden1, global_state=global1, index_state=index1)
    resumed.seal()

    complete = SupportSnapshot.load(path, lazy=True)
    assert complete.state == SnapshotState.SEALED
    assert [layer.layer_index for layer in complete.manifest.layers] == [0, 1]
    torch.testing.assert_close(
        complete.tensor(complete.layer(0).hidden), hidden0
    )
    torch.testing.assert_close(
        complete.tensor(complete.layer(1).hidden), hidden1
    )


def test_one_shot_snapshot_keeps_only_the_active_layer():
    snapshot = _snapshot(SnapshotProfile.ONE_SHOT, expected_layers=3)
    first_hidden, first_global, first_index = _layer_tensors(seed=1)
    first = snapshot.add_layer(
        0,
        first_hidden,
        global_state=first_global,
        index_state=first_index,
    )
    second_hidden, second_global, second_index = _layer_tensors(seed=2)
    second = snapshot.add_layer(
        1,
        second_hidden,
        global_state=second_global,
        index_state=second_index,
    )

    assert snapshot.active_layer == second
    assert len(snapshot.manifest.layers) == 1
    assert first.hidden not in snapshot.manifest.tensors
    assert all(
        not tensor_id.startswith("layer.000")
        for tensor_id in snapshot.manifest.tensors
    )


def test_on_demand_projection_matches_full_projection_with_duplicate_rows():
    torch.manual_seed(20261125)
    rows, embed_dim, heads = 19, 8, 2
    hidden = torch.randn(rows, embed_dim)
    qkv_weight = torch.randn(3 * embed_dim, embed_dim)
    qkv_bias = torch.randn(3 * embed_dim)
    weights = HeadProjectionWeights.from_qkv(qkv_weight, qkv_bias, heads)
    provider = DevicePageProvider(hidden, page_rows=4)
    row_ids = torch.tensor(
        [
            [[0, 3, 3], [7, 18, 0]],
            [[4, 4, 1], [18, 2, 9]],
        ],
        dtype=torch.int64,
    )

    projected = OnDemandHeadProjector(weights).project(provider, row_ids)
    key_full = F.linear(
        hidden,
        qkv_weight[embed_dim : 2 * embed_dim],
        qkv_bias[embed_dim : 2 * embed_dim],
    ).reshape(rows, heads, embed_dim // heads)
    value_full = F.linear(
        hidden,
        qkv_weight[2 * embed_dim :],
        qkv_bias[2 * embed_dim :],
    ).reshape(rows, heads, embed_dim // heads)
    expected_key = torch.stack(
        [key_full[:, head].index_select(0, row_ids[head].flatten()).reshape(2, 3, 4)
         for head in range(heads)]
    )
    expected_value = torch.stack(
        [value_full[:, head].index_select(0, row_ids[head].flatten()).reshape(2, 3, 4)
         for head in range(heads)]
    )
    torch.testing.assert_close(projected.key, expected_key)
    torch.testing.assert_close(projected.value, expected_value)


def test_page_cache_and_staging_pool_are_bounded():
    hidden = torch.arange(12 * 4, dtype=torch.float32).reshape(12, 4)
    cache = BoundedPageCache(max_pages=2, max_bytes=2 * 4 * 4 * 4)
    provider = DevicePageProvider(
        hidden, page_rows=4, cache=cache, namespace="hidden"
    )
    provider.page(0)
    provider.page(1)
    handle = provider.prefetch_rows(torch.tensor([0, 7, 11]))
    handle.wait()
    assert handle.completed
    assert handle.page_ids == (0, 1, 2)
    assert len(cache) == 2
    provider.page(0)
    provider.page(2)
    assert len(cache) == 2
    assert cache.get(("hidden", 1)) is None
    assert cache.hits >= 1
    assert cache.current_bytes <= cache.max_bytes
    assert cache.peak_bytes <= cache.max_bytes
    assert cache.peak_pages <= cache.max_pages

    pool = BoundedStagingPool(max_buffers=1, max_bytes=64, pin_memory=False)
    with pool.acquire((4, 4), torch.float32) as first:
        first.fill_(7)
        with pytest.raises(RuntimeError, match="currently leased"):
            with pool.acquire((4, 4), torch.float32):
                pass
    with pool.acquire((4, 4), torch.float32) as reused:
        assert reused.data_ptr() == first.data_ptr()
    assert pool.allocated_bytes == 64
    assert pool.peak_allocated_bytes <= pool.max_bytes
    assert pool.peak_leased_buffers <= pool.max_buffers


def test_host_page_provider_uses_explicit_bounded_copy_semantics():
    hidden = torch.arange(12 * 4, dtype=torch.float32).reshape(12, 4)
    cache = BoundedPageCache(max_pages=1, max_bytes=64)
    provider = HostPageProvider(
        hidden,
        page_rows=4,
        cache=cache,
        cache_page_mode="copy",
        mmap_backed=True,
    )
    cached = provider.page(0)
    assert provider.cache_page_mode == "copy"
    assert provider.mmap_backed
    assert cached.untyped_storage().data_ptr() != hidden.untyped_storage().data_ptr()
    hidden[0, 0] = -1
    assert cached[0, 0] == 0
    assert cache.current_bytes == 64
    assert cache.peak_bytes == 64


def test_online_hidden_runtime_evicts_layers_and_projects_from_snapshot(tmp_path):
    snapshot = _snapshot(SnapshotProfile.ONLINE_HIDDEN, expected_layers=2)
    expected_hidden = []
    for layer_index in range(2):
        hidden, global_state, index_state = _layer_tensors(seed=10 + layer_index)
        expected_hidden.append(hidden)
        snapshot.add_layer(
            layer_index,
            hidden,
            global_state=global_state,
            index_state=index_state,
        )
    snapshot.seal()
    path = snapshot.save(tmp_path / "snapshot")
    loaded = SupportSnapshot.load(path, lazy=True)
    cache = BoundedPageCache(max_pages=2, max_bytes=1024)
    runtime = HiddenSnapshotRuntime(
        loaded,
        location="host",
        page_rows=4,
        max_open_layers=1,
        page_cache=cache,
        staging_pool=BoundedStagingPool(1, 1024, pin_memory=False),
    )

    layer0 = runtime.open_layer(0)
    assert layer0.hidden_provider.read_rows(torch.tensor([0, 10])).shape == (2, 8)
    layer1 = runtime.open_layer(1)
    assert layer0.closed
    assert not layer1.closed
    assert list(runtime._open) == [1]
    assert layer1.hidden_provider.mmap_backed
    assert layer1.hidden_provider.cache_page_mode == "copy"

    torch.manual_seed(7)
    qkv_weight = torch.randn(24, 8)
    qkv_bias = torch.randn(24)
    weights = HeadProjectionWeights.from_qkv(qkv_weight, qkv_bias, 2)
    row_ids = torch.tensor([[0, 2, 10], [1, 2, 9]])
    projected = layer1.project(row_ids, weights)
    full_key = F.linear(expected_hidden[1], qkv_weight[8:16], qkv_bias[8:16])
    full_key = full_key.reshape(11, 2, 4)
    expected = torch.stack(
        [full_key[:, head].index_select(0, row_ids[head]) for head in range(2)]
    )
    torch.testing.assert_close(projected.key, expected)

    runtime.close()
    assert layer1.closed
    loaded.close()
    assert loaded.state == SnapshotState.CLOSED
