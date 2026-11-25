import pytest
import torch

from tabicl._model.embedding import ColEmbedding
from tabicl._model.interaction import RowInteraction
from tabicl._model.inference_config import MgrConfig
from tabicl._model.attention import flash_attn3_toggle
from tabicl._model.streaming_frontend import (
    DefaultRowStoreFactory,
    DeviceTensorRowSource,
    HostTensorRowSource,
    HostTensorRowStore,
    MmapRowStore,
    StreamingFrontendExecutor,
)


def _modules(
    *,
    regression=False,
    grouping=False,
    affine=False,
    dropout=0.0,
    ssmax="none",
    target_aware=True,
):
    col = ColEmbedding(
        embed_dim=12,
        num_blocks=2,
        nhead=3,
        dim_feedforward=24,
        num_inds=4,
        dropout=dropout,
        affine=affine,
        feature_group=grouping,
        feature_group_size=3,
        target_aware=target_aware,
        max_classes=0 if regression else 4,
        reserve_cls_tokens=2,
        ssmax=ssmax,
        zero_init=False,
    ).eval()
    row = RowInteraction(
        embed_dim=12,
        num_blocks=2,
        nhead=3,
        dim_feedforward=24,
        num_cls=2,
        dropout=dropout,
        zero_init=False,
    ).eval()
    return col, row


def _eager_frontend(col, row, x, y, device="cpu"):
    config = MgrConfig(device=device, use_amp=False, use_fa3=False, offload=False)
    return row(col(x, y, mgr_config=config), mgr_config=config)


@pytest.mark.parametrize(
    "regression,grouping,affine,ssmax,source_kind,intermediate",
    [
        (False, "same", False, "qassmax-mlp-elementwise", "device", "device"),
        (False, False, False, "none", "host", "host"),
        (True, False, True, "none", "host", "device"),
    ],
)
def test_streaming_frontend_matches_col_embedding_and_row_interaction(
    regression, grouping, affine, ssmax, source_kind, intermediate
):
    torch.manual_seed(20261125)
    col, row = _modules(
        regression=regression, grouping=grouping, affine=affine, ssmax=ssmax
    )
    x = torch.randn(2, 11, 5, dtype=torch.float32)
    y = torch.randn(2, 7) if regression else torch.randint(0, 4, (2, 7)).float()

    with torch.inference_mode():
        expected = _eager_frontend(col, row, x.clone(), y)
        source = (
            DeviceTensorRowSource(x)
            if source_kind == "device"
            else HostTensorRowSource(x)
        )
        actual = StreamingFrontendExecutor(
            col,
            row,
            row_tile_rows=3,
            intermediate_location=intermediate,
            output_location="device",
        ).run(source, y).tensor

    torch.testing.assert_close(actual, expected, rtol=4e-5, atol=4e-6)


def test_streaming_frontend_writes_caller_owned_host_store():
    torch.manual_seed(17)
    col, row = _modules()
    x = torch.randn(1, 8, 3)
    y = torch.randint(0, 4, (1, 5)).float()
    destination = HostTensorRowStore.empty((1, 8, 24), torch.float32)
    with torch.inference_mode():
        expected = _eager_frontend(col, row, x.clone(), y)
        returned = StreamingFrontendExecutor(
            col, row, row_tile_rows=3, output_location="host"
        ).run(HostTensorRowSource(x), y, output_store=destination)
    assert returned is destination
    torch.testing.assert_close(returned.tensor, expected, rtol=4e-5, atol=4e-6)


@pytest.mark.parametrize(
    "target_aware,affine,ssmax",
    [
        (False, False, "none"),
        (True, True, "qassmax-mlp-elementwise"),
    ],
)
def test_mmap_streaming_matches_eager_across_tiles(
    tmp_path, target_aware, affine, ssmax
):
    torch.manual_seed(31)
    col, row = _modules(
        target_aware=target_aware,
        affine=affine,
        ssmax=ssmax,
    )
    if not target_aware:
        for block in col.tf_col.blocks:
            block.skip_value = -7.0
    x = torch.randn(2, 17, 5)
    if not target_aware:
        x[:, :, 2] = -100.0
    y = torch.randint(0, 4, (2, 11)).float()
    with torch.inference_mode():
        expected = _eager_frontend(col, row, x.clone(), y)
        output = StreamingFrontendExecutor(
            col,
            row,
            row_tile_rows=3,
            intermediate_location="mmap",
            output_location="mmap",
            mmap_directory=tmp_path,
        ).run(HostTensorRowSource(x), y)
        assert isinstance(output, MmapRowStore)
        assert list(tmp_path.glob("*.mmap")) == [output.path]
        actual = output.tensor.clone()
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        output.close()
    assert output.closed
    assert not output.path.exists()
    assert list(tmp_path.iterdir()) == []


class _TrackingMmapFactory(DefaultRowStoreFactory):
    def __init__(self, directory):
        super().__init__(mmap_directory=directory)
        self.created = []
        self.max_open = 0

    def create(self, location, shape, dtype, work_device):
        store = super().create(location, shape, dtype, work_device)
        self.created.append(store)
        self.max_open = max(
            self.max_open,
            sum(isinstance(item, MmapRowStore) and not item.closed for item in self.created),
        )
        return store


def test_mmap_pipeline_never_allocates_full_memory_state(tmp_path):
    torch.manual_seed(37)
    col, row = _modules()
    x = torch.randn(1, 257, 6)
    y = torch.randint(0, 4, (1, 193)).float()
    factory = _TrackingMmapFactory(tmp_path)
    with torch.inference_mode():
        output = StreamingFrontendExecutor(
            col,
            row,
            row_tile_rows=17,
            intermediate_location="mmap",
            output_location="mmap",
            store_factory=factory,
        ).run(HostTensorRowSource(x), y)
    assert all(isinstance(store, MmapRowStore) for store in factory.created)
    assert factory.max_open <= 2
    assert all(store.closed for store in factory.created[:-1])
    assert factory.created[-1] is output and not output.closed
    assert len(list(tmp_path.glob("*.mmap"))) == 1
    output.close()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mmap_store_round_trip_and_cleanup(tmp_path, dtype):
    source = torch.arange(48, dtype=torch.float32).reshape(2, 6, 4).to(dtype)
    store = MmapRowStore.empty(source.shape, dtype, directory=tmp_path)
    path = store.path
    store.write_rows(0, 2, source[:, :2])
    store.write_rows(2, 6, source[:, 2:])
    torch.testing.assert_close(store.read_rows(0, 6, torch.device("cpu")), source)
    store.close()
    assert store.closed
    assert not path.exists()
    with pytest.raises(RuntimeError, match="closed"):
        store.read_rows(0, 1, torch.device("cpu"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_streaming_frontend_streams_mmap_rows_through_cuda(tmp_path):
    torch.manual_seed(23)
    col, row = _modules(grouping="same", ssmax="qassmax-mlp-elementwise")
    col, row = col.cuda(), row.cuda()
    x = torch.randn(2, 13, 4)
    y = torch.randint(0, 4, (2, 8)).float()
    with torch.inference_mode(), flash_attn3_toggle(False):
        expected = _eager_frontend(col, row, x, y, device="cuda").cpu()
        output = StreamingFrontendExecutor(
            col,
            row,
            row_tile_rows=4,
            intermediate_location="mmap",
            output_location="mmap",
            mmap_directory=tmp_path,
        ).run(HostTensorRowSource(x, pin_memory=True), y)
        actual = output.tensor.clone()
        output.close()
    torch.testing.assert_close(actual, expected, rtol=5e-5, atol=5e-6)
    assert list(tmp_path.iterdir()) == []


def test_streaming_frontend_rejects_semantic_changes():
    col, row = _modules()
    source = DeviceTensorRowSource(torch.randn(1, 8, 3))
    y = torch.randint(0, 4, (1, 5)).float()
    executor = StreamingFrontendExecutor(col, row, row_tile_rows=3)

    with pytest.raises(RuntimeError, match="autograd"):
        executor.run(source, y)
    with torch.inference_mode(), pytest.raises(NotImplementedError, match="embed_with_test"):
        executor.run(source, y, embed_with_test=True)
    with torch.inference_mode(), pytest.raises(NotImplementedError, match="ragged"):
        executor.run(source, y, d=torch.tensor([2]))
    with torch.inference_mode(), pytest.raises(NotImplementedError, match="feature-shuffle"):
        executor.run(source, y, feature_shuffles=[[0, 1, 2]])
    with torch.inference_mode(), pytest.raises(NotImplementedError, match="mixed-radix"):
        executor.run(source, torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.float32))


def test_streaming_frontend_rejects_dropout():
    col, row = _modules(dropout=0.1)
    with pytest.raises(NotImplementedError, match="dropout=0"):
        StreamingFrontendExecutor(col, row)


def test_streaming_frontend_rejects_unmatched_reserved_tokens():
    col, row = _modules()
    col.reserve_cls_tokens = row.num_cls + 1
    with pytest.raises(NotImplementedError, match="reserve_cls_tokens"):
        StreamingFrontendExecutor(col, row)
