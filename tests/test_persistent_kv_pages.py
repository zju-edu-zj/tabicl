import pytest
import torch

from tabicl._model.persistent_kv_pages import (
    PersistentKVPageStore,
    PersistentKVPageWriter,
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_persistent_pages_roundtrip_and_explicit_leaf_mapping(tmp_path, dtype):
    writer = PersistentKVPageWriter(
        tmp_path / "pages",
        num_heads=2,
        head_dim=4,
        page_rows=8,
        logical_leaf_rows=3,
        dtype=dtype,
    )
    expected = {}
    for head in range(2):
        for page, valid_rows in enumerate((8, 5)):
            key = torch.arange(valid_rows * 4, dtype=torch.float32).reshape(valid_rows, 4)
            key = (key + head * 100 + page * 10).to(dtype)
            value = -key
            assert writer.append(head, key, value, valid_rows=valid_rows) == page
            expected[head, page] = (key, value)
    store = writer.finalize()
    loaded = PersistentKVPageStore.load(tmp_path / "pages", verify=True)
    assert loaded.manifest == store.manifest

    pages = torch.tensor([[[1, 0]], [[0, 1]]])
    offsets = torch.tensor([[[1, 4]], [[2, 0]]])
    lengths = torch.tensor([[[3, 2]], [[3, 3]]])
    candidates = loaded.candidate_tensors(
        pages, offsets, lengths, device="cpu"
    )
    for head in range(2):
        for leaf in range(2):
            page = int(pages[head, 0, leaf])
            offset = int(offsets[head, 0, leaf])
            length = int(lengths[head, 0, leaf])
            start = leaf * 3
            torch.testing.assert_close(
                candidates.key[head, 0, start : start + length],
                expected[head, page][0][offset : offset + length],
            )
            torch.testing.assert_close(
                candidates.value[head, 0, start : start + length],
                expected[head, page][1][offset : offset + length],
            )
            assert candidates.valid[head, 0, start : start + length].all()


def test_writer_rejects_invalid_pages_and_checksum_corruption(tmp_path):
    writer = PersistentKVPageWriter(
        tmp_path / "pages", num_heads=1, head_dim=2, page_rows=4
    )
    with pytest.raises(ValueError, match="valid_rows"):
        writer.append(0, torch.randn(3, 2), torch.randn(3, 2), valid_rows=2)
    writer.append(0, torch.randn(4, 2), torch.randn(4, 2), valid_rows=4)
    writer.finalize()
    key_path = tmp_path / "pages" / "head-000-key.bin"
    with key_path.open("r+b") as stream:
        stream.seek(0)
        stream.write(b"broken!!")
    with pytest.raises(ValueError, match="checksum"):
        PersistentKVPageStore.load(tmp_path / "pages", verify=True)
