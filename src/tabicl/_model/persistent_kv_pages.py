"""Per-head persistent K/V pages with bounded construction and reads."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import BinaryIO

import torch
from torch import Tensor


FORMAT = "tabicl-dual-path-kv-pages"
SCHEMA_VERSION = 1
MANIFEST = "manifest.json"


_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _dtype_name(dtype: torch.dtype) -> str:
    for name, value in _DTYPES.items():
        if value == dtype:
            return name
    raise ValueError(f"unsupported K/V page dtype: {dtype}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_tensor(stream: BinaryIO, tensor: Tensor) -> None:
    raw = tensor.detach().contiguous().cpu().view(torch.uint8).numpy()
    stream.write(raw.tobytes())


@dataclass(frozen=True)
class PersistentCandidateTensors:
    key: Tensor
    value: Tensor
    valid: Tensor


class PersistentKVPageWriter:
    """Write fixed-size physical pages one head at a time.

    Pages for each head must be appended in page-ID order. Only the caller's
    current page is resident, so construction does not require a full K/V or
    row-to-position inverse tensor.
    """

    def __init__(
        self,
        path,
        *,
        num_heads: int,
        head_dim: int,
        page_rows: int = 128,
        logical_leaf_rows: int = 32,
        dtype: torch.dtype = torch.float32,
    ):
        if min(num_heads, head_dim, page_rows, logical_leaf_rows) < 1:
            raise ValueError("page dimensions must be positive")
        if page_rows < logical_leaf_rows:
            raise ValueError("a physical page must fit one logical leaf")
        self.path = Path(path)
        if self.path.exists() and (not self.path.is_dir() or any(self.path.iterdir())):
            raise FileExistsError(f"K/V page directory is not empty: {self.path}")
        self.path.mkdir(parents=True, exist_ok=True)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.page_rows = int(page_rows)
        self.logical_leaf_rows = int(logical_leaf_rows)
        self.dtype = dtype
        self.dtype_name = _dtype_name(dtype)
        self._streams = []
        self._pages = [0] * self.num_heads
        self._closed = False
        for head in range(self.num_heads):
            key_path = self.path / f"head-{head:03d}-key.bin.tmp"
            value_path = self.path / f"head-{head:03d}-value.bin.tmp"
            self._streams.append((key_path.open("wb"), value_path.open("wb")))

    def append(self, head: int, key: Tensor, value: Tensor, *, valid_rows: int) -> int:
        if self._closed:
            raise RuntimeError("cannot append to a finalized page store")
        if not 0 <= head < self.num_heads:
            raise IndexError("head is out of range")
        if key.shape != value.shape or key.ndim != 2 or key.shape[1] != self.head_dim:
            raise ValueError("page K/V must have matching shape [rows, head_dim]")
        if not 0 <= valid_rows <= self.page_rows or key.shape[0] != valid_rows:
            raise ValueError("valid_rows must match the supplied page rows")
        if key.dtype != self.dtype or value.dtype != self.dtype:
            raise ValueError("page K/V dtype does not match the store")
        if valid_rows:
            if not torch.isfinite(key).all() or not torch.isfinite(value).all():
                raise ValueError("page K/V must be finite")
        padded_key = torch.zeros(
            (self.page_rows, self.head_dim), dtype=self.dtype, device=key.device
        )
        padded_value = torch.zeros_like(padded_key)
        padded_key[:valid_rows] = key
        padded_value[:valid_rows] = value
        key_stream, value_stream = self._streams[head]
        _write_tensor(key_stream, padded_key)
        _write_tensor(value_stream, padded_value)
        page_id = self._pages[head]
        self._pages[head] += 1
        return page_id

    def finalize(self) -> "PersistentKVPageStore":
        if self._closed:
            raise RuntimeError("page writer is already finalized")
        heads = []
        for head, (key_stream, value_stream) in enumerate(self._streams):
            for stream in (key_stream, value_stream):
                stream.flush()
                os.fsync(stream.fileno())
                stream.close()
            key_tmp = self.path / f"head-{head:03d}-key.bin.tmp"
            value_tmp = self.path / f"head-{head:03d}-value.bin.tmp"
            key_path = key_tmp.with_suffix("")
            value_path = value_tmp.with_suffix("")
            os.replace(key_tmp, key_path)
            os.replace(value_tmp, value_path)
            expected = (
                self._pages[head]
                * self.page_rows
                * self.head_dim
                * torch.empty((), dtype=self.dtype).element_size()
            )
            if key_path.stat().st_size != expected or value_path.stat().st_size != expected:
                raise RuntimeError("K/V page file size does not match its manifest")
            heads.append({
                "head": head,
                "pages": self._pages[head],
                "key_file": key_path.name,
                "value_file": value_path.name,
                "key_sha256": _sha256(key_path),
                "value_sha256": _sha256(value_path),
                "nbytes": expected,
            })
        manifest = {
            "format": FORMAT,
            "schema_version": SCHEMA_VERSION,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "page_rows": self.page_rows,
            "logical_leaf_rows": self.logical_leaf_rows,
            "dtype": self.dtype_name,
            "heads": heads,
        }
        temporary = self.path / f".{MANIFEST}.tmp"
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, self.path / MANIFEST)
        self._closed = True
        return PersistentKVPageStore.load(self.path)

    def close(self) -> None:
        if self._closed:
            return
        for streams in self._streams:
            for stream in streams:
                if not stream.closed:
                    stream.close()
        self._closed = True


class PersistentKVPageStore:
    """Read mmap-backed K/V pages and materialize only selected leaves."""

    def __init__(self, path: Path, manifest: dict, key: list[Tensor], value: list[Tensor]):
        self.path = path
        self.manifest = manifest
        self.num_heads = int(manifest["num_heads"])
        self.head_dim = int(manifest["head_dim"])
        self.page_rows = int(manifest["page_rows"])
        self.logical_leaf_rows = int(manifest["logical_leaf_rows"])
        self.dtype = _DTYPES[manifest["dtype"]]
        self.key = key
        self.value = value

    @classmethod
    def load(cls, path, *, verify: bool = False) -> "PersistentKVPageStore":
        root = Path(path)
        manifest = json.loads((root / MANIFEST).read_text())
        if manifest.get("format") != FORMAT or manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported persistent K/V page format")
        dtype = _DTYPES.get(manifest.get("dtype"))
        if dtype is None:
            raise ValueError("unsupported persistent K/V page dtype")
        key, value = [], []
        heads = manifest.get("heads", [])
        if len(heads) != int(manifest.get("num_heads", 0)):
            raise ValueError("K/V page manifest is incomplete")
        page_size = int(manifest["page_rows"]) * int(manifest["head_dim"])
        for expected_head, metadata in enumerate(heads):
            if int(metadata.get("head", -1)) != expected_head:
                raise ValueError("K/V page heads are out of order")
            pages = int(metadata["pages"])
            key_path = root / metadata["key_file"]
            value_path = root / metadata["value_file"]
            expected_bytes = int(metadata["nbytes"])
            if key_path.stat().st_size != expected_bytes or value_path.stat().st_size != expected_bytes:
                raise ValueError("K/V page file size mismatch")
            if verify and (
                _sha256(key_path) != metadata["key_sha256"]
                or _sha256(value_path) != metadata["value_sha256"]
            ):
                raise ValueError("K/V page checksum mismatch")
            key.append(torch.from_file(str(key_path), shared=False, size=pages * page_size, dtype=dtype).reshape(
                pages, int(manifest["page_rows"]), int(manifest["head_dim"])
            ))
            value.append(torch.from_file(str(value_path), shared=False, size=pages * page_size, dtype=dtype).reshape_as(key[-1]))
        return cls(root, manifest, key, value)

    def candidate_tensors(
        self,
        physical_page_ids: Tensor,
        page_row_offsets: Tensor,
        leaf_lengths: Tensor,
        *,
        device: str | torch.device,
    ) -> PersistentCandidateTensors:
        """Fetch leaves addressed as ``[H,Q,L]`` from explicit page mappings."""

        if physical_page_ids.shape != page_row_offsets.shape or physical_page_ids.shape != leaf_lengths.shape:
            raise ValueError("page IDs, offsets, and leaf lengths must have the same shape")
        if physical_page_ids.ndim != 3 or physical_page_ids.shape[0] != self.num_heads:
            raise ValueError("candidate page mappings must have shape [H,Q,L]")
        target = torch.device(device)
        heads, queries, leaves = physical_page_ids.shape
        rows = self.logical_leaf_rows
        output_key = torch.empty(
            (heads, queries, leaves * rows, self.head_dim), dtype=self.dtype, device=target
        )
        output_value = torch.empty_like(output_key)
        valid = torch.zeros((heads, queries, leaves * rows), dtype=torch.bool, device=target)
        for head in range(heads):
            page_ids = physical_page_ids[head].detach().to("cpu", torch.long)
            offsets = page_row_offsets[head].detach().to("cpu", torch.long)
            lengths = leaf_lengths[head].detach().to("cpu", torch.long)
            for query in range(queries):
                for leaf in range(leaves):
                    length = int(lengths[query, leaf])
                    if not 0 <= length <= rows:
                        raise ValueError("logical leaf length is out of range")
                    page = int(page_ids[query, leaf])
                    offset = int(offsets[query, leaf])
                    if not 0 <= page < self.key[head].shape[0] or offset + length > self.page_rows:
                        raise IndexError("logical leaf mapping is outside its physical page")
                    start = leaf * rows
                    if length:
                        output_key[head, query, start : start + length] = self.key[head][
                            page, offset : offset + length
                        ].to(target)
                        output_value[head, query, start : start + length] = self.value[head][
                            page, offset : offset + length
                        ].to(target)
                        valid[head, query, start : start + length] = True
                    if length < rows:
                        output_key[head, query, start + length : start + rows].zero_()
                        output_value[head, query, start + length : start + rows].zero_()
        return PersistentCandidateTensors(output_key, output_value, valid)


__all__ = [
    "PersistentCandidateTensors",
    "PersistentKVPageWriter",
    "PersistentKVPageStore",
]
