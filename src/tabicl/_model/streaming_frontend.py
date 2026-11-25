"""Bounded-memory execution for the column and row front-end.

The column set transformer is streamed along the table-row dimension.  Each
ISAB block reads the support twice to compute the inducing attention with a
stable, exact softmax reduction, then writes the reverse attention one row tile
at a time.  The final column tile is passed directly to ``RowInteraction``.

This module is inference-only.  Unsupported modes are rejected explicitly so
that callers never receive an output with silently changed model semantics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import math
import mmap
import os
from pathlib import Path
import tempfile
from typing import Optional, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .embedding import ColEmbedding
from .interaction import RowInteraction
from .layers import InducedSelfAttentionBlock, MultiheadAttentionBlock


class RowSource(ABC):
    """Random-access provider whose second dimension is the table-row axis."""

    @property
    @abstractmethod
    def shape(self) -> torch.Size:
        raise NotImplementedError

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype:
        raise NotImplementedError

    @abstractmethod
    def read_rows(self, start: int, end: int, device: torch.device) -> Tensor:
        """Return ``tensor[:, start:end]`` on ``device``."""


class RowStore(RowSource):
    """Mutable row-addressable tensor store."""

    @abstractmethod
    def write_rows(self, start: int, end: int, rows: Tensor) -> None:
        """Write ``rows`` into ``tensor[:, start:end]``."""

    @property
    @abstractmethod
    def tensor(self) -> Tensor:
        """Return the backing tensor without copying."""

    @property
    def closed(self) -> bool:
        return False

    def flush(self) -> None:
        """Persist writes when the backing store requires it."""

    def close(self, *, unlink: bool = True) -> None:
        """Release resources owned by this store."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class DeviceTensorRowSource(RowSource):
    """Provider for a tensor already resident on the execution device."""

    def __init__(self, tensor: Tensor):
        if tensor.ndim < 2:
            raise ValueError("row tensors must have at least two dimensions")
        self._tensor = tensor

    @property
    def shape(self) -> torch.Size:
        return self._tensor.shape

    @property
    def dtype(self) -> torch.dtype:
        return self._tensor.dtype

    def read_rows(self, start: int, end: int, device: torch.device) -> Tensor:
        _validate_row_range(start, end, self.shape[1])
        if self._tensor.device != device:
            raise ValueError(
                f"device source is on {self._tensor.device}, requested {device}; "
                "use HostTensorRowSource for transfers"
            )
        return self._tensor[:, start:end]


class HostTensorRowSource(RowSource):
    """Provider for a CPU tensor, copied to the requested execution device."""

    def __init__(self, tensor: Tensor, *, pin_memory: bool = False):
        if tensor.ndim < 2:
            raise ValueError("row tensors must have at least two dimensions")
        if tensor.device.type != "cpu":
            raise ValueError("HostTensorRowSource requires a CPU tensor")
        if pin_memory and torch.cuda.is_available() and not tensor.is_pinned():
            tensor = tensor.pin_memory()
        self._tensor = tensor

    @property
    def shape(self) -> torch.Size:
        return self._tensor.shape

    @property
    def dtype(self) -> torch.dtype:
        return self._tensor.dtype

    def read_rows(self, start: int, end: int, device: torch.device) -> Tensor:
        _validate_row_range(start, end, self.shape[1])
        rows = self._tensor[:, start:end]
        return rows.to(device=device, non_blocking=rows.is_pinned())


class DeviceTensorRowStore(DeviceTensorRowSource, RowStore):
    """Row store backed by a tensor on the execution device."""

    @classmethod
    def empty(cls, shape: Sequence[int], dtype: torch.dtype, device: torch.device):
        return cls(torch.empty(tuple(shape), dtype=dtype, device=device))

    def write_rows(self, start: int, end: int, rows: Tensor) -> None:
        _validate_write(self.shape, start, end, rows)
        if rows.device != self._tensor.device:
            raise ValueError("device-store writes must already be on the store device")
        self._tensor[:, start:end].copy_(rows)

    @property
    def tensor(self) -> Tensor:
        return self._tensor


class HostTensorRowStore(HostTensorRowSource, RowStore):
    """Row store backed by optional pinned CPU memory."""

    @classmethod
    def empty(
        cls,
        shape: Sequence[int],
        dtype: torch.dtype,
        *,
        pin_memory: bool = False,
    ):
        pin_memory = pin_memory and torch.cuda.is_available()
        tensor = torch.empty(tuple(shape), dtype=dtype, device="cpu", pin_memory=pin_memory)
        return cls(tensor)

    def write_rows(self, start: int, end: int, rows: Tensor) -> None:
        _validate_write(self.shape, start, end, rows)
        # A non-blocking D2H copy needs explicit event ownership before CPU code
        # may consume it. Keep this provider synchronous and exact; an async
        # store can implement the RowStore interface separately.
        source = rows.detach().to(device="cpu")
        self._tensor[:, start:end].copy_(source)

    @property
    def tensor(self) -> Tensor:
        return self._tensor


class MmapRowStore(RowStore):
    """File-backed row store with explicit flush, close, and unlink lifecycle."""

    def __init__(
        self,
        path: Path,
        shape: Sequence[int],
        dtype: torch.dtype,
    ) -> None:
        self._path = Path(path)
        self._shape = torch.Size(shape)
        self._dtype = dtype
        self._closed = False
        numel = math.prod(self._shape)
        if numel < 1 or any(dimension < 1 for dimension in self._shape):
            raise ValueError("mmap store dimensions must be positive")
        size_bytes = numel * torch.empty((), dtype=dtype).element_size()
        descriptor = os.open(self._path, os.O_RDWR)
        try:
            self._mapping = mmap.mmap(descriptor, size_bytes, access=mmap.ACCESS_WRITE)
        finally:
            os.close(descriptor)
        self._tensor = torch.frombuffer(
            self._mapping, dtype=dtype, count=numel
        ).reshape(self._shape)

    @classmethod
    def empty(
        cls,
        shape: Sequence[int],
        dtype: torch.dtype,
        *,
        directory: Optional[os.PathLike | str] = None,
    ):
        directory_path = Path(directory) if directory is not None else Path(tempfile.gettempdir())
        directory_path.mkdir(parents=True, exist_ok=True)
        numel = math.prod(shape)
        if numel < 1 or any(dimension < 1 for dimension in shape):
            raise ValueError("mmap store dimensions must be positive")
        size_bytes = numel * torch.empty((), dtype=dtype).element_size()
        descriptor, staging = tempfile.mkstemp(
            prefix=".tabicl-frontend-", suffix=".partial", dir=directory_path
        )
        staging_path = Path(staging)
        final_path = staging_path.with_suffix(".mmap")
        try:
            os.ftruncate(descriptor, size_bytes)
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = -1
            os.replace(staging_path, final_path)
            return cls(final_path, shape, dtype)
        except Exception:
            if descriptor >= 0:
                os.close(descriptor)
            staging_path.unlink(missing_ok=True)
            final_path.unlink(missing_ok=True)
            raise

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("mmap row store is closed")

    @property
    def path(self) -> Path:
        return self._path

    @property
    def shape(self) -> torch.Size:
        return self._shape

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def tensor(self) -> Tensor:
        self._require_open()
        return self._tensor

    def read_rows(self, start: int, end: int, device: torch.device) -> Tensor:
        self._require_open()
        _validate_row_range(start, end, self.shape[1])
        rows = self._tensor[:, start:end]
        result = rows.clone() if device.type == "cpu" else rows.to(device=device)
        self._drop_cached_pages()
        return result

    def write_rows(self, start: int, end: int, rows: Tensor) -> None:
        self._require_open()
        _validate_write(self.shape, start, end, rows)
        self._tensor[:, start:end].copy_(rows.detach().to(device="cpu"))
        self.flush()
        self._drop_cached_pages()

    def flush(self) -> None:
        self._require_open()
        self._mapping.flush()

    def _drop_cached_pages(self) -> None:
        advice = getattr(mmap, "MADV_DONTNEED", None)
        if advice is None or not hasattr(self._mapping, "madvise"):
            return
        try:
            self._mapping.madvise(advice)
        except OSError:
            # Some filesystems do not implement mmap advice. Correctness does
            # not depend on eviction; this only bounds resident page cache.
            pass

    def close(self, *, unlink: bool = True) -> None:
        if self._closed:
            return
        mapping = self._mapping
        try:
            mapping.flush()
        finally:
            self._tensor = None
            self._mapping = None
            self._closed = True
            if unlink:
                self._path.unlink(missing_ok=True)
            try:
                mapping.close()
            except BufferError:
                # An external tensor view can keep the mapping alive. The file is
                # already unlinked and the view owns the remaining mapping lifetime.
                pass

    def unlink(self) -> None:
        self._path.unlink(missing_ok=True)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class RowStoreFactory(ABC):
    """Factory for executor-owned intermediate and output stores."""

    @abstractmethod
    def create(
        self,
        location: str,
        shape: Sequence[int],
        dtype: torch.dtype,
        work_device: torch.device,
    ) -> RowStore:
        raise NotImplementedError


class DefaultRowStoreFactory(RowStoreFactory):
    def __init__(self, *, mmap_directory: Optional[os.PathLike | str] = None):
        self.mmap_directory = mmap_directory

    def create(
        self,
        location: str,
        shape: Sequence[int],
        dtype: torch.dtype,
        work_device: torch.device,
    ) -> RowStore:
        if location == "device":
            return DeviceTensorRowStore.empty(shape, dtype, work_device)
        if location == "host":
            return HostTensorRowStore.empty(
                shape, dtype, pin_memory=work_device.type == "cuda"
            )
        if location == "mmap":
            return MmapRowStore.empty(shape, dtype, directory=self.mmap_directory)
        raise ValueError("store location must be 'device', 'host', or 'mmap'")


def _validate_row_range(start: int, end: int, size: int) -> None:
    if not 0 <= start < end <= size:
        raise ValueError(f"invalid row range [{start}, {end}) for {size} rows")


def _validate_write(shape: torch.Size, start: int, end: int, rows: Tensor) -> None:
    _validate_row_range(start, end, shape[1])
    expected = (shape[0], end - start, *shape[2:])
    if tuple(rows.shape) != expected:
        raise ValueError(f"row write expected shape {expected}, got {tuple(rows.shape)}")


def _module_device_dtype(module: nn.Module) -> tuple[torch.device, torch.dtype]:
    parameter = next(module.parameters(), None)
    if parameter is None:
        raise ValueError("front-end modules must contain parameters")
    return parameter.device, parameter.dtype


def _dropout_probabilities(module: nn.Module) -> list[tuple[str, float]]:
    found = []
    for name, child in module.named_modules():
        if isinstance(child, nn.Dropout) and child.p:
            found.append((name, float(child.p)))
        elif isinstance(child, nn.MultiheadAttention) and child.dropout:
            found.append((name, float(child.dropout)))
    return found


class ISABStreamingExecutor:
    """Execute one ISAB with bounded row memory and exact selected semantics."""

    def __init__(
        self,
        block: InducedSelfAttentionBlock,
        *,
        row_tile_rows: int,
        work_device: torch.device,
    ) -> None:
        if row_tile_rows < 1:
            raise ValueError("row_tile_rows must be positive")
        if not block.multihead_attn1.norm_first or not block.multihead_attn2.norm_first:
            raise NotImplementedError("streaming ISAB currently requires pre-norm blocks")
        self.block = block
        self.row_tile_rows = row_tile_rows
        self.work_device = work_device

    @staticmethod
    def _project_heads(src: Tensor, weight: Tensor, bias: Optional[Tensor], heads: int) -> Tensor:
        projected = F.linear(src, weight, bias)
        head_dim = projected.shape[-1] // heads
        return projected.view(*projected.shape[:-1], heads, head_dim).transpose(-3, -2)

    def _project_query(self, mab: MultiheadAttentionBlock, query: Tensor, support_rows: int) -> Tensor:
        attention = mab.attn
        embed_dim = attention.embed_dim
        q = self._project_heads(
            mab.norm1(query),
            attention.in_proj_weight[:embed_dim],
            attention.in_proj_bias[:embed_dim] if attention.in_proj_bias is not None else None,
            attention.num_heads,
        )
        if attention.ssmax_layer is not None:
            flat = q.reshape(-1, *q.shape[-3:])
            q = attention.ssmax_layer(flat, support_rows).view_as(q)
        return q

    @staticmethod
    def _project_key_value(mab: MultiheadAttentionBlock, src: Tensor) -> tuple[Tensor, Tensor]:
        attention = mab.attn
        embed_dim = attention.embed_dim
        normalized = mab.norm1(src)
        kv = F.linear(
            normalized,
            attention.in_proj_weight[embed_dim:],
            attention.in_proj_bias[embed_dim:] if attention.in_proj_bias is not None else None,
        )
        key, value = kv.chunk(2, dim=-1)
        head_dim = embed_dim // attention.num_heads
        key = key.view(*key.shape[:-1], attention.num_heads, head_dim).transpose(-3, -2)
        value = value.view(*value.shape[:-1], attention.num_heads, head_dim).transpose(-3, -2)
        return key, value

    def _support_tiles(self, source: RowSource, support_rows: int):
        for start in range(0, support_rows, self.row_tile_rows):
            end = min(start + self.row_tile_rows, support_rows)
            rows = source.read_rows(start, end, self.work_device)
            yield rows.transpose(1, 2), start, end

    def _inducing_hidden(self, source: RowSource, support_rows: int) -> Tensor:
        """Two-pass softmax: first global max, then denominator and value sum."""
        mab = self.block.multihead_attn1
        batch, _, groups, embed_dim = source.shape
        inducing = self.block.ind_vectors.expand(batch, groups, -1, -1)
        query = self._project_query(mab, inducing, support_rows)
        scale = query.shape[-1] ** -0.5

        maximum = torch.full(
            query.shape[:-1], -float("inf"), device=self.work_device, dtype=torch.float32
        )
        for tile, _, _ in self._support_tiles(source, support_rows):
            key, _ = self._project_key_value(mab, tile)
            score = torch.matmul(query.float(), key.float().transpose(-1, -2)) * scale
            maximum = torch.maximum(maximum, score.amax(dim=-1))

        denominator = torch.zeros_like(maximum)
        numerator = torch.zeros(
            (*maximum.shape, query.shape[-1]), device=self.work_device, dtype=torch.float32
        )
        for tile, _, _ in self._support_tiles(source, support_rows):
            key, value = self._project_key_value(mab, tile)
            score = torch.matmul(query.float(), key.float().transpose(-1, -2)) * scale
            probability = torch.exp(score - maximum.unsqueeze(-1))
            denominator += probability.sum(dim=-1)
            numerator += torch.matmul(probability, value.float())

        attention_heads = numerator / denominator.unsqueeze(-1)
        attention_output = attention_heads.to(query.dtype).transpose(-3, -2).flatten(-2)
        attention_output = F.linear(
            attention_output, mab.attn.out_proj.weight, mab.attn.out_proj.bias
        )
        hidden = inducing + attention_output
        return hidden + mab._ff_block(mab.norm2(hidden))

    def run(
        self,
        source: RowSource,
        destination: RowStore,
        *,
        support_rows: int,
        skip_mask: Tensor,
    ) -> Tensor:
        if source.shape != destination.shape:
            raise ValueError("ISAB source and destination shapes must match")
        hidden = self._inducing_hidden(source, support_rows)
        mab = self.block.multihead_attn2
        output_skip_mask = torch.ones_like(skip_mask)
        for start in range(0, source.shape[1], self.row_tile_rows):
            end = min(start + self.row_tile_rows, source.shape[1])
            tile = source.read_rows(start, end, self.work_device).transpose(1, 2)
            output = mab(tile, hidden, hidden)
            output = torch.where(
                skip_mask[..., None, None],
                torch.as_tensor(self.block.skip_value, device=output.device, dtype=output.dtype),
                output,
            )
            output_skip_mask &= (output == self.block.skip_value).all(dim=(2, 3))
            destination.write_rows(start, end, output.transpose(1, 2))
        return output_skip_mask


class StreamingFrontendExecutor:
    """Stream ``ColEmbedding`` into row-tiled ``RowInteraction`` execution."""

    def __init__(
        self,
        col_embedding: ColEmbedding,
        row_interaction: RowInteraction,
        *,
        row_tile_rows: int = 65536,
        intermediate_location: str = "mmap",
        output_location: str = "device",
        mmap_directory: Optional[os.PathLike | str] = None,
        store_factory: Optional[RowStoreFactory] = None,
    ) -> None:
        self.col_embedding = col_embedding
        self.row_interaction = row_interaction
        self.row_tile_rows = row_tile_rows
        self.intermediate_location = intermediate_location
        self.output_location = output_location
        if store_factory is not None and mmap_directory is not None:
            raise ValueError("mmap_directory cannot be combined with a custom store_factory")
        self.store_factory = store_factory or DefaultRowStoreFactory(
            mmap_directory=mmap_directory
        )
        self.work_device, self.dtype = _module_device_dtype(col_embedding)
        row_device, row_dtype = _module_device_dtype(row_interaction)
        if (row_device, row_dtype) != (self.work_device, self.dtype):
            raise ValueError("column and row modules must use the same device and dtype")
        self._validate_modules()

    def _validate_modules(self) -> None:
        if self.row_tile_rows < 1:
            raise ValueError("row_tile_rows must be positive")
        if self.col_embedding.training or self.row_interaction.training:
            raise NotImplementedError("streaming frontend is inference-only; call eval() first")
        dropout = _dropout_probabilities(self.col_embedding) + _dropout_probabilities(
            self.row_interaction
        )
        if dropout:
            raise NotImplementedError(f"streaming frontend requires dropout=0, found {dropout}")
        if self.row_interaction.recompute:
            raise NotImplementedError("streaming frontend does not support recompute=True")
        for block in self.col_embedding.tf_col.blocks:
            if not isinstance(block, InducedSelfAttentionBlock):
                raise TypeError("column transformer must contain ISAB blocks")
        for block in self.row_interaction.tf_row.blocks:
            if not block.norm_first:
                raise NotImplementedError("streaming row interaction requires pre-norm blocks")
        if self.col_embedding.reserve_cls_tokens != self.row_interaction.num_cls:
            raise NotImplementedError(
                "exact streaming requires reserve_cls_tokens == row num_cls"
            )

    def _new_store(self, location: str, shape: Sequence[int]) -> RowStore:
        store = self.store_factory.create(
            location, shape, self.dtype, self.work_device
        )
        if tuple(store.shape) != tuple(shape) or store.dtype != self.dtype:
            store.close()
            raise ValueError("store factory returned a store with the wrong shape or dtype")
        return store

    def _group_count(self, features: int) -> int:
        grouping = self.col_embedding.feature_group
        if not grouping or grouping is True or grouping == "same":
            return features
        if grouping == "valid":
            return math.ceil(features / self.col_embedding.feature_group_size)
        raise NotImplementedError(f"unsupported feature_group={grouping!r}")

    def _group_rows(self, rows: Tensor) -> Tensor:
        return self.col_embedding.feature_grouping(rows)

    def _label_embedding(self, labels: Tensor) -> Tensor:
        column = self.col_embedding
        if column.max_classes > 0:
            return column.y_encoder(labels.float())
        return column.y_encoder(labels.unsqueeze(-1))

    def _initial_state(
        self,
        source: RowSource,
        y_train: Tensor,
        groups: int,
    ) -> tuple[RowStore, Tensor]:
        batch, rows, _ = source.shape
        state = self._new_store(
            self.intermediate_location,
            (batch, rows, groups, self.col_embedding.embed_dim),
        )
        try:
            skip_mask = torch.ones((batch, groups), device=self.work_device, dtype=torch.bool)
            blocks = self.col_embedding.tf_col.blocks
            initial_skip_value = blocks[0].skip_value if blocks else -100.0
            support_rows = y_train.shape[1]
            for start in range(0, rows, self.row_tile_rows):
                end = min(start + self.row_tile_rows, rows)
                raw = source.read_rows(start, end, self.work_device)
                grouped = self._group_rows(raw)
                projected = self.col_embedding.in_linear(grouped)
                if self.col_embedding.target_aware and start < support_rows:
                    support_end = min(end, support_rows)
                    labels = y_train[:, start:support_end].to(self.work_device)
                    label_embedding = self._label_embedding(labels)
                    projected[:, : support_end - start] += label_embedding[:, :, None, :]
                skip_mask &= (projected == initial_skip_value).all(dim=(1, 3))
                state.write_rows(start, end, projected)
            return state, skip_mask
        except Exception:
            state.close()
            raise

    def _validate_request(
        self,
        source: RowSource,
        y_train: Tensor,
        *,
        embed_with_test: bool,
        d: Optional[Tensor],
        feature_shuffles,
    ) -> None:
        if torch.is_grad_enabled():
            raise RuntimeError("streaming frontend does not support autograd; use torch.inference_mode()")
        if embed_with_test:
            raise NotImplementedError("embed_with_test=True changes ISAB support and is not supported")
        if d is not None:
            raise NotImplementedError("ragged feature counts are not supported")
        if feature_shuffles is not None:
            raise NotImplementedError("feature-shuffle reuse is not supported")
        if len(source.shape) != 3:
            raise ValueError("row source must have shape [B, T, H]")
        if source.dtype != self.dtype:
            raise ValueError(f"source dtype {source.dtype} does not match module dtype {self.dtype}")
        if y_train.ndim != 2 or y_train.shape[0] != source.shape[0]:
            raise ValueError("y_train must have shape [B, train_size]")
        if not 0 < y_train.shape[1] <= source.shape[1]:
            raise ValueError("y_train must select a non-empty support prefix")
        if self.col_embedding.target_aware and self.col_embedding.max_classes > 0:
            labels = y_train.detach()
            if not torch.equal(labels, labels.round()) or labels.min().item() < 0:
                raise ValueError("classification labels must be non-negative integers")
            classes = int(labels.max().item()) + 1
            if classes > self.col_embedding.max_classes:
                raise NotImplementedError(
                    "streaming mixed-radix classification is not implemented"
                )
        if self.col_embedding.affine and self.col_embedding.in_linear.in_features != 1:
            raise NotImplementedError("affine column embedding with grouped features is unsupported")

    def run(
        self,
        source: RowSource,
        y_train: Tensor,
        *,
        output_store: Optional[RowStore] = None,
        embed_with_test: bool = False,
        d: Optional[Tensor] = None,
        feature_shuffles=None,
    ) -> RowStore:
        self._validate_request(
            source,
            y_train,
            embed_with_test=embed_with_test,
            d=d,
            feature_shuffles=feature_shuffles,
        )
        batch, rows, features = source.shape
        groups = self._group_count(features)
        output_shape = (
            batch,
            rows,
            self.row_interaction.num_cls * self.row_interaction.embed_dim,
        )
        if output_store is not None and (
            tuple(output_store.shape) != output_shape or output_store.dtype != self.dtype
        ):
            raise ValueError("output_store has the wrong shape or dtype")
        state = None
        owned_output = output_store is None
        try:
            state, skip_mask = self._initial_state(source, y_train, groups)
            for block in self.col_embedding.tf_col.blocks:
                next_state = self._new_store(self.intermediate_location, state.shape)
                try:
                    next_skip_mask = ISABStreamingExecutor(
                        block,
                        row_tile_rows=self.row_tile_rows,
                        work_device=self.work_device,
                    ).run(
                        state,
                        next_state,
                        support_rows=y_train.shape[1],
                        skip_mask=skip_mask,
                    )
                except Exception:
                    next_state.close()
                    raise
                state.close()
                state = next_state
                skip_mask = next_skip_mask

            if output_store is None:
                output_store = self._new_store(self.output_location, output_shape)
            reserve = self.col_embedding.reserve_cls_tokens
            for start in range(0, rows, self.row_tile_rows):
                end = min(start + self.row_tile_rows, rows)
                column_state = state.read_rows(start, end, self.work_device)
                if self.col_embedding.affine:
                    features_tile = self._group_rows(
                        source.read_rows(start, end, self.work_device)
                    )
                    weights = self.col_embedding.ln_w(self.col_embedding.out_w(column_state))
                    biases = self.col_embedding.ln_b(self.col_embedding.out_b(column_state))
                    column_state = features_tile * weights + biases
                reserved = torch.full(
                    (batch, end - start, reserve, self.col_embedding.embed_dim),
                    -100.0,
                    device=self.work_device,
                    dtype=self.dtype,
                )
                embeddings = torch.cat((reserved, column_state), dim=2)
                embeddings[:, :, : self.row_interaction.num_cls] = self.row_interaction.cls_tokens
                row_output = self.row_interaction._aggregate_embeddings(embeddings)
                output_store.write_rows(start, end, row_output)
            state.close()
            state = None
            return output_store
        except Exception:
            if state is not None:
                state.close()
            if owned_output and output_store is not None:
                output_store.close()
            raise


__all__ = [
    "RowSource",
    "RowStore",
    "DeviceTensorRowSource",
    "HostTensorRowSource",
    "DeviceTensorRowStore",
    "HostTensorRowStore",
    "MmapRowStore",
    "RowStoreFactory",
    "DefaultRowStoreFactory",
    "ISABStreamingExecutor",
    "StreamingFrontendExecutor",
]
