"""Bounded page access and on-demand head projection for V3 snapshots."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
import math
from typing import Hashable, Iterator, Literal, Mapping

import torch
from torch import Tensor
from torch.nn import functional as F

from .dual_path_snapshot import (
    LayerSnapshotReference,
    SnapshotProfile,
    SnapshotState,
    SupportSnapshot,
)


def _tensor_nbytes(tensor: Tensor) -> int:
    return tensor.numel() * tensor.element_size()


class BoundedPageCache:
    """Byte- and entry-bounded LRU cache for fixed-size tensor pages."""

    def __init__(self, max_pages: int, max_bytes: int):
        if max_pages < 1 or max_bytes < 1:
            raise ValueError("max_pages and max_bytes must be positive")
        self.max_pages = max_pages
        self.max_bytes = max_bytes
        self._items: OrderedDict[Hashable, Tensor] = OrderedDict()
        self._bytes = 0
        self.hits = 0
        self.misses = 0
        self.peak_bytes = 0
        self.peak_pages = 0

    @property
    def current_bytes(self) -> int:
        return self._bytes

    def __len__(self) -> int:
        return len(self._items)

    def get(self, key: Hashable) -> Tensor | None:
        value = self._items.get(key)
        if value is None:
            self.misses += 1
            return None
        self.hits += 1
        self._items.move_to_end(key)
        return value

    def put(self, key: Hashable, value: Tensor) -> None:
        value_bytes = _tensor_nbytes(value)
        existing = self._items.pop(key, None)
        if existing is not None:
            self._bytes -= _tensor_nbytes(existing)
        if value_bytes > self.max_bytes:
            return
        while self._items and (
            len(self._items) >= self.max_pages
            or self._bytes + value_bytes > self.max_bytes
        ):
            _, evicted = self._items.popitem(last=False)
            self._bytes -= _tensor_nbytes(evicted)
        self._items[key] = value
        self._bytes += value_bytes
        self.peak_bytes = max(self.peak_bytes, self._bytes)
        self.peak_pages = max(self.peak_pages, len(self._items))

    def discard_namespace(self, namespace: Hashable) -> None:
        for key in list(self._items):
            if isinstance(key, tuple) and key and key[0] == namespace:
                value = self._items.pop(key)
                self._bytes -= _tensor_nbytes(value)

    def clear(self) -> None:
        self._items.clear()
        self._bytes = 0


class BoundedStagingPool:
    """Bounded reusable CPU buffers for host-to-device staging.

    The initial implementation performs a synchronous transfer before returning
    a lease to the pool. That is correct for every backend and leaves async event
    ownership as a later transport optimization.
    """

    def __init__(self, max_buffers: int, max_bytes: int, pin_memory: bool = True):
        if max_buffers < 1 or max_bytes < 1:
            raise ValueError("max_buffers and max_bytes must be positive")
        self.max_buffers = max_buffers
        self.max_bytes = max_bytes
        self.pin_memory = pin_memory
        self._free: dict[tuple[tuple[int, ...], torch.dtype, bool], list[Tensor]] = {}
        self._leased: dict[int, tuple[tuple[int, ...], torch.dtype, bool]] = {}
        self._allocated_bytes = 0
        self._allocated_buffers = 0
        self.peak_allocated_bytes = 0
        self.peak_leased_buffers = 0

    @property
    def allocated_bytes(self) -> int:
        return self._allocated_bytes

    @property
    def leased_buffers(self) -> int:
        return len(self._leased)

    @staticmethod
    def _requested_bytes(shape: tuple[int, ...], dtype: torch.dtype) -> int:
        return math.prod(shape) * torch.empty((), dtype=dtype).element_size()

    def _evict_free(self, required_bytes: int) -> None:
        for key in list(self._free):
            pool = self._free[key]
            while pool and (
                self._allocated_buffers >= self.max_buffers
                or self._allocated_bytes + required_bytes > self.max_bytes
            ):
                buffer = pool.pop()
                self._allocated_buffers -= 1
                self._allocated_bytes -= _tensor_nbytes(buffer)
            if not pool:
                self._free.pop(key, None)

    @contextmanager
    def acquire(
        self, shape: tuple[int, ...], dtype: torch.dtype
    ) -> Iterator[Tensor]:
        use_pin = self.pin_memory and torch.cuda.is_available()
        key = (shape, dtype, use_pin)
        pool = self._free.get(key)
        if pool:
            buffer = pool.pop()
            if not pool:
                self._free.pop(key, None)
        else:
            required = self._requested_bytes(shape, dtype)
            if required > self.max_bytes:
                raise ValueError("requested staging buffer exceeds max_bytes")
            self._evict_free(required)
            if (
                self._allocated_buffers >= self.max_buffers
                or self._allocated_bytes + required > self.max_bytes
            ):
                raise RuntimeError("all bounded staging capacity is currently leased")
            buffer = torch.empty(shape, dtype=dtype, device="cpu", pin_memory=use_pin)
            self._allocated_buffers += 1
            self._allocated_bytes += required
            self.peak_allocated_bytes = max(
                self.peak_allocated_bytes, self._allocated_bytes
            )
        self._leased[id(buffer)] = key
        self.peak_leased_buffers = max(
            self.peak_leased_buffers, len(self._leased)
        )
        try:
            yield buffer
        finally:
            lease_key = self._leased.pop(id(buffer))
            self._free.setdefault(lease_key, []).append(buffer)

    def clear(self) -> None:
        if self._leased:
            raise RuntimeError("cannot clear staging pool while buffers are leased")
        self._free.clear()
        self._allocated_bytes = 0
        self._allocated_buffers = 0


@dataclass(frozen=True)
class PrefetchHandle:
    """Stable completion handle for synchronous or future asynchronous prefetch."""

    page_ids: tuple[int, ...]
    requested_bytes: int
    completed: bool = True

    def wait(self) -> None:
        """Wait for completion; synchronous providers are already complete."""

        return None


class PageProvider:
    """Read arbitrary rows through a page cache without exposing storage layout."""

    def __init__(
        self,
        tensor: Tensor,
        *,
        page_rows: int,
        cache: BoundedPageCache | None = None,
        namespace: Hashable | None = None,
        cache_page_mode: Literal["copy", "view"] = "view",
        mmap_backed: bool = False,
    ):
        if tensor.ndim != 2:
            raise ValueError("paged hidden tensor must have shape [rows, model_dim]")
        if page_rows < 1:
            raise ValueError("page_rows must be positive")
        if cache_page_mode not in ("copy", "view"):
            raise ValueError("cache_page_mode must be 'copy' or 'view'")
        self.tensor = tensor
        self.page_rows = page_rows
        self.cache = cache
        self.namespace = id(self) if namespace is None else namespace
        self.cache_page_mode = cache_page_mode
        self.mmap_backed = mmap_backed
        self.closed = False

    @property
    def rows(self) -> int:
        return self.tensor.shape[0]

    @property
    def model_dim(self) -> int:
        return self.tensor.shape[1]

    @property
    def source_device(self) -> torch.device:
        return self.tensor.device

    def _require_open(self) -> None:
        if self.closed:
            raise RuntimeError("page provider is closed")

    def _load_page(self, page_id: int) -> Tensor:
        start = page_id * self.page_rows
        end = min(start + self.page_rows, self.rows)
        page = self.tensor[start:end]
        if end - start == self.page_rows:
            return page
        padded = torch.zeros(
            (self.page_rows, self.model_dim),
            dtype=self.tensor.dtype,
            device=self.tensor.device,
        )
        padded[: end - start] = page
        return padded

    def page(self, page_id: int) -> Tensor:
        self._require_open()
        num_pages = (self.rows + self.page_rows - 1) // self.page_rows
        if not 0 <= page_id < num_pages:
            raise IndexError(f"page {page_id} is outside [0, {num_pages})")
        key = (self.namespace, page_id)
        cached = self.cache.get(key) if self.cache is not None else None
        if cached is not None:
            return cached
        value = self._load_page(page_id)
        if self.cache_page_mode == "copy":
            value = value.clone()
        if self.cache is not None:
            self.cache.put(key, value)
        return value

    def prefetch(self, page_ids: Tensor | Iterable[int]) -> PrefetchHandle:
        """Populate pages synchronously and return an async-compatible handle."""

        self._require_open()
        if isinstance(page_ids, Tensor):
            requested = page_ids.detach().to(device="cpu", dtype=torch.long).flatten()
            values = requested.tolist()
        else:
            values = [int(page_id) for page_id in page_ids]
        unique = tuple(sorted(set(values)))
        requested_bytes = 0
        for page_id in unique:
            page = self.page(page_id)
            requested_bytes += _tensor_nbytes(page)
        return PrefetchHandle(page_ids=unique, requested_bytes=requested_bytes)

    def prefetch_rows(self, row_ids: Tensor) -> PrefetchHandle:
        """Prefetch all physical pages containing the requested support rows."""

        ids = row_ids.detach().to(device="cpu", dtype=torch.long).flatten()
        if ids.numel() == 0:
            return PrefetchHandle(page_ids=(), requested_bytes=0)
        if ids.min().item() < 0 or ids.max().item() >= self.rows:
            raise IndexError("row_ids contain an out-of-range support row")
        pages = torch.div(ids, self.page_rows, rounding_mode="floor")
        return self.prefetch(pages)

    def _read_rows_source(self, row_ids: Tensor) -> Tensor:
        ids = row_ids.detach().to(device="cpu", dtype=torch.long)
        flat = ids.flatten()
        if flat.numel() == 0:
            return torch.empty(
                (*ids.shape, self.model_dim),
                dtype=self.tensor.dtype,
                device=self.source_device,
            )
        if flat.min().item() < 0 or flat.max().item() >= self.rows:
            raise IndexError("row_ids contain an out-of-range support row")
        output = torch.empty(
            (flat.numel(), self.model_dim),
            dtype=self.tensor.dtype,
            device=self.source_device,
        )
        page_ids = torch.div(flat, self.page_rows, rounding_mode="floor")
        offsets = flat % self.page_rows
        for page_id in torch.unique(page_ids, sorted=True).tolist():
            mask = page_ids == page_id
            page_offsets = offsets[mask].to(self.source_device)
            output[mask.to(self.source_device)] = self.page(page_id).index_select(
                0, page_offsets
            )
        return output.reshape(*ids.shape, self.model_dim)

    def read_rows(
        self, row_ids: Tensor, *, device: str | torch.device | None = None
    ) -> Tensor:
        value = self._read_rows_source(row_ids)
        if device is None:
            return value
        target = torch.device(device)
        return value if value.device == target else value.to(target)

    def close(self) -> None:
        if self.closed:
            return
        if self.cache is not None:
            self.cache.discard_namespace(self.namespace)
        self.tensor = torch.empty(0)
        self.closed = True


class DevicePageProvider(PageProvider):
    """Page provider for a tensor already resident on its execution device."""


class HostPageProvider(PageProvider):
    """CPU or mmap-backed provider with bounded transfer staging."""

    def __init__(
        self,
        tensor: Tensor,
        *,
        page_rows: int,
        cache: BoundedPageCache | None = None,
        staging_pool: BoundedStagingPool | None = None,
        namespace: Hashable | None = None,
        cache_page_mode: Literal["copy", "view"] = "copy",
        mmap_backed: bool = False,
    ):
        if tensor.device.type != "cpu":
            raise ValueError("HostPageProvider requires a CPU tensor")
        super().__init__(
            tensor,
            page_rows=page_rows,
            cache=cache,
            namespace=namespace,
            cache_page_mode=cache_page_mode,
            mmap_backed=mmap_backed,
        )
        self.staging_pool = staging_pool

    def read_rows(
        self, row_ids: Tensor, *, device: str | torch.device | None = None
    ) -> Tensor:
        value = self._read_rows_source(row_ids)
        target = torch.device("cpu" if device is None else device)
        if target.type == "cpu":
            return value
        if self.staging_pool is None:
            return value.to(target)
        with self.staging_pool.acquire(tuple(value.shape), value.dtype) as staging:
            staging.copy_(value)
            return staging.to(target, non_blocking=False)


@dataclass(frozen=True)
class HeadProjectionWeights:
    """Sparse K/V projection slices shared by every runtime layer query."""

    key_weight: Tensor
    value_weight: Tensor
    num_heads: int
    key_bias: Tensor | None = None
    value_bias: Tensor | None = None

    @classmethod
    def from_qkv(
        cls, in_proj_weight: Tensor, in_proj_bias: Tensor | None, num_heads: int
    ) -> "HeadProjectionWeights":
        if in_proj_weight.ndim != 2 or in_proj_weight.shape[0] % 3:
            raise ValueError("in_proj_weight must have shape [3 * embed_dim, embed_dim]")
        embed_dim = in_proj_weight.shape[0] // 3
        if in_proj_weight.shape[1] != embed_dim:
            raise ValueError("in_proj_weight must be square per Q/K/V projection")
        key_bias = value_bias = None
        if in_proj_bias is not None:
            if in_proj_bias.shape != (3 * embed_dim,):
                raise ValueError("in_proj_bias must have shape [3 * embed_dim]")
            key_bias = in_proj_bias[embed_dim : 2 * embed_dim]
            value_bias = in_proj_bias[2 * embed_dim :]
        return cls(
            key_weight=in_proj_weight[embed_dim : 2 * embed_dim],
            value_weight=in_proj_weight[2 * embed_dim :],
            key_bias=key_bias,
            value_bias=value_bias,
            num_heads=num_heads,
        )

    @property
    def embed_dim(self) -> int:
        return self.key_weight.shape[0]

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads

    def validate(self) -> None:
        embed_dim = self.embed_dim
        if self.num_heads < 1 or embed_dim % self.num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        if self.key_weight.shape != (embed_dim, embed_dim):
            raise ValueError("key_weight must have shape [embed_dim, embed_dim]")
        if self.value_weight.shape != (embed_dim, embed_dim):
            raise ValueError("value_weight must match key_weight")
        if self.value_weight.device != self.key_weight.device:
            raise ValueError("key_weight and value_weight must share a device")
        if self.value_weight.dtype != self.key_weight.dtype:
            raise ValueError("key_weight and value_weight must share a dtype")
        for name, bias in (("key_bias", self.key_bias), ("value_bias", self.value_bias)):
            if bias is not None and bias.shape != (embed_dim,):
                raise ValueError(f"{name} must have shape [embed_dim]")
            if bias is not None and bias.device != self.key_weight.device:
                raise ValueError(f"{name} must share the projection device")
            if bias is not None and bias.dtype != self.key_weight.dtype:
                raise ValueError(f"{name} must share the projection dtype")


@dataclass(frozen=True)
class ProjectedHeadKV:
    key: Tensor
    value: Tensor
    row_ids: Tensor


class OnDemandHeadProjector:
    """Read unique hidden rows once and project only their requested heads."""

    def __init__(self, weights: HeadProjectionWeights):
        weights.validate()
        self.weights = weights

    def project(
        self,
        provider: PageProvider,
        row_ids: Tensor,
        *,
        device: str | torch.device | None = None,
    ) -> ProjectedHeadKV:
        """Project row IDs shaped ``[heads, ...]`` into per-head sparse K/V."""

        if row_ids.ndim < 2 or row_ids.shape[0] != self.weights.num_heads:
            raise ValueError("row_ids must have shape [num_heads, ...]")
        target = self.weights.key_weight.device if device is None else torch.device(device)
        tensors = (
            self.weights.key_weight,
            self.weights.value_weight,
            self.weights.key_bias,
            self.weights.value_bias,
        )
        if any(tensor is not None and tensor.device != target for tensor in tensors):
            raise ValueError("projection weights and requested output device must match")
        if provider.model_dim != self.weights.embed_dim:
            raise ValueError("hidden provider model_dim does not match projection weights")

        ids = row_ids.detach().to(dtype=torch.long)
        unique, inverse = torch.unique(ids.flatten(), sorted=True, return_inverse=True)
        hidden_unique = provider.read_rows(unique, device=target).to(
            dtype=self.weights.key_weight.dtype
        )
        selected = hidden_unique.index_select(0, inverse.to(target)).reshape(
            *ids.shape, self.weights.embed_dim
        )
        output_shape = (*ids.shape, self.weights.head_dim)
        key = torch.empty(output_shape, dtype=selected.dtype, device=target)
        value = torch.empty_like(key)
        for head in range(self.weights.num_heads):
            start = head * self.weights.head_dim
            end = start + self.weights.head_dim
            key_bias = None if self.weights.key_bias is None else self.weights.key_bias[start:end]
            value_bias = (
                None if self.weights.value_bias is None else self.weights.value_bias[start:end]
            )
            key[head] = F.linear(
                selected[head], self.weights.key_weight[start:end], key_bias
            )
            value[head] = F.linear(
                selected[head], self.weights.value_weight[start:end], value_bias
            )
        return ProjectedHeadKV(key=key, value=value, row_ids=ids.to(target))


class HiddenLayerRuntime:
    """Runtime view over one layer of a hidden-profile snapshot."""

    def __init__(
        self,
        snapshot: SupportSnapshot,
        layer: LayerSnapshotReference,
        hidden_provider: PageProvider,
    ):
        self.snapshot = snapshot
        self.layer = layer
        self.hidden_provider = hidden_provider
        self.closed = False

    def _tensor_group(
        self, references: Mapping[str, str], device: str | torch.device | None
    ) -> dict[str, Tensor]:
        if self.closed:
            raise RuntimeError("layer runtime is closed")
        target = None if device is None else torch.device(device)
        result = {}
        for name, reference in references.items():
            tensor = self.snapshot.tensor(reference)
            result[name] = tensor if target is None or tensor.device == target else tensor.to(target)
        return result

    def global_state(
        self, device: str | torch.device | None = None
    ) -> dict[str, Tensor]:
        return self._tensor_group(self.layer.global_state, device)

    def index_state(
        self, device: str | torch.device | None = None
    ) -> dict[str, Tensor]:
        return self._tensor_group(self.layer.index_state, device)

    def project(
        self,
        row_ids: Tensor,
        weights: HeadProjectionWeights,
        *,
        device: str | torch.device | None = None,
    ) -> ProjectedHeadKV:
        if self.closed:
            raise RuntimeError("layer runtime is closed")
        return OnDemandHeadProjector(weights).project(
            self.hidden_provider, row_ids, device=device
        )

    def close(self) -> None:
        if self.closed:
            return
        self.hidden_provider.close()
        self.snapshot.release_tensor(self.layer.hidden)
        for reference in self.layer.global_state.values():
            self.snapshot.release_tensor(reference)
        for reference in self.layer.index_state.values():
            self.snapshot.release_tensor(reference)
        self.closed = True


class HiddenSnapshotRuntime:
    """Bounded-lifetime runtime for one-shot or online hidden snapshots.

    ``max_open_layers`` bounds resident hidden providers. With the default of one,
    online inference never retains twelve layers of hidden tensors or sparse K/V.
    """

    def __init__(
        self,
        snapshot: SupportSnapshot,
        *,
        location: str = "host",
        device: str | torch.device = "cpu",
        page_rows: int = 4096,
        max_open_layers: int = 1,
        page_cache: BoundedPageCache | None = None,
        staging_pool: BoundedStagingPool | None = None,
        host_cache_page_mode: Literal["copy", "view"] = "copy",
    ):
        if location not in ("device", "host"):
            raise ValueError("location must be 'device' or 'host'")
        if max_open_layers < 1:
            raise ValueError("max_open_layers must be positive")
        if snapshot.state == SnapshotState.CLOSED:
            raise RuntimeError("cannot open a closed snapshot")
        if (
            snapshot.profile == SnapshotProfile.ONLINE_HIDDEN
            and snapshot.state != SnapshotState.SEALED
        ):
            raise RuntimeError("online hidden runtime requires a sealed snapshot")
        self.snapshot = snapshot
        self.location = location
        self.device = torch.device(device)
        self.page_rows = page_rows
        self.max_open_layers = max_open_layers
        self.page_cache = page_cache
        self.staging_pool = staging_pool
        self.host_cache_page_mode = host_cache_page_mode
        self._open: OrderedDict[int, HiddenLayerRuntime] = OrderedDict()
        self.closed = False

    def open_layer(self, layer_index: int) -> HiddenLayerRuntime:
        if self.closed:
            raise RuntimeError("snapshot runtime is closed")
        existing = self._open.get(layer_index)
        if existing is not None:
            self._open.move_to_end(layer_index)
            return existing

        layer = self.snapshot.layer(layer_index)
        hidden = self.snapshot.tensor(layer.hidden)
        namespace = (self.snapshot.manifest.snapshot_id, layer_index, "hidden")
        if self.location == "device":
            resident = hidden if hidden.device == self.device else hidden.to(self.device)
            provider: PageProvider = DevicePageProvider(
                resident,
                page_rows=self.page_rows,
                cache=self.page_cache,
                namespace=namespace,
            )
        else:
            host = hidden if hidden.device.type == "cpu" else hidden.to("cpu")
            provider = HostPageProvider(
                host,
                page_rows=self.page_rows,
                cache=self.page_cache,
                staging_pool=self.staging_pool,
                namespace=namespace,
                cache_page_mode=self.host_cache_page_mode,
                mmap_backed=self.snapshot.is_mmap_backed(layer.hidden),
            )
        runtime = HiddenLayerRuntime(self.snapshot, layer, provider)
        self._open[layer_index] = runtime
        while len(self._open) > self.max_open_layers:
            _, evicted = self._open.popitem(last=False)
            evicted.close()
        return runtime

    def close_layer(self, layer_index: int) -> None:
        runtime = self._open.pop(layer_index, None)
        if runtime is not None:
            runtime.close()

    def close(self) -> None:
        if self.closed:
            return
        for runtime in self._open.values():
            runtime.close()
        self._open.clear()
        self.closed = True

    def __enter__(self) -> "HiddenSnapshotRuntime":
        if self.closed:
            raise RuntimeError("snapshot runtime is closed")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
