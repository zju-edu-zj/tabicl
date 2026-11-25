"""Persistent, external-memory content-tree index.

The builder accepts one head and one bounded row shard at a time. Finalization
replays each head from an on-disk record stream and builds the same recursive
content partition used by V3: sixteen rank samples choose an approximate
farthest pair, and its direction bisects the node. Rank selection and bisection
use bounded-memory sequential scans; a full ``[H,N,D]`` key tensor and a full
row-to-position inverse are never materialized.

Logical leaves and physical pages are independent. The immutable manifest is
published last, so an interrupted build cannot be loaded as a valid index.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
from typing import BinaryIO, Iterator

import numpy as np
import torch
from torch import Tensor


FORMAT_NAME = "tabicl-content-index-persistent"
SCHEMA_VERSION = 1
ALGORITHM_VERSION = 2
MANIFEST_NAME = "manifest.json"

_DTYPES = {
    "float32": np.dtype("<f4"),
    "int32": np.dtype("<i4"),
    "int64": np.dtype("<i8"),
}
_UINT32_SIGN = np.uint32(1 << 31)
_UINT64_SIGN = np.uint64(1 << 63)


@dataclass(frozen=True)
class PersistentIndexLayout:
    """Logical retrieval, physical storage, and bounded build parameters."""

    logical_leaf_rows: int = 32
    physical_page_rows: int = 128
    selected_logical_leaves: int = 32
    search_beam_leaves: int = 64
    proposal_leaves: int = 64
    exhaustive_leaf_limit: int = 256
    build_chunk_rows: int = 65536

    def __post_init__(self):
        if self.logical_leaf_rows < 2 or self.logical_leaf_rows % 2:
            raise ValueError("logical_leaf_rows must be a positive even number")
        if self.physical_page_rows < self.logical_leaf_rows:
            raise ValueError("physical_page_rows must fit one logical leaf")
        positive = (
            self.selected_logical_leaves,
            self.search_beam_leaves,
            self.proposal_leaves,
            self.exhaustive_leaf_limit,
            self.build_chunk_rows,
        )
        if any(value < 1 for value in positive):
            raise ValueError("layout budgets and build_chunk_rows must be positive")


@dataclass
class PersistentSearchResult:
    """Fixed-shape retrieval result with int64-safe identifiers."""

    logical_leaf_ids: Tensor
    physical_page_ids: Tensor
    page_row_offsets: Tensor
    candidate_rows: Tensor
    visited_nodes: Tensor


@dataclass
class _OpenArray:
    path: Path
    dtype: np.dtype
    shape: tuple[int, ...]
    array: np.memmap


@dataclass
class _HeadBuildState:
    stream: BinaryIO
    row_count: int = 0
    first_row_id: int | None = None
    last_row_id: int | None = None
    finalized: bool = False


@dataclass
class _PartitionTask:
    offset: int
    count: int
    capacity: int
    ordering_direction: np.ndarray


def _record_dtype(head_dim: int) -> np.dtype:
    return np.dtype([
        ("row_id", _DTYPES["int64"]),
        ("key", _DTYPES["float32"], (head_dim,)),
    ])


def _write_array(path: Path, value: np.ndarray) -> None:
    value = np.ascontiguousarray(value)
    with path.open("wb") as stream:
        value.tofile(stream)


def _array_spec(path: Path, dtype: np.dtype, shape: tuple[int, ...]) -> dict:
    return {
        "file": path.name,
        "dtype": next(name for name, item in _DTYPES.items() if item == dtype),
        "shape": list(shape),
        "nbytes": int(math.prod(shape) * dtype.itemsize),
    }


def _open_array(root: Path, spec: dict) -> _OpenArray:
    try:
        dtype = _DTYPES[spec["dtype"]]
        shape = tuple(int(value) for value in spec["shape"])
        path = root / spec["file"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid persistent content-index array specification") from error
    expected = int(math.prod(shape) * dtype.itemsize)
    if expected != int(spec.get("nbytes", -1)) or not path.is_file():
        raise ValueError(f"invalid or missing persistent content-index array: {path.name}")
    if path.stat().st_size != expected:
        raise ValueError(f"persistent content-index array has wrong size: {path.name}")
    array = np.memmap(path, mode="r", dtype=dtype, shape=shape)
    return _OpenArray(path=path, dtype=dtype, shape=shape, array=array)


def _iter_records(
    path: Path,
    dtype: np.dtype,
    offset: int,
    count: int,
    chunk_rows: int,
) -> Iterator[np.ndarray]:
    with path.open("rb") as stream:
        stream.seek(offset * dtype.itemsize)
        remaining = count
        while remaining:
            take = min(remaining, chunk_rows)
            records = np.fromfile(stream, dtype=dtype, count=take)
            if records.size != take:
                raise ValueError(f"truncated content-tree work file: {path.name}")
            yield records
            remaining -= take


def _score_order_keys(records: np.ndarray, direction: np.ndarray) -> np.ndarray:
    keys = torch.from_numpy(np.ascontiguousarray(records["key"]))
    vector = torch.from_numpy(np.ascontiguousarray(direction, dtype=np.float32))
    scores = (keys * vector).sum(-1).numpy()
    finite = np.finfo(np.float32)
    scores = np.nan_to_num(scores, nan=0.0, posinf=finite.max, neginf=finite.min)
    bits = scores.view(np.uint32)
    return np.where(
        bits & _UINT32_SIGN,
        np.bitwise_not(bits),
        bits ^ _UINT32_SIGN,
    ).astype(np.uint32, copy=False)


def _row_order_keys(records: np.ndarray) -> np.ndarray:
    rows = np.asarray(records["row_id"], dtype=np.int64).view(np.uint64)
    return rows ^ _UINT64_SIGN


def _locate_rank(counts: np.ndarray, rank: int) -> tuple[int, int]:
    cumulative = np.cumsum(counts, dtype=np.int64)
    bucket = int(np.searchsorted(cumulative, rank, side="right"))
    if bucket >= counts.size:
        raise RuntimeError("external rank selection exhausted its histogram")
    base = 0 if bucket == 0 else int(cumulative[bucket - 1])
    return bucket, rank - base


def _select_row_order_keys(
    path: Path,
    dtype: np.dtype,
    task: _PartitionTask,
    direction: np.ndarray,
    score_key: int,
    ranks: list[int],
    chunk_rows: int,
) -> list[int]:
    states = [(0, list(enumerate(ranks)))]
    result = [0] * len(ranks)
    for part in range(4):
        shift = 48 - part * 16
        prefix_shift = 64 - part * 16
        histograms = [np.zeros(1 << 16, dtype=np.int64) for _ in states]
        for records in _iter_records(
            path, dtype, task.offset, task.count, chunk_rows
        ):
            scores = _score_order_keys(records, direction)
            rows = _row_order_keys(records)
            for state_index, (prefix, _) in enumerate(states):
                mask = scores == np.uint32(score_key)
                if part:
                    mask &= (rows >> np.uint64(prefix_shift)) == np.uint64(prefix)
                buckets = ((rows[mask] >> np.uint64(shift)) & np.uint64(0xFFFF))
                histograms[state_index] += np.bincount(
                    buckets.astype(np.int64), minlength=1 << 16
                )
        next_states = []
        for (prefix, requests), counts in zip(states, histograms):
            grouped: dict[int, list[tuple[int, int]]] = {}
            for request, rank in requests:
                bucket, relative = _locate_rank(counts, rank)
                grouped.setdefault(bucket, []).append((request, relative))
            for bucket, requests in grouped.items():
                next_prefix = (prefix << 16) | bucket
                if part == 3:
                    for request, _ in requests:
                        result[request] = next_prefix
                else:
                    next_states.append((next_prefix, requests))
        states = next_states
    return result


def _select_records(
    path: Path,
    dtype: np.dtype,
    task: _PartitionTask,
    direction: np.ndarray,
    ranks: list[int],
    chunk_rows: int,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Select exact score/row-id order statistics by external radix scans."""

    if not ranks or min(ranks) < 0 or max(ranks) >= task.count:
        raise ValueError("rank is outside the content-tree node")
    high_counts = np.zeros(1 << 16, dtype=np.int64)
    for records in _iter_records(path, dtype, task.offset, task.count, chunk_rows):
        high = _score_order_keys(records, direction) >> np.uint32(16)
        high_counts += np.bincount(high.astype(np.int64), minlength=1 << 16)

    high_requests: dict[int, list[tuple[int, int]]] = {}
    for request, rank in enumerate(ranks):
        high, relative = _locate_rank(high_counts, rank)
        high_requests.setdefault(high, []).append((request, relative))
    low_counts = {
        high: np.zeros(1 << 16, dtype=np.int64) for high in high_requests
    }
    for records in _iter_records(path, dtype, task.offset, task.count, chunk_rows):
        keys = _score_order_keys(records, direction)
        high = keys >> np.uint32(16)
        low = keys & np.uint32(0xFFFF)
        for target, counts in low_counts.items():
            counts += np.bincount(
                low[high == target].astype(np.int64), minlength=1 << 16
            )

    score_keys = [0] * len(ranks)
    tie_ranks = [0] * len(ranks)
    tie_counts = [0] * len(ranks)
    for high, requests in high_requests.items():
        counts = low_counts[high]
        for request, relative in requests:
            low, tie_rank = _locate_rank(counts, relative)
            score_keys[request] = (high << 16) | low
            tie_ranks[request] = tie_rank
            tie_counts[request] = int(counts[low])

    row_keys: list[int | None] = [None] * len(ranks)
    by_score: dict[int, list[int]] = {}
    for request, score_key in enumerate(score_keys):
        by_score.setdefault(score_key, []).append(request)
    for score_key, requests in by_score.items():
        if tie_counts[requests[0]] == 1:
            continue
        selected = _select_row_order_keys(
            path,
            dtype,
            task,
            direction,
            score_key,
            [tie_ranks[request] for request in requests],
            chunk_rows,
        )
        for request, row_key in zip(requests, selected):
            row_keys[request] = row_key

    selected_records = np.empty(len(ranks), dtype=dtype)
    found = np.zeros(len(ranks), dtype=np.bool_)
    for records in _iter_records(path, dtype, task.offset, task.count, chunk_rows):
        chunk_scores = _score_order_keys(records, direction)
        chunk_rows_order = _row_order_keys(records)
        for request, score_key in enumerate(score_keys):
            if found[request]:
                continue
            mask = chunk_scores == np.uint32(score_key)
            if row_keys[request] is not None:
                mask &= chunk_rows_order == np.uint64(row_keys[request])
            matches = np.flatnonzero(mask)
            if matches.size:
                selected_records[request] = records[int(matches[0])]
                row_keys[request] = int(chunk_rows_order[int(matches[0])])
                found[request] = True
        if found.all():
            break
    if not found.all():
        raise RuntimeError("external content-tree selection lost a requested row")
    return selected_records, [
        (score_key, int(row_key))
        for score_key, row_key in zip(score_keys, row_keys)
    ]


def _farthest_direction(samples: np.ndarray) -> np.ndarray:
    points = torch.from_numpy(np.ascontiguousarray(samples["key"]))
    mean = points.mean(0, keepdim=True)
    a_index = (points - mean).square().sum(-1).argmax()
    a = points[a_index]
    z_index = (points - a).square().sum(-1).argmax()
    return (points[z_index] - a).numpy().astype(np.float32, copy=False)


def _initial_direction(head_dim: int) -> np.ndarray:
    direction = torch.arange(1, head_dim + 1, dtype=torch.float32)
    return torch.nn.functional.normalize(direction, dim=0).numpy()


class PersistentContentIndexBuilder:
    """Build a content-tree index from bounded per-head row shards."""

    def __init__(
        self,
        path,
        *,
        num_heads: int,
        head_dim: int,
        layout: PersistentIndexLayout | None = None,
        partition: str = "content_tree",
    ):
        if num_heads < 1 or head_dim < 1:
            raise ValueError("num_heads and head_dim must be positive")
        if partition != "content_tree":
            raise ValueError("persistent V3 currently requires partition='content_tree'")
        self.path = Path(path)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.layout = PersistentIndexLayout() if layout is None else layout
        self.partition = partition
        if self.path.exists():
            if not self.path.is_dir() or any(self.path.iterdir()):
                raise FileExistsError(
                    f"index build directory is not empty: {self.path}"
                )
        self.path.mkdir(parents=True, exist_ok=True)
        self._record_dtype = _record_dtype(self.head_dim)
        self._closed = False
        self._published = False
        self._states = [
            _HeadBuildState(self._work_path(head, "ingest.records").open("wb"))
            for head in range(self.num_heads)
        ]

    def _work_path(self, head: int, suffix: str) -> Path:
        return self.path / f".head-{head:03d}-{suffix}"

    def _array_path(self, head: int, suffix: str) -> Path:
        return self.path / f"head-{head:03d}-{suffix}"

    @property
    def finalized(self) -> bool:
        return self._published

    @torch.no_grad()
    def ingest(self, head: int, row_ids: Tensor, keys: Tensor) -> None:
        """Append one bounded row shard for one head."""

        if self._closed:
            raise RuntimeError("cannot ingest after builder close or finalization")
        if not 0 <= head < self.num_heads:
            raise IndexError("head is out of range")
        if row_ids.ndim != 1 or keys.ndim != 2:
            raise ValueError("row_ids must be [R] and keys must be [R,D]")
        if keys.shape != (row_ids.numel(), self.head_dim):
            raise ValueError("row shard and key shard shapes do not match")
        if row_ids.numel() > self.layout.build_chunk_rows:
            raise ValueError("ingest shard exceeds layout.build_chunk_rows")
        if not row_ids.numel():
            return
        if not bool(torch.isfinite(keys).all()):
            raise ValueError("content keys must be finite")
        ids = row_ids.detach().to(device="cpu", dtype=torch.int64).contiguous()
        if not bool((ids[1:] > ids[:-1]).all()):
            raise ValueError("row IDs must be strictly increasing within each shard")
        state = self._states[head]
        first, last = int(ids[0]), int(ids[-1])
        if state.last_row_id is not None and first <= state.last_row_id:
            raise ValueError("row shards must be strictly increasing for each head")
        records = np.empty(ids.numel(), dtype=self._record_dtype)
        records["row_id"] = ids.numpy()
        records["key"] = keys.detach().to(
            device="cpu", dtype=torch.float32
        ).contiguous().numpy()
        records.tofile(state.stream)
        state.row_count += ids.numel()
        if state.first_row_id is None:
            state.first_row_id = first
        state.last_row_id = last

    def _close_ingest(self, state: _HeadBuildState) -> None:
        state.stream.flush()
        os.fsync(state.stream.fileno())
        state.stream.close()

    def _copy_task(self, source: Path, task: _PartitionTask, output: BinaryIO) -> None:
        for records in _iter_records(
            source,
            self._record_dtype,
            task.offset,
            task.count,
            self.layout.build_chunk_rows,
        ):
            records.tofile(output)

    def _partition_task(
        self,
        source: Path,
        task: _PartitionTask,
        output: BinaryIO,
        output_offset: int,
        temporary_right: Path,
    ) -> tuple[list[_PartitionTask], int]:
        sample_ranks = [index * (task.count - 1) // 15 for index in range(16)]
        samples, _ = _select_records(
            source,
            self._record_dtype,
            task,
            task.ordering_direction,
            sample_ranks,
            self.layout.build_chunk_rows,
        )
        direction = _farthest_direction(samples)
        left_target = min(task.count, task.capacity // 2)
        left_task = _PartitionTask(
            output_offset, left_target, task.capacity // 2, direction
        )
        if left_target == task.count:
            self._copy_task(source, task, output)
            return [left_task], output_offset + task.count

        _, pivot_keys = _select_records(
            source,
            self._record_dtype,
            task,
            direction,
            [left_target - 1],
            self.layout.build_chunk_rows,
        )
        pivot_score, pivot_row = pivot_keys[0]
        left_count = 0
        right_count = 0
        with temporary_right.open("wb") as right_stream:
            for records in _iter_records(
                source,
                self._record_dtype,
                task.offset,
                task.count,
                self.layout.build_chunk_rows,
            ):
                scores = _score_order_keys(records, direction)
                rows = _row_order_keys(records)
                left = (scores < np.uint32(pivot_score)) | (
                    (scores == np.uint32(pivot_score))
                    & (rows <= np.uint64(pivot_row))
                )
                records[left].tofile(output)
                records[~left].tofile(right_stream)
                left_count += int(left.sum())
                right_count += int((~left).sum())
            right_stream.flush()
            os.fsync(right_stream.fileno())
        if left_count != left_target or left_count + right_count != task.count:
            raise RuntimeError("external content-tree partition is unbalanced")
        right_offset = output_offset + left_count
        with temporary_right.open("rb") as right_stream:
            shutil.copyfileobj(right_stream, output, length=8 << 20)
        temporary_right.unlink()
        return [
            left_task,
            _PartitionTask(right_offset, right_count, task.capacity // 2, direction),
        ], right_offset + right_count

    def _build_partition(self, head: int, rows: int) -> tuple[Path, list[_PartitionTask]]:
        leaves = (rows + self.layout.logical_leaf_rows - 1) // self.layout.logical_leaf_rows
        capacity = self.layout.logical_leaf_rows * (1 << (leaves - 1).bit_length())
        current = self._work_path(head, "ingest.records")
        tasks = [_PartitionTask(0, rows, capacity, _initial_direction(self.head_dim))]
        level = 0
        while tasks[0].capacity > self.layout.logical_leaf_rows:
            following = self._work_path(head, f"level-{level:03d}.records")
            next_tasks: list[_PartitionTask] = []
            output_offset = 0
            with following.open("wb") as output:
                for task_index, task in enumerate(tasks):
                    right = self._work_path(
                        head, f"right-{level:03d}-{task_index:08d}.records"
                    )
                    children, output_offset = self._partition_task(
                        current, task, output, output_offset, right
                    )
                    next_tasks.extend(children)
                output.flush()
                os.fsync(output.fileno())
            current.unlink()
            current = following
            tasks = next_tasks
            level += 1
        return current, tasks

    def _terminal_records(self, source: Path, task: _PartitionTask) -> np.ndarray:
        records = next(_iter_records(
            source,
            self._record_dtype,
            task.offset,
            task.count,
            self.layout.logical_leaf_rows,
        ))
        scores = _score_order_keys(records, task.ordering_direction)
        rows = _row_order_keys(records)
        return records[np.lexsort((rows, scores))]

    def _build_tree_arrays(
        self,
        head: int,
        leaf_representatives: np.ndarray,
        leaf_rep_counts: np.ndarray,
    ) -> tuple[dict[str, tuple[Path, np.ndarray]], int]:
        leaves = leaf_representatives.shape[0]
        capacity = 1 << (leaves - 1).bit_length()
        nodes = 2 * capacity - 1
        leaf_offset = capacity - 1
        sums_path = self._work_path(head, "tree-sums.f32")
        counts_path = self._work_path(head, "tree-counts.i64")
        reps_path = self._array_path(head, "tree-representatives.f32")
        rep_counts_path = self._array_path(head, "tree-rep-counts.i64")
        sums = np.memmap(
            sums_path, mode="w+", dtype=_DTYPES["float32"],
            shape=(nodes, self.head_dim),
        )
        counts = np.memmap(
            counts_path, mode="w+", dtype=_DTYPES["int64"], shape=(nodes,)
        )
        representatives = np.memmap(
            reps_path, mode="w+", dtype=_DTYPES["float32"],
            shape=(nodes, 2, self.head_dim),
        )
        rep_counts = np.memmap(
            rep_counts_path, mode="w+", dtype=_DTYPES["int64"],
            shape=(nodes, 2),
        )
        sums[:] = 0
        counts[:] = 0
        representatives[:] = 0
        rep_counts[:] = 0
        for start in range(0, leaves, self.layout.build_chunk_rows):
            end = min(start + self.layout.build_chunk_rows, leaves)
            reps = leaf_representatives[start:end]
            local_counts = leaf_rep_counts[start:end]
            node_slice = slice(leaf_offset + start, leaf_offset + end)
            sums[node_slice] = (reps * local_counts[..., None]).sum(1)
            counts[node_slice] = local_counts.sum(1)
            representatives[node_slice] = reps
            rep_counts[node_slice] = local_counts
        level_start = leaf_offset
        level_width = capacity
        while level_width > 1:
            parent_start = (level_start - 1) // 2
            parent_width = level_width // 2
            for start in range(0, parent_width, self.layout.build_chunk_rows):
                end = min(start + self.layout.build_chunk_rows, parent_width)
                left_ids = level_start + 2 * np.arange(start, end)
                right_ids = left_ids + 1
                parent_ids = parent_start + np.arange(start, end)
                left_sums = np.asarray(sums[left_ids])
                right_sums = np.asarray(sums[right_ids])
                left_counts = np.asarray(counts[left_ids])
                right_counts = np.asarray(counts[right_ids])
                sums[parent_ids] = left_sums + right_sums
                counts[parent_ids] = left_counts + right_counts
                representatives[parent_ids, 0] = (
                    left_sums / np.maximum(left_counts, 1)[:, None]
                )
                representatives[parent_ids, 1] = (
                    right_sums / np.maximum(right_counts, 1)[:, None]
                )
                rep_counts[parent_ids, 0] = left_counts
                rep_counts[parent_ids, 1] = right_counts
            level_start = parent_start
            level_width = parent_width
        representatives.flush()
        rep_counts.flush()
        sums.flush()
        counts.flush()
        del representatives, rep_counts, sums, counts
        sums_path.unlink()
        counts_path.unlink()
        return {
            "tree_representatives": (
                reps_path,
                np.memmap(
                    reps_path, mode="r", dtype=_DTYPES["float32"],
                    shape=(nodes, 2, self.head_dim),
                ),
            ),
            "tree_rep_counts": (
                rep_counts_path,
                np.memmap(
                    rep_counts_path, mode="r", dtype=_DTYPES["int64"],
                    shape=(nodes, 2),
                ),
            ),
        }, capacity

    def _finalize_head(self, head: int, state: _HeadBuildState) -> dict:
        if state.finalized:
            raise RuntimeError("head was already finalized")
        if state.row_count < 1:
            raise ValueError(f"head {head} has no ingested rows")
        self._close_ingest(state)
        source, tasks = self._build_partition(head, state.row_count)
        row_ids_path = self._array_path(head, "row-ids.i64")
        leaf_counts_path = self._array_path(head, "leaf-counts.i32")
        leaf_reps_path = self._array_path(head, "leaf-representatives.f32")
        leaf_rep_counts_path = self._array_path(head, "leaf-rep-counts.i64")
        leaf_counts = np.empty(len(tasks), dtype=_DTYPES["int32"])
        leaf_representatives = np.zeros(
            (len(tasks), 2, self.head_dim), dtype=_DTYPES["float32"]
        )
        leaf_rep_counts = np.zeros((len(tasks), 2), dtype=_DTYPES["int64"])
        micro_rows = self.layout.logical_leaf_rows // 2
        with row_ids_path.open("wb") as row_stream:
            for leaf, task in enumerate(tasks):
                records = self._terminal_records(source, task)
                records["row_id"].astype(_DTYPES["int64"], copy=False).tofile(row_stream)
                leaf_counts[leaf] = task.count
                first = min(task.count, micro_rows)
                second = task.count - first
                leaf_rep_counts[leaf] = (first, second)
                leaf_representatives[leaf, 0] = records["key"][:first].mean(0)
                if second:
                    leaf_representatives[leaf, 1] = records["key"][first:].mean(0)
            row_stream.flush()
            os.fsync(row_stream.fileno())
        source.unlink()
        _write_array(leaf_counts_path, leaf_counts)
        _write_array(leaf_reps_path, leaf_representatives)
        _write_array(leaf_rep_counts_path, leaf_rep_counts)

        leaf_row_offsets = np.zeros(len(tasks) + 1, dtype=_DTYPES["int64"])
        np.cumsum(leaf_counts, dtype=np.int64, out=leaf_row_offsets[1:])
        leaf_page_ids = np.empty(len(tasks), dtype=_DTYPES["int64"])
        leaf_page_offsets = np.empty(len(tasks), dtype=_DTYPES["int32"])
        page_leaf_ids = []
        page_leaf_offsets = [0]
        page_row_counts = []
        page = 0
        page_rows = 0
        for leaf, count_value in enumerate(leaf_counts):
            count = int(count_value)
            if page_rows and page_rows + count > self.layout.physical_page_rows:
                page_row_counts.append(page_rows)
                page_leaf_offsets.append(len(page_leaf_ids))
                page += 1
                page_rows = 0
            leaf_page_ids[leaf] = page
            leaf_page_offsets[leaf] = page_rows
            page_leaf_ids.append(leaf)
            page_rows += count
        page_row_counts.append(page_rows)
        page_leaf_offsets.append(len(page_leaf_ids))

        generated: dict[str, tuple[Path, np.ndarray]] = {
            "leaf_row_offsets": (
                self._array_path(head, "leaf-row-offsets.i64"), leaf_row_offsets
            ),
            "leaf_page_ids": (
                self._array_path(head, "leaf-page-ids.i64"), leaf_page_ids
            ),
            "leaf_page_offsets": (
                self._array_path(head, "leaf-page-row-offsets.i32"), leaf_page_offsets
            ),
            "page_leaf_offsets": (
                self._array_path(head, "page-leaf-offsets.i64"),
                np.asarray(page_leaf_offsets, dtype=_DTYPES["int64"]),
            ),
            "page_leaf_ids": (
                self._array_path(head, "page-leaf-ids.i64"),
                np.asarray(page_leaf_ids, dtype=_DTYPES["int64"]),
            ),
            "page_row_counts": (
                self._array_path(head, "page-row-counts.i32"),
                np.asarray(page_row_counts, dtype=_DTYPES["int32"]),
            ),
        }
        tree_generated, leaf_capacity = self._build_tree_arrays(
            head, leaf_representatives, leaf_rep_counts
        )
        generated.update(tree_generated)
        arrays = {
            "row_ids": _array_spec(row_ids_path, _DTYPES["int64"], (state.row_count,)),
            "leaf_counts": _array_spec(
                leaf_counts_path, _DTYPES["int32"], (len(tasks),)
            ),
            "leaf_representatives": _array_spec(
                leaf_reps_path,
                _DTYPES["float32"],
                (len(tasks), 2, self.head_dim),
            ),
            "leaf_rep_counts": _array_spec(
                leaf_rep_counts_path, _DTYPES["int64"], (len(tasks), 2)
            ),
        }
        for name, (path, value) in generated.items():
            if name not in tree_generated:
                _write_array(path, value)
            arrays[name] = _array_spec(path, value.dtype, value.shape)
        state.finalized = True
        return {
            "head": head,
            "rows": state.row_count,
            "leaves": len(tasks),
            "leaf_capacity": leaf_capacity,
            "pages": len(page_row_counts),
            "min_row_id": state.first_row_id,
            "max_row_id": state.last_row_id,
            "arrays": arrays,
        }

    def finalize(self) -> "PersistentContentIndex":
        """Build all heads and atomically publish the immutable manifest."""

        if self._closed:
            raise RuntimeError("builder is already closed or finalized")
        heads = [
            self._finalize_head(head, state)
            for head, state in enumerate(self._states)
        ]
        manifest = {
            "format": FORMAT_NAME,
            "schema_version": SCHEMA_VERSION,
            "algorithm_version": ALGORITHM_VERSION,
            "partition": self.partition,
            "partition_semantics": "v3-content-tree-rank-sampled-farthest-pair",
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "row_id_dtype": "int64",
            "layout": {
                field: getattr(self.layout, field)
                for field in self.layout.__dataclass_fields__
            },
            "heads": heads,
        }
        temporary = self.path / f".{MANIFEST_NAME}.tmp"
        temporary.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.path / MANIFEST_NAME)
        self._closed = True
        self._published = True
        return PersistentContentIndex.load(self.path)

    def close(self) -> None:
        if self._closed:
            return
        for state in self._states:
            if not state.stream.closed:
                state.stream.close()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


class PersistentContentIndex:
    """Read-only mmap-backed content-tree index."""

    def __init__(self, path: Path, manifest: dict, arrays: list[dict[str, _OpenArray]]):
        self.path = path
        self.manifest = manifest
        self.num_heads = int(manifest["num_heads"])
        self.head_dim = int(manifest["head_dim"])
        self.layout = PersistentIndexLayout(**manifest["layout"])
        self._arrays = arrays

    @classmethod
    def load(cls, path) -> "PersistentContentIndex":
        root = Path(path)
        manifest_path = root / MANIFEST_NAME
        if not manifest_path.is_file():
            raise ValueError("persistent content index is missing manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != FORMAT_NAME:
            raise ValueError("not a TabICL persistent content index")
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported content-index schema version: "
                f"{manifest.get('schema_version')!r}"
            )
        if manifest.get("algorithm_version") != ALGORITHM_VERSION:
            raise ValueError("unsupported content-index algorithm version")
        if manifest.get("partition") != "content_tree":
            raise ValueError("persistent V3 index is not content-tree compatible")
        num_heads = int(manifest.get("num_heads", 0))
        heads = manifest.get("heads", [])
        if num_heads < 1 or len(heads) != num_heads:
            raise ValueError("persistent content-index head manifest is incomplete")
        arrays = []
        for expected_head, head in enumerate(heads):
            if int(head.get("head", -1)) != expected_head:
                raise ValueError("persistent content-index heads are out of order")
            arrays.append({
                name: _open_array(root, spec)
                for name, spec in head["arrays"].items()
            })
        return cls(root, manifest, arrays)

    def save(self, path) -> "PersistentContentIndex":
        """Copy data files first and atomically publish the copied manifest."""

        destination = Path(path)
        if destination.resolve() == self.path.resolve():
            return self
        if destination.exists():
            if not destination.is_dir() or any(destination.iterdir()):
                raise FileExistsError(
                    f"index destination is not empty: {destination}"
                )
        destination.mkdir(parents=True, exist_ok=True)
        for source in self.path.iterdir():
            if source.is_file() and source.name != MANIFEST_NAME:
                shutil.copyfile(source, destination / source.name)
        temporary = destination / f".{MANIFEST_NAME}.tmp"
        shutil.copyfile(self.path / MANIFEST_NAME, temporary)
        os.replace(temporary, destination / MANIFEST_NAME)
        return self.load(destination)

    def _head(self, head: int) -> tuple[dict, dict[str, _OpenArray]]:
        if not 0 <= head < self.num_heads:
            raise IndexError("head is out of range")
        return self.manifest["heads"][head], self._arrays[head]

    @staticmethod
    def _tensor(value: np.ndarray, device, dtype=None) -> Tensor:
        return torch.as_tensor(np.asarray(value).copy(), device=device, dtype=dtype)

    def _score_representatives(
        self, query: Tensor, representatives: Tensor, counts: Tensor
    ) -> Tensor:
        logits = torch.einsum("qd,lrd->qlr", query.float(), representatives)
        logits = logits / math.sqrt(self.head_dim)
        logits = logits + counts.clamp_min(1).log()[None]
        logits = logits.masked_fill(counts[None] == 0, -float("inf"))
        return torch.logsumexp(logits, dim=-1)

    def _leaf_scores(self, head: int, query: Tensor, leaf_ids: Tensor) -> Tensor:
        _, arrays = self._head(head)
        ids = leaf_ids.detach().to("cpu", torch.int64).numpy()
        reps = self._tensor(
            arrays["leaf_representatives"].array[ids], query.device, torch.float32
        )
        counts = self._tensor(
            arrays["leaf_rep_counts"].array[ids], query.device, torch.float32
        )
        logits = (query[:, None, None, :] * reps).sum(-1) / math.sqrt(self.head_dim)
        logits = logits + counts.clamp_min(1).log()
        logits = logits.masked_fill(counts == 0, -float("inf"))
        return torch.logsumexp(logits, dim=-1)

    def _tree_proposals(
        self, head: int, query: Tensor
    ) -> tuple[Tensor, Tensor, int]:
        metadata, arrays = self._head(head)
        capacity = int(metadata["leaf_capacity"])
        nodes = torch.zeros((query.shape[0], 1), device=query.device, dtype=torch.int64)
        visited = 0
        for _ in range(capacity.bit_length() - 1):
            children = (
                nodes[..., None] * 2
                + 1
                + torch.arange(2, device=query.device)
            ).flatten(-2)
            next_nodes = []
            for row in range(query.shape[0]):
                ids = children[row].to("cpu").numpy()
                reps = self._tensor(
                    arrays["tree_representatives"].array[ids],
                    query.device,
                    torch.float32,
                )
                counts = self._tensor(
                    arrays["tree_rep_counts"].array[ids],
                    query.device,
                    torch.float32,
                )
                scores = self._score_representatives(
                    query[row : row + 1], reps, counts
                )[0]
                keep = min(self.layout.search_beam_leaves, scores.numel())
                next_nodes.append(children[row].gather(0, scores.topk(keep).indices))
            nodes = torch.stack(next_nodes)
            visited += children.shape[-1]
        leaves = nodes - (capacity - 1)
        valid = leaves < int(metadata["leaves"])
        keep = min(self.layout.proposal_leaves, leaves.shape[-1])
        return (
            leaves[:, :keep].masked_fill(~valid[:, :keep], 0),
            valid[:, :keep],
            visited,
        )

    def _reference_leaf_ids_for_head(
        self, head: int, query: Tensor, selected: int, chunk_leaves: int
    ) -> Tensor:
        metadata, arrays = self._head(head)
        best_scores = torch.full(
            (query.shape[0], selected), -float("inf"),
            device=query.device, dtype=torch.float32,
        )
        best_ids = torch.full(
            (query.shape[0], selected), -1,
            device=query.device, dtype=torch.int64,
        )
        for start in range(0, int(metadata["leaves"]), chunk_leaves):
            end = min(start + chunk_leaves, int(metadata["leaves"]))
            reps = self._tensor(
                arrays["leaf_representatives"].array[start:end],
                query.device,
                torch.float32,
            )
            counts = self._tensor(
                arrays["leaf_rep_counts"].array[start:end],
                query.device,
                torch.float32,
            )
            scores = self._score_representatives(query, reps, counts)
            ids = torch.arange(
                start, end, device=query.device, dtype=torch.int64
            )[None].expand(query.shape[0], -1)
            merged_scores = torch.cat((best_scores, scores), -1)
            merged_ids = torch.cat((best_ids, ids), -1)
            top = merged_scores.topk(selected, -1)
            best_scores = top.values
            best_ids = merged_ids.gather(-1, top.indices)
        return best_ids

    def _result_from_leaf_ids(
        self, leaf_ids: Tensor, visited_nodes: Tensor
    ) -> PersistentSearchResult:
        pages = []
        offsets = []
        candidate_rows = []
        for head in range(self.num_heads):
            _, arrays = self._head(head)
            ids = leaf_ids[head].to("cpu").numpy()
            pages.append(self._tensor(
                arrays["leaf_page_ids"].array[ids], leaf_ids.device, torch.int64
            ))
            offsets.append(self._tensor(
                arrays["leaf_page_offsets"].array[ids], leaf_ids.device, torch.int64
            ))
            candidate_rows.append(self._tensor(
                arrays["leaf_counts"].array[ids], leaf_ids.device, torch.int64
            ).sum(-1))
        return PersistentSearchResult(
            logical_leaf_ids=leaf_ids,
            physical_page_ids=torch.stack(pages),
            page_row_offsets=torch.stack(offsets),
            candidate_rows=torch.stack(candidate_rows),
            visited_nodes=visited_nodes,
        )

    @torch.no_grad()
    def search(
        self, query: Tensor, *, selected_leaves: int | None = None
    ) -> PersistentSearchResult:
        if (
            query.ndim != 3
            or query.shape[0] != self.num_heads
            or query.shape[-1] != self.head_dim
        ):
            raise ValueError("query must have shape [num_heads, queries, head_dim]")
        if not bool(torch.isfinite(query).all()):
            raise ValueError("query must be finite")
        selected = (
            self.layout.selected_logical_leaves
            if selected_leaves is None else int(selected_leaves)
        )
        if selected < 1:
            raise ValueError("selected_leaves must be positive")
        selected = min(
            selected, min(int(head["leaves"]) for head in self.manifest["heads"])
        )
        results = []
        visited = torch.empty(
            (self.num_heads, query.shape[1]),
            device=query.device, dtype=torch.int64,
        )
        for head, metadata in enumerate(self.manifest["heads"]):
            if int(metadata["leaves"]) <= self.layout.exhaustive_leaf_limit:
                ids = self._reference_leaf_ids_for_head(
                    head, query[head], selected, self.layout.build_chunk_rows
                )
                visited[head].fill_(int(metadata["leaves"]))
            else:
                proposals, valid, count = self._tree_proposals(head, query[head])
                scores = self._leaf_scores(
                    head, query[head], proposals
                ).masked_fill(~valid, -float("inf"))
                keep = min(selected, proposals.shape[-1])
                ids = proposals.gather(-1, scores.topk(keep, -1).indices)
                if keep < selected:
                    ids = self._reference_leaf_ids_for_head(
                        head, query[head], selected, self.layout.build_chunk_rows
                    )
                visited[head].fill_(count)
            results.append(ids)
        return self._result_from_leaf_ids(torch.stack(results), visited)

    @torch.no_grad()
    def reference_search(
        self,
        query: Tensor,
        *,
        selected_leaves: int | None = None,
        chunk_leaves: int = 65536,
    ) -> PersistentSearchResult:
        if (
            query.ndim != 3
            or query.shape[0] != self.num_heads
            or query.shape[-1] != self.head_dim
        ):
            raise ValueError("query must have shape [num_heads, queries, head_dim]")
        if chunk_leaves < 1:
            raise ValueError("chunk_leaves must be positive")
        selected = (
            self.layout.selected_logical_leaves
            if selected_leaves is None else int(selected_leaves)
        )
        if selected < 1:
            raise ValueError("selected_leaves must be positive")
        selected = min(
            selected, min(int(head["leaves"]) for head in self.manifest["heads"])
        )
        ids = torch.stack([
            self._reference_leaf_ids_for_head(
                head, query[head], selected, chunk_leaves
            )
            for head in range(self.num_heads)
        ])
        visited = torch.empty(
            (self.num_heads, query.shape[1]),
            device=query.device, dtype=torch.int64,
        )
        for head, metadata in enumerate(self.manifest["heads"]):
            visited[head].fill_(int(metadata["leaves"]))
        return self._result_from_leaf_ids(ids, visited)

    def rows_for_leaves(self, head: int, leaf_ids: Tensor) -> tuple[Tensor, Tensor]:
        _, arrays = self._head(head)
        ids = leaf_ids.detach().to("cpu", torch.int64).reshape(-1).numpy()
        offsets = arrays["leaf_row_offsets"].array
        rows = arrays["row_ids"].array
        output = np.full(
            (ids.size, self.layout.logical_leaf_rows), -1,
            dtype=_DTYPES["int64"],
        )
        valid = np.zeros(output.shape, dtype=np.bool_)
        for index, leaf in enumerate(ids):
            start, end = int(offsets[leaf]), int(offsets[leaf + 1])
            count = end - start
            output[index, :count] = rows[start:end]
            valid[index, :count] = True
        shape = (*leaf_ids.shape, self.layout.logical_leaf_rows)
        return (
            torch.as_tensor(output.reshape(shape), device=leaf_ids.device),
            torch.as_tensor(valid.reshape(shape), device=leaf_ids.device),
        )

    def rows_for_pages(self, head: int, page_ids: Tensor) -> tuple[Tensor, Tensor]:
        _, arrays = self._head(head)
        pages = page_ids.detach().to("cpu", torch.int64).reshape(-1).numpy()
        page_offsets = arrays["page_leaf_offsets"].array
        page_leaves = arrays["page_leaf_ids"].array
        leaf_page_offsets = arrays["leaf_page_offsets"].array
        output = np.full(
            (pages.size, self.layout.physical_page_rows), -1,
            dtype=_DTYPES["int64"],
        )
        valid = np.zeros(output.shape, dtype=np.bool_)
        for index, page in enumerate(pages):
            leaves = np.asarray(
                page_leaves[int(page_offsets[page]) : int(page_offsets[page + 1])]
            )
            rows, row_valid = self.rows_for_leaves(
                head, torch.as_tensor(leaves, dtype=torch.int64)
            )
            for local, leaf in enumerate(leaves):
                start = int(leaf_page_offsets[leaf])
                count = int(row_valid[local].sum())
                output[index, start : start + count] = rows[local, :count].numpy()
                valid[index, start : start + count] = True
        shape = (*page_ids.shape, self.layout.physical_page_rows)
        return (
            torch.as_tensor(output.reshape(shape), device=page_ids.device),
            torch.as_tensor(valid.reshape(shape), device=page_ids.device),
        )
