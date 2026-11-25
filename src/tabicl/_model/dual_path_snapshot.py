"""Versioned support snapshots for Dual-Path one-shot and online runtimes.

The online hidden profile persists one support hidden tensor per layer together
with the layer's global state and content-index tensors. It deliberately has no
K/V page field: sparse K/V is projected from selected hidden rows at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping
import uuid

import torch
from torch import Tensor


SNAPSHOT_SCHEMA_VERSION = 1


class SnapshotProfile(str, Enum):
    """Persistence profiles supported by the V3 runtime."""

    ONE_SHOT = "one_shot"
    ONLINE_HIDDEN = "online_hidden"


class SnapshotState(str, Enum):
    """Snapshot lifecycle states."""

    BUILDING = "building"
    SEALED = "sealed"
    CLOSED = "closed"


@dataclass(frozen=True)
class TensorReference:
    """Serializable reference to one independently stored tensor."""

    tensor_id: str
    role: str
    shape: tuple[int, ...]
    dtype: str
    file: str
    byte_size: int | None = None
    sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tensor_id": self.tensor_id,
            "role": self.role,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "file": self.file,
            "byte_size": self.byte_size,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TensorReference":
        return cls(
            tensor_id=str(value["tensor_id"]),
            role=str(value["role"]),
            shape=tuple(int(item) for item in value["shape"]),
            dtype=str(value["dtype"]),
            file=str(value["file"]),
            byte_size=(
                None if value.get("byte_size") is None else int(value["byte_size"])
            ),
            sha256=(None if value.get("sha256") is None else str(value["sha256"])),
        )


@dataclass(frozen=True)
class LayerSnapshotReference:
    """References needed to execute one support layer."""

    layer_index: int
    hidden: str
    global_state: dict[str, str]
    index_state: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_index": self.layer_index,
            "hidden": self.hidden,
            "global_state": dict(sorted(self.global_state.items())),
            "index_state": dict(sorted(self.index_state.items())),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LayerSnapshotReference":
        return cls(
            layer_index=int(value["layer_index"]),
            hidden=str(value["hidden"]),
            global_state={str(k): str(v) for k, v in value["global_state"].items()},
            index_state={str(k): str(v) for k, v in value["index_state"].items()},
        )


@dataclass
class SupportSnapshotManifest:
    """Versioned manifest describing an immutable or resumable support state."""

    snapshot_id: str
    generation: int
    profile: SnapshotProfile
    state: SnapshotState
    support_size: int
    model_dim: int
    batch_size: int
    expected_layers: int
    model_fingerprint: str
    layout: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    tensors: dict[str, TensorReference] = field(default_factory=dict)
    layers: list[LayerSnapshotReference] = field(default_factory=list)
    schema_version: int = SNAPSHOT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "snapshot_id": self.snapshot_id,
            "generation": self.generation,
            "profile": self.profile.value,
            "state": self.state.value,
            "support_size": self.support_size,
            "model_dim": self.model_dim,
            "batch_size": self.batch_size,
            "expected_layers": self.expected_layers,
            "model_fingerprint": self.model_fingerprint,
            "layout": self.layout,
            "metadata": self.metadata,
            "tensors": {
                key: value.to_dict() for key, value in sorted(self.tensors.items())
            },
            "layers": [layer.to_dict() for layer in self.layers],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SupportSnapshotManifest":
        version = int(value["schema_version"])
        if version > SNAPSHOT_SCHEMA_VERSION:
            raise ValueError(
                f"Snapshot schema {version} is newer than supported schema "
                f"{SNAPSHOT_SCHEMA_VERSION}"
            )
        if version < 1:
            raise ValueError(f"Unsupported snapshot schema version: {version}")
        return cls(
            schema_version=version,
            snapshot_id=str(value["snapshot_id"]),
            generation=int(value["generation"]),
            profile=SnapshotProfile(value["profile"]),
            state=SnapshotState(value["state"]),
            support_size=int(value["support_size"]),
            model_dim=int(value["model_dim"]),
            batch_size=int(value.get("batch_size", 1)),
            expected_layers=int(value["expected_layers"]),
            model_fingerprint=str(value["model_fingerprint"]),
            layout=dict(value["layout"]),
            metadata=dict(value.get("metadata", {})),
            tensors={
                str(key): TensorReference.from_dict(ref)
                for key, ref in value.get("tensors", {}).items()
            },
            layers=[
                LayerSnapshotReference.from_dict(layer)
                for layer in value.get("layers", [])
            ],
        )

    def validate(self, *, require_integrity: bool = False) -> None:
        if self.schema_version != SNAPSHOT_SCHEMA_VERSION:
            raise ValueError(
                f"Snapshot schema {self.schema_version} must be migrated before use"
            )
        if self.generation < 0:
            raise ValueError("snapshot generation must be non-negative")
        if self.support_size < 1 or self.model_dim < 1:
            raise ValueError("support_size and model_dim must be positive")
        if self.batch_size != 1:
            raise ValueError(
                "V3 support snapshots currently require batch_size=1; "
                "store each dataset as an independent snapshot"
            )
        if self.expected_layers < 1:
            raise ValueError("expected_layers must be positive")
        json.dumps(self.layout)
        json.dumps(self.metadata)

        if self.profile == SnapshotProfile.ONE_SHOT and len(self.layers) > 1:
            raise ValueError("one-shot snapshots may retain only one active layer")
        if self.profile == SnapshotProfile.ONLINE_HIDDEN:
            actual = [layer.layer_index for layer in self.layers]
            if actual != list(range(len(actual))):
                raise ValueError("online hidden layers must be added sequentially from zero")
            if self.state == SnapshotState.SEALED and len(actual) != self.expected_layers:
                raise ValueError("sealed online hidden snapshots must contain every layer")

        referenced: set[str] = set()
        for layer in self.layers:
            if not 0 <= layer.layer_index < self.expected_layers:
                raise ValueError(f"invalid layer index: {layer.layer_index}")
            referenced.add(layer.hidden)
            referenced.update(layer.global_state.values())
            referenced.update(layer.index_state.values())
            hidden = self.tensors.get(layer.hidden)
            if hidden is None:
                raise ValueError(f"missing hidden tensor reference: {layer.hidden}")
            if hidden.role != "hidden":
                raise ValueError("layer hidden reference must have role='hidden'")
            if hidden.shape != (self.support_size, self.model_dim):
                raise ValueError(
                    f"layer {layer.layer_index} hidden has shape {hidden.shape}; expected "
                    f"({self.support_size}, {self.model_dim})"
                )

        missing = referenced.difference(self.tensors)
        if missing:
            raise ValueError(f"manifest references missing tensors: {sorted(missing)}")
        orphaned = set(self.tensors).difference(referenced)
        if orphaned:
            raise ValueError(f"manifest contains orphaned tensors: {sorted(orphaned)}")
        for ref in self.tensors.values():
            if ref.role != "hidden" and not ref.role.startswith(("global:", "index:")):
                raise ValueError(f"unsupported tensor role in hidden snapshot: {ref.role}")
            if require_integrity:
                if ref.byte_size is None or ref.byte_size < 1:
                    raise ValueError(f"tensor file size is missing: {ref.tensor_id}")
                if ref.sha256 is None or len(ref.sha256) != 64:
                    raise ValueError(f"tensor SHA-256 is missing: {ref.tensor_id}")


class SupportSnapshot:
    """Own tensor references and lifecycle for one support-set generation.

    Building snapshots retain caller-provided tensors without cloning. Serialized
    tensors are independent files so loading can stay lazy and mmap-backed.
    """

    MANIFEST_FILE = "manifest.json"

    def __init__(
        self,
        manifest: SupportSnapshotManifest,
        assets: Mapping[str, Tensor | Path] | None = None,
        *,
        root: Path | None = None,
        map_location: str | torch.device = "cpu",
        mmap: bool = True,
    ):
        manifest.validate()
        self.manifest = manifest
        self._assets: dict[str, Tensor | Path] = dict(assets or {})
        self._loaded: dict[str, Tensor] = {}
        self._root = root
        self._map_location = map_location
        self._mmap = mmap
        self._stale_files: list[Path] = []

    @classmethod
    def create(
        cls,
        *,
        profile: SnapshotProfile | str,
        support_size: int,
        model_dim: int,
        expected_layers: int,
        model_fingerprint: str,
        layout: Mapping[str, Any],
        batch_size: int = 1,
        generation: int = 0,
        snapshot_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        directory: str | os.PathLike[str] | None = None,
    ) -> "SupportSnapshot":
        manifest = SupportSnapshotManifest(
            snapshot_id=snapshot_id or uuid.uuid4().hex,
            generation=generation,
            profile=SnapshotProfile(profile),
            state=SnapshotState.BUILDING,
            support_size=support_size,
            model_dim=model_dim,
            batch_size=batch_size,
            expected_layers=expected_layers,
            model_fingerprint=model_fingerprint,
            layout=dict(layout),
            metadata=dict(metadata or {}),
        )
        if directory is None:
            return cls(manifest)
        root = Path(directory)
        if root.exists():
            raise FileExistsError(f"snapshot directory already exists: {root}")
        root.mkdir(parents=True)
        snapshot = cls(manifest, root=root)
        snapshot._write_manifest()
        return snapshot

    @property
    def state(self) -> SnapshotState:
        return self.manifest.state

    @property
    def profile(self) -> SnapshotProfile:
        return self.manifest.profile

    def _require_open(self) -> None:
        if self.state == SnapshotState.CLOSED:
            raise RuntimeError("snapshot is closed")

    def _require_building(self) -> None:
        self._require_open()
        if self.state != SnapshotState.BUILDING:
            raise RuntimeError("snapshot is sealed and cannot be modified")

    @staticmethod
    def _safe_name(name: str) -> str:
        safe = "".join(char if char.isalnum() or char in "_-" else "_" for char in name)
        if not safe:
            raise ValueError("tensor names must contain at least one safe character")
        return safe

    def _register_tensor(
        self, layer_index: int, category: str, name: str, tensor: Tensor
    ) -> str:
        if not isinstance(tensor, Tensor):
            raise TypeError(f"{category}:{name} must be a torch.Tensor")
        safe_name = self._safe_name(name)
        tensor_id = f"layer.{layer_index:03d}.{category}.{safe_name}"
        if tensor_id in self.manifest.tensors:
            raise ValueError(f"duplicate tensor reference: {tensor_id}")
        role = "hidden" if category == "hidden" else f"{category}:{name}"
        ref = TensorReference(
            tensor_id=tensor_id,
            role=role,
            shape=tuple(tensor.shape),
            dtype=str(tensor.dtype),
            file=f"tensors/layer_{layer_index:03d}/{category}_{safe_name}.pt",
        )
        self.manifest.tensors[tensor_id] = ref
        if self._root is None:
            self._assets[tensor_id] = tensor.detach()
        else:
            destination = self._root / ref.file
            ref = self._write_tensor_file(tensor, destination, ref)
            self.manifest.tensors[tensor_id] = ref
            self._assets[tensor_id] = destination
        return tensor_id

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def _write_tensor_file(
        cls, tensor: Tensor, destination: Path, reference: TensorReference
    ) -> TensorReference:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(
            f".{destination.name}.tmp-{uuid.uuid4().hex}"
        )
        try:
            torch.save(tensor.detach().to("cpu").contiguous(), temporary)
            byte_size = temporary.stat().st_size
            sha256 = cls._sha256_file(temporary)
            os.replace(temporary, destination)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        return replace(reference, byte_size=byte_size, sha256=sha256)

    def _drop_layer(self, layer: LayerSnapshotReference) -> None:
        tensor_ids = {
            layer.hidden,
            *layer.global_state.values(),
            *layer.index_state.values(),
        }
        for tensor_id in tensor_ids:
            self.manifest.tensors.pop(tensor_id, None)
            asset = self._assets.pop(tensor_id, None)
            if isinstance(asset, Path):
                self._stale_files.append(asset)
            self._loaded.pop(tensor_id, None)

    def _write_manifest(self) -> None:
        if self._root is None:
            return
        self.manifest.validate(require_integrity=True)
        manifest_path = self._root / self.MANIFEST_FILE
        temporary = manifest_path.with_name(f".{manifest_path.name}.tmp-{uuid.uuid4().hex}")
        temporary.write_text(
            json.dumps(self.manifest.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, manifest_path)
        for stale in self._stale_files:
            try:
                stale.unlink()
            except FileNotFoundError:
                pass
        self._stale_files.clear()

    def add_layer(
        self,
        layer_index: int,
        hidden: Tensor,
        *,
        global_state: Mapping[str, Tensor],
        index_state: Mapping[str, Tensor],
    ) -> LayerSnapshotReference:
        """Add an online layer or advance the active one-shot layer.

        Online hidden snapshots append all layers. One-shot snapshots evict the
        previous active layer before registering the next layer.
        """

        self._require_building()
        if tuple(hidden.shape) != (self.manifest.support_size, self.manifest.model_dim):
            raise ValueError(
                f"hidden must have shape ({self.manifest.support_size}, "
                f"{self.manifest.model_dim})"
            )
        if not 0 <= layer_index < self.manifest.expected_layers:
            raise ValueError(f"invalid layer index: {layer_index}")

        if self.profile == SnapshotProfile.ONLINE_HIDDEN:
            if layer_index != len(self.manifest.layers):
                raise ValueError("online hidden layers must be added sequentially")
        elif self.manifest.layers:
            previous = self.manifest.layers[0]
            if layer_index <= previous.layer_index:
                raise ValueError("one-shot layer index must advance monotonically")
            self._drop_layer(previous)
            self.manifest.layers.clear()

        hidden_ref = self._register_tensor(layer_index, "hidden", "support", hidden)
        global_refs = {
            name: self._register_tensor(layer_index, "global", name, tensor)
            for name, tensor in sorted(global_state.items())
        }
        index_refs = {
            name: self._register_tensor(layer_index, "index", name, tensor)
            for name, tensor in sorted(index_state.items())
        }
        layer = LayerSnapshotReference(
            layer_index=layer_index,
            hidden=hidden_ref,
            global_state=global_refs,
            index_state=index_refs,
        )
        self.manifest.layers.append(layer)
        self.manifest.validate()
        self._write_manifest()
        return layer

    def seal(self) -> None:
        self._require_building()
        if not self.manifest.layers:
            raise ValueError("cannot seal an empty snapshot")
        self.manifest.state = SnapshotState.SEALED
        self.manifest.validate()
        self._write_manifest()

    def layer(self, layer_index: int) -> LayerSnapshotReference:
        self._require_open()
        for layer in self.manifest.layers:
            if layer.layer_index == layer_index:
                return layer
        raise KeyError(f"layer {layer_index} is not present in the snapshot")

    @property
    def active_layer(self) -> LayerSnapshotReference:
        if self.profile != SnapshotProfile.ONE_SHOT:
            raise RuntimeError("active_layer is defined only for one-shot snapshots")
        if not self.manifest.layers:
            raise RuntimeError("one-shot snapshot has no active layer")
        return self.manifest.layers[0]

    def tensor(self, reference: str | TensorReference) -> Tensor:
        self._require_open()
        tensor_id = reference.tensor_id if isinstance(reference, TensorReference) else reference
        if tensor_id not in self.manifest.tensors:
            raise KeyError(f"unknown tensor reference: {tensor_id}")
        asset = self._assets.get(tensor_id)
        if isinstance(asset, Tensor):
            return asset
        if tensor_id in self._loaded:
            return self._loaded[tensor_id]
        if not isinstance(asset, Path):
            raise RuntimeError(f"tensor asset is unavailable: {tensor_id}")
        kwargs: dict[str, Any] = {
            "map_location": self._map_location,
            "weights_only": True,
        }
        if self._mmap:
            kwargs["mmap"] = True
        value = torch.load(asset, **kwargs)
        if not isinstance(value, Tensor):
            raise TypeError(f"snapshot asset {asset} did not contain a tensor")
        ref = self.manifest.tensors[tensor_id]
        if tuple(value.shape) != ref.shape or str(value.dtype) != ref.dtype:
            raise ValueError(f"snapshot tensor metadata mismatch: {tensor_id}")
        self._loaded[tensor_id] = value
        return value

    def release_tensor(self, reference: str | TensorReference) -> None:
        """Release a lazily loaded tensor while retaining its file reference."""

        tensor_id = reference.tensor_id if isinstance(reference, TensorReference) else reference
        if isinstance(self._assets.get(tensor_id), Path):
            self._loaded.pop(tensor_id, None)

    def is_mmap_backed(self, reference: str | TensorReference) -> bool:
        """Return whether a lazy CPU tensor uses explicit ``torch.load(mmap=True)``."""

        tensor_id = reference.tensor_id if isinstance(reference, TensorReference) else reference
        asset = self._assets.get(tensor_id)
        return (
            self._mmap
            and isinstance(asset, Path)
            and torch.device(self._map_location).type == "cpu"
        )

    def save(self, directory: str | os.PathLike[str]) -> Path:
        """Atomically write the manifest and independently loadable tensors."""

        self._require_open()
        self.manifest.validate()
        if self.profile == SnapshotProfile.ONLINE_HIDDEN and self.state != SnapshotState.SEALED:
            raise RuntimeError("online hidden snapshots must be sealed before serialization")

        target = Path(directory)
        if self._root is not None and target.resolve() == self._root.resolve():
            self._write_manifest()
            return target
        if target.exists():
            raise FileExistsError(f"snapshot directory already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.parent / f".{target.name}.tmp-{uuid.uuid4().hex}"
        temporary.mkdir(parents=False)
        try:
            for tensor_id, ref in sorted(self.manifest.tensors.items()):
                destination = temporary / ref.file
                tensor = self.tensor(tensor_id).detach().to("cpu").contiguous()
                updated = self._write_tensor_file(tensor, destination, ref)
                self.manifest.tensors[tensor_id] = updated
            self.manifest.validate(require_integrity=True)
            manifest_path = temporary / self.MANIFEST_FILE
            manifest_path.write_text(
                json.dumps(self.manifest.to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, target)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return target

    @classmethod
    def load(
        cls,
        directory: str | os.PathLike[str],
        *,
        map_location: str | torch.device = "cpu",
        lazy: bool = True,
        mmap: bool = True,
        strict: bool = False,
    ) -> "SupportSnapshot":
        root = Path(directory)
        manifest_path = root / cls.MANIFEST_FILE
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = SupportSnapshotManifest.from_dict(value)
        manifest.validate(require_integrity=True)
        assets: dict[str, Path] = {}
        for tensor_id, ref in manifest.tensors.items():
            path = root / ref.file
            if not path.is_file():
                raise FileNotFoundError(f"snapshot tensor is missing: {path}")
            if strict:
                if path.stat().st_size != ref.byte_size:
                    raise ValueError(f"snapshot tensor byte size mismatch: {tensor_id}")
                if cls._sha256_file(path) != ref.sha256:
                    raise ValueError(f"snapshot tensor SHA-256 mismatch: {tensor_id}")
            assets[tensor_id] = path
        snapshot = cls(
            manifest,
            assets,
            root=root,
            map_location=map_location,
            mmap=mmap,
        )
        if not lazy:
            for tensor_id in manifest.tensors:
                snapshot.tensor(tensor_id)
        return snapshot

    def close(self) -> None:
        if self.state == SnapshotState.CLOSED:
            return
        self._loaded.clear()
        self._assets.clear()
        self.manifest.state = SnapshotState.CLOSED

    def __enter__(self) -> "SupportSnapshot":
        self._require_open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
