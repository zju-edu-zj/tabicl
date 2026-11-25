"""Physical page stores for Dual-Path V3 inference."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from .content_index_v3 import ContentIndexV3, DualPathLayout


@dataclass
class CandidateTensors:
    key: Tensor
    value: Tensor
    valid: Tensor


@dataclass
class PreparedDualPathSupport:
    index: ContentIndexV3
    page_store: "DualPathPageStore"
    global_state: dict
    support_size: int


class DualPathPageStore:
    """Read-only K/V pages with logical-leaf addressing."""

    def __init__(self, key: Tensor, value: Tensor, support_size: int, layout: DualPathLayout):
        if key.shape != value.shape or key.ndim != 5:
            raise ValueError("page K/V must have shape [B,H,P,128,D]")
        if key.shape[-2] != layout.storage_page_rows:
            raise ValueError("physical page rows do not match the layout")
        self.key = key
        self.value = value
        self.support_size = support_size
        self.layout = layout

    @property
    def device(self):
        return self.key.device

    @property
    def dtype(self):
        return self.key.dtype

    @classmethod
    @torch.no_grad()
    def pack(
        cls,
        key: Tensor,
        value: Tensor,
        index: ContentIndexV3,
        location: str = "device",
    ):
        if location not in ("device", "host"):
            raise ValueError("page store location must be 'device' or 'host'")
        b, h, support, d = key.shape
        page_rows = index.layout.storage_page_rows
        pages = (support + page_rows - 1) // page_rows
        gather = index.order.long()[..., None].expand(b, h, support, d)
        sorted_key = key.gather(2, gather)
        sorted_value = value.gather(2, gather)
        pad = pages * page_rows - support
        sorted_key = F.pad(sorted_key, (0, 0, 0, pad)).reshape(b, h, pages, page_rows, d)
        sorted_value = F.pad(sorted_value, (0, 0, 0, pad)).reshape(b, h, pages, page_rows, d)
        if location == "host":
            sorted_key = sorted_key.to("cpu", non_blocking=False)
            sorted_value = sorted_value.to("cpu", non_blocking=False)
            if torch.cuda.is_available():
                sorted_key = sorted_key.pin_memory()
                sorted_value = sorted_value.pin_memory()
        return cls(sorted_key, sorted_value, support, index.layout)

    def flattened(self) -> tuple[Tensor, Tensor]:
        return self.key.flatten(2, 3), self.value.flatten(2, 3)

    def candidate_tensors(self, logical_leaf_ids: Tensor, device) -> CandidateTensors:
        """Fetch selected leaves while masking unselected rows in fetched pages."""
        b, h, q, selected = logical_leaf_ids.shape
        d = self.key.shape[-1]
        leaf_ids = logical_leaf_ids.to(device=device, dtype=torch.long)
        page_ids = torch.div(
            leaf_ids, self.layout.leaves_per_page, rounding_mode="floor"
        )
        slots = leaf_ids % self.layout.leaves_per_page
        rows = slots[..., None] * self.layout.logical_leaf_rows + torch.arange(
            self.layout.logical_leaf_rows, device=device
        )

        # Query tiles keep this gather bounded. Host-backed pages are copied only
        # after page IDs are known; resident pages use the same semantics.
        pages = self.key.shape[2]
        flat_page = page_ids.clamp(0, pages - 1)
        batch = torch.arange(b, device=device)[:, None, None, None].expand_as(flat_page)
        head = torch.arange(h, device=device)[None, :, None, None].expand_as(flat_page)
        address = ((batch * h + head) * pages + flat_page).reshape(-1)
        unique, inverse = torch.unique(address, sorted=True, return_inverse=True)
        unique_batch = torch.div(unique, h * pages, rounding_mode="floor")
        remainder = unique % (h * pages)
        unique_head = torch.div(remainder, pages, rounding_mode="floor")
        unique_page = remainder % pages
        if self.key.device.type == "cpu" and device.type != "cpu":
            host_index = (unique_batch.cpu(), unique_head.cpu(), unique_page.cpu())
            unique_key = self.key[host_index].to(device=device, non_blocking=True)
            unique_value = self.value[host_index].to(device=device, non_blocking=True)
        else:
            unique_key = self.key[unique_batch, unique_head, unique_page].to(device=device)
            unique_value = self.value[unique_batch, unique_head, unique_page].to(device=device)
        page_rows = self.layout.storage_page_rows
        flat_positions = (
            inverse.reshape(b, h, q, selected, 1) * page_rows + rows
        ).reshape(-1)
        key = unique_key.flatten(0, 1).index_select(0, flat_positions).reshape(
            b, h, q, selected * self.layout.logical_leaf_rows, d
        )
        value = unique_value.flatten(0, 1).index_select(0, flat_positions).reshape_as(key)
        sorted_position = (
            leaf_ids[..., None] * self.layout.logical_leaf_rows
            + torch.arange(self.layout.logical_leaf_rows, device=device)
        )
        valid = (sorted_position < self.support_size).flatten(-2)
        return CandidateTensors(key=key, value=value, valid=valid)

    def size_bytes(self) -> int:
        return (
            self.key.numel() * self.key.element_size()
            + self.value.numel() * self.value.element_size()
        )
