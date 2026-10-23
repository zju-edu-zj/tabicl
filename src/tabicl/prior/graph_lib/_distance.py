"""CPU nearest-center kernels that reduce distances directly to indices."""
import torch


def _nearest_center_1d(x, centers, p):
    """Use sorted neighbors in 1-D; retain the first original center on ties."""
    values, order = centers[:, 0].sort(stable=True)
    positions = torch.searchsorted(values, x[:, 0].contiguous()).clamp_(0, len(values) - 1)
    right = positions
    left = (positions - 1).clamp_min(0)
    left_distance = (x[:, 0] - values[left]).abs()
    right_distance = (x[:, 0] - values[right]).abs()
    choose_left = left_distance <= right_distance
    best = torch.where(choose_left, order[left], order[right])

    minimum = torch.minimum(left_distance, right_distance)
    maximum = torch.maximum(left_distance, right_distance)
    ambiguous = ((positions > 0) & (positions < len(values)) &
                 (maximum <= minimum * (1 + 1e-5) + 1e-12))
    # cdist's pow/sum rounding can break near-ties differently from |x-c|.
    if ambiguous.any():
        rows = ambiguous.nonzero(as_tuple=False).flatten()
        best[rows] = torch.cdist(x[rows], centers, p=p).argmin(dim=-1)
    return best


def nearest_center(x, centers, p):
    # Keep the Euclidean cdist path because its GEMM implementation is efficient.
    if (x.device.type != "cpu" or x.dtype not in (torch.float32, torch.float64)
            or not 0.5 <= p <= 4 or p == 2 or len(x) == 0):
        return torch.cdist(x, centers, p=p).argmin(dim=-1)
    if not torch.isfinite(x).all() or not torch.isfinite(centers).all():
        return torch.cdist(x, centers, p=p).argmin(dim=-1)
    if x.shape[1] == 1:
        if x.abs().max() > 1e8 or centers.abs().max() > 1e8:
            return torch.cdist(x, centers, p=p).argmin(dim=-1)
        return _nearest_center_1d(x, centers, p)
    # cdist's vectorized CPU kernel is materially faster than broadcast/pow
    # reduction for multidimensional points. Chunk only its output matrix.
    rows = max(1, (4 * 1024**2) // (len(centers) * x.element_size()))
    return torch.cat([torch.cdist(chunk, centers, p=p).argmin(dim=-1)
                      for chunk in x.split(rows)])
