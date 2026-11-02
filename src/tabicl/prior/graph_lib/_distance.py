"""Distance execution preserving the selected prior compatibility contract.

Only CPU non-Euclidean, finite inputs are compacted. In that kernel each
point pair is independent of the number/order of other rows. In particular,
do NOT use this transformation on the size-dependent p=2 GEMM path.
"""
import os
import torch

from ._cpu_budget import parallel_distance


_native_loaded = False


def _nearest_center_1d(x, centers, p):
    """Historical sorted-neighbor behavior, retained for prior compatibility.

    This is intentionally not advertised as equivalent to cdist's argmin.
    In particular, duplicate centers and underflow can give different indices.
    """
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
    if ambiguous.any():
        rows = ambiguous.nonzero(as_tuple=False).flatten()
        best[rows] = torch.cdist(x[rows], centers, p=p).argmin(dim=-1)
    return best


def _native_available():
    global _native_loaded
    library = os.environ.get("TABICL_PRIOR_DISTANCE_LIBRARY")
    if not library:
        return False
    if not _native_loaded:
        torch.ops.load_library(library)
        _native_loaded = True
    return True


def _compact_rows(x, comparisons):
    """Evaluate repeated rows once; use a deterministic, RNG-free cost gate."""
    if len(x) < 128 or comparisons < 8 or x.shape[1] == 0:
        return x, None
    # Scalar torch.unique is cheaper than hash-map allocation; native hashing
    # wins for full multidimensional rows, including the continuous-row probe.
    if (x.shape[1] > 1 and os.environ.get('TABICL_PRIOR_COMPACT', 'hash') == 'hash'
            and _native_available() and hasattr(torch.ops.tabicl_prior_exact, 'compact_rows_cpu')):
        unique, inverse = torch.ops.tabicl_prior_exact.compact_rows_cpu(x)
        return unique, inverse if inverse.numel() else None
    if x.shape[1] == 1:
        unique, inverse = torch.unique(x[:, 0], sorted=True, return_inverse=True)
        unique = unique[:, None]
    else:
        # Equal full rows must be equal in these columns. False positives in
        # this cheap probe only cost a full unique(); they cannot affect data.
        probe = x[:64, :min(4, x.shape[1])]
        if len(torch.unique(probe, dim=0)) > 56:
            return x, None
        unique, inverse = torch.unique(x, dim=0, sorted=True, return_inverse=True)
    if len(unique) * 8 > len(x) * 7:
        return x, None
    return unique, inverse


def compact_cdist(x, centers, p, *, vectorized=False):
    """Full cdist matrix, preserving EM's distances and subsequent shapes."""
    if (x.ndim != 2 or centers.ndim != 2 or x.device.type != "cpu"
            or x.dtype not in (torch.float32, torch.float64) or p == 2
            or not 0.5 <= p <= 4 or not torch.isfinite(x).all()
            or not torch.isfinite(centers).all()):
        return torch.cdist(x, centers, p=p)
    compact, inverse = _compact_rows(x, len(centers))
    with parallel_distance(compact, centers, p):
        if vectorized and _native_available() and hasattr(torch.ops.tabicl_prior_exact, 'cdist_vector_cpu'):
            distances = torch.ops.tabicl_prior_exact.cdist_vector_cpu(compact, centers, p)
        else:
            distances = torch.cdist(compact, centers, p=p)
    return distances if inverse is None else distances[inverse]


def compact_em_probabilities(x, centers, stds, constants, p, q, inverse_std_q):
    """Complete EM posterior on distinct rows; restore AFTER the linear map.

    p=2 retains the historical size-dependent GEMM dispatch. Unsupported native
    libraries retain the existing distance/softmax calculation on compact rows.
    Return inverse separately so downstream output work is compacted as well.
    """
    eligible = (x.ndim == 2 and centers.ndim == 2 and x.device.type == 'cpu'
        and x.dtype in (torch.float32, torch.float64) and p != 2 and .5 <= p <= 4
        and torch.isfinite(x).all() and torch.isfinite(centers).all()
        and torch.isfinite(inverse_std_q).all() and torch.isfinite(constants).all())
    compact, inverse = _compact_rows(x, len(centers)) if eligible else (x, None)
    available = eligible and _native_available()
    # The existing kernel vectors over centers, the fused one over rows.
    # Wide, nonrepeated inputs with no scalar center tail favor the former.
    # Both evaluate the same complete metric; this gate changes execution only.
    lanes = 8 if x.dtype == torch.float32 else 4
    prefer_distances = (inverse is None and x.shape[1] >= 32
                        and len(centers) % lanes == 0 and p != q)
    matching_parameters = all(t.dtype == x.dtype and t.device == x.device
                              for t in (stds, inverse_std_q, constants))
    if (available and matching_parameters and not prefer_distances
            and hasattr(torch.ops.tabicl_prior_exact, 'em_probabilities_cpu')):
        with parallel_distance(compact, centers, p):
            result = torch.ops.tabicl_prior_exact.em_probabilities_cpu(compact, centers, inverse_std_q, constants, p, q)
    else:
        with parallel_distance(compact, centers, p):
            if available and hasattr(torch.ops.tabicl_prior_exact, 'cdist_vector_cpu'):
                dists = torch.ops.tabicl_prior_exact.cdist_vector_cpu(compact, centers, p)
            else:
                dists = torch.cdist(compact, centers, p=p)
        result = torch.softmax(constants - torch.clamp(dists / stds[None, :], min=0.) ** q, dim=-1)
    return result, inverse


def nearest_center(x, centers, p, *, vectorized=False):
    """Accelerate distances while making 1-D compatibility explicit.

    Default ``TABICL_PRIOR_1D_MODE=legacy`` retains the current prior's
    sorted-neighbor behavior. ``reference`` instead retains original
    cdist(...).argmin(-1), including floating-point and first-index ties.
    Multidimensional dispatch is identical in both modes.
    """
    if (x.ndim != 2 or centers.ndim != 2 or x.device.type != "cpu"
            or x.dtype not in (torch.float32, torch.float64)
            or not 0.5 <= p <= 4 or p == 2 or len(x) == 0
            or len(centers) == 0):
        return torch.cdist(x, centers, p=p).argmin(dim=-1)
    if not torch.isfinite(x).all() or not torch.isfinite(centers).all():
        with parallel_distance(x, centers, p):
            return torch.cdist(x, centers, p=p).argmin(dim=-1)
    if x.shape[1] == 1:
        mode = os.environ.get("TABICL_PRIOR_1D_MODE", "legacy")
        if mode not in ("legacy", "reference"):
            raise ValueError("TABICL_PRIOR_1D_MODE must be legacy or reference")
        if mode == "legacy":
            if x.abs().max() > 1e8 or centers.abs().max() > 1e8:
                with parallel_distance(x, centers, p):
                    return torch.cdist(x, centers, p=p).argmin(dim=-1)
            return _nearest_center_1d(x, centers, p)
    compact, inverse = _compact_rows(x, len(centers))
    # Keep original center order; otherwise all floating-point tie cases,
    # not merely duplicate-center ties, would need special handling.
    rows = max(1, (4 * 1024**2) // (len(centers) * x.element_size()))
    with parallel_distance(compact, centers, p):
        if vectorized and _native_available() and hasattr(torch.ops.tabicl_prior_exact, 'nearest_vector_cpu'):
            result = torch.ops.tabicl_prior_exact.nearest_vector_cpu(compact, centers, p)
        elif _native_available():
            result = torch.ops.tabicl_prior_exact.nearest_cpu(compact, centers, p)
        else:
            result = torch.cat([torch.cdist(chunk, centers, p=p).argmin(dim=-1)
                                for chunk in compact.split(rows)])
    return result if inverse is None else result[inverse]
