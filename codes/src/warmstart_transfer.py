"""Warm-Start & Parameter Transfer Operators for QAOA (Verified Primary-Source Citations).

Implements:
1. `wurtz_love`: Published tree-subgraph fixed angles from:
   - J. Wurtz & P. J. Love, "MaxCut quantum approximate optimization algorithm performance
     with p > 2", Phys. Rev. A 103, 042612 (2021), DOI: 10.1103/PhysRevA.103.042612;
   - J. Wurtz & D. Lykov, "Fixed-angle conjectures for the quantum approximate optimization
     algorithm on regular MaxCut graphs", Phys. Rev. A 104, 052419 (2021), Table I,
     DOI: 10.1103/PhysRevA.104.052419.
2. `shaydulin_rescaled`: Edge-weight RMS rescaling rule
   gamma_weighted = gamma_fixed / sqrt((1/|E|) * sum_{e in E} w_e^2) from:
   - R. Shaydulin, P. C. Lotshaw, J. Larson, J. Ostrowski & T. S. Humble, "Parameter Transfer
     for Quantum Approximate Optimization of Weighted MaxCut", ACM Trans. Quantum Comput.
     4(3), Article 19 (2023), DOI: 10.1145/3584706;
   - S. H. Sureshbabu, D. Herman, R. Shaydulin et al., "Parameter Setting in Quantum
     Approximate Optimization of Weighted Problems", Quantum 8, 1231 (2024),
     DOI: 10.22331/q-2024-01-18-1231;
   - T. Hao, Z. He, R. Shaydulin, J. Larson & M. Pistoia, "End-to-end protocol for
     high-quality quantum approximate optimization algorithm parameters with few shots",
     Phys. Rev. Research 7, 033181 (2025), DOI: 10.1103/24gg-7p8z.
3. `zhou_interp`: Linear interpolation operator (`INTERP`) from L. Zhou, S.-T. Wang,
   S. Choi, H. Pichler & M. D. Lukin, Phys. Rev. X 10, 021067 (2020), Eq. (B1),
   DOI: 10.1103/PhysRevX.10.021067.
4. `layer_padding`: Appends zero-initialized layer p to optimized depth-(p-1) parameters.
"""
from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np

from qaoa_core import get_bounds, get_default_x0


def zhou_interp_from_lower_depth(theta_prev: np.ndarray, p_target: int) -> np.ndarray:
    """Apply Zhou et al. (Phys. Rev. X 10, 021067, 2020, Eq. B1) INTERP operator from p-1 to p."""
    arr = np.asarray(theta_prev, dtype=float).ravel()
    p_prev = len(arr) // 2
    if p_prev == p_target:
        return arr.copy()
    g_prev = arr[:p_prev]
    b_prev = arr[p_prev:]

    while p_prev < p_target:
        p_next = p_prev + 1
        g_next = np.zeros(p_next, dtype=float)
        b_next = np.zeros(p_next, dtype=float)
        for i in range(1, p_next + 1):
            w_left = (i - 1) / float(p_prev)
            w_right = (p_prev - i + 1) / float(p_prev)
            g_l = g_prev[i - 2] if i - 2 >= 0 else 0.0
            g_r = g_prev[i - 1] if i - 1 < p_prev else 0.0
            b_l = b_prev[i - 2] if i - 2 >= 0 else 0.0
            b_r = b_prev[i - 1] if i - 1 < p_prev else 0.0
            g_next[i - 1] = w_left * g_l + w_right * g_r
            b_next[i - 1] = w_left * b_l + w_right * b_r
        g_prev, b_prev = g_next, b_next
        p_prev = p_next

    bounds = get_bounds(p_target)
    return np.clip(np.concatenate([g_prev, b_prev]), bounds[:, 0], bounds[:, 1])


def layer_padding_from_lower_depth(theta_prev: np.ndarray, p_target: int) -> np.ndarray:
    """Pad depth-(p-1) parameters with small non-zero tail angles to reach depth p_target."""
    arr = np.asarray(theta_prev, dtype=float).ravel()
    p_prev = len(arr) // 2
    if p_prev >= p_target:
        return arr[: 2 * p_target].copy()
    pad_len = p_target - p_prev
    g_new = np.concatenate([arr[:p_prev], np.full(pad_len, 0.08)])
    b_new = np.concatenate([arr[p_prev:], np.full(pad_len, 0.08)])
    bounds = get_bounds(p_target)
    return np.clip(np.concatenate([g_new, b_new]), bounds[:, 0], bounds[:, 1])


def compute_initial_parameters(
    p: int,
    init_mode: str = "naive",
    edges: Optional[List[Tuple[int, int, float]]] = None,
    theta_lower_depth: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Unified parameter initialization dispatcher for warm-start & transfer experiments."""
    if init_mode in ("naive", "cold_naive"):
        return get_default_x0(p, "naive", edges)
    elif init_mode in ("wurtz_love", "wurtz_love_fixed"):
        return get_default_x0(p, "wurtz_love", edges)
    elif init_mode in ("shaydulin_rescaled", "shaydulin_weighted"):
        return get_default_x0(p, "shaydulin_rescaled", edges)
    elif init_mode == "zhou_interp":
        base = (
            theta_lower_depth
            if theta_lower_depth is not None
            else get_default_x0(max(1, p - 1), "shaydulin_rescaled", edges)
        )
        return zhou_interp_from_lower_depth(base, p)
    elif init_mode == "layer_padding":
        base = (
            theta_lower_depth
            if theta_lower_depth is not None
            else get_default_x0(max(1, p - 1), "shaydulin_rescaled", edges)
        )
        return layer_padding_from_lower_depth(base, p)
    raise ValueError(f"Unknown init_mode: {init_mode}")
