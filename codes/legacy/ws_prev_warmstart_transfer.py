"""Parameter-transfer and multi-layer QAOA initialization schemes (Workstream 7).

Implements:
1. `wurtz_love_fixed`: Published tree-subgraph fixed angles from Wurtz & Love, Phys. Rev. A 104, 052419 (2021).
2. `shaydulin_rescaled`: Weighted-MaxCut parameter transfer with inverse RMS edge-weight scaling
   gamma_weighted = gamma_fixed / sqrt((1/|E|) * sum_{e in E} w_e^2) from Shaydulin et al.,
   Quantum 8, 1231 (2024) and Hao et al., Phys. Rev. Research 7 (2025) Eq. (16).
3. `zhou_interp`: Linear interpolation operator (`INTERP`) from Zhou, Wang, Choi, Pichler & Lukin,
   Phys. Rev. X 10, 021067 (2020) Eq. (B1), mapping optimized depth-(p-1) angles to depth p:
     gamma_i^[p] = ((i - 1) / p) * gamma_{i-1}^[p-1] + ((p - i + 1) / p) * gamma_i^[p-1]
   for i = 1..p (with boundary convention gamma_0^[p-1] = gamma_p^[p-1] = 0).
4. `layer_padding`: Appends a small near-identity layer (gamma_p = 0.05, beta_p = 0.05) to
   optimized depth-(p-1) parameters.
5. `cold_naive`: Uninformed baseline initialization [0.30, ..., 0.50, ...].
"""
from __future__ import annotations
import math
from typing import List, Optional, Tuple
import numpy as np

from qaoa_core import get_bounds, get_default_x0


def zhou_interp_vector(theta_prev: np.ndarray) -> np.ndarray:
    """Apply Zhou et al. (Phys. Rev. X 10, 021067, 2020, Eq. B1) INTERP operator from p-1 to p."""
    t_prev = np.asarray(theta_prev, dtype=float).ravel()
    p_prev = len(t_prev) // 2
    p_new = p_prev + 1
    g_prev = t_prev[:p_prev]
    b_prev = t_prev[p_prev:]

    g_new = np.zeros(p_new, dtype=float)
    b_new = np.zeros(p_new, dtype=float)
    for i in range(1, p_new + 1):
        # 1-indexed i in {1, ..., p_new}, p_prev = p_new - 1
        w_left = (i - 1.0) / p_new
        w_right = (p_new - i + 1.0) / p_new
        g_left = g_prev[i - 2] if (i - 2) >= 0 else 0.0
        g_right = g_prev[i - 1] if (i - 1) < p_prev else 0.0
        b_left = b_prev[i - 2] if (i - 2) >= 0 else 0.0
        b_right = b_prev[i - 1] if (i - 1) < p_prev else 0.0
        g_new[i - 1] = w_left * g_left + w_right * g_right
        b_new[i - 1] = w_left * b_left + w_right * b_right

    bounds = get_bounds(p_new)
    return np.clip(np.concatenate([g_new, b_new]), bounds[:, 0], bounds[:, 1])


def layer_padding_vector(theta_prev: np.ndarray, pad_gamma: float = 0.05, pad_beta: float = 0.05) -> np.ndarray:
    """Pad an optimized depth-(p-1) parameter vector with a small layer p initialization."""
    t_prev = np.asarray(theta_prev, dtype=float).ravel()
    p_prev = len(t_prev) // 2
    p_new = p_prev + 1
    g_new = np.concatenate([t_prev[:p_prev], [pad_gamma]])
    b_new = np.concatenate([t_prev[p_prev:], [pad_beta]])
    bounds = get_bounds(p_new)
    return np.clip(np.concatenate([g_new, b_new]), bounds[:, 0], bounds[:, 1])


def compute_initial_parameters(
    p: int,
    strategy: str,
    edges: Optional[List[Tuple[int, int, float]]] = None,
    theta_lower_depth: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute initial parameter vector for depth p under the specified transfer strategy."""
    if strategy == "custom":
        if theta_lower_depth is not None:
            return np.asarray(theta_lower_depth, dtype=float).ravel().copy()
        return get_default_x0(p, "shaydulin_rescaled", edges)
        return np.asarray(theta_lower_depth, dtype=float).ravel().copy()
    if strategy in ("naive", "cold_naive"):
        return get_default_x0(p, "naive", edges)
    elif strategy in ("wurtz_love", "wurtz_love_fixed"):
        return get_default_x0(p, "wurtz_love", edges)
    elif strategy == "shaydulin_rescaled":
        return get_default_x0(p, "shaydulin_rescaled", edges)
    elif strategy == "zhou_interp":
        if theta_lower_depth is not None:
            t_cur = np.asarray(theta_lower_depth, dtype=float).ravel()
            while len(t_cur) // 2 < p:
                t_cur = zhou_interp_vector(t_cur)
            return t_cur
        # If no lower-depth optimized vector is supplied, interpolate from p=1 rescaled fixed angle
        t1 = get_default_x0(1, "shaydulin_rescaled", edges)
        while len(t1) // 2 < p:
            t1 = zhou_interp_vector(t1)
        return t1
    elif strategy == "layer_padding":
        if theta_lower_depth is not None:
            t_cur = np.asarray(theta_lower_depth, dtype=float).ravel()
            while len(t_cur) // 2 < p:
                t_cur = layer_padding_vector(t_cur)
            return t_cur
        t1 = get_default_x0(1, "shaydulin_rescaled", edges)
        while len(t1) // 2 < p:
            t1 = layer_padding_vector(t1)
        return t1
    raise ValueError(f"Unknown warm-start strategy: {strategy}")
