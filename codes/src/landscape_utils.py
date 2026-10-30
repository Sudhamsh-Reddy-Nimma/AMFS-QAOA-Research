"""Global-minimum finder for the exact p-layer QAOA loss, used to build *strict* global nulls.

`RetrospectiveEvaluator.compute_exact_qaoa_ceiling` (qaoa_core.py) is a short COBYLA estimate: it is
exceeded by some optimizers at larger n and p (2 of 540 scaling trials by up to 0.0103).  A strict global
null needs an *upper bound on how low any theta can go*, i.e. a minimum that is not too high.  This module
searches harder (dense grid + local polish for p=1; many random + structured starts for p>=2) and returns
the lowest exact loss found over the full parameter box.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.optimize import minimize

from qaoa_core import RetrospectiveEvaluator, get_bounds, get_default_x0


def global_min_loss(ev: RetrospectiveEvaluator, p: int, seed: int = 0, n_starts_p2: int = 40) -> Tuple[float, np.ndarray]:
    b = get_bounds(p)
    bl = list(map(tuple, b))
    best_loss, best_x = np.inf, None

    def f(x):
        return ev.true_loss(np.asarray(x, dtype=float))

    if p == 1:
        G = np.linspace(b[0, 0], b[0, 1], 121)
        Bt = np.linspace(b[1, 0], b[1, 1], 61)
        vals = np.array([[f([g, bb]) for bb in Bt] for g in G])
        order = np.dstack(np.unravel_index(np.argsort(vals, axis=None)[:5], vals.shape))[0]
        starts = [np.array([G[i], Bt[j]]) for i, j in order]
        gi, bi = np.unravel_index(np.argmin(vals), vals.shape)
        best_loss, best_x = float(vals[gi, bi]), np.array([G[gi], Bt[bi]])
    else:
        rng = np.random.default_rng(seed + 991 * p)
        starts = [get_default_x0(p, "wurtz_love", ev.engine.edges), get_default_x0(p, "naive", ev.engine.edges)]
        starts += [rng.uniform(b[:, 0], b[:, 1]) for _ in range(n_starts_p2 * (1 if p == 2 else 2))]

    for x0 in starts:
        r = minimize(f, x0, method="L-BFGS-B", bounds=bl, options={"maxiter": 150})
        if r.fun < best_loss:
            best_loss, best_x = float(r.fun), np.asarray(r.x, dtype=float)
    return best_loss, best_x
