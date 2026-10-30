"""Genuine Multi-Fidelity Surrogate Fusion via Autoregressive Co-Kriging (Workstream 8).

Implements the three explicit fidelity levels defined in Section 11.1:
  - Fidelity 0 (`F0`, 0 quantum shots): Classical local 1-hop tree-subgraph correlator predictor
    `f_0(theta)` from `engine.classical_fidelity0_loss(theta)`.
  - Fidelity 1 (`F1`, 20 quantum shots): Pilot noisy quantum evaluation `y_1(theta)` with variance `v_1`.
  - Fidelity 2 (`F2`, 150 quantum shots): Higher-fidelity quantum evaluation `y_2(theta)` with variance `v_2`.

Implements the Kennedy & O'Hagan (Biometrika 2000) / Forrester et al. (Proc. R. Soc. A 2007)
autoregressive Gaussian-process co-kriging model:
  f_high(x) = rho * f_low(x) + discrepancy(x),   discrepancy(x) ~ GP(mu_d, k_Matern52(x, x'))
across the hierarchy F0 -> F1 -> F2:
  1. Level 0 -> 1: Learns scaling `rho_01` and discrepancy GP `delta_01(x) = y_1(x) - rho_01 * f_0(x)`
     from paired evaluations `(f_0(x_i), y_1(x_i))`.
  2. Level 1 -> 2: Learns scaling `rho_12` and discrepancy GP `delta_12(x) = y_2(x) - rho_12 * mu_1(x)`
     from higher-fidelity evaluations `y_2(x_j)`.
  3. Negative-Transfer / Correlation Gate: Computes the empirical correlation `r_01` between `f_0` and `y_1`
     on the pilot sample; if `r_01 < r_min` (e.g. `0.25`, which occurs when triangles/4-cycles or
     multi-layer interference invert the classical tree predictor), sets `rho_01 = 0.0` (falling back
     smoothly to a pure discrepancy GP) so misleading classical priors never corrupt optimization!
"""
from __future__ import annotations
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.stats import spearmanr

from qaoa_core import OperationalQAOAOracle, QAOACircuitEngine


def _matern52_cov(X1: np.ndarray, X2: np.ndarray, ls: float, var: float) -> np.ndarray:
    diff = (X1[:, None, :] - X2[None, :, :]) / max(ls, 1e-3)
    r = np.sqrt(np.sum(diff ** 2, axis=-1))
    s5r = math.sqrt(5.0) * r
    return var * (1.0 + s5r + (5.0 / 3.0) * (r ** 2)) * np.exp(-s5r)


def _fit_gp_1level(
    X: np.ndarray,
    y: np.ndarray,
    noise_vars: np.ndarray,
    X_pred: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit Matern-5/2 GP to residuals `y` with known heteroscedastic noise variances `noise_vars`."""
    if len(X) == 0:
        return np.zeros(len(X_pred)), np.full(len(X_pred), 0.1)
    mu_y = float(np.mean(y))
    sd_y = float(np.std(y))
    if sd_y < 1e-6:
        sd_y = 1.0
    y_std = (y - mu_y) / sd_y
    v_std = noise_vars / (sd_y ** 2) + 1e-4

    best_lml = -math.inf
    best_tuple = None
    for ls in (0.35, 0.75, 1.30):
        for var_s in (0.5, 1.0):
            K = _matern52_cov(X, X, ls, var_s) + np.diag(v_std)
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                continue
            al = np.linalg.solve(L.T, np.linalg.solve(L, y_std))
            lml = -0.5 * float(y_std @ al) - float(np.sum(np.log(np.diag(L))))
            if lml > best_lml:
                best_lml = lml
                best_tuple = (ls, var_s, L, al)

    if best_tuple is None:
        ls, var_s = 0.75, 1.0
        K = _matern52_cov(X, X, ls, var_s) + np.diag(v_std + 1e-2)
        L = np.linalg.cholesky(K)
        al = np.linalg.solve(L.T, np.linalg.solve(L, y_std))
    else:
        ls, var_s, L, al = best_tuple

    Ks = _matern52_cov(X, X_pred, ls, var_s)
    pred_mean = (Ks.T @ al) * sd_y + mu_y
    proj = np.linalg.solve(L, Ks)
    pred_var = np.maximum(var_s - np.sum(proj ** 2, axis=0), 1e-8) * (sd_y ** 2)
    return pred_mean, pred_var


@dataclass
class CoKrigingDiagnostics:
    rho_01: float
    rho_12: float
    pearson_r01: float
    spearman_r01: float
    gate_triggered_negative_transfer: bool
    rmse_f0_only: float
    rmse_cokriging: float
    coverage_90pct: float
    coverage_95pct: float
    fit_wall_seconds: float


class ThreeLevelCoKrigingSurrogate:
    """Kennedy-O'Hagan 3-level autoregressive co-kriging model (F0 -> F1 -> F2)."""

    def __init__(
        self,
        engine: QAOACircuitEngine,
        use_f0: bool = True,
        use_f1: bool = True,
        use_f2: bool = True,
        correlation_gate_threshold: float = 0.25,
    ):
        self.engine = engine
        self.use_f0 = bool(use_f0)
        self.use_f1 = bool(use_f1)
        self.use_f2 = bool(use_f2)
        self.correlation_gate_threshold = float(correlation_gate_threshold)

        self.X1: List[np.ndarray] = []
        self.y1: List[float] = []
        self.v1: List[float] = []
        self.f0_at_X1: List[float] = []

        self.X2: List[np.ndarray] = []
        self.y2: List[float] = []
        self.v2: List[float] = []

        self.rho_01: float = 0.0
        self.rho_12: float = 1.0
        self.pearson_r01: float = 0.0
        self.spearman_r01: float = 0.0
        self.gate_triggered: bool = False
        self.total_fit_seconds: float = 0.0

    def eval_f0_batch(self, X_cand: np.ndarray) -> np.ndarray:
        if not self.use_f0:
            return np.full(len(X_cand), 0.5)
        return np.array([self.engine.classical_fidelity0_loss(x) for x in X_cand], dtype=float)

    def add_f1_observation(self, theta: np.ndarray, losses: np.ndarray) -> None:
        if len(losses) == 0:
            return
        t_c = np.asarray(theta, dtype=float).ravel().copy()
        self.X1.append(t_c)
        self.y1.append(float(np.mean(losses)))
        var_m = float(np.var(losses, ddof=1) / len(losses)) if len(losses) > 1 else 0.01
        self.v1.append(max(var_m, 1e-5))
        self.f0_at_X1.append(self.engine.classical_fidelity0_loss(t_c) if self.use_f0 else 0.5)

    def add_f2_observation(self, theta: np.ndarray, losses: np.ndarray) -> None:
        if len(losses) == 0:
            return
        t_c = np.asarray(theta, dtype=float).ravel().copy()
        self.X2.append(t_c)
        self.y2.append(float(np.mean(losses)))
        var_m = float(np.var(losses, ddof=1) / len(losses)) if len(losses) > 1 else 0.002
        self.v2.append(max(var_m, 1e-6))

    def predict(self, X_cand: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict posterior mean and standard deviation at highest active fidelity on `X_cand`."""
        t0 = time.perf_counter()
        f0_cand = self.eval_f0_batch(X_cand)

        if not self.use_f1 or len(self.X1) < 2:
            # Pure F0 or pure F2 fallback
            if self.use_f2 and len(self.X2) >= 2:
                X2_arr = np.asarray(self.X2)
                y2_arr = np.asarray(self.y2)
                v2_arr = np.asarray(self.v2)
                if self.use_f0:
                    f0_x2 = self.eval_f0_batch(X2_arr)
                    if np.std(f0_x2) > 1e-6 and np.std(y2_arr) > 1e-6:
                        r = float(np.corrcoef(f0_x2, y2_arr)[0, 1])
                        self.pearson_r01 = r
                        if r >= self.correlation_gate_threshold:
                            self.rho_01 = float(np.cov(f0_x2, y2_arr)[0, 1] / np.var(f0_x2))
                        else:
                            self.rho_01 = 0.0
                            self.gate_triggered = True
                    res2 = y2_arr - self.rho_01 * f0_x2
                else:
                    self.rho_01 = 0.0
                    res2 = y2_arr
                mu_d, var_d = _fit_gp_1level(X2_arr, res2, v2_arr, X_cand)
                self.total_fit_seconds += time.perf_counter() - t0
                return self.rho_01 * f0_cand + mu_d, np.sqrt(var_d)
            self.total_fit_seconds += time.perf_counter() - t0
            return f0_cand, np.full(len(X_cand), 0.05)

        X1_arr = np.asarray(self.X1)
        y1_arr = np.asarray(self.y1)
        v1_arr = np.asarray(self.v1)
        f0_x1 = np.asarray(self.f0_at_X1)

        if self.use_f0 and np.std(f0_x1) > 1e-6 and np.std(y1_arr) > 1e-6:
            r_p = float(np.corrcoef(f0_x1, y1_arr)[0, 1])
            r_s = float(spearmanr(f0_x1, y1_arr).statistic)
            self.pearson_r01 = r_p
            self.spearman_r01 = r_s if not math.isnan(r_s) else 0.0
            if r_p >= self.correlation_gate_threshold and self.spearman_r01 >= 0.15:
                raw_rho = float(np.cov(f0_x1, y1_arr)[0, 1] / np.var(f0_x1))
                self.rho_01 = float(np.clip(raw_rho, 0.0, 2.0))
                self.gate_triggered = False
            else:
                # Negative transfer protection: zero out low-fidelity weight when poorly correlated
                self.rho_01 = 0.0
                self.gate_triggered = True
        else:
            self.rho_01 = 0.0

        delta_01 = y1_arr - self.rho_01 * f0_x1
        mu_d01, var_d01 = _fit_gp_1level(X1_arr, delta_01, v1_arr, X_cand)
        mu_1 = self.rho_01 * f0_cand + mu_d01
        var_1 = var_d01

        if not self.use_f2 or len(self.X2) == 0:
            self.total_fit_seconds += time.perf_counter() - t0
            return mu_1, np.sqrt(np.maximum(var_1, 1e-8))

        # Level 1 -> Level 2 autoregressive fusion: f_2(x) = rho_12 * f_1(x) + delta_12(x)
        X2_arr = np.asarray(self.X2)
        y2_arr = np.asarray(self.y2)
        v2_arr = np.asarray(self.v2)
        f0_x2 = self.eval_f0_batch(X2_arr)
        mu_d01_at_x2, var_d01_at_x2 = _fit_gp_1level(X1_arr, delta_01, v1_arr, X2_arr)
        mu_1_at_x2 = self.rho_01 * f0_x2 + mu_d01_at_x2

        if len(X2_arr) >= 3 and np.std(mu_1_at_x2) > 1e-6 and np.std(y2_arr) > 1e-6:
            r12 = float(np.corrcoef(mu_1_at_x2, y2_arr)[0, 1])
            if r12 >= 0.25:
                self.rho_12 = float(np.clip(np.cov(mu_1_at_x2, y2_arr)[0, 1] / np.var(mu_1_at_x2), 0.2, 1.5))
            else:
                self.rho_12 = 1.0
        else:
            self.rho_12 = 1.0

        delta_12 = y2_arr - self.rho_12 * mu_1_at_x2
        comb_v2 = v2_arr + (self.rho_12 ** 2) * var_d01_at_x2 * 0.25
        mu_d12, var_d12 = _fit_gp_1level(X2_arr, delta_12, comb_v2, X_cand)

        mu_2 = self.rho_12 * mu_1 + mu_d12
        var_2 = (self.rho_12 ** 2) * var_1 + var_d12
        self.total_fit_seconds += time.perf_counter() - t0
        return mu_2, np.sqrt(np.maximum(var_2, 1e-8))
