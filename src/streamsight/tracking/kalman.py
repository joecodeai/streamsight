"""Constant-velocity Kalman filter in (cx, cy, aspect, height) space.

State is 8-D: position (cx, cy, a, h) and their velocities. Noise is scaled by
box height so large, close objects tolerate larger pixel motion than small,
distant ones.
"""

from __future__ import annotations

import numpy as np

_STD_POS = 1.0 / 20
_STD_VEL = 1.0 / 160


class KalmanFilter:
    ndim = 4

    def __init__(self) -> None:
        self._motion = np.eye(8)
        for i in range(4):
            self._motion[i, 4 + i] = 1.0
        self._update = np.eye(4, 8)

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = np.r_[measurement, np.zeros(4)]
        h = measurement[3]
        std = [
            2 * _STD_POS * h,
            2 * _STD_POS * h,
            1e-2,
            2 * _STD_POS * h,
            10 * _STD_VEL * h,
            10 * _STD_VEL * h,
            1e-5,
            10 * _STD_VEL * h,
        ]
        return mean, np.diag(np.square(std))

    def predict(self, mean: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = mean[3]
        std = [_STD_POS * h, _STD_POS * h, 1e-2, _STD_POS * h, _STD_VEL * h, _STD_VEL * h, 1e-5, _STD_VEL * h]
        q = np.diag(np.square(std))
        mean = self._motion @ mean
        cov = self._motion @ cov @ self._motion.T + q
        return mean, cov

    def multi_predict(self, means: np.ndarray, covs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Vectorised predict for (N, 8) means and (N, 8, 8) covariances."""
        h = means[:, 3]
        std = np.stack(
            [
                _STD_POS * h,
                _STD_POS * h,
                np.full_like(h, 1e-2),
                _STD_POS * h,
                _STD_VEL * h,
                _STD_VEL * h,
                np.full_like(h, 1e-5),
                _STD_VEL * h,
            ],
            axis=1,
        )
        q = np.zeros((len(means), 8, 8))
        idx = np.arange(8)
        q[:, idx, idx] = np.square(std)
        means = means @ self._motion.T
        covs = self._motion @ covs @ self._motion.T + q
        return means, covs

    def update(
        self, mean: np.ndarray, cov: np.ndarray, measurement: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        h = mean[3]
        r = np.diag(np.square([_STD_POS * h, _STD_POS * h, 1e-1, _STD_POS * h]))
        proj_mean = self._update @ mean
        proj_cov = self._update @ cov @ self._update.T + r
        gain = np.linalg.solve(proj_cov, (cov @ self._update.T).T).T
        mean = mean + gain @ (measurement - proj_mean)
        cov = cov - gain @ proj_cov @ gain.T
        return mean, cov


def xyxy_to_xyah(box: np.ndarray) -> np.ndarray:
    w = box[2] - box[0]
    h = max(box[3] - box[1], 1e-6)
    return np.array([box[0] + w / 2, box[1] + h / 2, w / h, h], dtype=np.float64)


def xyah_to_xyxy(xyah: np.ndarray) -> np.ndarray:
    cx, cy, a, h = xyah[:4]
    w = a * h
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)
