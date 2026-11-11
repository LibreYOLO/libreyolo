"""Kalman filters with constant-velocity models for bounding box tracking.

State space: (cx, cy, aspect_ratio, height, v_cx, v_cy, v_a, v_h)
Measurement: (cx, cy, aspect_ratio, height)
"""

import numpy as np

try:
    import scipy.linalg
except ImportError as e:
    raise ImportError(
        'scipy is required for tracking. Install with: pip install "libreyolo[tracking]"'
    ) from e


class KalmanFilterXYAH:
    """8-dimensional Kalman filter for bounding box tracking.

    Uses a constant-velocity model with adaptive noise proportional
    to the bounding box height.
    """

    _std_weight_position = 1.0 / 20
    _std_weight_velocity = 1.0 / 160

    def __init__(self):
        ndim = 4
        dt = 1.0

        # State transition matrix (constant velocity).
        self._motion_mat = np.eye(2 * ndim, dtype=np.float64)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        # Observation matrix (we only observe position components).
        self._update_mat = np.eye(ndim, 2 * ndim, dtype=np.float64)

    def initiate(self, measurement: np.ndarray):
        """Initialize track state from an unassociated measurement.

        Args:
            measurement: (cx, cy, a, h) bounding box center, aspect ratio, height.

        Returns:
            (mean, covariance) tuple with shapes (8,) and (8, 8).
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.concatenate([mean_pos, mean_vel])

        h = measurement[3]
        std = [
            2 * self._std_weight_position * h,
            2 * self._std_weight_position * h,
            1e-2,
            2 * self._std_weight_position * h,
            10 * self._std_weight_velocity * h,
            10 * self._std_weight_velocity * h,
            1e-5,
            10 * self._std_weight_velocity * h,
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray):
        """Run Kalman filter prediction step.

        Args:
            mean: (8,) state vector.
            covariance: (8, 8) state covariance.

        Returns:
            (predicted_mean, predicted_covariance).
        """
        h = max(mean[3], 1e-2)
        std_pos = [
            self._std_weight_position * h,
            self._std_weight_position * h,
            1e-2,
            self._std_weight_position * h,
        ]
        std_vel = [
            self._std_weight_velocity * h,
            self._std_weight_velocity * h,
            1e-5,
            self._std_weight_velocity * h,
        ]
        motion_cov = np.diag(np.square(np.concatenate([std_pos, std_vel])))

        mean = self._motion_mat @ mean
        # Clamp covariance to prevent overflow from repeated prediction
        # without updates (e.g., lost tracks). The near-zero aspect-ratio
        # velocity variance can trigger spurious numpy matmul warnings.
        covariance = np.clip(covariance, -1e10, 1e10)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov
        return mean, covariance

    def multi_predict(self, mean: np.ndarray, covariance: np.ndarray):
        """Vectorized prediction for multiple tracks.

        Args:
            mean: (N, 8) state vectors.
            covariance: (N, 8, 8) state covariances.

        Returns:
            (predicted_means, predicted_covariances) with same shapes.
        """
        h = np.maximum(mean[:, 3], 1e-2)
        std_pos = [
            self._std_weight_position * h,
            self._std_weight_position * h,
            1e-2 * np.ones_like(h),
            self._std_weight_position * h,
        ]
        std_vel = [
            self._std_weight_velocity * h,
            self._std_weight_velocity * h,
            1e-5 * np.ones_like(h),
            self._std_weight_velocity * h,
        ]
        sqr = np.square(np.column_stack(std_pos + std_vel))
        motion_cov = np.array([np.diag(s) for s in sqr])

        mean = np.einsum("ij,nj->ni", self._motion_mat, mean)
        covariance = np.clip(covariance, -1e10, 1e10)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            covariance = (
                np.einsum(
                    "ij,njk,lk->nil",
                    self._motion_mat,
                    covariance,
                    self._motion_mat,
                )
                + motion_cov
            )
        return mean, covariance

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray):
        """Run Kalman filter correction step.

        Args:
            mean: (8,) predicted state vector.
            covariance: (8, 8) predicted state covariance.
            measurement: (4,) observed (cx, cy, a, h).

        Returns:
            (corrected_mean, corrected_covariance).
        """
        h = mean[3]
        std = [
            self._std_weight_position * h,
            self._std_weight_position * h,
            1e-1,
            self._std_weight_position * h,
        ]
        innovation_cov = np.diag(np.square(std))

        projected_mean = self._update_mat @ mean
        projected_cov = (
            self._update_mat @ covariance @ self._update_mat.T + innovation_cov
        )

        # Solve via Cholesky decomposition for numerical stability.
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False
        )
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            (covariance @ self._update_mat.T).T,
            check_finite=False,
        ).T

        innovation = measurement - projected_mean
        new_mean = mean + innovation @ kalman_gain.T
        new_covariance = covariance - kalman_gain @ projected_cov @ kalman_gain.T
        return new_mean, new_covariance


class KalmanFilterXYWH(KalmanFilterXYAH):
    """BoT-SORT Kalman filter using independent width and height states.

    State space: ``(cx, cy, w, h, vx, vy, vw, vh)``. Process,
    measurement, and initialization noise scale independently with the box
    width and height. This avoids coupling width changes to an aspect-ratio
    state as ByteTrack's XYAH filter does.
    """

    @staticmethod
    def _scales(measurement: np.ndarray) -> tuple[float, float]:
        return max(abs(float(measurement[2])), 1e-2), max(
            abs(float(measurement[3])), 1e-2
        )

    def initiate(self, measurement: np.ndarray):
        """Initialize from a ``(cx, cy, width, height)`` measurement."""
        mean = np.concatenate([measurement, np.zeros_like(measurement)])
        w, h = self._scales(measurement)
        sp = self._std_weight_position
        sv = self._std_weight_velocity
        std = np.array(
            [
                2 * sp * w,
                2 * sp * h,
                2 * sp * w,
                2 * sp * h,
                10 * sv * w,
                10 * sv * h,
                10 * sv * w,
                10 * sv * h,
            ],
            dtype=np.float64,
        )
        return mean, np.diag(np.square(std))

    def predict(self, mean: np.ndarray, covariance: np.ndarray):
        """Run a scale-aware XYWH prediction step."""
        w, h = self._scales(mean)
        sp = self._std_weight_position
        sv = self._std_weight_velocity
        std = np.array(
            [sp * w, sp * h, sp * w, sp * h, sv * w, sv * h, sv * w, sv * h],
            dtype=np.float64,
        )
        motion_cov = np.diag(np.square(std))
        mean = self._motion_mat @ mean
        covariance = np.clip(covariance, -1e10, 1e10)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            covariance = self._motion_mat @ covariance @ self._motion_mat.T
            covariance += motion_cov
        return mean, covariance

    def multi_predict(self, mean: np.ndarray, covariance: np.ndarray):
        """Vectorized scale-aware XYWH prediction."""
        if len(mean) == 0:
            return mean, covariance
        w = np.maximum(np.abs(mean[:, 2]), 1e-2)
        h = np.maximum(np.abs(mean[:, 3]), 1e-2)
        sp = self._std_weight_position
        sv = self._std_weight_velocity
        std = np.column_stack(
            [sp * w, sp * h, sp * w, sp * h, sv * w, sv * h, sv * w, sv * h]
        )
        motion_cov = np.array([np.diag(np.square(row)) for row in std])
        mean = np.einsum("ij,nj->ni", self._motion_mat, mean)
        covariance = np.clip(covariance, -1e10, 1e10)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            covariance = (
                np.einsum(
                    "ij,njk,lk->nil",
                    self._motion_mat,
                    covariance,
                    self._motion_mat,
                )
                + motion_cov
            )
        return mean, covariance

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray):
        """Correct the state with a ``(cx, cy, width, height)`` measurement."""
        w, h = self._scales(mean)
        sp = self._std_weight_position
        std = np.array([sp * w, sp * h, sp * w, sp * h], dtype=np.float64)
        innovation_cov = np.diag(np.square(std))
        projected_mean = self._update_mat @ mean
        projected_cov = (
            self._update_mat @ covariance @ self._update_mat.T + innovation_cov
        )
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False
        )
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            (covariance @ self._update_mat.T).T,
            check_finite=False,
        ).T
        innovation = measurement - projected_mean
        new_mean = mean + innovation @ kalman_gain.T
        # Joseph form preserves symmetry and positive semi-definiteness better
        # over long sequences than the abbreviated P - KSK^T update.
        identity_minus_gain = np.eye(8) - kalman_gain @ self._update_mat
        new_covariance = (
            identity_minus_gain @ covariance @ identity_minus_gain.T
            + kalman_gain @ innovation_cov @ kalman_gain.T
        )
        return new_mean, new_covariance
