"""gap_classifier.py — Trajectory-based Straight/Rolling gap classifier."""

import math
from typing import List, Tuple


class GapClassifier:
    """
    Classifies a pedestrian crossing as Straight Gap or Rolling Gap.

    Algorithm
    ---------
    1. Compute per-frame Euclidean displacement (speed proxy).
    2. Find mean speed over the full trajectory.
    3. Identify contiguous runs where speed < speed_ratio * mean_speed.
    4. If any such run lasts ≥ ceil(min_duration * fps) frames → Rolling.
    5. Otherwise → Straight.

    Edge cases
    ----------
    - Empty or single-point trajectory → "Straight"
    - Zero mean speed (pedestrian stationary) → "Straight"
    """

    def __init__(
        self,
        fps: float,
        speed_ratio: float = 0.20,
        min_duration: float = 0.5,
    ) -> None:
        self._fps = fps
        self._speed_ratio = speed_ratio
        self._min_frames = math.ceil(min_duration * fps)

    # ── Public API ────────────────────────────────────────────────────────

    def classify(self, trajectory: List[Tuple[float, float]]) -> str:
        """
        Returns "Straight" or "Rolling".
        trajectory: list of (cx, cy) centroids in frame order.
        """
        speeds = self.compute_speeds(trajectory)
        if not speeds:
            return "Straight"

        mean_speed = sum(speeds) / len(speeds)
        if mean_speed == 0.0:
            return "Straight"

        threshold = self._speed_ratio * mean_speed
        slow = [s < threshold for s in speeds]

        # Find longest contiguous run of True
        max_run = 0
        current_run = 0
        for flag in slow:
            if flag:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0

        return "Rolling" if max_run >= self._min_frames else "Straight"

    def compute_speeds(
        self, trajectory: List[Tuple[float, float]]
    ) -> List[float]:
        """
        Returns a list of N-1 non-negative Euclidean displacements
        (pixels per frame) for a trajectory of N points.
        """
        if len(trajectory) < 2:
            return []
        speeds = []
        for i in range(len(trajectory) - 1):
            dx = trajectory[i + 1][0] - trajectory[i][0]
            dy = trajectory[i + 1][1] - trajectory[i][1]
            speeds.append(math.sqrt(dx * dx + dy * dy))
        return speeds
