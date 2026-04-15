"""vehicle_metrics.py — Vehicle entry/exit tracking and gap metric computation."""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class VehicleEvent:
    frame_idx: int
    timestamp: float   # seconds = frame_idx / fps
    event_type: str    # "entry" | "exit"
    vehicle_id: int


@dataclass
class GapMetrics:
    gap_seconds: float
    time_headway: float
    vehicle_speed_px_per_s: float


class VehicleMetricsExtractor:
    """
    Tracks vehicle presence in the Road_Region frame-by-frame and computes:
    - Gap duration at any given frame
    - Time headway (mean inter-entry interval)
    - Vehicle speed (pixel displacement per second)
    """

    def __init__(self, fps: float) -> None:
        self._fps = fps
        self._events: List[VehicleEvent] = []
        # track_id → last known centroid for speed estimation
        self._prev_centroids: Dict[int, Tuple[float, float]] = {}
        # track_id → last estimated speed
        self._vehicle_speeds: Dict[int, float] = {}
        # track_ids currently in region
        self._in_region_prev: Set[int] = set()
        # frame_idx → set of vehicle ids present
        self._presence: Dict[int, Set[int]] = {}

    # ── Per-frame update ──────────────────────────────────────────────────

    def update(
        self,
        frame_idx: int,
        vehicle_tracks: list,   # list[Track] — avoid circular import
        in_region: Set[int],
    ) -> None:
        """
        Called each frame with vehicle tracks whose centroids are in Road_Region.
        `in_region` is the set of vehicle track_ids currently inside the polygon.
        """
        self._presence[frame_idx] = set(in_region)

        # Detect entries (new IDs)
        for vid in in_region - self._in_region_prev:
            ts = frame_idx / self._fps
            self._events.append(
                VehicleEvent(frame_idx, ts, "entry", vid)
            )

        # Detect exits (IDs that left)
        for vid in self._in_region_prev - in_region:
            ts = frame_idx / self._fps
            self._events.append(
                VehicleEvent(frame_idx, ts, "exit", vid)
            )

        # Update speed estimates for vehicles in region
        track_map = {t.track_id: t for t in vehicle_tracks}
        for vid in in_region:
            if vid in track_map:
                cx, cy = track_map[vid].centroid
                if vid in self._prev_centroids:
                    px, py = self._prev_centroids[vid]
                    dist = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
                    self._vehicle_speeds[vid] = dist * self._fps
                self._prev_centroids[vid] = (cx, cy)

        self._in_region_prev = set(in_region)

    # ── Query methods ─────────────────────────────────────────────────────

    def get_gap_at_frame(self, frame_idx: int) -> GapMetrics:
        """
        Returns GapMetrics for the gap period that contains frame_idx.
        gap_seconds = duration of the clear (vehicle-free) window around frame_idx.
        """
        ts = frame_idx / self._fps
        gap_seconds = self._compute_gap_at_time(ts)
        time_headway = self.compute_time_headway()
        # Use mean of all recorded vehicle speeds as representative speed
        speeds = list(self._vehicle_speeds.values())
        vehicle_speed = sum(speeds) / len(speeds) if speeds else 0.0
        return GapMetrics(gap_seconds, time_headway, vehicle_speed)

    def compute_time_headway(self) -> float:
        """
        Mean time interval between consecutive vehicle entries.
        Returns 0.0 if fewer than 2 entry events.
        """
        entry_times = sorted(
            e.timestamp for e in self._events if e.event_type == "entry"
        )
        if len(entry_times) < 2:
            return 0.0
        intervals = [entry_times[i + 1] - entry_times[i] for i in range(len(entry_times) - 1)]
        return sum(intervals) / len(intervals)

    @staticmethod
    def compute_speed(
        cx1: float, cy1: float, cx2: float, cy2: float, fps: float
    ) -> float:
        """Euclidean displacement × fps → pixels per second."""
        return math.sqrt((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2) * fps

    # ── Internal ──────────────────────────────────────────────────────────

    def _compute_gap_at_time(self, ts: float) -> float:
        """
        Find the vehicle-free gap window that contains timestamp ts.
        Returns the duration of that window in seconds.
        """
        # Build sorted list of (timestamp, event_type) for entries and exits
        sorted_events = sorted(self._events, key=lambda e: e.timestamp)

        # Find the most recent exit before ts
        last_exit_ts: Optional[float] = None
        for ev in sorted_events:
            if ev.event_type == "exit" and ev.timestamp <= ts:
                last_exit_ts = ev.timestamp

        # Find the next entry after ts
        next_entry_ts: Optional[float] = None
        for ev in sorted_events:
            if ev.event_type == "entry" and ev.timestamp > ts:
                next_entry_ts = ev.timestamp
                break

        if last_exit_ts is None and next_entry_ts is None:
            # No vehicles at all — gap is the full video duration so far
            return ts

        gap_start = last_exit_ts if last_exit_ts is not None else 0.0
        gap_end = next_entry_ts if next_entry_ts is not None else ts

        return max(0.0, gap_end - gap_start)
