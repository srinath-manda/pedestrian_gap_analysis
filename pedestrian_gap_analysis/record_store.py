"""record_store.py — In-memory accumulation of per-pedestrian crossing records."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class PedestrianRecord:
    track_id: int
    gender: str = "Unknown"
    age_group: str = "Unknown"
    platoon: str = "Alone"
    gap_seconds: float = 0.0
    time_headway: float = 0.0
    vehicle_speed: float = 0.0
    gap_type: str = ""           # "Straight" | "Rolling"
    gap_type_binary: int = -1    # 1 = Straight, 0 = Rolling; -1 = not yet set
    trajectory: List[Tuple[float, float]] = field(default_factory=list)
    entry_frame: int = -1
    exit_frame: int = -1
    complete: bool = False       # True only when track exits Road_Region cleanly


class RecordStore:
    """
    Manages PedestrianRecord objects keyed by track_id.
    Provides helpers for the frame loop to update trajectories and finalise records.
    """

    def __init__(self) -> None:
        self._records: Dict[int, PedestrianRecord] = {}

    # ── Record lifecycle ──────────────────────────────────────────────────

    def get_or_create(self, track_id: int, entry_frame: int) -> PedestrianRecord:
        """Return existing record or create a new one for this track."""
        if track_id not in self._records:
            self._records[track_id] = PedestrianRecord(
                track_id=track_id,
                entry_frame=entry_frame,
            )
        return self._records[track_id]

    def append_trajectory(
        self, track_id: int, centroid: Tuple[float, float]
    ) -> None:
        """Append a centroid to the track's trajectory (only while in region)."""
        if track_id in self._records:
            self._records[track_id].trajectory.append(centroid)

    def set_attributes(
        self, track_id: int, gender: str, age_group: str
    ) -> None:
        if track_id in self._records:
            rec = self._records[track_id]
            rec.gender = gender
            rec.age_group = age_group

    def set_platoon(self, track_id: int, platoon: str) -> None:
        if track_id in self._records:
            self._records[track_id].platoon = platoon

    def set_gap_metrics(
        self,
        track_id: int,
        gap_seconds: float,
        time_headway: float,
        vehicle_speed: float,
    ) -> None:
        if track_id in self._records:
            rec = self._records[track_id]
            rec.gap_seconds = gap_seconds
            rec.time_headway = time_headway
            rec.vehicle_speed = vehicle_speed

    def finalise(self, track_id: int, exit_frame: int, gap_type: str) -> None:
        """Mark a record as complete when the pedestrian exits the Road_Region."""
        if track_id in self._records:
            rec = self._records[track_id]
            rec.exit_frame = exit_frame
            rec.gap_type = gap_type
            rec.gap_type_binary = 1 if gap_type == "Straight" else 0
            rec.complete = True

    # ── Queries ───────────────────────────────────────────────────────────

    def get(self, track_id: int) -> Optional[PedestrianRecord]:
        return self._records.get(track_id)

    def get_complete_records(self) -> List[PedestrianRecord]:
        """Return only records where complete=True."""
        return [r for r in self._records.values() if r.complete]

    def all_records(self) -> List[PedestrianRecord]:
        return list(self._records.values())

    def active_track_ids(self) -> List[int]:
        return list(self._records.keys())
