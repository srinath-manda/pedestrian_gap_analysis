"""platoon_detector.py — Detects group vs solo pedestrian crossings."""

from typing import Set


class PlatoonDetector:
    """
    Determines whether each pedestrian crossed alone or as part of a group.

    A track is flagged "Group" if, in any frame during its Road_Region presence,
    at least one other pedestrian track was simultaneously present.
    Once flagged "Group", the flag is permanent.
    """

    def __init__(self) -> None:
        # track_id → "Group" | "Alone"
        self._flags: dict[int, str] = {}

    def update(self, frame_idx: int, track_ids_in_region: Set[int]) -> None:
        """
        Called each frame with the set of pedestrian track IDs currently
        inside the Road_Region polygon.
        """
        ids = set(track_ids_in_region)

        for tid in ids:
            # Ensure the track has an entry
            if tid not in self._flags:
                self._flags[tid] = "Alone"

        if len(ids) >= 2:
            # All co-present tracks become "Group" (permanent)
            for tid in ids:
                self._flags[tid] = "Group"

    def get_platoon_flag(self, track_id: int) -> str:
        """
        Returns "Group" or "Alone" for the given track_id.
        Returns "Alone" for unknown track IDs (defensive default).
        """
        return self._flags.get(track_id, "Alone")
