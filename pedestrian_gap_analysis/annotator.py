"""annotator.py — Draws overlays on frames and writes the annotated output video."""

import logging
import os
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class Annotator:
    """
    Renders bounding boxes, track IDs, Road_Region polygon, and
    pedestrian attribute labels onto each frame, then writes to MP4.
    """

    # Colours (BGR)
    _COLOUR_BOX = (0, 255, 0)       # green — pedestrian bbox
    _COLOUR_POLY = (255, 165, 0)    # orange — road region polygon
    _COLOUR_LABEL = (255, 255, 255) # white — text
    _COLOUR_BG = (0, 0, 0)         # black — text background

    def __init__(
        self,
        output_dir: str,
        fps: float,
        width: int,
        height: int,
        codec: str = "mp4v",
        filename: str = "annotated_output.mp4",
    ) -> None:
        self._writer: Optional[cv2.VideoWriter] = None
        self._writer_ok = False
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, filename)

        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
            if not writer.isOpened():
                raise IOError(f"VideoWriter failed to open '{out_path}' with codec '{codec}'")
            self._writer = writer
            self._writer_ok = True
        except Exception as exc:
            logger.error(
                "[Annotator] Could not initialise VideoWriter: %s. "
                "Annotated video will NOT be saved.",
                exc,
            )

    # ── Public API ────────────────────────────────────────────────────────

    def annotate_frame(
        self,
        frame: np.ndarray,
        tracks: list,           # list[Track]
        polygon: np.ndarray,    # (N,2) int32
        record_store,           # RecordStore — avoid circular import
    ) -> np.ndarray:
        """
        Draw all overlays onto a copy of `frame` and return the annotated copy.
        """
        out = frame.copy()

        # Draw Road_Region polygon
        if polygon is not None and len(polygon) >= 3:
            cv2.polylines(
                out,
                [polygon.reshape((-1, 1, 2))],
                isClosed=True,
                color=self._COLOUR_POLY,
                thickness=2,
            )

        # Draw each pedestrian track
        for track in tracks:
            x1, y1, x2, y2 = (int(v) for v in track.bbox)
            tid = track.track_id

            # Bounding box
            cv2.rectangle(out, (x1, y1), (x2, y2), self._COLOUR_BOX, 2)

            # Build label string
            rec = record_store.get(tid)
            if rec:
                gender = rec.gender if rec.gender != "Unknown" else "?"
                age_grp = rec.age_group if rec.age_group != "Unknown" else "?"
                gap_lbl = rec.gap_type if rec.gap_type else ""
                label = f"ID:{tid} {gender}/{age_grp}"
                if gap_lbl:
                    label += f" [{gap_lbl}]"
            else:
                label = f"ID:{tid}"

            self._put_label(out, label, (x1, y1 - 5))

        return out

    def write_frame(self, frame: np.ndarray) -> None:
        """Write an annotated frame to the output video."""
        if self._writer_ok and self._writer is not None:
            self._writer.write(frame)

    def release(self) -> None:
        """Finalise and close the video file."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    # ── Internal ──────────────────────────────────────────────────────────

    def _put_label(
        self, img: np.ndarray, text: str, pos: tuple, font_scale: float = 0.5
    ) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        x, y = pos
        y = max(y, th + 4)
        # Background rectangle
        cv2.rectangle(img, (x, y - th - 4), (x + tw, y + baseline), self._COLOUR_BG, -1)
        cv2.putText(img, text, (x, y), font, font_scale, self._COLOUR_LABEL, thickness)
