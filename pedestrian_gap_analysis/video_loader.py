"""video_loader.py — VideoLoader: opens a video file and exposes frame iteration."""

import sys
import cv2
import numpy as np


class VideoLoader:
    def __init__(self, path: str) -> None:
        self._path = path
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        """Open the video file. Exits with a descriptive message on failure."""
        cap = cv2.VideoCapture(self._path)
        if not cap.isOpened():
            sys.exit(
                f"[VideoLoader] ERROR: Cannot open video file '{self._path}'. "
                "Check that the path exists and the file is a valid video."
            )
        self._cap = cap

    # ── Metadata properties (available after open()) ──────────────────────

    @property
    def fps(self) -> float:
        self._require_open()
        return self._cap.get(cv2.CAP_PROP_FPS)

    @property
    def frame_count(self) -> int:
        self._require_open()
        count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # MTS/AVCHD files sometimes report -1 — return 0 as safe fallback
        return max(count, 0)

    @property
    def width(self) -> int:
        self._require_open()
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        self._require_open()
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ── Frame iteration ───────────────────────────────────────────────────

    def read_frame(self) -> tuple[bool, np.ndarray | None]:
        """Read the next frame. Returns (True, frame) or (False, None) at end."""
        self._require_open()
        ret, frame = self._cap.read()
        if not ret:
            return False, None
        return True, frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # ── Internal ──────────────────────────────────────────────────────────

    def _require_open(self) -> None:
        if self._cap is None:
            raise RuntimeError("[VideoLoader] Video is not open. Call open() first.")
