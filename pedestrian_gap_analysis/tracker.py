"""tracker.py — ByteTrack-based pedestrian tracker using external detections."""

from dataclasses import dataclass
import numpy as np

from pedestrian_gap_analysis.detector import Detection


@dataclass
class Track:
    track_id: int
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    centroid: tuple[float, float]


class PedestrianTracker:
    """
    Runs ByteTrack on externally-provided pedestrian detections.
    Does NOT re-run YOLO — uses the detections passed in directly.
    This ensures person-on-vehicle filtering applied upstream is respected.
    """

    def __init__(self, model_path: str = "yolov8n.pt", max_age: int = 30, device: str = "0") -> None:
        self._max_age = max_age
        self._device = device
        self._tracker = None
        self._next_id = 1
        self._tracks: dict[int, dict] = {}   # id → {bbox, conf, lost_frames}
        self._iou_threshold = 0.3

    def update(
        self, detections: list[Detection], frame: np.ndarray
    ) -> list[Track]:
        """
        Match detections to existing tracks using IoU, assign IDs.
        Returns active Track objects for this frame.
        """
        # Filter to pedestrians only (safety check)
        ped_dets = [d for d in detections if d.class_id == 0]

        if not ped_dets:
            # Age out all tracks
            lost = [tid for tid, t in self._tracks.items()
                    if t["lost"] >= self._max_age]
            for tid in lost:
                del self._tracks[tid]
            for t in self._tracks.values():
                t["lost"] += 1
            return []

        det_boxes = np.array([d.bbox for d in ped_dets], dtype=np.float32)

        # Try to use ultralytics ByteTrack if available
        try:
            return self._update_bytetrack(ped_dets, frame)
        except Exception:
            return self._update_iou(ped_dets, det_boxes)

    def _update_bytetrack(self, ped_dets: list[Detection], frame: np.ndarray) -> list[Track]:
        """Use ultralytics ByteTrack via custom tracker on pre-filtered detections."""
        from ultralytics import YOLO
        from ultralytics.trackers.byte_tracker import BYTETracker
        from types import SimpleNamespace

        if self._tracker is None:
            args = SimpleNamespace(
                track_high_thresh=0.25,
                track_low_thresh=0.1,
                new_track_thresh=0.25,
                track_buffer=self._max_age,
                match_thresh=0.8,
                fuse_score=True,
            )
            self._tracker = BYTETracker(args, frame_rate=25)

        # Build detection array: [x1,y1,x2,y2,conf,cls]
        det_array = np.array(
            [[d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3], d.confidence, 0]
             for d in ped_dets],
            dtype=np.float32,
        )

        h, w = frame.shape[:2]
        online_targets = self._tracker.update(det_array, (h, w), (h, w))

        tracks = []
        for t in online_targets:
            x1, y1, x2, y2 = t.tlbr
            tracks.append(Track(
                track_id=int(t.track_id),
                bbox=(float(x1), float(y1), float(x2), float(y2)),
                confidence=float(t.score),
                centroid=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
            ))
        return tracks

    def _update_iou(self, ped_dets: list[Detection], det_boxes: np.ndarray) -> list[Track]:
        """Fallback: simple IoU-based tracker."""
        matched_det = set()
        matched_track = set()

        track_ids = list(self._tracks.keys())
        if track_ids:
            track_boxes = np.array([self._tracks[tid]["bbox"] for tid in track_ids])
            iou_matrix = self._iou_batch(det_boxes, track_boxes)

            for di in range(len(ped_dets)):
                best_tid_idx = int(np.argmax(iou_matrix[di]))
                if iou_matrix[di, best_tid_idx] >= self._iou_threshold:
                    tid = track_ids[best_tid_idx]
                    if tid not in matched_track:
                        matched_det.add(di)
                        matched_track.add(tid)
                        self._tracks[tid]["bbox"] = ped_dets[di].bbox
                        self._tracks[tid]["conf"] = ped_dets[di].confidence
                        self._tracks[tid]["lost"] = 0

        # New tracks for unmatched detections
        for di, det in enumerate(ped_dets):
            if di not in matched_det:
                self._tracks[self._next_id] = {
                    "bbox": det.bbox, "conf": det.confidence, "lost": 0
                }
                self._next_id += 1

        # Age out lost tracks
        lost = [tid for tid, t in self._tracks.items()
                if t["lost"] >= self._max_age]
        for tid in lost:
            del self._tracks[tid]
        for tid in set(self._tracks) - matched_track:
            self._tracks[tid]["lost"] += 1

        tracks = []
        for tid, t in self._tracks.items():
            if t["lost"] == 0:
                x1, y1, x2, y2 = t["bbox"]
                tracks.append(Track(
                    track_id=tid,
                    bbox=t["bbox"],
                    confidence=t["conf"],
                    centroid=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                ))
        return tracks

    @staticmethod
    def _iou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
        """Compute IoU matrix between two sets of boxes."""
        ax1, ay1, ax2, ay2 = boxes_a[:, 0:1], boxes_a[:, 1:2], boxes_a[:, 2:3], boxes_a[:, 3:4]
        bx1, by1, bx2, by2 = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]

        ix1 = np.maximum(ax1, bx1)
        iy1 = np.maximum(ay1, by1)
        ix2 = np.minimum(ax2, bx2)
        iy2 = np.minimum(ay2, by2)

        inter = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter
        return inter / np.maximum(union, 1e-6)
