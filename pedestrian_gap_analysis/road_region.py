"""road_region.py — RoadRegionSelector: interactive polygon definition on first frame."""

import cv2
import numpy as np


class RoadRegionSelector:
    """
    Displays the first video frame and lets the user click polygon vertices.
    Left-click  → add vertex
    Right-click → remove last vertex
    Enter/Space → confirm (requires ≥ 3 vertices)
    R           → reset all vertices
    """

    def select(self, frame: np.ndarray) -> np.ndarray:
        """
        Show the frame and collect polygon vertices interactively.
        Loops until the user confirms a polygon with ≥ 3 vertices.
        Returns an (N, 2) int32 numpy array of [x, y] vertices.
        """
        vertices: list[tuple[int, int]] = []
        window = "Define Road Region — Left-click to add, Enter to confirm, R to reset"

        def mouse_cb(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                vertices.append((x, y))
            elif event == cv2.EVENT_RBUTTONDOWN and vertices:
                vertices.pop()

        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window, mouse_cb)

        while True:
            display = frame.copy()
            # Draw existing vertices and edges
            for i, pt in enumerate(vertices):
                cv2.circle(display, pt, 5, (0, 255, 0), -1)
                if i > 0:
                    cv2.line(display, vertices[i - 1], pt, (0, 255, 0), 2)
            if len(vertices) >= 3:
                cv2.line(display, vertices[-1], vertices[0], (0, 255, 0), 2)
                cv2.putText(
                    display,
                    f"Vertices: {len(vertices)} — Press Enter to confirm",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
            else:
                cv2.putText(
                    display,
                    f"Vertices: {len(vertices)} — Need at least 3",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

            cv2.imshow(window, display)
            key = cv2.waitKey(20) & 0xFF

            if key in (13, 32):  # Enter or Space
                if len(vertices) >= 3:
                    break
                # else keep looping — not enough vertices
            elif key == ord("r") or key == ord("R"):
                vertices.clear()

        cv2.destroyWindow(window)
        return np.array(vertices, dtype=np.int32)

    @staticmethod
    def point_in_region(point: tuple[float, float], polygon: np.ndarray) -> bool:
        """
        Returns True if `point` (x, y) is inside or on the boundary of `polygon`.
        Uses cv2.pointPolygonTest (returns ≥ 0 for inside/on-boundary).
        """
        poly = polygon.reshape((-1, 1, 2)).astype(np.float32)
        result = cv2.pointPolygonTest(poly, (float(point[0]), float(point[1])), False)
        return result >= 0
