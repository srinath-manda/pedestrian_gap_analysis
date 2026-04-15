"""main.py — Pedestrian Gap Acceptance Analysis pipeline orchestrator."""

import argparse
import os
import sys
import time

import numpy as np

import pedestrian_gap_analysis.config as cfg
from pedestrian_gap_analysis.annotator import Annotator
from pedestrian_gap_analysis.attribute_classifier import AttributeClassifier
from pedestrian_gap_analysis.dataset_exporter import DatasetExporter
from pedestrian_gap_analysis.detector import Detector
from pedestrian_gap_analysis.gap_classifier import GapClassifier
from pedestrian_gap_analysis.logit_model import LogitModel
from pedestrian_gap_analysis.platoon_detector import PlatoonDetector
from pedestrian_gap_analysis.record_store import RecordStore
from pedestrian_gap_analysis.road_region import RoadRegionSelector
from pedestrian_gap_analysis.tracker import PedestrianTracker
from pedestrian_gap_analysis.vehicle_metrics import VehicleMetricsExtractor
from pedestrian_gap_analysis.video_loader import VideoLoader
from pedestrian_gap_analysis.visualizer import Visualizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pedestrian Gap Acceptance Analysis Pipeline"
    )
    parser.add_argument(
        "--video", default=cfg.VIDEO_PATH,
        help="Path to input video file (default: %(default)s)"
    )
    parser.add_argument(
        "--output", default=cfg.OUTPUT_DIR,
        help="Output directory (default: %(default)s)"
    )
    parser.add_argument(
        "--conf", type=float, default=cfg.YOLO_CONF_THRESHOLD,
        help="YOLO confidence threshold (default: %(default)s)"
    )
    parser.add_argument(
        "--device", default=cfg.DEVICE,
        help="Compute device: '0' for GPU 0, 'cpu' for CPU (default: %(default)s)"
    )
    parser.add_argument(
        "--polygon", default=None,
        help="Pre-defined polygon as space-separated x,y pairs e.g. '100,200 300,200 300,400 100,400'"
    )
    return parser.parse_args()


def main(
    video_path: str | None = None,
    output_dir: str | None = None,
    conf_threshold: float | None = None,
    road_polygon: np.ndarray | None = None,
    device: str | None = None,
) -> dict:
    """
    Run the full pipeline.

    Parameters
    ----------
    video_path     : override VIDEO_PATH from config / CLI
    output_dir     : override OUTPUT_DIR from config / CLI
    conf_threshold : override YOLO_CONF_THRESHOLD
    road_polygon   : pre-defined polygon (skips interactive selection — for tests)

    Returns
    -------
    dict with keys: csv_path, summary_path, record_count
    """
    # ── Resolve config ────────────────────────────────────────────────────
    vpath = video_path or cfg.VIDEO_PATH
    odir = output_dir or cfg.OUTPUT_DIR
    conf = conf_threshold if conf_threshold is not None else cfg.YOLO_CONF_THRESHOLD
    dev = device if device is not None else cfg.DEVICE
    os.makedirs(odir, exist_ok=True)

    # ── GPU check ─────────────────────────────────────────────────────────
    try:
        import torch
        if dev != "cpu" and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(int(dev) if dev.isdigit() else 0)
            print(f"[Pipeline] GPU detected: {gpu_name} — using device '{dev}'")
        elif dev != "cpu":
            print("[Pipeline] WARNING: CUDA not available, falling back to CPU")
            dev = "cpu"
        else:
            print("[Pipeline] Running on CPU")
    except Exception:
        dev = "cpu"

    # ── Step 1: Load video ────────────────────────────────────────────────
    loader = VideoLoader(vpath)
    loader.open()
    fps = loader.fps
    width = loader.width
    height = loader.height
    print(f"[Pipeline] Video: {vpath}  FPS={fps:.1f}  {width}x{height}  "
          f"Frames={loader.frame_count}")

    # ── Step 2: Road region ───────────────────────────────────────────────
    if road_polygon is None:
        ret, first_frame = loader.read_frame()
        if not ret:
            sys.exit("[Pipeline] ERROR: Could not read first frame.")
        selector = RoadRegionSelector()
        polygon = selector.select(first_frame)
        # Reopen to process from frame 0
        loader.release()
        loader = VideoLoader(vpath)
        loader.open()
    else:
        polygon = road_polygon

    # ── Step 3: Initialise components ─────────────────────────────────────
    detector = Detector(cfg.YOLO_MODEL, conf, cfg.YOLO_CLASSES, device=dev)
    tracker = PedestrianTracker(cfg.YOLO_MODEL, cfg.BYTETRACK_MAX_AGE, device=dev)
    attr_clf = AttributeClassifier()
    gap_clf = GapClassifier(fps, cfg.ROLLING_GAP_SPEED_RATIO, cfg.ROLLING_GAP_MIN_DURATION)
    veh_metrics = VehicleMetricsExtractor(fps)
    platoon_det = PlatoonDetector()
    record_store = RecordStore()
    annotator = Annotator(odir, fps, width, height, cfg.OUTPUT_VIDEO_CODEC)

    selector_obj = RoadRegionSelector()

    # Track IDs that were inside the region in the previous frame
    prev_in_region: set[int] = set()

    # ── Step 4: Frame loop ────────────────────────────────────────────────
    frame_idx = 0
    total_frames = loader.frame_count
    start_time = time.time()
    print("[Pipeline] Processing frames…")

    while True:
        ret, frame = loader.read_frame()
        if not ret:
            break

        if frame_idx % 300 == 0 and frame_idx > 0:
            elapsed = time.time() - start_time
            fps_proc = frame_idx / elapsed
            if total_frames > 0:
                remaining = (total_frames - frame_idx) / fps_proc if fps_proc > 0 else 0
                pct = frame_idx / total_frames * 100
                print(
                    f"  Frame {frame_idx}/{total_frames} ({pct:.1f}%) | "
                    f"{fps_proc:.1f} fps | ETA {remaining/60:.1f} min"
                )
            else:
                # MTS files may not report total frame count
                print(
                    f"  Frame {frame_idx} | {fps_proc:.1f} fps | "
                    f"Elapsed {elapsed/60:.1f} min"
                )

        # Detect
        detections = detector.detect(frame)

        # ── Filter out persons riding on vehicles (IoU overlap check) ────
        veh_dets = [d for d in detections if d.class_id in cfg.VEHICLE_CLASSES]
        ped_dets_raw = [d for d in detections if d.class_id == 0]

        def _bbox_overlap_ratio(pb, vb):
            """Return what fraction of person bbox pb is covered by vehicle bbox vb."""
            ix1 = max(pb[0], vb[0]); iy1 = max(pb[1], vb[1])
            ix2 = min(pb[2], vb[2]); iy2 = min(pb[3], vb[3])
            if ix2 <= ix1 or iy2 <= iy1:
                return 0.0
            inter = (ix2 - ix1) * (iy2 - iy1)
            p_area = max((pb[2]-pb[0]) * (pb[3]-pb[1]), 1)
            return inter / p_area

        # Discard any person whose bbox is >50% covered by a vehicle bbox
        ped_dets = []
        for pd_ in ped_dets_raw:
            on_vehicle = any(
                _bbox_overlap_ratio(pd_.bbox, vd.bbox) > 0.50
                for vd in veh_dets
            )
            if not on_vehicle:
                ped_dets.append(pd_)

        # Track only clean pedestrian detections
        tracks = tracker.update(ped_dets, frame)

        # Spatial filter — pedestrian tracks in region
        ped_in_region: set[int] = set()
        for t in tracks:
            if selector_obj.point_in_region(t.centroid, polygon):
                ped_in_region.add(t.track_id)

        # Vehicle tracks in region
        # veh_dets already computed above
        veh_in_region: set[int] = set()
        veh_track_list = []

        # Use detection index as a pseudo vehicle ID (no persistent tracking for vehicles)
        for i, vd in enumerate(veh_dets):
            cx = (vd.bbox[0] + vd.bbox[2]) / 2
            cy = (vd.bbox[1] + vd.bbox[3]) / 2
            if selector_obj.point_in_region((cx, cy), polygon):
                # Use a stable pseudo-ID based on class + position bucket
                pseudo_id = hash((vd.class_id, int(cx // 50), int(cy // 50))) % 100000
                veh_in_region.add(pseudo_id)

                class _VT:
                    def __init__(self, tid, centroid):
                        self.track_id = tid
                        self.centroid = centroid

                veh_track_list.append(_VT(pseudo_id, (cx, cy)))

        veh_metrics.update(frame_idx, veh_track_list, veh_in_region)

        # Platoon detection
        platoon_det.update(frame_idx, ped_in_region)

        # Per-track processing
        for t in tracks:
            in_region = t.track_id in ped_in_region
            was_in_region = t.track_id in prev_in_region

            if in_region:
                rec = record_store.get_or_create(t.track_id, frame_idx)

                # Classify attributes on first entry
                if rec.gender == "Unknown" and rec.age_group == "Unknown" and not rec.complete:
                    x1, y1, x2, y2 = (int(v) for v in t.bbox)
                    crop = frame[max(0, y1):y2, max(0, x1):x2]
                    if crop.size > 0:
                        attrs = attr_clf.classify(crop)
                        record_store.set_attributes(t.track_id, attrs.gender, attrs.age_group)

                record_store.append_trajectory(t.track_id, t.centroid)

            elif was_in_region:
                # Track just exited — finalise
                rec = record_store.get(t.track_id)
                if rec and not rec.complete:
                    gap_type = gap_clf.classify(rec.trajectory)
                    gap_m = veh_metrics.get_gap_at_frame(rec.entry_frame)
                    record_store.set_gap_metrics(
                        t.track_id,
                        gap_m.gap_seconds,
                        gap_m.time_headway,
                        gap_m.vehicle_speed_px_per_s,
                    )
                    record_store.set_platoon(
                        t.track_id, platoon_det.get_platoon_flag(t.track_id)
                    )
                    record_store.finalise(t.track_id, frame_idx, gap_type)

        prev_in_region = set(ped_in_region)

        # Annotate and write frame
        # Only show tracks that are in region OR have previously entered region
        polygon_tracks = [t for t in tracks
                          if t.track_id in ped_in_region
                          or record_store.get(t.track_id) is not None]
        annotated = annotator.annotate_frame(frame, polygon_tracks, polygon, record_store)
        annotator.write_frame(annotated)

        frame_idx += 1

    loader.release()
    annotator.release()
    print(f"[Pipeline] Processed {frame_idx} frames.")

    # ── Step 5: Export dataset ────────────────────────────────────────────
    complete_records = record_store.get_complete_records()
    print(f"[Pipeline] Complete crossing records: {len(complete_records)}")

    exporter = DatasetExporter()
    csv_path = exporter.export(complete_records, odir)
    print(f"[Pipeline] Dataset saved → {csv_path}")

    # ── Step 6: Logistic regression ───────────────────────────────────────
    summary_path = None
    odds_df = None
    if len(complete_records) > 0:
        try:
            import pandas as pd
            lm = LogitModel()
            results = lm.fit(csv_path)
            summary_path = lm.save_summary(results, odir)
            odds_df = lm.get_odds_ratios(results)
            print(f"[Pipeline] Model summary saved → {summary_path}")
        except Exception as exc:
            print(f"[Pipeline] WARNING: Logistic regression failed: {exc}")

    # ── Step 7: Visualisations ────────────────────────────────────────────
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        viz = Visualizer(odir, cfg.PLOT_DPI)
        if odds_df is not None:
            viz.generate_all(df, odds_df)
        else:
            # Generate plots that don't need the model
            viz.plot_gap_type_distribution(df)
            viz.plot_gender_vs_gap_type(df)
            viz.plot_age_group_vs_gap_type(df)
            viz.plot_platoon_vs_gap_type(df)
            viz.plot_gap_duration_boxplot(df)
        print(f"[Pipeline] Plots saved to {odir}")
    except Exception as exc:
        print(f"[Pipeline] WARNING: Visualisation failed: {exc}")

    print("[Pipeline] Done.")
    return {
        "csv_path": csv_path,
        "summary_path": summary_path,
        "record_count": len(complete_records),
    }


if __name__ == "__main__":
    args = parse_args()

    # Parse --polygon if provided
    pre_polygon = None
    if args.polygon:
        try:
            pts = [tuple(int(v) for v in p.split(",")) for p in args.polygon.strip().split()]
            pre_polygon = np.array(pts, dtype=np.int32)
            print(f"[Pipeline] Using pre-defined polygon: {pts}")
        except Exception as e:
            sys.exit(f"[Pipeline] ERROR: Invalid --polygon format: {e}\n"
                     "Expected format: '100,200 300,200 300,400 100,400'")

    main(video_path=args.video, output_dir=args.output,
         conf_threshold=args.conf, device=args.device,
         road_polygon=pre_polygon)
