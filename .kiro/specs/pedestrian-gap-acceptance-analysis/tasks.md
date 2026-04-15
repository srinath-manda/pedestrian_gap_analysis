# Implementation Plan: Pedestrian Gap Acceptance Analysis

## Overview

Implement the full pedestrian gap acceptance analysis pipeline in Python. The pipeline processes a 2-hour field video, detects and tracks pedestrians and vehicles, classifies crossing behavior and demographics, extracts traffic metrics, exports a structured dataset, fits a Binary Logistic Regression model, and produces annotated video and visualization plots. All 15 modules are implemented incrementally, each wired into the pipeline before moving to the next.

## Tasks

- [x] 1. Project setup — directory structure and dependencies
  - Create the `pedestrian_gap_analysis/` package directory with `__init__.py`
  - Create `pedestrian_gap_analysis/tests/` directory with `__init__.py`
  - Create `requirements.txt` with all dependencies:
    `ultralytics`, `deepface`, `statsmodels`, `pandas`, `numpy`, `opencv-python`, `matplotlib`, `scipy`, `pytest`, `hypothesis`
  - _Requirements: all_

- [x] 2. Implement `config.py` — centralized constants
  - [x] 2.1 Create `pedestrian_gap_analysis/config.py` with all constants
    - Define `VIDEO_PATH`, `OUTPUT_DIR`, `YOLO_MODEL`, `YOLO_CONF_THRESHOLD`, `YOLO_CLASSES`
    - Define `BYTETRACK_MAX_AGE`, `ROLLING_GAP_SPEED_RATIO`, `ROLLING_GAP_MIN_DURATION`
    - Define `AGE_YOUNG_MAX`, `AGE_MIDDLE_MAX`, `MIN_RECORDS_FOR_REGRESSION`, `PLOT_DPI`, `OUTPUT_VIDEO_CODEC`
    - _Requirements: all (configuration)_

- [x] 3. Implement `video_loader.py` — `VideoLoader`
  - [x] 3.1 Create `pedestrian_gap_analysis/video_loader.py` with `VideoLoader` class
    - Implement `__init__`, `open`, `fps`, `frame_count`, `width`, `height`, `read_frame`, `release`
    - `open()` raises `SystemExit` with descriptive message if file not found or unreadable
    - Expose metadata before processing begins
    - _Requirements: 1.1, 1.2, 1.3_
  - [x]* 3.2 Write property test for VideoLoader metadata accuracy
    - **Property 1: VideoLoader metadata accuracy**
    - Generate synthetic video files with known fps/dimensions using OpenCV `VideoWriter`
    - Assert `loader.fps`, `loader.frame_count`, `loader.width`, `loader.height` match the synthetic video's actual values
    - **Validates: Requirements 1.3**
  - [x]* 3.3 Write unit tests for VideoLoader
    - Test successful open and metadata exposure
    - Test `SystemExit` raised for missing file path
    - Test `read_frame` returns `(True, ndarray)` for valid frames and `(False, None)` at end
    - _Requirements: 1.1, 1.2, 1.3_

- [x] 4. Implement `road_region.py` — `RoadRegionSelector`
  - [x] 4.1 Create `pedestrian_gap_analysis/road_region.py` with `RoadRegionSelector` class
    - Implement `select(frame)` using OpenCV `setMouseCallback`; loop until ≥ 3 vertices collected
    - Implement `point_in_region(point, polygon)` using `cv2.pointPolygonTest`
    - Enforce minimum 3 vertices before returning (re-prompt if fewer)
    - _Requirements: 4.1, 4.2, 4.3, 4.4_
  - [x]* 4.2 Write property test for spatial filtering correctness
    - **Property 4: Spatial filtering correctness**
    - Generate random convex polygons (≥ 3 vertices) and random 2D points
    - Assert `point_in_region` returns `True` for points strictly inside and `False` for points strictly outside, consistent with `cv2.pointPolygonTest`
    - **Validates: Requirements 4.3**
  - [x]* 4.3 Write unit tests for RoadRegionSelector
    - Test `point_in_region` with known inside/outside/boundary points for a fixed polygon
    - _Requirements: 4.3_

- [x] 5. Implement `detector.py` — `Detector` and `Detection`
  - [x] 5.1 Create `pedestrian_gap_analysis/detector.py` with `Detection` dataclass and `Detector` class
    - `Detection` fields: `bbox`, `confidence`, `class_id`, `class_name`
    - `Detector.__init__` loads YOLOv8n with pretrained COCO weights
    - `detect(frame)` runs inference, filters to `target_classes`, returns `list[Detection]`
    - Returns empty list when no target-class objects detected
    - _Requirements: 2.1, 2.2, 2.3, 2.4_
  - [ ]* 5.2 Write unit tests for Detector
    - Test `detect` returns empty list for a blank frame (no detections)
    - Test returned `Detection` objects have correct field types
    - Mock YOLOv8 inference to avoid requiring GPU in CI
    - _Requirements: 2.1, 2.3_

- [x] 6. Implement `tracker.py` — `PedestrianTracker` and `Track`
  - [x] 6.1 Create `pedestrian_gap_analysis/tracker.py` with `Track` dataclass and `PedestrianTracker` class
    - `Track` fields: `track_id`, `bbox`, `confidence`, `centroid`
    - `PedestrianTracker.update(detections, frame)` wraps ByteTrack via `ultralytics`
    - Only pedestrian detections (class_id == 0) are passed to the tracker
    - Returns active `list[Track]` per frame
    - _Requirements: 3.1, 3.2, 3.3, 3.4_
  - [ ]* 6.2 Write property test for track output structure completeness
    - **Property 3: Track output structure completeness**
    - For any non-empty list of pedestrian detections, every `Track` in the returned list shall have non-negative integer `track_id`, 4-element float `bbox`, float `confidence` in [0,1], and 2-element float `centroid`
    - **Validates: Requirements 3.3**
  - [ ]* 6.3 Write unit tests for PedestrianTracker
    - Test that non-pedestrian detections are filtered out before tracking
    - Test centroid is computed as center of bbox
    - _Requirements: 3.1, 3.3_

- [x] 7. Implement `attribute_classifier.py` — `AttributeClassifier` and `PedestrianAttributes`
  - [x] 7.1 Create `pedestrian_gap_analysis/attribute_classifier.py`
    - `PedestrianAttributes` dataclass: `gender`, `age_group`
    - `classify(crop)` calls DeepFace; catches all exceptions and returns `"Unknown"` for failed fields
    - `age_to_group(age)` static method: Young ≤ 29, Middle 30–59, Old ≥ 60
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_
  - [x]* 7.2 Write property test for gender output domain
    - **Property 5: Gender output domain**
    - For any image crop (including random noise arrays), `classify().gender` shall always be one of `{"Male", "Female", "Unknown"}`
    - Mock DeepFace to simulate both success and exception paths
    - **Validates: Requirements 5.2**
  - [x]* 7.3 Write property test for age group bucketing correctness
    - **Property 6: Age group bucketing correctness**
    - For any integer age in [0, 150], `age_to_group(age)` returns `"Young"` if age ≤ 29, `"Middle"` if 30 ≤ age ≤ 59, `"Old"` if age ≥ 60
    - Output is always one of `{"Young", "Middle", "Old"}`
    - **Validates: Requirements 5.3**
  - [x]* 7.4 Write unit tests for AttributeClassifier
    - Test `age_to_group` boundary values: 0, 29, 30, 59, 60, 150
    - Test `classify` returns `"Unknown"` when DeepFace raises an exception
    - _Requirements: 5.3, 5.4_

- [x] 8. Implement `gap_classifier.py` — `GapClassifier`
  - [x] 8.1 Create `pedestrian_gap_analysis/gap_classifier.py` with `GapClassifier` class
    - `compute_speeds(trajectory)` returns list of N−1 Euclidean displacements
    - `classify(trajectory)` implements the speed-drop algorithm: threshold = 0.20 × mean_speed; rolling if any contiguous run ≥ ceil(0.5 × fps) frames
    - Edge cases: empty/single-point trajectory → `"Straight"`; zero mean speed → `"Straight"`
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_
  - [x]* 8.2 Write property test for speed sequence computation
    - **Property 8: Speed sequence computation**
    - For any trajectory of N ≥ 2 centroids, `compute_speeds()` returns exactly N−1 non-negative floats where each equals the Euclidean distance between consecutive pairs
    - **Validates: Requirements 6.2**
  - [x]* 8.3 Write property test for gap classification correctness
    - **Property 9: Gap classification correctness**
    - For any trajectory, if speed sequence contains a contiguous run of frames with speed < 20% of mean lasting ≥ ceil(0.5 × fps) frames → result is `"Rolling"`; otherwise → `"Straight"`
    - Result is always one of `{"Straight", "Rolling"}`
    - **Validates: Requirements 6.3, 6.4**
  - [x]* 8.4 Write unit tests for GapClassifier
    - Test straight trajectory (constant speed) → `"Straight"`
    - Test trajectory with clear pause → `"Rolling"`
    - Test single-point and empty trajectory → `"Straight"`
    - Test zero-speed trajectory → `"Straight"`
    - _Requirements: 6.2, 6.3, 6.4_

- [x] 9. Implement `vehicle_metrics.py` — `VehicleMetricsExtractor`, `VehicleEvent`, `GapMetrics`
  - [x] 9.1 Create `pedestrian_gap_analysis/vehicle_metrics.py`
    - `VehicleEvent` dataclass: `frame_idx`, `timestamp`, `event_type`, `vehicle_id`
    - `GapMetrics` dataclass: `gap_seconds`, `time_headway`, `vehicle_speed_px_per_s`
    - `update(frame_idx, vehicle_tracks, in_region)` records entry/exit events per vehicle ID
    - `get_gap_at_frame(frame_idx)` returns `GapMetrics` for the gap period containing that frame
    - `compute_time_headway()` returns mean inter-entry interval; 0.0 if fewer than 2 entries
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_
  - [x]* 9.2 Write property test for vehicle event recording correctness
    - **Property 10: Vehicle event recording correctness**
    - For any sequence of per-frame vehicle presence updates, an `"entry"` event is recorded exactly when a vehicle ID first appears and an `"exit"` event exactly when it last appears — no duplicate events per entry/exit cycle
    - **Validates: Requirements 7.1**
  - [x]* 9.3 Write property test for gap duration computation
    - **Property 11: Gap duration computation**
    - For any ordered list of vehicle entry/exit timestamps, gap duration for a given window equals the time between the most recent vehicle exit before the window and the next vehicle entry after the window start
    - **Validates: Requirements 7.2**
  - [x]* 9.4 Write property test for time headway computation
    - **Property 12: Time headway computation**
    - For any list of N entry timestamps with N ≥ 2, `compute_time_headway()` returns the arithmetic mean of consecutive differences; for N < 2 returns 0.0
    - **Validates: Requirements 7.3**
  - [x]* 9.5 Write property test for vehicle speed computation
    - **Property 13: Vehicle speed computation**
    - For any two consecutive centroids and known fps, estimated speed equals `sqrt((cx2−cx1)² + (cy2−cy1)²) × fps`
    - **Validates: Requirements 7.4**
  - [x]* 9.6 Write unit tests for VehicleMetricsExtractor
    - Test entry/exit event recording for a vehicle that enters and exits
    - Test gap duration when no vehicles present
    - Test time headway with 0, 1, and 2+ entry events
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

- [x] 10. Implement `platoon_detector.py` — `PlatoonDetector`
  - [x] 10.1 Create `pedestrian_gap_analysis/platoon_detector.py` with `PlatoonDetector` class
    - `update(frame_idx, track_ids_in_region)` records co-presence per frame
    - `get_platoon_flag(track_id)` returns `"Group"` if ever co-present with another track, else `"Alone"`
    - Once flagged `"Group"`, a track retains that flag permanently
    - _Requirements: 8.1, 8.2, 8.3_
  - [x]* 10.2 Write property test for platoon detection correctness
    - **Property 14: Platoon detection correctness**
    - For any sequence of frames, if a track_id was present in any frame with ≥ 1 other track → `"Group"`; if never co-present → `"Alone"`; result always one of `{"Group", "Alone"}`
    - **Validates: Requirements 8.1, 8.2**
  - [x]* 10.3 Write unit tests for PlatoonDetector
    - Test solo pedestrian → `"Alone"`
    - Test two simultaneous pedestrians → both `"Group"`
    - Test track that starts solo then joins a group → `"Group"`
    - _Requirements: 8.1, 8.2, 8.3_

- [x] 11. Implement `record_store.py` — `RecordStore` and `PedestrianRecord`
  - [x] 11.1 Create `pedestrian_gap_analysis/record_store.py`
    - `PedestrianRecord` dataclass with all fields: `track_id`, `gender`, `age_group`, `platoon`, `gap_seconds`, `time_headway`, `vehicle_speed`, `gap_type`, `gap_type_binary`, `trajectory`, `entry_frame`, `exit_frame`, `complete`
    - `RecordStore` with methods to create/update records and retrieve complete records
    - `get_complete_records()` returns only records with `complete=True`
    - _Requirements: 5.5, 6.5, 8.3, 9.5_
  - [x]* 11.2 Write property test for RecordStore attribute round-trip
    - **Property 7: RecordStore attribute round-trip**
    - For any track_id and any combination of gender, age_group, platoon, gap_type values stored in RecordStore, retrieving the record returns exactly the same values — no mutation, no loss
    - **Validates: Requirements 5.5, 6.5, 8.3**
  - [x]* 11.3 Write unit tests for RecordStore
    - Test that incomplete records are excluded from `get_complete_records()`
    - Test that stored attributes are retrieved unchanged
    - _Requirements: 9.5_

- [x] 12. Implement `dataset_exporter.py` — `DatasetExporter`
  - [x] 12.1 Create `pedestrian_gap_analysis/dataset_exporter.py` with `DatasetExporter` class
    - `export(records, output_dir)` writes only `complete=True` records to `gap_acceptance_dataset.csv`
    - CSV columns: `track_id`, `gender`, `age_group`, `platoon`, `gap_seconds`, `time_headway`, `vehicle_speed`, `gap_type`
    - `gap_type` encoded as binary integer: 1 = Straight, 0 = Rolling
    - Returns path to written CSV file
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_
  - [x]* 12.2 Write property test for CSV export completeness and filtering
    - **Property 15: CSV export completeness and filtering**
    - For any list of PedestrianRecord objects with mixed `complete` flags, exported CSV has exactly as many rows as records with `complete=True`, and no row for `complete=False` records
    - **Validates: Requirements 9.1, 9.5**
  - [x]* 12.3 Write property test for CSV schema and binary encoding correctness
    - **Property 16: CSV schema and binary encoding correctness**
    - For any list of complete records, CSV contains exactly the 8 required columns, and `gap_type` is 1 for `"Straight"` and 0 for `"Rolling"` in every row
    - **Validates: Requirements 9.2, 9.3**
  - [x]* 12.4 Write unit tests for DatasetExporter
    - Test output file is named `gap_acceptance_dataset.csv`
    - Test all 8 required columns are present
    - Test binary encoding of gap_type
    - Test incomplete records are excluded
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

- [x] 13. Checkpoint — core pipeline modules complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 14. Implement `logit_model.py` — `LogitModel`
  - [x] 14.1 Create `pedestrian_gap_analysis/logit_model.py` with `LogitModel` class
    - `fit(csv_path)` loads CSV, encodes categoricals with `pd.get_dummies(drop_first=True)`, fits `statsmodels` Logit
    - Warns if record count < 30 before fitting
    - `save_summary(results, output_dir)` writes `logit_model_summary.txt`
    - `get_odds_ratios(results)` returns DataFrame with predictor, coefficient, odds_ratio columns
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6_
  - [x]* 14.2 Write property test for odds ratio computation
    - **Property 17: Odds ratio computation**
    - For any fitted LogitModel, odds ratio for each predictor equals `exp(coefficient)` from the statsmodels summary; odds ratio is always a positive float
    - Use synthetic CSV data to fit the model
    - **Validates: Requirements 10.4**
  - [x]* 14.3 Write unit tests for LogitModel
    - Test warning is printed when fewer than 30 records
    - Test `logit_model_summary.txt` is created in output directory
    - Test `get_odds_ratios` returns positive floats for all predictors
    - Use a small synthetic CSV (≥ 30 rows) for fitting tests
    - _Requirements: 10.1, 10.3, 10.5, 10.6_

- [x] 15. Implement `visualizer.py` — `Visualizer`
  - [x] 15.1 Create `pedestrian_gap_analysis/visualizer.py` with `Visualizer` class
    - Implement all 6 plot methods: `plot_gap_type_distribution`, `plot_gender_vs_gap_type`, `plot_age_group_vs_gap_type`, `plot_platoon_vs_gap_type`, `plot_gap_duration_boxplot`, `plot_odds_ratios`
    - Each plot: non-empty title, x-axis label, y-axis label, legend
    - `generate_all(df, odds_df)` calls all 6 methods
    - Save each plot as PNG at ≥ 150 DPI to output directory
    - _Requirements: 11.1, 11.2, 11.3_
  - [x]* 15.2 Write property test for plot metadata completeness
    - **Property 18: Plot metadata completeness**
    - For any valid DataFrame passed to any Visualizer plot method, the resulting matplotlib Figure has non-empty title, non-empty x-axis label, non-empty y-axis label, and at least one legend entry on the primary axes
    - Test all 6 plot methods
    - **Validates: Requirements 11.2**
  - [x]* 15.3 Write unit tests for Visualizer
    - Test all 6 PNG files are created in output directory
    - Test saved files are non-empty
    - Use a minimal synthetic DataFrame
    - _Requirements: 11.1, 11.3_

- [x] 16. Implement `annotator.py` — `Annotator`
  - [x] 16.1 Create `pedestrian_gap_analysis/annotator.py` with `Annotator` class
    - `__init__` initializes `cv2.VideoWriter` for `annotated_output.mp4`; catches codec failure and logs without terminating
    - `annotate_frame(frame, tracks, polygon, record_store)` draws bounding boxes, track IDs, Road_Region polygon, and gender/age/gap_type labels
    - `write_frame(frame)` writes annotated frame to video
    - `release()` finalizes the video file
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5_
  - [x]* 16.2 Write unit tests for Annotator
    - Test `annotate_frame` returns an ndarray of the same shape as input
    - Test that codec failure is caught and logged without raising an exception
    - Test `annotated_output.mp4` is created in output directory after `release()`
    - _Requirements: 12.1, 12.4, 12.5_

- [x] 17. Implement `main.py` — pipeline orchestrator
  - [x] 17.1 Create `pedestrian_gap_analysis/main.py` wiring all components together
    - Parse args / load config
    - `VideoLoader.open()` → `RoadRegionSelector.select(first_frame)`
    - Initialize all components: `Detector`, `PedestrianTracker`, `AttributeClassifier`, `GapClassifier`, `VehicleMetricsExtractor`, `PlatoonDetector`, `RecordStore`, `Annotator`
    - Frame loop: detect → track → spatial filter → attribute classify (first entry) → update trajectory → update vehicle metrics → update platoon → annotate
    - On track exit: `GapClassifier.classify(trajectory)` → update record
    - Post-loop: `DatasetExporter.export()` → `LogitModel.fit()` → `Visualizer.generate_all()` → `Annotator.release()`
    - _Requirements: all_
  - [ ]* 17.2 Write unit tests for main pipeline wiring
    - Test that `main()` can be imported without error
    - Test argument parsing produces correct config values
    - _Requirements: all_

- [x] 18. Integration smoke test
  - [x] 18.1 Create `pedestrian_gap_analysis/tests/test_integration.py`
    - Generate a short synthetic video (20 frames, 640×480) using `cv2.VideoWriter` with a few colored rectangles simulating pedestrians and vehicles
    - Run the full pipeline end-to-end (mocking `RoadRegionSelector.select` to return a fixed polygon and `AttributeClassifier.classify` to return fixed attributes)
    - Assert all expected output files are created: `gap_acceptance_dataset.csv`, `logit_model_summary.txt`, `annotated_output.mp4`, and all 6 PNG plots
    - Assert CSV has the 8 required columns and correct data types
    - _Requirements: all_

- [x] 19. Final checkpoint — all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- Each task references specific requirements for traceability
- Property tests use Hypothesis with minimum 100 iterations per property
- Property test files are prefixed `test_properties_` in the `tests/` directory
- Unit test files are prefixed `test_` in the `tests/` directory
- All 18 correctness properties from the design document are covered by property-based tests
- Checkpoints at tasks 13 and 19 ensure incremental validation
