# Requirements Document

## Introduction

This project automates the analysis of pedestrian gap acceptance behavior at urban intersections using computer vision and statistical modeling. The input is a 2-hour field video recorded at Kompally intersection, Hyderabad. The system detects and tracks pedestrians and vehicles, classifies pedestrian attributes (gender, age group), detects platoon behavior, classifies gap type (Straight vs Rolling), extracts vehicle traffic metrics, compiles a structured dataset, and runs a Binary Logistic Regression model to identify factors influencing gap selection. All outputs — annotated video, CSV dataset, model results, and visualization plots — are suitable for academic submission and viva.

## Glossary

- **System**: The complete pedestrian gap acceptance analysis pipeline
- **Video_Loader**: The component responsible for reading and decoding the input video file using OpenCV
- **Detector**: The YOLOv8n-based object detection component that identifies persons and vehicles in each frame
- **Tracker**: The ByteTrack-based component that assigns and maintains consistent IDs for pedestrians across frames
- **Road_Region**: The user-defined polygonal area on the video frame representing the pedestrian crossing zone
- **Attribute_Classifier**: The DeepFace-based component that predicts gender and age group from cropped pedestrian bounding boxes
- **Gap_Classifier**: The trajectory analysis component that classifies each pedestrian crossing as a Straight Gap or Rolling Gap
- **Vehicle_Metrics_Extractor**: The component that computes gap duration, time headway, and estimated vehicle speed from vehicle detections
- **Platoon_Detector**: The component that identifies whether a pedestrian crossed alone or as part of a group
- **Dataset_Exporter**: The component that compiles all per-pedestrian records into a structured CSV file
- **Logit_Model**: The Binary Logistic Regression model built using statsmodels that identifies factors influencing gap type selection
- **Visualizer**: The component that generates statistical plots from the dataset and model results
- **Annotator**: The component that renders bounding boxes, track IDs, labels, and region overlays onto video frames and saves the output video
- **Straight Gap**: A crossing type where the pedestrian moves continuously through the road region without significant speed reduction
- **Rolling Gap**: A crossing type where the pedestrian pauses or slows to below 20% of their mean crossing speed for 0.5 seconds or more before completing the crossing
- **Time Headway**: The mean time interval between consecutive vehicle detections passing through the Road_Region
- **Platoon**: A group of 2 or more pedestrians simultaneously present within the Road_Region

---

## Requirements

### Requirement 1: Video Input Loading

**User Story:** As a researcher, I want the system to load the field video file, so that all subsequent processing operates on the correct input footage.

#### Acceptance Criteria

1. WHEN a valid video file path is provided, THE Video_Loader SHALL open the video and make frames available for sequential processing.
2. IF the provided video file path does not exist or cannot be opened, THEN THE Video_Loader SHALL display a descriptive error message and terminate the pipeline.
3. THE Video_Loader SHALL expose the video's frame rate, total frame count, and frame dimensions to downstream components before processing begins.

---

### Requirement 2: Object Detection

**User Story:** As a researcher, I want the system to detect pedestrians and vehicles in every frame, so that their positions and movements can be tracked and analyzed.

#### Acceptance Criteria

1. WHEN a video frame is received, THE Detector SHALL run YOLOv8n inference and return bounding boxes, confidence scores, and class labels for all detected persons, cars, buses, trucks, and motorcycles.
2. THE Detector SHALL use the pretrained COCO weights for YOLOv8n without requiring additional training.
3. WHEN no objects of the target classes are present in a frame, THE Detector SHALL return an empty detection list for that frame without raising an error.
4. THE Detector SHALL process every frame of the input video in sequential order.

---

### Requirement 3: Pedestrian Tracking

**User Story:** As a researcher, I want each pedestrian to be assigned a consistent ID across frames, so that per-pedestrian behavioral metrics can be computed over time.

#### Acceptance Criteria

1. WHEN pedestrian detections are received from the Detector, THE Tracker SHALL assign a unique integer track ID to each pedestrian and maintain that ID across consecutive frames.
2. WHEN a pedestrian temporarily leaves the frame and re-enters within the ByteTrack re-identification window, THE Tracker SHALL reassign the same track ID to that pedestrian.
3. THE Tracker SHALL output, for each active track per frame, the track ID, bounding box coordinates, and confidence score.
4. WHEN a track has not been matched for more than the configured maximum age threshold, THE Tracker SHALL mark that track as lost and cease outputting it.

---

### Requirement 4: Road Region Definition

**User Story:** As a researcher, I want to define the crossing zone interactively on the first frame, so that all behavioral analysis is restricted to the actual pedestrian crossing area.

#### Acceptance Criteria

1. WHEN the pipeline starts, THE System SHALL display the first video frame and prompt the user to click polygon vertices defining the Road_Region.
2. WHEN the user completes the polygon (minimum 3 vertices), THE System SHALL store the Road_Region coordinates and use them for all subsequent spatial filtering.
3. WHILE processing frames, THE System SHALL restrict pedestrian crossing analysis, gap classification, platoon detection, and vehicle metric extraction to detections whose bounding box centroids fall within the Road_Region polygon.
4. IF the user defines a polygon with fewer than 3 vertices, THEN THE System SHALL prompt the user to redefine the Road_Region before proceeding.

---

### Requirement 5: Pedestrian Attribute Classification

**User Story:** As a researcher, I want each pedestrian's gender and age group to be automatically classified, so that demographic factors can be included in the gap acceptance model.

#### Acceptance Criteria

1. WHEN a pedestrian track is first detected within the Road_Region, THE Attribute_Classifier SHALL crop the pedestrian bounding box from the frame and pass it to DeepFace for analysis.
2. THE Attribute_Classifier SHALL classify each pedestrian's gender as Male or Female.
3. THE Attribute_Classifier SHALL classify each pedestrian's age group as Young (under 30), Middle (30–59), or Old (60 and above) based on DeepFace's estimated age value.
4. IF DeepFace fails to return a result for a given crop (e.g., face not detected), THEN THE Attribute_Classifier SHALL assign the label "Unknown" for that attribute and continue processing without terminating.
5. THE Attribute_Classifier SHALL store the classified attributes against the pedestrian's track ID for later dataset export.

---

### Requirement 6: Gap Type Classification

**User Story:** As a researcher, I want each pedestrian crossing to be automatically classified as a Straight Gap or Rolling Gap, so that the dependent variable for the regression model is accurately captured.

#### Acceptance Criteria

1. WHEN a pedestrian track exits the Road_Region after having entered it, THE Gap_Classifier SHALL analyze the full trajectory of that track within the Road_Region.
2. THE Gap_Classifier SHALL compute the per-frame displacement of the pedestrian centroid within the Road_Region to derive a frame-by-frame speed sequence.
3. WHEN the pedestrian's speed drops below 20% of their mean crossing speed for a continuous duration of 0.5 seconds or more, THE Gap_Classifier SHALL classify the crossing as a Rolling Gap.
4. WHEN no such speed drop occurs during the crossing, THE Gap_Classifier SHALL classify the crossing as a Straight Gap.
5. THE Gap_Classifier SHALL store the gap type classification against the pedestrian's track ID for later dataset export.

---

### Requirement 7: Vehicle Metrics Extraction

**User Story:** As a researcher, I want vehicle gap duration, time headway, and estimated speed to be extracted automatically, so that traffic conditions at the moment of each crossing can be quantified.

#### Acceptance Criteria

1. WHEN vehicle detections (cars, buses, trucks, motorcycles) are present within the Road_Region, THE Vehicle_Metrics_Extractor SHALL record the frame timestamps of each vehicle's entry and exit.
2. THE Vehicle_Metrics_Extractor SHALL compute gap duration as the time in seconds between the exit of one vehicle and the entry of the next vehicle within the Road_Region.
3. THE Vehicle_Metrics_Extractor SHALL compute time headway as the mean time interval in seconds between consecutive vehicle entries into the Road_Region over the full video.
4. THE Vehicle_Metrics_Extractor SHALL estimate vehicle speed in pixels per second from the displacement of the vehicle bounding box centroid between consecutive frames.
5. WHEN no vehicles are detected in the Road_Region for a given time window, THE Vehicle_Metrics_Extractor SHALL record that window as a gap period with the corresponding duration.

---

### Requirement 8: Platoon Detection

**User Story:** As a researcher, I want to know whether each pedestrian crossed alone or as part of a group, so that platoon behavior can be included as a factor in the model.

#### Acceptance Criteria

1. WHEN 2 or more pedestrian tracks are simultaneously present within the Road_Region in the same frame, THE Platoon_Detector SHALL flag all co-present tracks as platoon crossings.
2. WHEN only 1 pedestrian track is present within the Road_Region at a given time, THE Platoon_Detector SHALL flag that track as a solo crossing.
3. THE Platoon_Detector SHALL store the platoon flag (Group or Alone) against each pedestrian's track ID for later dataset export.

---

### Requirement 9: Dataset Export

**User Story:** As a researcher, I want all extracted per-pedestrian records compiled into a structured CSV file, so that the dataset can be used for statistical modeling and academic reporting.

#### Acceptance Criteria

1. WHEN all frames have been processed, THE Dataset_Exporter SHALL compile one record per completed pedestrian crossing into a CSV file.
2. THE Dataset_Exporter SHALL include the following columns in the CSV: `track_id`, `gender`, `age_group`, `platoon`, `gap_seconds`, `time_headway`, `vehicle_speed`, `gap_type`.
3. THE Dataset_Exporter SHALL encode `gap_type` as a binary integer (1 for Straight Gap, 0 for Rolling Gap) in the CSV.
4. THE Dataset_Exporter SHALL save the CSV file to a user-configurable output directory with a fixed filename of `gap_acceptance_dataset.csv`.
5. IF a pedestrian track does not complete a full crossing of the Road_Region (e.g., track lost mid-crossing), THEN THE Dataset_Exporter SHALL exclude that record from the CSV.

---

### Requirement 10: Binary Logistic Regression Model

**User Story:** As a researcher, I want a Binary Logistic Regression model built on the dataset, so that I can identify which factors significantly influence straight gap selection and report findings for academic submission.

#### Acceptance Criteria

1. WHEN the CSV dataset is available, THE Logit_Model SHALL load it and encode categorical variables (gender, age_group, platoon) as dummy variables before fitting.
2. THE Logit_Model SHALL use `gap_type` as the binary dependent variable and `gender`, `age_group`, `platoon`, `gap_seconds`, `time_headway`, and `vehicle_speed` as independent variables.
3. WHEN the model is fitted using statsmodels Logit, THE Logit_Model SHALL report the coefficient, standard error, z-statistic, p-value, and 95% confidence interval for each predictor.
4. THE Logit_Model SHALL compute and report the odds ratio (exponentiated coefficient) for each predictor.
5. THE Logit_Model SHALL save the full model summary as a text file named `logit_model_summary.txt` in the output directory.
6. IF the dataset contains fewer than 30 complete records, THEN THE Logit_Model SHALL display a warning that the sample size may be insufficient for reliable regression results before proceeding.

---

### Requirement 11: Visualization

**User Story:** As a researcher, I want statistical plots generated from the dataset and model results, so that findings can be presented visually in the academic report and viva.

#### Acceptance Criteria

1. WHEN the dataset and model results are available, THE Visualizer SHALL generate the following 6 plots and save each as a PNG file in the output directory:
   - Gap type distribution (bar chart of Straight vs Rolling counts)
   - Gender vs gap type (grouped bar chart)
   - Age group vs gap type (grouped bar chart)
   - Platoon vs gap type (grouped bar chart)
   - Gap duration distribution by gap type (box plot)
   - Odds ratio plot for logistic regression predictors (horizontal bar chart with 95% CI error bars)
2. THE Visualizer SHALL label all axes, include a title, and include a legend on each plot.
3. THE Visualizer SHALL save each plot at a minimum resolution of 150 DPI.

---

### Requirement 12: Annotated Video Output

**User Story:** As a researcher, I want an annotated output video showing detections, tracks, and region overlays, so that the analysis can be demonstrated visually during the viva.

#### Acceptance Criteria

1. WHEN processing each frame, THE Annotator SHALL draw bounding boxes and track IDs for all active pedestrian tracks.
2. THE Annotator SHALL overlay the Road_Region polygon on every frame.
3. THE Annotator SHALL display the classified gender, age group, and gap type label alongside each pedestrian's bounding box once those attributes are determined.
4. THE Annotator SHALL save the annotated video to the output directory as `annotated_output.mp4` using the same frame rate as the input video.
5. IF the output video file cannot be written (e.g., codec unavailable), THEN THE Annotator SHALL log a descriptive error and continue saving all other outputs without terminating the pipeline.
