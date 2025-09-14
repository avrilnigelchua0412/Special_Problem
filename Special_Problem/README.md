## README for Analysis Paralysis By GPT (Help Me)
###### BOGO
### Preprocessing

1. Polygon → Bounding Box Conversion
 - Some annotations are polygons; YOLO requires bounding boxes.
 - Conversion may introduce noise (e.g., elongated bounding boxes for irregular clusters).
 - Challenge: preserve accuracy while standardizing format.

2. Region Removal Logic (RRL)
 - Strategy for removing background noise or irrelevant regions.
 - Directly linked to problem in (1): bounding boxes may capture non-informative areas.
 - Idea: test filtering rules or preprocessing masks to clean training data.

### Modeling Approaches

1. Cluster-Only Model

 - Detect cell clusters exclusively.
 - Focus: adequacy assessment via presence/absence of clusters.

2. Thyrocyte-Only Model

 - Detect individual thyrocytes.
 - Hypothesis: capturing fine-grained features may improve generalization and reduce false negatives.

3. Segmentation Model (U-Net or Variants)

 - Alternative to bounding boxes.
 - Useful for pixel-level precision when clusters have irregular boundaries.
 - Still exploratory — requires further investigation.

### Technical Research Directions (Potential Publication)

1. Model Comparison
 - YOLOv5 vs YOLOv7 vs YOLOvX (Latest)
 - Goal: benchmark different YOLO versions for cytology data.
 - Metrics: Precision, Recall, F1, mAP@0.5, mAP@0.5:0.95.

2. Integration of Attention Layers

 - Reference: PI-YOLO: Dynamic Sparse Attention and Lightweight Convolutional-Based YOLO for Vessel Detection in Pathological Images (Li et al.).
 - Hypothesis: Attention improves detection of small/thin clusters in noisy slides.
 - Experiment: Add custom attention modules to YOLO backbone or head.

3. Segmentation with U-Net (or U-Net++/Attention U-Net)

 - Investigate U-Net’s applicability for FNAB cytology images.
 - Compare segmentation vs. detection in terms of adequacy evaluation.
 - May complement detection models rather than replace them.

### Microscope Effects:

1. The microscope and camera setup (magnification, lighting, resolution) can change how images look.

2. This may affect model performance — a model trained on one microscope’s images may not work as well on another.

3. For my future technical paper, the microscope type and settings should be reported.