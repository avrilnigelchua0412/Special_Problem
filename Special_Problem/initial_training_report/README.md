| **Model Summary** | **Value** |
| ----------------- | --------- |
| Layers            | 157       |
| Parameters        | 7,015,519 |
| Gradients         | 0         |
| GFLOPs            | 15.8      |

| **Class**     | **Images** | **Instances** | **Precision (P)** | **Recall (R)** | **mAP@50** | **mAP@50–95** |
| ------------- | ---------- | ------------- | ----------------- | -------------- | ---------- | ------------- |
| **All**       | 1,831      | 146,945       | 0.536             | 0.220          | 0.283      | 0.165         |
| **Cluster**   | 1,831      | 1,186         | 0.557             | 0.347          | 0.368      | 0.255         |
| **Thyrocyte** | 1,831      | 145,759       | 0.514             | 0.0932         | 0.197      | 0.0755        |

#### Phase 1: Baseline Tiled Training (Current Results)

The initial YOLOv5 model was trained using all tiled versions of both the original and augmented images.
No filtering was applied to ensure that augmented tiles contained cluster annotations, resulting in a dataset where thyrocyte tiles dominated the training distribution.

To partially compensate for this imbalance, per-class weights were introduced during loss computation:
```py
# Define per-class positive weights (for imbalanced datasets)
# Example: 2 classes — "cluster" (weight=5.0), "thyrocyte" (weight=1.0)
class_weights = torch.tensor([5.0, 1.0], device=device)
```
A higher weight (5.0) was assigned to the cluster class to emphasize its contribution to the loss function, while thyrocyte retained a baseline weight of 1.0.
All other optimization parameters were set according to the project’s hyp.yaml configuration file.

Despite using per-class weighting, the imbalance in training data distribution remained a limiting factor.
Although the model achieved moderately higher recall and mAP for the minority cluster class, the overall performance suggests that class imbalance and data redundancy (from non-filtered augmentation) still hindered robust small-object learning.

#### Phase 2: Refined Dataset Strategy (Next Steps)
Tile Filtering (Targeted Oversampling)
Retain only augmented tiles that contain cluster annotations.
This ensures that augmentation increases the effective sample diversity of minority-class regions (clusters) without artificially inflating majority-class (thyrocyte) data.

#### Phase 3: Hierarchical Evaluation Strategy

Following prior literature on tile-based small-object detection (Unel et al., 2019; BMC Med Imaging, 2023), the evaluation process will proceed in two hierarchical stages:
 - Tile-level evaluation first → ensure local feature learning → then merge for global evaluation.
 1. Tile-level evaluation:
The refined model will first be validated at the tile level to ensure that it can reliably detect small thyrocyte clusters in localized regions.

 2. Merged (whole-image) evaluation:
Once significant tile-level performance is achieved, tile outputs will be merged back into the original image space using coordinate mapping and non-max suppression to reconstruct whole-image predictions.

This staged evaluation ensures that improvements in small-object detection are verified locally before assessing global (whole-image) performance, consistent with prior research demonstrating that tiling substantially improves recall and mAP for tiny targets by preserving fine spatial detail.