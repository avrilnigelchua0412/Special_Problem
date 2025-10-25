### Experimental Results and Analysis

Model 1 – Baseline (No Pretrained Weights, Default Hyperparameters)
**Precision (P)** | **Recall (R)** | **mAP@50** | **mAP@50–95** |
----------------- | -------------- | ---------- | ------------- |
0.431             | 0.252          | 0.245      | 0.131         |
0.657             | 0.273          | 0.316      | 0.197         |
0.205             | 0.232          | 0.173      | 0.0654        |

This baseline model was trained with a dataset filtered to 30% thyrocytes. The results show limited performance with low recall and mAP scores, suggesting that the model struggles to generalize. The high cluster precision likely reflects the more visually distinct and larger “cluster” regions, making them easier for the model to identify despite class imbalance.

Model 2 – Pretrained YOLOv5s (Transfer Learning)
**Precision (P)** | **Recall (R)** | **mAP@50** | **mAP@50–95** |
----------------- | -------------- | ---------- | ------------- |
0.412             | 0.271          | 0.292      | 0.172         |
0.542             | 0.378          | 0.394      | 0.271         |
0.281             | 0.164          | 0.191      | 0.0735        |

Using YOLOv5s pretrained weights improved the results slightly, despite only being trained for 30 epochs. This indicates that transfer learning is essential for the task, as the pretrained backbone provides better feature representations even with limited fine-tuning.

Model 3 – Pretrained YOLOv5s with Weighted Classification Loss
**Precision (P)** | **Recall (R)** | **mAP@50** | **mAP@50–95** |
----------------- | -------------- | ---------- | ------------- |
0.494             | 0.233          | 0.291      | 0.168         |
0.588             | 0.352          | 0.386      | 0.262         |
0.399             | 0.114          | 0.196      | 0.0749        |

This model used a modified class loss weight cls_pw: [3.0, 1.0] to emphasize the underrepresented cluster class. However, performance did not improve significantly. It is likely that over-weighting a class loss did not help because of the inherent feature overlap between clusters and thyrocytes.

Model 4 – Fine-Tuned Model 2 (100 Epochs)
**Precision (P)** | **Recall (R)** | **mAP@50** | **mAP@50–95** |
----------------- | -------------- | ---------- | ------------- |
0.456             | 0.271          | 0.302      | 0.167         |
0.571             | 0.412          | 0.417      | 0.263         |
0.342             | 0.13           | 0.187      | 0.0713        |

Fine-tuning Model 2 to 100 epochs yielded modest improvements in recall, particularly for the cluster class, though precision slightly fluctuated. This is expected, as longer training allows the model to refine boundaries, though at the risk of minor overfitting. Overall, this phase validated the value of extended fine-tuning after transfer learning.

Model 5 – Fine-Tuned with --image-weights (20 Epochs)
**Precision (P)** | **Recall (R)** | **mAP@50** | **mAP@50–95** |
----------------- | -------------- | ---------- | ------------- |
0.448             | 0.259          | 0.291      | 0.161         |
0.591             | 0.373          | 0.404      | 0.255         |
0.304             | 0.146          | 0.179      | 0.0682        |

Applying --image-weights during fine-tuning did not yield a notable improvement. This option prioritizes images based on class frequency but lacks explicit control over weight scaling, leading to marginal or inconsistent benefits when class imbalance is severe.

Model 6 – Fine-Tuned with WeightedRandomSampler (20 Epochs)
**Precision (P)** | **Recall (R)** | **mAP@50** | **mAP@50–95** |
----------------- | -------------- | ---------- | ------------- |
0.417             | 0.275          | 0.292      | 0.166         |
0.562             | 0.38           | 0.398      | 0.262         |
0.273             | 0.17           | 0.187      | 0.0707        |

Using WeightedRandomSampler resulted in poorer outcomes. Although the sampler provides more explicit control over class balancing, it likely emphasized cluster samples excessively, reinforcing the hierarchical labeling issue where “cluster” is merely an aggregation of “thyrocytes.” This redundancy confuses the model’s representation learning.

Observations and Hypotheses
1. Hierarchical Labeling Problem: The “cluster” class essentially represents a collection of “thyrocytes,” causing strong feature overlap. The model struggles to differentiate them, as both share visual and contextual cues.
2. Annotation Noise: Original cluster annotations were polygon-based. Converting them into bounding boxes introduced spatial noise, especially with tiled inputs, leading to inconsistent training signals.
3. Underrepresented but Redundant Classes: Attempting to rebalance clusters through sampling or weighting yields minimal benefit, as the issue lies more in semantic redundancy than class count imbalance.


Future Directions
1. Single-Class Thyrocyte Model: Simplify the task by focusing only on thyrocyte detection. This avoids the hierarchical ambiguity and could yield cleaner feature extraction.
2. Dedicated Cluster Model (Polygon-Supported): Train a separate model that supports polygonal annotation for more accurate cluster localization.
3. Model Fusion (Late Fusion Strategy): Combine predictions from the Thyrocyte and Cluster models. While this may increase inference time, it allows specialized models to handle distinct visual features.
4. Curriculum Learning Approach: Train the thyrocyte model first, then fine-tune it to detect clusters. This progressive learning strategy may help the model understand simpler patterns (individual cells) before handling aggregated forms (clusters).