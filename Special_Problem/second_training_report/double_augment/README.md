| **Model Summary** | **Value** |
| ----------------- | --------- |
| Layers            | 157       |
| Parameters        | 7,015,519 |
| Gradients         | 0         |
| GFLOPs            | 15.8      |

| **Class**     | **Images** | **Instances** | **Precision (P)** | **Recall (R)** | **mAP@50** | **mAP@50–95** |
| ------------- | ---------- | ------------- | ----------------- | -------------- | ---------- | ------------- |
| **All**       | 1,831      | 146,945       | 0.45              | 0.247          | 0.289      | 0.165         |
| **Cluster**   | 1,831      | 1,186         | 0.535             | 0.362          | 0.372      | 0.255         |
| **Thyrocyte** | 1,831      | 145,759       | 0.364             | 0.132          | 0.205      | 0.0756        |

| Metric                    | Before | After | Change  | Observation                                                 |
| ------------------------- | ------ | ----- | ------- | ----------------------------------------------------------- |
| **Cluster Precision (P)** | 0.557  | 0.535 | ↓ 0.022 | Slight drop — model more willing to predict positives       |
| **Cluster Recall (R)**    | 0.347  | 0.362 | ↑ 0.015 | Slight increase — model caught more true positives          |
| **Cluster mAP@50**        | 0.368  | 0.372 | ↑ 0.004 | Minor improvement                                           |
| **Cluster mAP@50–95**     | 0.255  | 0.255 | —       | Stable performance                                          |
| **Thyrocyte Recall (R)**  | 0.0932 | 0.132 | ↑ 0.039 | Slightly improved detection of thyrocytes despite filtering |
| **Overall mAP@50**        | 0.283  | 0.289 | ↑ 0.006 | Small but consistent improvement                            |

📊 Interpretation

Stable performance = preprocessing didn’t harm generalization.
The metrics are nearly the same, meaning the tiling filter (keeping only cluster-containing tiles) didn’t reduce the model’s ability to recognize thyrocytes or clusters. That’s good — it means the filtering isn’t causing major bias or data imbalance issues.

Small recall gain for both classes suggests that the double augmentation/tiling strategy might have made cluster regions more visible (less background/noise).

Minor trade-off in precision (especially cluster class) indicates that the model is predicting slightly more positives, which is often fine early in medical detection — recall is typically more valuable for identifying all potential diagnostic areas.

  After applying a filtering step that retains only tiles containing “cluster” annotations, the YOLO model was retrained with identical   hyperparameters. The performance remained generally stable, with a slight improvement in cluster recall and mAP@50.
  
  This suggests that focusing training on diagnostically relevant regions (cluster-containing tiles) helps the model better detect target   structures without negatively impacting thyrocyte recognition.
  
  Furthermore, class imbalance was handled by assigning a class weight of 5.0 to “cluster” and 1.0 to “thyrocyte.” The same hyperparameter   configuration was used (refer to hyp.yaml).