| **Setup**                  | **YOLO augment?**           | **Manual (Albumentations)?** | **Cluster Recall** | **Cluster mAP@50** | **All mAP@50–95** |
| -------------------------- | --------------------------- | ---------------------------- | ------------------ | ------------------ | ----------------- |
| **Run 1 (Double augment)** | ✅ Light (hsv, rot, deg=10°) | ✅ Rich manual                | **0.362**          | **0.372**          | **0.165**         |
| **Run 2 (Manual only)**    | ❌ None                      | ✅ Rich manual                | **0.363**          | **0.383**          | **0.17**          |

In this iteration, two augmentation strategies were compared:
(1) a double augmentation approach combining manual pre-tiling Albumentations and YOLO’s in-training augmentations, and (2) manual-only augmentation with all in-training augmentations disabled.

Both achieved comparable performance, but the manual-only setup yielded slightly higher overall precision (0.528 vs. 0.45) and marginally higher mAP@50–95 (0.17 vs. 0.165), suggesting that excessive augmentation may introduce redundant or noisy variations, particularly in cytology datasets where color consistency carries diagnostic relevance.

Therefore, subsequent experiments will employ manual augmentations for geometric and photometric diversity, with only minimal in-training color jitter to preserve stain consistency.