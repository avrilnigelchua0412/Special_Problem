| **Class**     | **Images** | **Instances** | **Precision (P)** | **Recall (R)** | **mAP@50** | **mAP@50–95** |
| ------------- | ---------- | ------------- | ----------------- | -------------- | ---------- | ------------- |
| **All**       | 1,831      | 146,945       | 0.424             | 0.262          | 0.248      | 0.136         |
| **Cluster**   | 1,831      | 1,186         | 0.653             | 0.277          | 0.325      | 0.209         |
| **Thyrocyte** | 1,831      | 145,759       | 0.194             | 0.246          | 0.172      | 0.0639        |

| **Parameter**                 | **Description**                                                                                       |
| ----------------------------- | ----------------------------------------------------------------------------------------------------- |
| **Model**                     | YOLOv5s (unfrozen backbone)                                                                           |
| **Training Data**             | Filtered tiles (only tiles containing cluster annotations retained)                                   |
| **Sampler**                   | `torch.utils.data.WeightedRandomSampler` with class-balanced probabilities                            |
| **Loss Class Weights**        | Default `[1.0, 1.0]` (no manual weighting)                                                            |
| **Augmentation**              | Manual pre-tiling augmentations (Albumentations: rotation, brightness, stain-preserving color jitter) |
| **Epochs**                    | 100                                                                                                   |
| **Optimizer / LR / Hyp.yaml** | Identical to Phase 1                                                                                  |
| **Objective**                 | Evaluate the impact of sample-level rebalancing on minority (cluster) detection                       |

#### Observations and Analysis

Cluster precision increased markedly (0.653 ↑ from 0.557, from initial report) — the sampler caused the model to predict clusters more confidently, likely from repeated exposure to minority samples.

Cluster recall dropped slightly (0.277 ↓ from 0.347, from initial report), consistent with mild overfitting: the model memorized minority examples rather than generalizing to unseen clusters.

Thyrocyte recall improved (0.246 ↑ from 0.0932) but at the expense of precision, indicating noisier detections after exposure balancing.

Overall mAP@50–95 decreased, showing that although the model became more sensitive to minority tiles, its overall generalization suffered.

Interpretation — WeightedRandomSampler, without complementary class weights or regularization, overcompensated the class imbalance. The training distribution became less representative of real-world frequency, leading to reduced precision and unstable convergence.

#### Conclusion and Next Steps

The WeightedRandomSampler is not inherently detrimental; rather, it requires careful tuning and sufficient minority diversity to prevent overfitting.

Current evidence suggests that oversampling alone cannot outperform filtered manual augmentation or class-weighted training when the minority class (clusters) is limited and visually homogeneous.

The next experimental phase will therefore focus on:

Undersampling the thyrocyte-only tiles (retaining ≈ 30–50%) while keeping all cluster-containing tiles.

Evaluating a dual-model pipeline:

Model A — Thyrocyte-only detector

Model B — Cluster-only detector

Combine via late-fusion (NMS-based) to enhance overall diagnostic coverage.

#### Hypothesis
The targeted dataset composition — specifically undersampling excessive majority tiles while preserving cluster-rich tiles and applying careful targeted augmentations — will produce better generalization on small-object detection (thyrocyte clusters) than naive oversampling via a WeightedRandomSampler. Overexposure of a small set of minority tiles (replacement sampling) risks memorization and reduces mAP@50–95 on held-out images; preserving representative background/majority tiles while increasing the diversity (not just the count) of minority samples should mitigate this.

Planned evaluation:
comparing three strategies: (1) filtered + default weights (current best), (2) filtered + WeightedRandomSampler, and (3) filtered + undersampling of majority tiles (30–50% retention) with targeted augmentations. Primary metric: cluster recall (tile-level and merged whole-image). Secondary metrics: mAP@50 and mAP@50–95. Running each config with multiple seeds and report mean ± std and confidence intervals; acceptance will be defined as an absolute improvement in cluster recall ≥ 0.05 (?) without dropping overall mAP@50–95 by more than 0.03(?).

#### Note: 
Augmenting cluster tiles necessarily augments thyrocyte instances present in those tiles; consequently, some degree of majority-class duplication is unavoidable. The key is where augmentation occurs: augmenting whole original images before tiling preserves context; augmenting only tiles can lose context and create unnatural partial objects. The experiments explicitly separate these strategies (pre-tiling augment → tile later vs augment tiles directly) to quantify the effect.