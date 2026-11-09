# Research Summary: ALDR — AI-Driven Law & Road Monitoring
**A Unified Framework for Multi-Camera Vehicle Re-Identification and Fine-Grained Automotive Modification Detection**

---

## Abstract

Automated road enforcement typically relies on automated license plate readers (ALPR) or coarse vehicle category classifiers. However, these systems are vulnerable to obscured or counterfeit license plates and fail to inspect structural safety modifications that violate transportation regulations. In this paper, we propose **ALDR (AI-Driven Law & Road Monitoring)**, a six-stage roadside computer vision pipeline that unifies vehicle detection, metric Re-Identification (Re-ID), 11-class fine-grained modification detection, spatio-temporal cross-camera association, and probabilistic multi-view evidence fusion. Experimental evaluation demonstrates that ALDR achieves 0.681 mAP@0.5 across 11 modification categories on real roadside imagery, achieves 100% cross-camera association consistency on multi-view validation sequences (mean cosine similarity of 0.807), and produces automated forensic evidence reports calibrated for human-in-the-loop judicial screening.

---

## 1. Methodology & Pipeline

The ALDR architecture operates across six sequential stages:

1. **Vehicle Localization**: A YOLO-based detector localized on diverse traffic datasets identifies vehicle instances while filtering pedestrians and static road furniture. High-resolution bounding boxes are normalized and cropped.
2. **Metric Re-ID Representation**: Vehicle crops are mapped into a 512-dimensional metric embedding space $\mathbf{z} \in \mathbb{R}^{512}$ via a ResNet50 backbone followed by Global Average Pooling, linear projection, batch normalization, and unit $\ell_2$-normalization:
   $$\mathbf{z} = \frac{\mathbf{W}_e \operatorname{GAP}(f(\mathbf{x}))}{\|\mathbf{W}_e \operatorname{GAP}(f(\mathbf{x}))\|_2}$$
   Pairwise vehicle identity similarity between observations $u$ and $v$ is computed directly as the cosine inner product $\mathcal{S}(u, v) = \mathbf{z}_u^\top \mathbf{z}_v$.
3. **Cross-View Spatio-Temporal Association**: Observations across distributed roadside cameras are associated using an adjacency graph where edges satisfy $\mathcal{S}(u, v) \ge \tau_{\text{sim}}$ subject to temporal window constraints $|t_u - t_v| \le \Delta T_{\max}$ and physical travel-time non-contradiction rules. Connected component clustering identifies unique vehicle profiles.
4. **Fine-Grained Modification Detection**: An 11-class unified object detector scans vehicle crops to identify structural violations: bumpers, custom wheels, aftermarket exhausts, additional lights, aftermarket lights, colored headlights, bull bars, aftermarket spoilers, aftermarket tyres, body kits, and illegal number plates.
5. **Probabilistic Evidence Fusion**: To handle occlusion, view angle limitations, and varying detection confidences across camera angles, ALDR applies an independent probabilistic Noisy-OR aggregation model:
   $$\mathcal{P}_{\text{fused}}(m \mid \mathcal{V}) = 1 - \prod_{k \in \mathcal{V}} (1 - c_{k, m})$$
   where $c_{k, m}$ denotes the detector confidence for modification $m$ in observation view $k$. Categories without visual coverage are explicitly flagged as `NOT_VISIBLE`, preventing false negatives from masking uninspected vehicle sections.
6. **Forensic Profile Generation**: Comprehensive JSON reports, tabular summaries, and composite multi-panel evidence sheets are generated to support manual adjudication by traffic authorities.

---

## 2. Experimental Results & Discussion

### A. Modification Detection Performance
On the held-out validation set (47 images, 110 instances), the unified 11-class detector achieved:
- **Mean Precision**: 69.51%
- **Mean Recall**: 66.92%
- **mAP@0.5**: 68.12%
- **mAP@0.5:0.95**: 39.40%

Highly distinctive structural components such as `bull_bar` (mAP50: 0.995, mAP50-95: 0.845) and `wheel` (mAP50: 0.813) demonstrated exceptional detection accuracy, while subtle color shifts (`colored_headlight`: mAP50 0.313) exhibited higher variance under varying daylight conditions.

### B. Multi-View Association & Vehicle Retrieval
Using the VeRi multi-camera benchmark across non-overlapping roadside viewpoints with the trained 80-epoch metric embedding model:
- **Rank-1 Retrieval Accuracy**: **42.86%**
- **Rank-5 Retrieval Accuracy**: **71.43%**
- **mAP**: **28.81%**
- **Multi-Camera Tracklet Association**: 100% precision and recall on ground-truth multi-camera pairs (Camera 1 vs Camera 12), with an intra-vehicle cosine similarity score of $\mu = 0.8068 \pm 0.04$, compared to inter-vehicle similarity $\mu < 0.45$.

### C. Training Throughput Optimization
By eliminating Google Drive cloud FUSE random I/O and caching deterministic resized representations on local NVMe scratch space (`/tmp/veri_train_cache`), per-epoch training duration decreased by **96.8%** (from ~620 seconds to ~19 seconds per epoch), enabling scalable iterative model training in cloud environments.

---

## 3. Paper Contribution Highlights

1. **End-to-End Multi-Task Architecture**: First unified pipeline linking multi-camera vehicle Re-ID with fine-grained modification detection and probabilistic evidence aggregation.
2. **Statistically Sound Evidence Fusion**: Differentiates between non-detection and non-visibility, ensuring fairness in judicial road compliance screening.
3. **Reproducible Open Architecture**: Modularized codebase and open evaluation benchmarks allowing direct deployment across roadside camera networks.
