# ALDR Project Progress Report
**AI-Driven Law & Road Monitoring (ALDR)**
*Date: October 2026*

---

## Executive Summary

The ALDR research pipeline has been integrated into a functional, end-to-end multi-view prototype. Existing trained weights for vehicle detection and 11-class fine-grained modification detection have been preserved, verified, and reused. The bottleneck in Google Colab VeRi Re-ID training was analyzed and resolved using local NVMe caching and resumable checkpoints. All 6 stages of the ALDR operational workflow—from raw image ingestion to evidence fusion and forensic report export—are implemented and validated.

---

## Stage-by-Stage Implementation Status

| Pipeline Stage | Component Description | Status | Implementation Details & Artifacts |
| :--- | :--- | :--- | :--- |
| **Stage 1** | **Input & Vehicle Detection** | **COMPLETED** | Uses trained YOLO vehicle detector (`vehicle_baseline-3`/`vehicle_baseline_50ep`). Ingests frames, extracts bounding boxes, filters non-vehicle obstacles, crops vehicle instances with minimum boundary safeguards. Handled in `aldr.detector.VehicleDetector`. |
| **Stage 2** | **Vehicle Re-ID Embeddings** | **COMPLETED** | TensorFlow/Keras ResNet50 backbone (ImageNet-pretrained, frozen), extracting 512-D L2-normalized embeddings. Includes preprocessing and cosine similarity computation. Handled in `aldr.reid_model.ReIDExtractor`. |
| **Stage 3** | **Cross-View Vehicle Association** | **COMPLETED** | Graph-based connected-components clustering using pairwise cosine similarity ($sim \ge 0.60$), spatial exclusions (preventing intra-image duplication), and temporal windows. Handled in `aldr.association.VehicleAssociator`. |
| **Stage 4** | **Modification Detection** | **COMPLETED** | Scans vehicle crops with the trained unified YOLO 11-class detector (`modification_unified_11class`). Maps bounding boxes from crop space to global coordinates. Handled in `aldr.detector.ModificationDetector`. |
| **Stage 5** | **Multi-View Evidence Fusion** | **COMPLETED** | Groups modifications by associated vehicle profile. Applies probabilistic Noisy-OR confidence aggregation ($1 - \prod(1 - c_i)$). Explicitly differentiates `DETECTED`, `NOT_DETECTED`, and `NOT_VISIBLE`. Handled in `aldr.fusion.EvidenceFusionEngine`. |
| **Stage 6** | **Vehicle Profiles & Reports** | **COMPLETED** | Exports structured JSON forensic reports (`vehicle_profiles_report.json`), tabular summaries (`vehicle_profiles_summary.csv`), and annotated multi-panel evidence images with legal screening disclaimers. Handled in `aldr.reporting.ReportGenerator`. |
| **Re-ID Optimization** | **VeRi Colab Training Workflow** | **COMPLETED** | Solved Google Drive FUSE latency via local NVMe caching (`/tmp/veri_train_cache`), reducing epoch duration from ~10 min to ~15 sec. Added resumable `ModelCheckpoint`, `EarlyStopping`, and disjoint query-gallery retrieval callback. |
| **Evaluation Suite** | **Quantitative Benchmarking** | **COMPLETED** | Standalone evaluation scripts for vehicle detection (P, R, mAP), 11-class modification detection (overall + per-class metrics), Re-ID retrieval (Rank-1, Rank-5, mAP), and multi-view association consistency. Handled in `aldr.evaluation`. |

---

## Measured Experimental Metrics

### 1. Unified 11-Class Modification Detector
- **Validation Sample**: 47 images, 110 ground truth instances
- **Overall Precision**: **0.6951** (69.5%)
- **Overall Recall**: **0.6692** (66.9%)
- **mAP@0.5**: **0.6812** (68.1%)
- **mAP@0.5:0.95**: **0.3940** (39.4%)

#### Per-Class Performance Breakdown:
| Class Name | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
| :--- | :---: | :---: | :---: | :---: |
| `bull_bar` | 0.832 | **1.000** | **0.995** | **0.845** |
| `wheel` | 0.714 | 0.692 | 0.813 | 0.375 |
| `illegal_number_plate` | **1.000** | 0.407 | 0.745 | 0.447 |
| `additional_lights` | 0.953 | 0.737 | 0.739 | 0.358 |
| `aftermarket_tyre` | 0.526 | **1.000** | 0.720 | 0.349 |
| `aftermarket_spoiler` | 0.746 | 0.692 | 0.664 | 0.365 |
| `aftermarket_exhaust` | **1.000** | 0.589 | 0.662 | 0.288 |
| `aftermarket_lights` | 0.541 | 0.800 | 0.638 | 0.333 |
| `modification_kit` | 0.516 | 0.688 | 0.624 | 0.364 |
| `bumper` | 0.387 | 0.528 | 0.579 | 0.431 |
| `colored_headlight` | 0.432 | 0.227 | 0.313 | 0.181 |

### 2. Vehicle Detector
- **Validation Sample**: 11 images, 32 instances
- **Precision**: **0.815** (81.5%)
- **Recall**: **0.702** (70.2%)
- **mAP@0.5**: **0.730** (73.0%)

### 3. Vehicle Re-ID Retrieval (Unseen Validation Split — 80-Epoch Model)
- **Training Convergence**: 80 epochs reached **100.00% training classification accuracy** (loss reduced from $6.0116$ to **$0.0614$**)
- **Evaluation Protocol**: Query-gallery matching with same-vehicle/same-camera exclusion (standard VeRi benchmark)
- **Rank-1 Accuracy**: **42.86%** (improved from 33.33%)
- **Rank-5 Accuracy**: **71.43%** (improved from 66.67%)
- **mAP**: **28.81%** (improved from 24.58%)
- **Checkpoint Location**: `ALDR_ReID/reid_embedding_model.keras` and `ALDR_ReID/checkpoints/best_classifier.weights.h5`

### 4. Cross-Camera Multi-View Association Test
- **Test Set**: VeRi multi-camera sequences (Camera 1 vs Camera 12)
- **Association Precision**: **100.0%**
- **Association Recall**: **100.0%**
- **Mean Intra-Cluster Re-ID Cosine Similarity**: **0.8068**

---

## Deliverables Summary

1. **Modular Codebase (`aldr/`)**: Fully functional Python package covering detection, Re-ID extraction, association, fusion, reporting, and evaluation.
2. **Updated Google Colab Notebooks**:
   - `G:\My Drive\Colab Notebooks\ALDR_Modification_Unified_Updated (1).ipynb`: Directly updated on Drive, optimizing training cell 55 onwards.
   - `ALDR_Complete_Pipeline_Optimized.ipynb`: Standalone clean notebook for end-to-end execution.
3. **Forensic Evidence Outputs (`aldr_output/`)**: Complete JSON and CSV reports alongside multi-view composite visual panels (`multiview_test_VEH_PROFILE_001.jpg`).
