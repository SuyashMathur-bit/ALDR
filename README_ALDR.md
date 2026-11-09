# ALDR: AI-Driven Law & Road Monitoring

A unified roadside computer vision framework for **Vehicle Detection**, **Vehicle Re-Identification (Re-ID)**, **Fine-Grained Modification Detection**, **Cross-View Association**, **Multi-View Evidence Fusion**, and **Automated Vehicle Profile Reporting**.

---

## 1. System Architecture & Pipeline Stages

```
                        Roadside Camera Frames / Videos
                                      │
                                      ▼
             ┌──────────────────────────────────────────────────┐
             │       STAGE 1: Input & Vehicle Detection         │
             │   (YOLO-based Indian Vehicle Dataset Detector)   │
             └────────────────────────┬─────────────────────────┘
                                      │ Vehicle Bounding Boxes & Crops
                                      ▼
             ┌──────────────────────────────────────────────────┐
             │         STAGE 2: Vehicle Re-ID Embeddings        │
             │     (ResNet50 Backbone -> 512-D L2-Normalized)   │
             └────────────────────────┬─────────────────────────┘
                                      │ Feature Vectors (Cosine Similarity)
                                      ▼
             ┌──────────────────────────────────────────────────┐
             │     STAGE 3: Cross-View Vehicle Association      │
             │  (Graph Clustering + Temporal & Cam Constraints) │
             └────────────────────────┬─────────────────────────┘
                                      │ Associated Vehicle Profiles
                                      ▼
             ┌──────────────────────────────────────────────────┐
             │        STAGE 4: Modification Detection           │
             │     (Unified YOLO 11-Class Modification Model)   │
             └────────────────────────┬─────────────────────────┘
                                      │ Per-view Modification Evidence
                                      ▼
             ┌──────────────────────────────────────────────────┐
             │       STAGE 5: Multi-View Evidence Fusion        │
             │  (Noisy-OR Probabilistic Aggregation + Tracking) │
             └────────────────────────┬─────────────────────────┘
                                      │ Fused Profile Records
                                      ▼
             ┌──────────────────────────────────────────────────┐
             │       STAGE 6: Vehicle Profile & Reporting       │
             │   (JSON Reports, CSV Summaries, Evidence Panels) │
             └──────────────────────────────────────────────────┘
```

---

## 2. Target Modification Categories (11 Classes)

Preserving the exact class order and zero-indexed labeling:
0. `bumper`
1. `wheel`
2. `aftermarket_exhaust`
3. `additional_lights`
4. `aftermarket_lights`
5. `colored_headlight`
6. `bull_bar`
7. `aftermarket_spoiler`
8. `aftermarket_tyre`
9. `modification_kit`
10. `illegal_number_plate`

---

## 3. Installation & Setup

### Environment Requirements
- Python 3.10+ or 3.11
- PyTorch 2.0+ & Torchvision
- TensorFlow 2.15+ (with Keras 3)
- Ultralytics (`pip install ultralytics`)
- OpenCV, Pandas, NumPy, Matplotlib

### Quick Install
```bash
pip install ultralytics tensorflow opencv-python pandas numpy matplotlib pyyaml
```

---

## 4. Execution Commands

### A. End-to-End Pipeline Demonstration
Runs the full 6-stage pipeline on roadside camera frames, associating vehicles, detecting fine-grained modifications, performing probabilistic fusion, and generating structured JSON/CSV reports with visual evidence panels:
```bash
python aldr/run_aldr.py --demo --samples 4
```

### B. Multi-View Ground Truth Test
Tests cross-camera association on VeRi multi-camera tracklets (Camera 1 vs Camera 12):
```bash
python aldr/test_multiview.py
```

### C. Evaluate Vehicle Re-ID Retrieval
Evaluates Rank-1, Rank-5, and mAP on held-out unseen validation identities with same-camera exclusions:
```bash
python aldr/run_aldr.py --eval-reid
```

### D. Evaluate 11-Class Modification Detector
Evaluates overall and per-class Precision, Recall, mAP@0.5, and mAP@0.5:0.95:
```bash
python aldr/run_aldr.py --eval-mod
```

### E. Evaluate Vehicle Detector
Evaluates vehicle detection on the Indian Vehicle Dataset:
```bash
python aldr/run_aldr.py --eval-detector
```

### F. High-Performance Re-ID Model Training
Trains the classification head on top of the ImageNet-pretrained frozen ResNet50 backbone with early stopping, NVMe caching, and resumable checkpoints:
```bash
# Smoke test (2 epochs, 200 samples)
python aldr/run_aldr.py --smoke-test

# Full training
python aldr/run_aldr.py --train-reid --epochs 15 --batch-size 32
```

---

## 5. Google Colab Execution

Two updated notebooks are available:
1. **`ALDR_Complete_Pipeline_Optimized.ipynb`**: Complete, self-contained notebook covering setup, Re-ID optimization, full 6-stage pipeline, evaluation, and visual reporting.
2. **`ALDR_Modification_Unified_Updated (1).ipynb`**: Synchronized directly with your Google Drive (`/content/drive/MyDrive/Colab Notebooks`), replacing the slow training loop at cell 55 with the optimized, cached, and resumable pipeline.

### Why VeRi Re-ID Training Was Slow in Google Colab (And How It Is Fixed)
1. **Google Drive FUSE Latency**: Colab accesses `/content/drive/MyDrive` over a cloud FUSE mount. Reading 30,519 individual JPEG files per epoch created massive I/O delays (>10 minutes/epoch).
   - **Fix**: Images are cached locally on Colab NVMe scratch space (`/tmp/veri_train_cache`). Epoch 1 writes the decoded array; subsequent epochs read instantly from NVMe in ~15 seconds.
2. **Missing Checkpoints**: The original loop lacked a `ModelCheckpoint` callback.
   - **Fix**: Added `ModelCheckpoint(filepath=..., save_best_only=True)` so weights persist directly to Google Drive after each epoch.
3. **Disjoint Unseen Identity Split**: Softmax classification requires training labels in range `[0, num_classes - 1]`. The validation split contains unseen vehicle IDs (up to 767) and must be evaluated via query-gallery retrieval (Rank-1/Rank-5/mAP) rather than training loss.

---

## 6. Output Artifacts Directory (`aldr_output/`)

- `vehicle_profiles_report.json`: Full machine-readable forensic report with evidence records.
- `vehicle_profiles_summary.csv`: Tabular spreadsheet for road safety officers and analysts.
- `evidence_sheet_*.jpg`: Annotated multi-panel evidence summaries showing vehicle observations, bounding boxes, Re-ID similarity badges, and legal screening disclaimers.

---

## 7. Operational Limitations & Legal Notice

1. **Evidentiary Screening Only**: Computer vision detections of automotive modifications indicate visual non-compliance indicators. Final legal citations require certified inspection or manual officer confirmation.
2. **Camera Angles & Visibility**: A missing detection does not prove absence of modification. Parts not visible in camera perspective are explicitly classified as `NOT_VISIBLE`.
3. **Illumination & Weather**: Extreme glare, nighttime shadows, or heavy rain degrade Re-ID embedding quality and may require adaptive similarity thresholds.
