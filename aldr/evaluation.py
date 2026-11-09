"""
ALDR Evaluation Module
Comprehensive Evaluation: Detectors, Re-ID Retrieval (Rank-1/5, mAP), Association, and Fusion
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from ultralytics import YOLO

from aldr.config import (
    VEHICLE_DETECTOR_WEIGHTS,
    MODIFICATION_DETECTOR_WEIGHTS,
    INDIAN_VEHICLE_DATASET_DIR,
    MODIFICATION_DATASET_DIR,
    VERI_ROOT_DIR,
    ALDR_REID_DIR,
    REID_IMG_HEIGHT,
    REID_IMG_WIDTH,
    MODIFICATION_CLASSES
)
from aldr.reid_model import ReIDExtractor


def _prepare_local_yaml(base_dir: Path, nc: int, names: list) -> str:
    """Generates an absolute-path data.yaml for reliable evaluation on current OS."""
    resolved_path = str(base_dir.resolve()).replace("\\", "/")
    tmp_yaml = base_dir / "eval_resolved.yaml"
    content = f"""path: {resolved_path}
train: train/images
val: valid/images
test: test/images
nc: {nc}
names: {names}
"""
    with open(tmp_yaml, "w") as f:
        f.write(content)
    return str(tmp_yaml)


def evaluate_vehicle_detector(
    weights_path: Optional[Path] = None,
    dataset_yaml: Optional[Path] = None,
    split: str = "val",
    device: str = "cpu"
) -> Dict[str, float]:
    """Evaluates YOLO vehicle detector and reports Precision, Recall, mAP50, mAP50-95."""
    weights = str(weights_path or VEHICLE_DETECTOR_WEIGHTS)
    from aldr.config import VEHICLE_CLASSES
    data_yaml = _prepare_local_yaml(INDIAN_VEHICLE_DATASET_DIR, len(VEHICLE_CLASSES), VEHICLE_CLASSES)

    print(f"\n[Evaluation] Evaluating Vehicle Detector on {split} set...")
    model = YOLO(weights)
    res = model.val(data=data_yaml, split=split, device=device, verbose=False)

    metrics = {
        "precision": float(res.box.mp),
        "recall": float(res.box.mr),
        "mAP50": float(res.box.map50),
        "mAP50_95": float(res.box.map)
    }

    print("===== VEHICLE DETECTOR METRICS =====")
    print(f"Precision  : {metrics['precision']:.4f}")
    print(f"Recall     : {metrics['recall']:.4f}")
    print(f"mAP@0.5    : {metrics['mAP50']:.4f}")
    print(f"mAP@0.5:0.95: {metrics['mAP50_95']:.4f}")
    return metrics


def evaluate_modification_detector(
    weights_path: Optional[Path] = None,
    dataset_yaml: Optional[Path] = None,
    split: str = "val",
    device: str = "cpu"
) -> Dict[str, Any]:
    """Evaluates 11-class vehicle modification detector including per-class metrics."""
    weights = str(weights_path or MODIFICATION_DETECTOR_WEIGHTS)
    data_yaml = _prepare_local_yaml(MODIFICATION_DATASET_DIR, len(MODIFICATION_CLASSES), MODIFICATION_CLASSES)

    print(f"\n[Evaluation] Evaluating 11-Class Modification Detector on {split} set...")
    model = YOLO(weights)
    res = model.val(data=data_yaml, split=split, device=device, verbose=False)

    overall_metrics = {
        "precision": float(res.box.mp),
        "recall": float(res.box.mr),
        "mAP50": float(res.box.map50),
        "mAP50_95": float(res.box.map)
    }

    per_class = {}
    print("\n===== MODIFICATION DETECTOR OVERALL =====")
    print(f"Precision  : {overall_metrics['precision']:.4f}")
    print(f"Recall     : {overall_metrics['recall']:.4f}")
    print(f"mAP@0.5    : {overall_metrics['mAP50']:.4f}")
    print(f"mAP@0.5:0.95: {overall_metrics['mAP50_95']:.4f}")

    print("\n===== PER-CLASS MODIFICATION METRICS =====")
    for cls_id, name in model.names.items():
        p = float(res.box.p[cls_id]) if cls_id < len(res.box.p) else 0.0
        r = float(res.box.r[cls_id]) if cls_id < len(res.box.r) else 0.0
        ap50 = float(res.box.ap50[cls_id]) if cls_id < len(res.box.ap50) else 0.0
        ap = float(res.box.ap[cls_id]) if cls_id < len(res.box.ap) else 0.0
        per_class[name] = {"precision": p, "recall": r, "mAP50": ap50, "mAP50_95": ap}
        print(f"  {name:22s} | P: {p:.3f} | R: {r:.3f} | mAP50: {ap50:.3f} | mAP50-95: {ap:.3f}")

    return {"overall": overall_metrics, "per_class": per_class}


def evaluate_reid_retrieval(
    reid_extractor: ReIDExtractor,
    val_metadata_csv: Optional[Path] = None,
    image_dir: Optional[Path] = None,
    max_queries: int = 150,
    gallery_size: int = 600
) -> Dict[str, float]:
    """
    Standard VeRi Vehicle Re-ID retrieval evaluation:
    Calculates Rank-1, Rank-5, and mAP on held-out unseen identities
    following query-gallery protocol with camera exclusion.
    """
    csv_path = val_metadata_csv or (ALDR_REID_DIR / "val_metadata.csv")
    img_dir = image_dir or (VERI_ROOT_DIR / "image_train")

    if not csv_path.exists():
        raise FileNotFoundError(f"Validation metadata not found at {csv_path}")

    val_df = pd.read_csv(csv_path)

    # Filter vehicle identities that have at least 2 distinct observations
    valid_ids = val_df.groupby("vehicle_id").filter(lambda g: len(g) >= 2)["vehicle_id"].unique()
    val_subset = val_df[val_df["vehicle_id"].isin(valid_ids)]

    # Partition into query and gallery
    queries = []
    galleries = []

    for vid, group in val_subset.groupby("vehicle_id"):
        rows = group.to_dict("records")
        queries.append(rows[0])
        galleries.extend(rows[1:])

    queries = queries[:max_queries]
    if len(galleries) > gallery_size:
        galleries = galleries[:gallery_size]

    print(f"\n[Evaluation] Evaluating Re-ID Retrieval: {len(queries)} queries vs {len(galleries)} gallery images...")

    # Load and extract embeddings
    import cv2
    def extract_crop_emb(row):
        fpath = str(img_dir / row["filename"])
        img = cv2.imread(fpath)
        if img is None:
            return None
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return reid_extractor.extract_embedding(img_rgb)

    query_embs = []
    valid_queries = []
    for q in queries:
        emb = extract_crop_emb(q)
        if emb is not None:
            query_embs.append(emb)
            valid_queries.append(q)

    gallery_embs = []
    valid_gallery = []
    for g in galleries:
        emb = extract_crop_emb(g)
        if emb is not None:
            gallery_embs.append(emb)
            valid_gallery.append(g)

    query_embs = np.array(query_embs)
    gallery_embs = np.array(gallery_embs)

    # Similarity matrix (num_queries, num_gallery)
    sim_mat = np.matmul(query_embs, gallery_embs.T)

    rank1_hits = 0
    rank5_hits = 0
    aps = []

    for i, q in enumerate(valid_queries):
        q_id = q["vehicle_id"]
        q_cam = q["camera_id"]

        sims = sim_mat[i].copy()
        # Exclude same identity from same camera (standard VeRi evaluation protocol)
        for j, g in enumerate(valid_gallery):
            if g["vehicle_id"] == q_id and g["camera_id"] == q_cam:
                sims[j] = -999.0

        sorted_indices = np.argsort(-sims)
        sorted_gallery_ids = np.array([valid_gallery[idx]["vehicle_id"] for idx in sorted_indices])
        sorted_sims = sims[sorted_indices]

        # Filter out junk/excluded entries
        valid_mask = sorted_sims > -900.0
        sorted_gallery_ids = sorted_gallery_ids[valid_mask]

        matches = (sorted_gallery_ids == q_id)
        if not np.any(matches):
            continue

        if matches[0]:
            rank1_hits += 1
        if np.any(matches[:5]):
            rank5_hits += 1

        # Calculate Average Precision (AP)
        cum_matches = np.cumsum(matches)
        ranks = np.arange(1, len(matches) + 1)
        precision_at_k = cum_matches / ranks
        ap = np.sum(precision_at_k * matches) / np.sum(matches)
        aps.append(ap)

    n_q = len(aps)
    rank1 = (rank1_hits / max(n_q, 1)) * 100
    rank5 = (rank5_hits / max(n_q, 1)) * 100
    map_score = (np.mean(aps) if aps else 0.0) * 100

    print("===== VEHICLE RE-ID RETRIEVAL METRICS =====")
    print(f"Rank-1 Accuracy : {rank1:.2f}%")
    print(f"Rank-5 Accuracy : {rank5:.2f}%")
    print(f"mAP             : {map_score:.2f}%")
    print(f"Evaluated Queries: {n_q}")

    return {"rank1": rank1, "rank5": rank5, "mAP": map_score, "queries_evaluated": n_q}


def evaluate_association_and_fusion(
    pipeline,
    multi_view_samples: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Evaluates multi-view association precision, recall, and evidence fusion consistency.
    """
    print(f"\n[Evaluation] Evaluating Cross-View Association on {len(multi_view_samples)} observations...")
    result = pipeline.process_frames(multi_view_samples)

    # Count ground-truth identity pairs vs predicted clusters
    gt_pairs = 0
    tp_pairs = 0
    fp_pairs = 0

    n = len(multi_view_samples)
    for i in range(n):
        for j in range(i + 1, n):
            gt_same = multi_view_samples[i].get("ground_truth_id") == multi_view_samples[j].get("ground_truth_id")
            if gt_same:
                gt_pairs += 1

            # Check if assigned to same profile
            prof_i = None
            prof_j = None
            for p in result["raw_profiles"]:
                obs_ids = [o["detection_id"] for o in p["observations"]]
                if multi_view_samples[i].get("source_id") in [o.get("source_file") for o in p["observations"]]:
                    prof_i = p["vehicle_profile_id"]
                if multi_view_samples[j].get("source_id") in [o.get("source_file") for o in p["observations"]]:
                    prof_j = p["vehicle_profile_id"]

            if prof_i is not None and prof_i == prof_j:
                if gt_same:
                    tp_pairs += 1
                else:
                    fp_pairs += 1

    precision = tp_pairs / max(tp_pairs + fp_pairs, 1)
    recall = tp_pairs / max(gt_pairs, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)

    print("===== ASSOCIATION & FUSION METRICS =====")
    print(f"Association Precision : {precision * 100:.2f}%")
    print(f"Association Recall    : {recall * 100:.2f}%")
    print(f"Association F1-Score  : {f1 * 100:.2f}%")

    return {
        "association_precision": precision,
        "association_recall": recall,
        "association_f1": f1
    }
