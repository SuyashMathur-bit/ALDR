"""
ALDR Vehicle Gallery Retrieval & Multi-View Evidence Consolidation Module
Performs explicit Query-to-Gallery Vehicle Retrieval:
1. Detect vehicle + crop it
2. Extract Re-ID embedding
3. Compare against gallery database (cosine similarity ranking) to find the same vehicle
4. Combine cross-camera views + detected modification evidence
5. Fuse evidence (Noisy-OR) and generate final vehicle profile & visual report
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import cv2
import numpy as np

from aldr.config import (
    REID_SIMILARITY_THRESHOLD,
    MODIFICATION_CONF_THRESHOLD,
    VEHICLE_CONF_THRESHOLD,
    OUTPUT_DIR
)
from aldr.reid_model import ReIDExtractor
from aldr.detector import ModificationDetector, VehicleDetector
from aldr.fusion import EvidenceFusionEngine
from aldr.reporting import ReportGenerator


class VehicleGalleryRetriever:
    """
    Manages a gallery of vehicle observations across camera networks
    and executes query-to-gallery vehicle matching, multi-view consolidation,
    and forensic evidence reporting.
    """

    def __init__(
        self,
        similarity_threshold: float = REID_SIMILARITY_THRESHOLD,
        reid_extractor: Optional[ReIDExtractor] = None,
        modification_detector: Optional[ModificationDetector] = None,
        vehicle_detector: Optional[VehicleDetector] = None,
        output_dir: Optional[Union[str, Path]] = None,
        device: str = "cpu"
    ):
        self.similarity_threshold = similarity_threshold
        self.device = device

        self.reid_extractor = reid_extractor or ReIDExtractor()
        self.modification_detector = modification_detector or ModificationDetector(device=device)
        self.vehicle_detector = vehicle_detector or VehicleDetector(device=device)
        self.fusion_engine = EvidenceFusionEngine(aggregation_rule="noisy_or")
        self.report_generator = ReportGenerator(Path(output_dir or OUTPUT_DIR))

        # Gallery storage
        self.gallery: List[Dict[str, Any]] = []

    def add_to_gallery(
        self,
        image_or_crop: Union[str, Path, np.ndarray],
        camera_id: str,
        timestamp: Optional[float] = None,
        source_file: Optional[str] = None,
        bbox: Optional[List[int]] = None,
        is_crop: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Ingests a frame or crop into the gallery database.
        Extracts Re-ID embedding and detects any modifications.
        """
        # Load image if path
        if isinstance(image_or_crop, (str, Path)):
            source_file = source_file or Path(image_or_crop).name
            img = cv2.imread(str(image_or_crop))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = image_or_crop
            source_file = source_file or "frame"

        h, w = img_rgb.shape[:2]
        added_entries = []

        if is_crop:
            crop = img_rgb
            crop_bbox = bbox or [0, 0, w, h]
            emb = self.reid_extractor.extract_embedding(crop)
            mods = self.modification_detector.detect(crop, crop_bbox)
            entry = {
                "gallery_id": f"gal_{len(self.gallery) + 1:04d}",
                "camera_id": camera_id,
                "timestamp": timestamp,
                "source_file": source_file,
                "bbox": crop_bbox,
                "crop": crop,
                "embedding": emb,
                "modifications": mods
            }
            self.gallery.append(entry)
            added_entries.append(entry)
        else:
            # Full frame: detect vehicles first
            detections = self.vehicle_detector.detect(
                image=img_rgb,
                conf_threshold=VEHICLE_CONF_THRESHOLD,
                source_id=source_file,
                camera_id=camera_id,
                timestamp=timestamp
            )
            for det in detections:
                crop = det["crop"]
                emb = self.reid_extractor.extract_embedding(crop)
                mods = self.modification_detector.detect(crop, det["bbox"])
                entry = {
                    "gallery_id": f"gal_{len(self.gallery) + 1:04d}",
                    "camera_id": camera_id,
                    "timestamp": timestamp,
                    "source_file": source_file,
                    "bbox": det["bbox"],
                    "crop": crop,
                    "embedding": emb,
                    "modifications": mods
                }
                self.gallery.append(entry)
                added_entries.append(entry)

        return added_entries

    def search_gallery(
        self,
        query_crop: np.ndarray,
        query_camera_id: str,
        top_k: int = 5,
        exclude_same_camera: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Compares query vehicle embedding against all gallery embeddings using cosine similarity.
        Returns ranked list of matching gallery entries.
        """
        if not self.gallery:
            return []

        # Step 2: Extract query Re-ID embedding
        q_emb = self.reid_extractor.extract_embedding(query_crop)

        # Step 3: Compare against gallery
        g_embs = np.array([g["embedding"] for g in self.gallery], dtype=np.float32)
        sims = np.dot(g_embs, q_emb)

        results = []
        for idx, sim in enumerate(sims):
            entry = self.gallery[idx]
            if exclude_same_camera and entry["camera_id"] == query_camera_id:
                continue
            results.append({
                "gallery_entry": entry,
                "similarity": float(sim),
                "is_match": bool(sim >= self.similarity_threshold)
            })

        # Rank descending by cosine similarity
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    def retrieve_and_profile(
        self,
        query_image_or_crop: Union[str, Path, np.ndarray],
        query_camera_id: str = "query_cam",
        timestamp: Optional[float] = None,
        source_file: Optional[str] = None,
        is_crop: bool = False,
        profile_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Complete Pipeline Workflow:
        1. Detect vehicle + crop it
        2. Extract Re-ID embedding
        3. Compare against gallery and find the same vehicle
        4. Combine camera views + modification evidence
        5. Generate final vehicle profile and report
        """
        # Step 1: Detect & Crop
        if isinstance(query_image_or_crop, (str, Path)):
            source_file = source_file or Path(query_image_or_crop).name
            img = cv2.imread(str(query_image_or_crop))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = query_image_or_crop
            source_file = source_file or "query_input"

        h, w = img_rgb.shape[:2]

        if is_crop:
            q_crop = img_rgb
            q_bbox = [0, 0, w, h]
        else:
            dets = self.vehicle_detector.detect(
                img_rgb, conf_threshold=VEHICLE_CONF_THRESHOLD,
                source_id=source_file, camera_id=query_camera_id
            )
            if not dets:
                return {"status": "NO_VEHICLE_DETECTED"}
            q_crop = dets[0]["crop"]
            q_bbox = dets[0]["bbox"]

        # Step 2: Extract Re-ID embedding
        q_emb = self.reid_extractor.extract_embedding(q_crop)
        q_mods = self.modification_detector.detect(q_crop, q_bbox)

        query_obs = {
            "detection_id": f"query_{source_file[:10]}",
            "source_file": source_file,
            "camera_id": query_camera_id,
            "timestamp": timestamp or 0.0,
            "bbox": q_bbox,
            "crop": q_crop,
            "embedding": q_emb,
            "modifications": q_mods
        }

        # Step 3: Compare against gallery and find the same vehicle
        search_results = self.search_gallery(
            query_crop=q_crop,
            query_camera_id=query_camera_id,
            top_k=5,
            exclude_same_camera=True
        )

        matched_gallery = [r for r in search_results if r["is_match"]]
        top_match = search_results[0] if search_results else None

        # Step 4: Combine camera views + modification evidence
        all_views = [query_obs]
        matched_cam_ids = []
        similarities = []

        for m in matched_gallery:
            g_entry = m["gallery_entry"]
            similarities.append(m["similarity"])
            matched_cam_ids.append(g_entry["camera_id"])
            all_views.append({
                "detection_id": g_entry["gallery_id"],
                "source_file": g_entry["source_file"],
                "camera_id": g_entry["camera_id"],
                "timestamp": g_entry["timestamp"],
                "bbox": g_entry["bbox"],
                "crop": g_entry["crop"],
                "embedding": g_entry["embedding"],
                "modifications": g_entry["modifications"]
            })

        mean_sim = float(np.mean(similarities)) if similarities else 1.0
        pid = profile_id or f"RETRIEVED_VEH_{len(self.gallery):03d}"

        raw_profile = {
            "vehicle_profile_id": pid,
            "observations": all_views,
            "num_observations": len(all_views),
            "cameras": [query_camera_id] + matched_cam_ids,
            "mean_similarity": round(mean_sim, 4)
        }

        # Step 5: Probabilistic Evidence Fusion (Noisy-OR) & Final Profile
        fused_profile = self.fusion_engine.fuse_vehicle_profile(raw_profile)

        # Generate Visual Evidence Sheet
        vis_sheet_path = self.report_generator.render_visual_evidence(
            raw_profile, fused_profile, filename_prefix="retrieval_evidence"
        )

        # Export updated JSON and CSV reports
        json_path = self.report_generator.export_json([fused_profile], filename="retrieval_report.json")
        csv_path = self.report_generator.export_csv([fused_profile], filename="retrieval_summary.csv")

        return {
            "status": "MATCH_FOUND" if matched_gallery else "NEW_UNIQUE_VEHICLE",
            "query_camera": query_camera_id,
            "matched_gallery_count": len(matched_gallery),
            "matched_cameras": matched_cam_ids,
            "top_similarity": round(top_match["similarity"], 4) if top_match else 0.0,
            "all_ranked_matches": [
                {
                    "gallery_id": r["gallery_entry"]["gallery_id"],
                    "camera_id": r["gallery_entry"]["camera_id"],
                    "similarity": round(r["similarity"], 4),
                    "is_match": r["is_match"]
                }
                for r in search_results
            ],
            "raw_profile": raw_profile,
            "fused_profile": fused_profile,
            "visual_evidence_path": str(vis_sheet_path) if vis_sheet_path else None,
            "json_report_path": str(json_path),
            "csv_report_path": str(csv_path)
        }
