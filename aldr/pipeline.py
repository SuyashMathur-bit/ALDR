"""
ALDR Integrated Pipeline Module
Full End-to-End System Integration: Stages 1 through 6
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
import cv2
import numpy as np

from aldr.config import (
    VEHICLE_CONF_THRESHOLD,
    MODIFICATION_CONF_THRESHOLD,
    REID_SIMILARITY_THRESHOLD,
    OUTPUT_DIR
)
from aldr.detector import VehicleDetector, ModificationDetector
from aldr.reid_model import ReIDExtractor
from aldr.association import VehicleAssociator
from aldr.fusion import EvidenceFusionEngine
from aldr.reporting import ReportGenerator
from aldr.retrieval import VehicleGalleryRetriever


class ALDRPipeline:
    """
    End-to-End AI-Driven Law & Road Monitoring Pipeline.
    Integrates Vehicle Detection, Re-ID, Modification Detection,
    Cross-View Association, Evidence Fusion, and Reporting.
    """

    def __init__(
        self,
        vehicle_detector_weights: Optional[Union[str, Path]] = None,
        modification_detector_weights: Optional[Union[str, Path]] = None,
        reid_model_path: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        reid_similarity_threshold: float = REID_SIMILARITY_THRESHOLD,
        fusion_rule: str = "noisy_or",
        device: str = "cpu"
    ):
        print("[ALDR Pipeline] Initializing components...")
        self.device = device
        self.vehicle_detector = VehicleDetector(vehicle_detector_weights, device=device)
        self.modification_detector = ModificationDetector(modification_detector_weights, device=device)
        self.reid_extractor = ReIDExtractor(reid_model_path)
        self.associator = VehicleAssociator(similarity_threshold=reid_similarity_threshold)
        self.fusion_engine = EvidenceFusionEngine(aggregation_rule=fusion_rule)
        self.report_generator = ReportGenerator(Path(output_dir or OUTPUT_DIR))
        self.retriever = VehicleGalleryRetriever(
            similarity_threshold=reid_similarity_threshold,
            reid_extractor=self.reid_extractor,
            modification_detector=self.modification_detector,
            vehicle_detector=self.vehicle_detector,
            output_dir=output_dir,
            device=device
        )
        print("[ALDR Pipeline] Ready for inference and analysis.")

    def process_frames(
        self,
        frame_inputs: List[Dict[str, Any]],
        conf_veh: float = VEHICLE_CONF_THRESHOLD,
        conf_mod: float = MODIFICATION_CONF_THRESHOLD
    ) -> Dict[str, Any]:
        """
        Processes a list of roadside camera frames.
        Each frame_input dict:
            {
                "image": np.ndarray or str path,
                "source_id": str,
                "camera_id": str,
                "timestamp": float (optional)
            }
        """
        all_vehicle_observations = []

        # ==========================================
        # STAGE 1: Input & Vehicle Detection
        # ==========================================
        print(f"\n[ALDR Pipeline] Stage 1: Detecting vehicles across {len(frame_inputs)} frames...")
        for frame_info in frame_inputs:
            img = frame_info["image"]
            src_id = frame_info.get("source_id", "img")
            cam_id = frame_info.get("camera_id", "cam_0")
            ts = frame_info.get("timestamp")

            detections = self.vehicle_detector.detect(
                image=img,
                conf_threshold=conf_veh,
                source_id=src_id,
                camera_id=cam_id,
                timestamp=ts
            )
            for det in detections:
                all_vehicle_observations.append(det)

        print(f"[ALDR Pipeline] Stage 1 Complete: {len(all_vehicle_observations)} vehicle crops extracted.")

        if not all_vehicle_observations:
            print("[ALDR Pipeline] No vehicles found in input.")
            return {
                "num_frames": len(frame_inputs),
                "num_vehicle_observations": 0,
                "num_associated_profiles": 0,
                "raw_profiles": [],
                "fused_profiles": [],
                "json_report": "",
                "csv_report": "",
                "visual_evidence_paths": []
            }

        # ==========================================
        # STAGE 2: Vehicle Re-ID Embeddings
        # ==========================================
        print(f"[ALDR Pipeline] Stage 2: Extracting 512-D Re-ID embeddings...")
        crops = [obs["crop"] for obs in all_vehicle_observations]
        embeddings = self.reid_extractor.extract_batch(crops)

        for obs, emb in zip(all_vehicle_observations, embeddings):
            obs["embedding"] = emb

        # ==========================================
        # STAGE 3: Cross-View Vehicle Association
        # ==========================================
        print(f"[ALDR Pipeline] Stage 3: Associating observations across views...")
        raw_profiles = self.associator.associate(all_vehicle_observations)
        print(f"[ALDR Pipeline] Stage 3 Complete: Formed {len(raw_profiles)} unique vehicle profiles.")

        # ==========================================
        # STAGE 4: Modification Detection
        # ==========================================
        print(f"[ALDR Pipeline] Stage 4: Scanning vehicle crops for 11 modification classes...")
        for obs in all_vehicle_observations:
            mods = self.modification_detector.detect(
                vehicle_crop=obs["crop"],
                vehicle_bbox=obs["bbox"],
                conf_threshold=conf_mod
            )
            obs["modifications"] = mods

        # ==========================================
        # STAGE 5: Multi-View Evidence Fusion
        # ==========================================
        print(f"[ALDR Pipeline] Stage 5: Fusing multi-view modification evidence...")
        fused_profiles = []
        for profile in raw_profiles:
            fused = self.fusion_engine.fuse_vehicle_profile(profile)
            fused_profiles.append(fused)

        # ==========================================
        # STAGE 6: Vehicle Profile & Report Generation
        # ==========================================
        print(f"[ALDR Pipeline] Stage 6: Exporting reports and visual evidence...")
        json_path = self.report_generator.export_json(fused_profiles)
        csv_path = self.report_generator.export_csv(fused_profiles)

        visual_paths = []
        for raw_p, fused_p in zip(raw_profiles, fused_profiles):
            vis_img = self.report_generator.render_visual_evidence(raw_p, fused_p)
            if vis_img:
                visual_paths.append(str(vis_img))

        return {
            "num_frames": len(frame_inputs),
            "num_vehicle_observations": len(all_vehicle_observations),
            "num_associated_profiles": len(raw_profiles),
            "raw_profiles": raw_profiles,
            "fused_profiles": fused_profiles,
            "json_report": str(json_path),
            "csv_report": str(csv_path),
            "visual_evidence_paths": visual_paths
        }

    def query_gallery_and_profile(
        self,
        query_image_or_crop: Union[str, Path, np.ndarray],
        query_camera_id: str = "query_cam",
        timestamp: Optional[float] = None,
        is_crop: bool = False,
        profile_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes explicit query-to-gallery vehicle retrieval:
        1. Detect vehicle + crop it
        2. Extract Re-ID embedding
        3. Compare against gallery database (cosine similarity ranking) to find the same vehicle
        4. Combine cross-camera views + detected modification evidence
        5. Fuse evidence (Noisy-OR) and generate final vehicle profile & visual report
        """
        return self.retriever.retrieve_and_profile(
            query_image_or_crop=query_image_or_crop,
            query_camera_id=query_camera_id,
            timestamp=timestamp,
            is_crop=is_crop,
            profile_id=profile_id
        )
