"""
ALDR Multi-View Evidence Fusion Module
Stage 5: Aggregation, Uncertainty Modeling, Visibility Tracking
"""

from typing import List, Dict, Any, Optional
from collections import defaultdict
from aldr.config import MODIFICATION_CLASSES, MODIFICATION_ID_TO_NAME

# Part visibility mapping based on typical roadside camera angles / detected features
PART_VIEW_AFFINITY = {
    "bull_bar": ["front"],
    "colored_headlight": ["front"],
    "aftermarket_exhaust": ["rear", "side"],
    "aftermarket_spoiler": ["rear"],
    "wheel": ["side", "front", "rear"],
    "aftermarket_tyre": ["side"],
    "bumper": ["front", "rear"],
    "additional_lights": ["front", "top"],
    "aftermarket_lights": ["front", "rear"],
    "modification_kit": ["side", "front", "rear"],
    "illegal_number_plate": ["front", "rear"]
}


class EvidenceFusionEngine:
    """
    Fuses multi-view and temporal observations for an associated vehicle profile.
    Distinguishes: DETECTED, NOT_DETECTED, NOT_VISIBLE.
    Applies statistically sound confidence aggregation (Noisy-OR).
    """

    def __init__(
        self,
        aggregation_rule: str = "noisy_or",
        min_confirmed_conf: float = 0.60,
        multi_view_boost: bool = True
    ):
        self.aggregation_rule = aggregation_rule  # "noisy_or", "max", or "mean"
        self.min_confirmed_conf = min_confirmed_conf
        self.multi_view_boost = multi_view_boost

    def aggregate_confidences(self, conf_list: List[float]) -> float:
        """
        Combines independent observation confidences using Noisy-OR:
        P_fused = 1 - Product(1 - c_i)
        """
        if not conf_list:
            return 0.0

        if self.aggregation_rule == "max":
            return max(conf_list)
        elif self.aggregation_rule == "mean":
            return sum(conf_list) / len(conf_list)
        else:
            # Default: Noisy-OR (independent probabilistic evidence combination)
            prob_none = 1.0
            for c in conf_list:
                prob_none *= (1.0 - min(max(c, 0.0), 0.99))
            fused = 1.0 - prob_none
            return round(fused, 4)

    def fuse_vehicle_profile(
        self,
        vehicle_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Processes all observations of a single vehicle and generates
        a fused modification profile.
        """
        observations = vehicle_profile.get("observations", [])
        num_obs = len(observations)

        # Group detected modifications by class name
        mod_evidence = defaultdict(list)
        views_present = set()

        for obs in observations:
            cam_id = obs.get("camera_id", "cam_0")
            views_present.add(cam_id)
            mods = obs.get("modifications", [])

            for mod in mods:
                cname = mod["class_name"]
                mod_evidence[cname].append({
                    "detection_id": obs.get("detection_id"),
                    "source_file": obs.get("source_file"),
                    "camera_id": cam_id,
                    "timestamp": obs.get("timestamp"),
                    "confidence": mod["confidence"],
                    "crop_bbox": mod["crop_bbox"],
                    "image_bbox": mod["image_bbox"]
                })

        # Evaluate status for all 11 target categories
        fused_modifications = []
        has_any_violation = False

        for target_mod in MODIFICATION_CLASSES:
            evidence_list = mod_evidence.get(target_mod, [])

            if evidence_list:
                confidences = [e["confidence"] for e in evidence_list]
                fused_conf = self.aggregate_confidences(confidences)
                supporting_cameras = list({e["camera_id"] for e in evidence_list})

                # Boost confidence slightly if verified across distinct physical cameras
                if self.multi_view_boost and len(supporting_cameras) > 1:
                    fused_conf = min(1.0, fused_conf * 1.05)

                status = "DETECTED"
                review_flag = "REQUIRES_VERIFICATION"
                if fused_conf >= self.min_confirmed_conf:
                    if len(supporting_cameras) > 1 or fused_conf > 0.80:
                        review_flag = "HIGH_CONFIDENCE_FLAG"
                    else:
                        review_flag = "SINGLE_VIEW_SUSPECT"

                has_any_violation = True
                fused_modifications.append({
                    "modification_class": target_mod,
                    "status": status,
                    "fused_confidence": round(fused_conf, 4),
                    "observation_count": len(evidence_list),
                    "supporting_cameras": supporting_cameras,
                    "review_flag": review_flag,
                    "evidence_items": evidence_list
                })
            else:
                # Part was not detected. Determine if it was visible or not visible.
                # If we have multiple views, common parts like wheels/bumpers are assumed visible but compliant.
                # Otherwise marked NOT_DETECTED or NOT_VISIBLE.
                if num_obs >= 2:
                    status = "NOT_DETECTED"  # Likely compliant / factory standard
                else:
                    status = "NOT_VISIBLE"   # Insufficient multi-view coverage

                fused_modifications.append({
                    "modification_class": target_mod,
                    "status": status,
                    "fused_confidence": 0.0,
                    "observation_count": 0,
                    "supporting_cameras": [],
                    "review_flag": "COMPLIANT_OR_UNOBSERVED",
                    "evidence_items": []
                })

        # Filter only active detections for high-level summary
        active_violations = [m for m in fused_modifications if m["status"] == "DETECTED"]

        return {
            "vehicle_profile_id": vehicle_profile.get("vehicle_profile_id"),
            "num_observations": num_obs,
            "cameras_observed": list(views_present),
            "mean_reid_similarity": vehicle_profile.get("mean_similarity", 1.0),
            "has_violations": len(active_violations) > 0,
            "detected_modification_count": len(active_violations),
            "detected_modifications": active_violations,
            "all_category_statuses": fused_modifications,
            "overall_review_status": "FLAGGED_FOR_MANUAL_REVIEW" if active_violations else "CLEAR_LEGAL"
        }
