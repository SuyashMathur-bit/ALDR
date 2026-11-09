"""
ALDR Vehicle Profile and Report Generation Module
Stage 6: JSON/CSV Reports, Evidence Packaging, Visual Summaries
"""

import os
import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
import cv2
import numpy as np


class ReportGenerator:
    """
    Generates structured forensic reports and visual evidence summaries.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = Path(output_dir or "./aldr_output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.evidence_dir = self.output_dir / "evidence_crops"
        self.evidence_dir.mkdir(parents=True, exist_ok=True)

    def export_json(
        self,
        fused_profiles: List[Dict[str, Any]],
        filename: str = "vehicle_profiles_report.json"
    ) -> Path:
        """Exports full structured profile data to JSON."""
        out_path = self.output_dir / filename
        serializable_data = []

        for p in fused_profiles:
            # Clean non-serializable fields
            clean_p = json.loads(json.dumps(p, default=lambda o: str(o)))
            serializable_data.append(clean_p)

        report = {
            "project": "ALDR - AI-Driven Law & Road Monitoring",
            "system_version": "1.0.0",
            "timestamp": str(np.datetime64('now')),
            "total_vehicle_profiles": len(fused_profiles),
            "flagged_profiles_count": sum(1 for p in fused_profiles if p.get("has_violations", False)),
            "profiles": serializable_data,
            "disclaimer": "Automated AI detection report. For evidentiary screening only. Requires certified manual officer review."
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        print(f"[ALDR Report] JSON report saved to: {out_path}")
        return out_path

    def export_csv(
        self,
        fused_profiles: List[Dict[str, Any]],
        filename: str = "vehicle_profiles_summary.csv"
    ) -> Path:
        """Exports tabular summary of vehicle profiles and detected modifications."""
        out_path = self.output_dir / filename
        fieldnames = [
            "vehicle_profile_id",
            "num_observations",
            "cameras_observed",
            "mean_reid_similarity",
            "overall_review_status",
            "has_violations",
            "detected_modification_count",
            "detected_modification_classes",
            "max_confidence",
            "all_detected_modifications_with_conf"
        ]

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for p in fused_profiles:
                mods = p.get("detected_modifications", [])
                mod_names = [m["modification_class"] for m in mods]
                mod_str = "; ".join([f"{m['modification_class']} ({m['fused_confidence']:.2f})" for m in mods])
                max_c = max([m["fused_confidence"] for m in mods], default=0.0)

                writer.writerow({
                    "vehicle_profile_id": p.get("vehicle_profile_id"),
                    "num_observations": p.get("num_observations", 1),
                    "cameras_observed": ", ".join(p.get("cameras_observed", [])),
                    "mean_reid_similarity": p.get("mean_reid_similarity", 1.0),
                    "overall_review_status": p.get("overall_review_status"),
                    "has_violations": p.get("has_violations", False),
                    "detected_modification_count": p.get("detected_modification_count", 0),
                    "detected_modification_classes": ", ".join(mod_names),
                    "max_confidence": round(max_c, 3),
                    "all_detected_modifications_with_conf": mod_str
                })

        print(f"[ALDR Report] CSV report saved to: {out_path}")
        return out_path

    def render_visual_evidence(
        self,
        vehicle_profile: Dict[str, Any],
        fused_data: Dict[str, Any],
        filename_prefix: str = "evidence_sheet"
    ) -> Optional[Path]:
        """
        Creates an annotated multi-panel evidence image showing all views,
        associated Re-ID bounding boxes, detected modifications, and legal disclaimers.
        """
        observations = vehicle_profile.get("observations", [])
        if not observations:
            return None

        crops_to_draw = []
        for obs in observations:
            crop = obs.get("crop")
            if crop is None:
                continue
            crop_copy = crop.copy()
            # Draw modification boxes
            for mod in obs.get("modifications", []):
                mx1, my1, mx2, my2 = mod["crop_bbox"]
                cname = mod["class_name"]
                conf = mod["confidence"]

                # Bounding box in vibrant amber / green
                cv2.rectangle(crop_copy, (mx1, my1), (mx2, my2), (0, 215, 255), 2)
                label = f"{cname} {conf:.2f}"
                cv2.putText(
                    crop_copy, label, (mx1, max(my1 - 6, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 215, 255), 1, cv2.LINE_AA
                )

            # Resize to standard height for horizontal montage
            h, w = crop_copy.shape[:2]
            target_h = 300
            target_w = int(w * (target_h / max(h, 1)))
            resized_crop = cv2.resize(crop_copy, (target_w, target_h))

            # Add source caption
            cam = obs.get("camera_id", "cam_0")
            cv2.putText(
                resized_crop, f"{cam} ({obs.get('source_file', '')[:12]})",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
            )
            crops_to_draw.append(resized_crop)

        if not crops_to_draw:
            return None

        # Horizontal stitch of vehicle observations
        montage = np.hstack(crops_to_draw)
        mh, mw = montage.shape[:2]

        # Top banner for metadata
        banner_h = 100
        canvas = np.zeros((mh + banner_h, mw, 3), dtype=np.uint8)
        canvas[:banner_h, :] = (35, 35, 35)
        canvas[banner_h:, :] = cv2.cvtColor(montage, cv2.COLOR_RGB2BGR)

        # Header info
        pid = vehicle_profile.get("vehicle_profile_id")
        n_obs = len(observations)
        status = fused_data.get("overall_review_status", "PENDING")
        mean_sim = fused_data.get("mean_reid_similarity", 1.0)
        mods = [f"{m['modification_class']} ({m['fused_confidence']:.2f})" for m in fused_data.get("detected_modifications", [])]
        mod_summary = ", ".join(mods) if mods else "None Detected (Compliant)"

        color = (0, 0, 255) if fused_data.get("has_violations") else (0, 200, 0)

        cv2.putText(canvas, f"ALDR Vehicle Evidence Profile: {pid}", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(canvas, f"Status: {status} | Re-ID Mean Sim: {mean_sim:.2f} | Views: {n_obs}", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(canvas, f"Modifications: {mod_summary[:80]}", (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        out_img_path = self.output_dir / f"{filename_prefix}_{pid}.jpg"
        cv2.imwrite(str(out_img_path), canvas)
        return out_img_path
