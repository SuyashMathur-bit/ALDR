"""
Test Multi-View Association and Evidence Fusion on VeRi Multi-Camera Ground Truth
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import pandas as pd
from aldr.reid_model import ReIDExtractor
from aldr.association import VehicleAssociator
from aldr.fusion import EvidenceFusionEngine
from aldr.detector import ModificationDetector
from aldr.reporting import ReportGenerator
from aldr.config import VERI_ROOT_DIR, ALDR_REID_DIR, OUTPUT_DIR

def test_multiview_association():
    val_df = pd.read_csv(ALDR_REID_DIR / "val_metadata.csv")
    veh4 = val_df[val_df["vehicle_id"] == 4]
    print(f"[Multi-View Test] Vehicle 4 instances: {len(veh4)}, cameras: {veh4['camera_id'].unique().tolist()}")

    # Pick 2 frames from Camera 1 and 2 frames from Camera 12
    cam1_rows = veh4[veh4["camera_id"] == 1].head(2)
    cam12_rows = veh4[veh4["camera_id"] == 12].head(2)
    sample_df = pd.concat([cam1_rows, cam12_rows])

    extractor = ReIDExtractor()
    mod_detector = ModificationDetector()
    associator = VehicleAssociator(similarity_threshold=0.60)
    fusion = EvidenceFusionEngine()
    reporter = ReportGenerator(OUTPUT_DIR)

    observations = []
    for i, (_, row) in enumerate(sample_df.iterrows()):
        img_path = str(VERI_ROOT_DIR / "image_train" / row["filename"])
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        emb = extractor.extract_embedding(img_rgb)
        mods = mod_detector.detect(img_rgb)

        observations.append({
            "detection_id": f"obs_{i}_cam{row['camera_id']}",
            "source_file": row["filename"],
            "camera_id": f"cam_{row['camera_id']:02d}",
            "timestamp": 1000.0 + i * 30.0,
            "bbox": [0, 0, img.shape[1], img.shape[0]],
            "crop": img_rgb,
            "embedding": emb,
            "modifications": mods
        })

    # Associate across views
    profiles = associator.associate(observations)
    print(f"\n[Multi-View Test] Successfully associated into {len(profiles)} vehicle profiles:")

    for p in profiles:
        fused = fusion.fuse_vehicle_profile(p)
        print(f"Profile: {p['vehicle_profile_id']}")
        print(f"  Obs count: {p['num_observations']} from cameras: {p['cameras']}")
        print(f"  Mean Re-ID similarity: {p['mean_similarity']:.4f}")
        print(f"  Review status: {fused['overall_review_status']}")
        print(f"  Detected mods: {[m['modification_class'] for m in fused['detected_modifications']]}")

        # Render composite evidence
        vis_path = reporter.render_visual_evidence(p, fused, filename_prefix="multiview_test")
        print(f"  Evidence visualization: {vis_path}")

if __name__ == "__main__":
    test_multiview_association()
