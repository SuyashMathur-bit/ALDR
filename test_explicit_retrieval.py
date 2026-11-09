"""
Test Explicit Query-to-Gallery Vehicle Retrieval
1. Detect vehicle + crop it
2. Extract Re-ID embedding
3. Compare against gallery and find the same vehicle
4. Combine camera views + modification evidence
5. Generate final vehicle profile and report
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from aldr.retrieval import VehicleGalleryRetriever
from aldr.config import VERI_ROOT_DIR, ALDR_REID_DIR

def run_retrieval_test():
    print("=" * 70)
    print("RUNNING EXPLICIT QUERY-TO-GALLERY RETRIEVAL & PROFILING PIPELINE")
    print("=" * 70)

    val_df = pd.read_csv(ALDR_REID_DIR / "val_metadata.csv")
    
    # Let's populate the gallery with diverse vehicles across different cameras
    # Vehicle 4 (cam 12), Vehicle 5, Vehicle 6, Vehicle 7, Vehicle 8
    retriever = VehicleGalleryRetriever(similarity_threshold=0.60)

    # 1. Build Gallery
    print("\n--- Step A: Building Multi-Camera Gallery Database ---")
    available_vids = val_df['vehicle_id'].unique()[:8]  # [4, 11, 15, 16, 17, 39, 40, 50]
    for vid in available_vids:
        rows = val_df[val_df["vehicle_id"] == vid]
        # Pick a non-camera-1 instance for the gallery
        non_cam1 = rows[rows["camera_id"] != 1]
        sample_row = non_cam1.iloc[0] if len(non_cam1) > 0 else rows.iloc[-1]
        img_path = VERI_ROOT_DIR / "image_train" / sample_row["filename"]
        cam = f"cam_{sample_row['camera_id']:02d}"
        
        retriever.add_to_gallery(
            image_or_crop=img_path,
            camera_id=cam,
            source_file=sample_row["filename"],
            is_crop=True
        )
        print(f"Added to Gallery: Vehicle ID {vid:04d} from {cam} ({sample_row['filename']})")

    print(f"\nTotal Gallery size: {len(retriever.gallery)} indexed vehicle observations.")

    # 2. Prepare Query Vehicle (Vehicle 4 from Camera 01)
    print("\n--- Step B: Input Query Vehicle (Detected at Camera 01) ---")
    veh4_cam1 = val_df[(val_df["vehicle_id"] == 4) & (val_df["camera_id"] == 1)].iloc[0]
    query_img_path = VERI_ROOT_DIR / "image_train" / veh4_cam1["filename"]
    print(f"Query Image: {veh4_cam1['filename']} from cam_01 (True Vehicle ID: 4)")

    # 3. Execute Complete Workflow:
    # Detect + Crop -> Extract Re-ID -> Search Gallery -> Combine Views + Evidence -> Final Profile
    print("\n--- Step C: Executing Query-to-Gallery Retrieval & Evidence Fusion ---")
    result = retriever.retrieve_and_profile(
        query_image_or_crop=query_img_path,
        query_camera_id="cam_01",
        source_file=veh4_cam1["filename"],
        is_crop=True,
        profile_id="VEH_PROFILE_004"
    )

    print("\n" + "=" * 70)
    print("RETRIEVAL & CONSOLIDATION RESULTS:")
    print("=" * 70)
    print(f"Status                  : {result['status']}")
    print(f"Query Camera            : {result['query_camera']}")
    print(f"Matched Gallery Count   : {result['matched_gallery_count']}")
    print(f"Matched Cameras         : {result['matched_cameras']}")
    print(f"Top Cosine Similarity   : {result['top_similarity']:.4f}")
    
    print("\nRanked Gallery Search Results (Cosine Similarity):")
    for idx, r in enumerate(result["all_ranked_matches"], 1):
        match_str = "[MATCH FOUND]" if r["is_match"] else "[NO MATCH]"
        print(f"  Rank {idx}: {r['gallery_id']} | {r['camera_id']} | Similarity: {r['similarity']:.4f} {match_str}")

    fused = result["fused_profile"]
    print("\n--- Step D: Combined Multi-View Vehicle Profile & Evidence ---")
    print(f"Profile ID              : {fused['vehicle_profile_id']}")
    print(f"Total Observed Views    : {fused['num_observations']}")
    print(f"Cameras Observed        : {fused['cameras_observed']}")
    print(f"Mean Re-ID Similarity   : {fused['mean_reid_similarity']:.4f}")
    print(f"Has Violations          : {fused['has_violations']}")
    print(f"Review Status           : {fused['overall_review_status']}")
    print(f"Visual Evidence Sheet   : {result['visual_evidence_path']}")
    print(f"JSON Report             : {result['json_report_path']}")
    print(f"CSV Summary             : {result['csv_report_path']}")

if __name__ == "__main__":
    run_retrieval_test()
