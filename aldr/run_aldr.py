"""
ALDR Main CLI Execution & Demonstration Script
AI-Driven Law & Road Monitoring (ALDR)
"""

import os
import sys
import glob
import argparse
from pathlib import Path

# Ensure package root is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
from aldr.config import (
    INDIAN_VEHICLE_DATASET_DIR,
    MODIFICATION_DATASET_DIR,
    VERI_ROOT_DIR,
    ALDR_REID_DIR,
    OUTPUT_DIR
)
from aldr.pipeline import ALDRPipeline
from aldr.reid_train import train_reid
from aldr.evaluation import (
    evaluate_vehicle_detector,
    evaluate_modification_detector,
    evaluate_reid_retrieval
)


def run_demonstration(num_samples: int = 5):
    """
    Runs end-to-end ALDR demonstration on multi-view roadside images.
    """
    print("\n" + "=" * 65)
    print("      ALDR: AI-DRIVEN LAW & ROAD MONITORING — END-TO-END DEMO      ")
    print("=" * 65)

    pipeline = ALDRPipeline()

    # Collect sample test images
    test_img_dir = INDIAN_VEHICLE_DATASET_DIR / "test" / "images"
    preferred = [str(p) for p in test_img_dir.glob("*24_jpg*")]
    all_imgs = sorted(glob.glob(str(test_img_dir / "*.jpg")))
    test_images = preferred + [img for img in all_imgs if img not in preferred]
    test_images = test_images[:num_samples]

    if not test_images:
        # Fallback to any images in modification dataset
        test_img_dir = MODIFICATION_DATASET_DIR / "test" / "images"
        test_images = sorted(glob.glob(str(test_img_dir / "*.jpg")))[:num_samples]

    if not test_images:
        print(f"[Demo Error] No test images found in {test_img_dir}")
        return

    print(f"[Demo] Selected {len(test_images)} roadside test images for processing:")
    frame_inputs = []
    cameras = ["cam_overhead_01", "cam_lane_02", "cam_overhead_01", "cam_junction_03", "cam_lane_02"]

    for idx, fpath in enumerate(test_images):
        cam = cameras[idx % len(cameras)]
        print(f"  Frame {idx + 1}: {Path(fpath).name} (Simulated Camera: {cam})")
        frame_inputs.append({
            "image": fpath,
            "source_id": Path(fpath).name,
            "camera_id": cam,
            "timestamp": 1000.0 + (idx * 20.0)  # Simulated timestamps (20s apart)
        })

    # Execute full pipeline
    results = pipeline.process_frames(frame_inputs)

    print("\n" + "=" * 65)
    print("                      DEMONSTRATION RESULTS                      ")
    print("=" * 65)
    print(f"Input Frames Analyzed          : {results['num_frames']}")
    print(f"Vehicle Instances Detected     : {results['num_vehicle_observations']}")
    print(f"Unique Associated Profiles     : {results['num_associated_profiles']}")
    print(f"JSON Report Exported           : {results['json_report']}")
    print(f"CSV Summary Exported           : {results['csv_report']}")
    print(f"Visual Evidence Panels Saved   : {len(results['visual_evidence_paths'])}")
    for p in results['visual_evidence_paths']:
        print(f"   -> {p}")
    print("=" * 65)

    # Print summary table of profiles
    print("\nProfile Summary:")
    for prof in results["fused_profiles"]:
        pid = prof["vehicle_profile_id"]
        status = prof["overall_review_status"]
        mods = [f"{m['modification_class']} ({m['fused_confidence']:.2f})" for m in prof["detected_modifications"]]
        mod_str = ", ".join(mods) if mods else "No Illegal Modifications Detected"
        print(f"  [{pid}] Views: {prof['num_observations']} | Status: {status:28s} | Mods: {mod_str}")


def main():
    parser = argparse.ArgumentParser(description="ALDR Main Execution CLI")
    parser.add_argument("--demo", action="store_true", help="Run end-to-end demonstration")
    parser.add_argument("--train-reid", action="store_true", help="Train Vehicle Re-ID model")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick 2-epoch Re-ID smoke test")
    parser.add_argument("--eval-detector", action="store_true", help="Evaluate vehicle detector")
    parser.add_argument("--eval-mod", action="store_true", help="Evaluate modification detector")
    parser.add_argument("--eval-reid", action="store_true", help="Evaluate Re-ID retrieval on held-out validation set")
    parser.add_argument("--eval-all", action="store_true", help="Run full evaluation across all components")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--samples", type=int, default=4, help="Number of demo images")
    args = parser.parse_args()

    if args.demo:
        run_demonstration(num_samples=args.samples)
    elif args.train_reid or args.smoke_test:
        train_reid(
            epochs=2 if args.smoke_test else args.epochs,
            sample_size=200 if args.smoke_test else None,
            batch_size=args.batch_size
        )
    elif args.eval_detector:
        evaluate_vehicle_detector()
    elif args.eval_mod:
        evaluate_modification_detector()
    elif args.eval_reid:
        from aldr.reid_model import ReIDExtractor
        extractor = ReIDExtractor()
        evaluate_reid_retrieval(extractor)
    elif args.eval_all:
        evaluate_vehicle_detector()
        evaluate_modification_detector()
        from aldr.reid_model import ReIDExtractor
        extractor = ReIDExtractor()
        evaluate_reid_retrieval(extractor)
    else:
        # Default behavior: run demo
        run_demonstration(num_samples=args.samples)


if __name__ == "__main__":
    main()
