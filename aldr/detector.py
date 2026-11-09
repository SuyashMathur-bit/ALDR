"""
ALDR Detector Module
Vehicle Detection (Stage 1) & Fine-Grained Modification Detection (Stage 4)
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
import cv2
import numpy as np
from ultralytics import YOLO

from aldr.config import (
    VEHICLE_DETECTOR_WEIGHTS,
    MODIFICATION_DETECTOR_WEIGHTS,
    VEHICLE_CONF_THRESHOLD,
    MODIFICATION_CONF_THRESHOLD,
    MODIFICATION_CLASSES,
    TARGET_VEHICLE_CLASSES,
    NON_VEHICLE_CLASSES
)


class VehicleDetector:
    """YOLO-based Vehicle Detector trained on Indian Vehicle Dataset."""

    def __init__(self, weights_path: Optional[Union[str, Path]] = None, device: str = "cpu"):
        weights = str(weights_path or VEHICLE_DETECTOR_WEIGHTS)
        if not os.path.exists(weights):
            raise FileNotFoundError(f"Vehicle detector weights not found at: {weights}")
        self.device = device
        self.model = YOLO(weights)
        self.class_names = self.model.names

    def detect(
        self,
        image: Union[str, np.ndarray],
        conf_threshold: float = VEHICLE_CONF_THRESHOLD,
        source_id: str = "image_0",
        camera_id: Optional[str] = None,
        timestamp: Optional[float] = None,
        min_crop_size: int = 16,
        use_tracker: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Detect vehicles in an image/frame using YOLO and maintain track IDs with ByteTrack.
        """
        if isinstance(image, str):
            orig_img = cv2.imread(image)
            if orig_img is None:
                raise ValueError(f"Could not load image at {image}")
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            filename = os.path.basename(image)
        else:
            orig_img = image.copy()
            filename = source_id

        h, w = orig_img.shape[:2]

        # Use ByteTrack tracking if requested, otherwise standard prediction
        if use_tracker:
            results = self.model.track(
                source=orig_img,
                conf=conf_threshold,
                tracker="bytetrack.yaml",
                persist=True,
                device=self.device,
                verbose=False
            )
        else:
            results = self.model.predict(
                source=orig_img,
                conf=conf_threshold,
                device=self.device,
                verbose=False
            )

        detections = []
        if not results or len(results) == 0:
            return detections

        r = results[0]
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            # Boundary clipping
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))

            crop_w = x2 - x1
            crop_h = y2 - y1

            # Validate crop dimensions
            if crop_w < min_crop_size or crop_h < min_crop_size:
                continue

            cls_id = int(box.cls[0].item())
            cls_name = self.class_names.get(cls_id, f"class_{cls_id}").lower()
            confidence = float(box.conf[0].item())

            # Filter non-vehicle obstacles if desired
            if cls_name in NON_VEHICLE_CLASSES:
                continue

            # ByteTrack track ID
            track_id = int(box.id[0].item()) if (box.id is not None) else (i + 1)

            crop = orig_img[y1:y2, x1:x2].copy()

            detection_record = {
                "detection_id": f"{source_id}_veh_{i}",
                "track_id": track_id,
                "source_file": filename,
                "camera_id": camera_id or "cam_0",
                "timestamp": timestamp,
                "bbox": [x1, y1, x2, y2],
                "bbox_normalized": [x1 / w, y1 / h, x2 / w, y2 / h],
                "confidence": confidence,
                "class_id": cls_id,
                "class_name": cls_name,
                "crop": crop,
                "crop_shape": (crop_h, crop_w, 3)
            }
            detections.append(detection_record)

        return detections


class ModificationDetector:
    """Unified YOLO 11-class Vehicle Modification Detector."""

    def __init__(self, weights_path: Optional[Union[str, Path]] = None, device: str = "cpu"):
        weights = str(weights_path or MODIFICATION_DETECTOR_WEIGHTS)
        if not os.path.exists(weights):
            raise FileNotFoundError(f"Modification detector weights not found at: {weights}")
        self.device = device
        self.model = YOLO(weights)
        self.class_names = self.model.names
        # Ensure class mapping matches our 11 target categories
        self.expected_classes = MODIFICATION_CLASSES

    def detect(
        self,
        vehicle_crop: np.ndarray,
        vehicle_bbox: Optional[List[int]] = None,
        conf_threshold: float = MODIFICATION_CONF_THRESHOLD
    ) -> List[Dict[str, Any]]:
        """
        Detect illegal and fine-grained vehicle modifications on a vehicle crop.
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return []

        h, w = vehicle_crop.shape[:2]
        vx1, vy1 = (vehicle_bbox[0], vehicle_bbox[1]) if vehicle_bbox else (0, 0)

        results = self.model.predict(
            source=vehicle_crop,
            conf=conf_threshold,
            device=self.device,
            verbose=False
        )

        modifications = []
        if not results or len(results) == 0:
            return modifications

        r = results[0]
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            return modifications

        for i, box in enumerate(boxes):
            mx1, my1, mx2, my2 = map(int, box.xyxy[0].tolist())
            mx1 = max(0, min(mx1, w - 1))
            my1 = max(0, min(my1, h - 1))
            mx2 = max(0, min(mx2, w))
            my2 = max(0, min(my2, h))

            cls_id = int(box.cls[0].item())
            cls_name = self.class_names.get(cls_id, f"mod_{cls_id}")
            confidence = float(box.conf[0].item())

            # Map to global image coordinates
            gx1, gy1 = vx1 + mx1, vy1 + my1
            gx2, gy2 = vx1 + mx2, vy1 + my2

            mod_record = {
                "mod_id": f"mod_{i}_{cls_name}",
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": confidence,
                "crop_bbox": [mx1, my1, mx2, my2],
                "image_bbox": [gx1, gy1, gx2, gy2]
            }
            modifications.append(mod_record)

        return modifications
