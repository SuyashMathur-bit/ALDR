from ultralytics import YOLO
import cv2
from pathlib import Path

weights = Path("G:/My Drive/ALDR_Part2/vehicle_baseline-3/weights/best.pt")
model = YOLO(str(weights))
test_dir = Path("G:/My Drive/Indian Vehicle Dataset by Amit Juyal.v2i.yolov11extracted/test/images")
test_imgs = list(test_dir.glob("*.jpg"))[:2]

for idx, p in enumerate(test_imgs):
    img = cv2.imread(str(p))
    res = model.track(source=img, tracker="bytetrack.yaml", persist=True, verbose=False)[0]
    print(f"Frame {idx} ({p.name}): {len(res.boxes)} vehicles tracked.")
    for b in res.boxes[:3]:
        tid = int(b.id[0].item()) if b.id is not None else "static_1"
        cls_id = int(b.cls[0].item())
        conf = float(b.conf[0].item())
        print(f"  Track ID: {tid} | Class: {model.names[cls_id]} | Conf: {conf:.2f} | BBox: {[round(x, 1) for x in b.xyxy[0].tolist()]}")
