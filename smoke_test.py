# -*- coding: utf-8 -*-
"""Smoke test for the local YOLOv8 (ultralytics 8.0.29) checkout.

Verifies: import, CUDA availability, loading pretrained yolov8n.pt,
and a real inference pass over the bundled bus.jpg sample.
"""
import platform
import sys

print("=" * 70)
print(f"python      : {sys.version.split()[0]} ({platform.machine()})")

import torch  # noqa: E402

print(f"torch       : {torch.__version__}")
print(f"cuda avail  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu         : {torch.cuda.get_device_name(0)}")
    print(f"capability  : {torch.cuda.get_device_capability(0)}")

import ultralytics  # noqa: E402

print(f"ultralytics : {ultralytics.__version__}  (path: {ultralytics.__file__})")

from ultralytics import YOLO  # noqa: E402

model = YOLO("yolov8n.pt")  # local file, no network needed
results = model.predict("ultralytics/assets/bus.jpg", save=True)

r = results[0]
names = model.model.names  # ultralytics 8.0.29 keeps class names on the model
print("-" * 70)
print(f"detections  : {len(r.boxes)}")
for b in r.boxes:
    cls_id = int(b.cls.item())
    print(f"  - {names[cls_id]:<8} conf={b.conf.item():.3f} xyxy={b.xyxy[0].tolist()}")

import glob  # noqa: E402

saved = sorted(glob.glob("runs/detect/predict*/bus.jpg"))
print(f"saved image : {saved[-1] if saved else 'NOT FOUND'}")
print("=" * 70)
print("SMOKE TEST PASSED")
