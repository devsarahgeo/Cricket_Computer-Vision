"""
This is not accurate, WIP........

Test script for checking what my keypoint detection model outputs on the first video frame.

It prints every keypoint index + (x, y, confidence) for the first
video frame and saves an annotated image so you can match index numbers
to physical pitch locations.

Usage:
    python probe_keypoints.py
Output:
    Terminal: keypoint index list
    File:     ../output/keypoint_probe.jpg
"""

import cv2
import os
import numpy as np
from ultralytics import YOLO

base_dir   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_path = os.path.join(base_dir, "models", "keypoint_detection.pt")
video_path = os.path.join(base_dir, "input_video", "shrt_video.mp4")
out_path   = os.path.join(base_dir, "output",      "keypoint_probe.jpg")
os.makedirs(os.path.join(base_dir, "output"), exist_ok=True)

# Load model 
model = YOLO(model_path)
print(f"\nModel task : {model.task}")
print(f"Model info : {model_path}\n")

# Grab first frame 
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()
if not ret:
    raise RuntimeError("Could not read video frame.")

# Run inference 
results = results = model(frame, verbose=False)[0]

# Pose / keypoint output 
if results.keypoints is not None:
    kp_data = results.keypoints.data   # (n_detections, n_keypoints, 3)
    print(f"Detections : {len(kp_data)}")
    print(f"Keypoints per detection: {kp_data.shape[1]}\n")

    # Pick the highest-confidence detection
    mean_confs = kp_data[:, :, 2].mean(dim=1)
    best = int(mean_confs.argmax().item())
    kps  = kp_data[best].cpu().numpy()   # (n_kp, 3)

    print(f"{'Index':>5}  {'x':>7}  {'y':>7}  {'conf':>6}")
    print("-" * 35)
    for i, (x, y, c) in enumerate(kps):
        print(f"  {i:>3}    {x:7.1f}  {y:7.1f}  {c:6.3f}")

    # Draw on frame
    for i, (x, y, c) in enumerate(kps):
        ix, iy = int(x), int(y)
        color = (0, 255, 255) if c > 0.4 else (100, 100, 100)
        cv2.circle(frame, (ix, iy), 7, color, -1)
        cv2.putText(frame, str(i), (ix + 8, iy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(frame, str(i), (ix + 8, iy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

    print(f"\nAnnotated image saved → {out_path}")
    print("\nYellow = confidence > 0.4 (high)  |  Grey = low confidence")
    print("\nLook at keypoint_probe.jpg, match index numbers to pitch")
    print("corners/creases, then edit KEYPOINT_WORLD in keypoint_detection.py.")

# Plain detection fallback 
elif results.boxes is not None and len(results.boxes) > 0:
    print("Model outputs bounding boxes (not pose keypoints).")
    print(f"Detections : {len(results.boxes)}\n")
    print(f"{'Index':>5}  {'cx':>7}  {'cy':>7}  {'conf':>6}  {'class':>5}")
    print("-" * 42)
    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        conf  = float(box.conf[0])
        cls   = int(box.cls[0])
        print(f"  {i:>3}    {cx:7.1f}  {cy:7.1f}  {conf:6.3f}  {cls:5d}")
        cv2.circle(frame, (int(cx), int(cy)), 7, (0, 255, 255), -1)
        cv2.putText(frame, str(i), (int(cx) + 8, int(cy) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    print(f"\nAnnotated image saved → {out_path}")
    print("\nIn keypoint_detection.py, KEYPOINT_WORLD index = box index above.")

else:
    print("No detections on first frame — try a frame further into the video.")

cv2.imwrite(out_path, frame)
