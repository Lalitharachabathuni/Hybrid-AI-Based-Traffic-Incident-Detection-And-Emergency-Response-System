"""
video_detection.py
--------------------
Computer Vision module: Vehicle Detection & Tracking + Speed Estimation from
raw traffic-camera video, exactly as described in the System Architecture /
Methodology slides (YOLOv11 for detection, ByteTrack for multi-object
tracking, Optical Flow for speed/motion analysis).

This module is OPTIONAL / BONUS: the core sensor pipeline (preprocess.py ->
train_model.py -> simulate_stream.py) uses synthesized speed/occupancy/flow
data, not camera footage, so it never calls this file. If you have your own
traffic-camera clip, you can run this script against it to see the CV half
of the architecture in action:

    pip install ultralytics
    python backend/video_detection.py --source path/to/video.mp4

It uses Ultralytics YOLO (v8/v11 architecture) with built-in ByteTrack
tracking, and estimates each tracked vehicle's relative speed with
Farneback dense optical flow computed on its bounding box.
"""
import argparse
import sys

import cv2
import numpy as np


def estimate_speed(prev_gray, curr_gray, bbox):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(x1, 0), max(y1, 0)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    prev_roi = prev_gray[y1:y2, x1:x2]
    curr_roi = curr_gray[y1:y2, x1:x2]
    if prev_roi.size == 0 or curr_roi.size == 0 or prev_roi.shape != curr_roi.shape:
        return 0.0
    flow = cv2.calcOpticalFlowFarneback(prev_roi, curr_roi, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return float(np.mean(mag))


def run(source: str, weights: str = "yolo11n.pt", show: bool = True):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("This bonus module requires the 'ultralytics' package: pip install ultralytics")
        sys.exit(1)

    model = YOLO(weights)  # auto-downloads pretrained COCO weights on first use
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Could not open video source: {source}")
        sys.exit(1)

    prev_gray = None
    vehicle_classes = {"car", "truck", "bus", "motorcycle"}

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)[0]

        if results.boxes is not None:
            for box in results.boxes:
                cls_name = model.names[int(box.cls[0])]
                if cls_name not in vehicle_classes:
                    continue
                track_id = int(box.id[0]) if box.id is not None else -1
                xyxy = box.xyxy[0].tolist()
                speed_proxy = estimate_speed(prev_gray, gray, xyxy) if prev_gray is not None else 0.0

                x1, y1, x2, y2 = [int(v) for v in xyxy]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame, f"ID {track_id} {cls_name} v~{speed_proxy:.1f}",
                    (x1, max(y1 - 8, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
                )

        prev_gray = gray
        if show:
            cv2.imshow("Vehicle Detection & Tracking (YOLOv11 + ByteTrack + Optical Flow)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Path to a video file or 0 for webcam")
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    run(args.source, args.weights, show=not args.no_show)
