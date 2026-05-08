#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from ultralytics import YOLO

# Resolve v4/runs/detections relative to the script's parent (v4/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) # Get parent directory of the script's directory
OUTPUT_DIR = os.path.join(PROJECT_ROOT,"runs", "detections")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="0")
    parser.add_argument("--half", action="store_true",
                        help="FP16 inference — required for .engine exported with half=True, optional for .pt on GPU")
    parser.add_argument("--display-size", type=int, nargs=2, metavar=("W", "H"), default=None,
                        help="Resize the display/output frames to this width and height, e.g. --display-size 1280 720")
    
    args = parser.parse_args()

    import cv2
    model = YOLO(args.weights, task="detect")
    # Handle video inputs
    if args.source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        cap = cv2.VideoCapture(args.source)
     
        frame_times = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            t0 = time.perf_counter()
            # Run inference on the current frame
            results = model.predict(
                source=frame,
                imgsz=args.imgsz,
                conf=args.conf,
                device=args.device,
                half=args.half,
                show=False,
                save=False,
            )
            res = results[0]
            for box, cls, conf in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy(), res.boxes.conf.cpu().numpy()):
                x1, y1, x2, y2 = map(int, box)
                class_id = int(cls)
                label = res.names[class_id] if hasattr(res, 'names') else str(class_id)
                # Set color: door (id 0) blue, window (id 1) green
                if class_id == 0:
                    color = (255, 0, 0)  # Blue for door
                elif class_id == 1:
                    color = (0, 255, 0)  # Green for window
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                thickness = 2
                text = f"{conf:.2f}"
                (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                cv2.putText(frame, text, (x1, y1 - 10 if y1 - 10 > 10 else y1 + th + 10), font, font_scale, color, thickness, cv2.LINE_AA)
            elapsed = time.perf_counter() - t0
            fps = 1.0 / elapsed
            frame_times.append(elapsed)

            fps_text = f"FPS: {fps:.1f}"
            fps_font = cv2.FONT_HERSHEY_SIMPLEX
            fps_scale = 2.0
            fps_thick = 3
            (tw, th), _ = cv2.getTextSize(fps_text, fps_font, fps_scale, fps_thick)
            margin = 20
            tx = frame.shape[1] - tw - margin
            ty = th + margin
            cv2.rectangle(frame, (tx - 6, margin - 6), (tx + tw + 6, ty + 6), (0, 0, 0), -1)
            cv2.putText(frame, fps_text, (tx, ty), fps_font, fps_scale, (0, 255, 255), fps_thick, cv2.LINE_AA)

            display_frame = frame
            if args.display_size is not None:
                w, h = args.display_size
                display_frame = cv2.resize(frame, (w, h))
            cv2.imshow('Detections', display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        
        if frame_times:
            steady = frame_times[1:]  # skip first frame (CUDA/TRT warmup — slower on any model)
            if steady:
                avg_fps = len(steady) / sum(steady)
                print(f"Average FPS: {avg_fps:.2f}  ({len(steady)} frames, warmup excluded)")
            else:
                print(f"Average FPS: {1.0/frame_times[0]:.2f}  (1 frame only)")

if __name__ == "__main__":
    main()
