#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time

from sahi import AutoDetectionModel
from sahi.models.ultralytics import UltralyticsDetectionModel
from sahi.predict import get_sliced_prediction


class TRTDetectionModel(UltralyticsDetectionModel):
    """Thin wrapper that skips model.to(device) for TensorRT engines.

    SAHI's default loader calls model.to(device) which fails on .engine files.
    Device is already forwarded via the predict() kwargs, so skipping it is safe.
    """

    def load_model(self):
        from ultralytics import YOLO
        try:
            model = YOLO(self.model_path)
            if not self.model_path.endswith(".engine"):
                model.to(self.device)
            self.set_model(model)
        except Exception as e:
            raise TypeError("model_path is not a valid Ultralytics model path: ", e)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Model inference size per tile")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--display-size", type=int, nargs=2, metavar=("W", "H"), default=None,
                        help="Resize the display/output frames to this width and height, e.g. --display-size 1280 720")
    parser.add_argument("--slice-height", type=int, default=640,
                        help="Tile height in pixels (default: 640)")
    parser.add_argument("--slice-width", type=int, default=640,
                        help="Tile width in pixels (default: 640)")
    parser.add_argument("--overlap-height-ratio", type=float, default=0.10,
                        help="Vertical overlap between tiles, 0.0-0.5 (default: 0.10)")
    parser.add_argument("--overlap-width-ratio", type=float, default=0.10,
                        help="Horizontal overlap between tiles, 0.0-0.5 (default: 0.10)")
    parser.add_argument("--half", action="store_true",
                        help="FP16 inference — required for .engine exported with half=True, "
                             "optional for .pt on GPU")
    parser.add_argument("--postprocess-type", default="NMM",
                        choices=["NMS", "NMM", "GREEDYNMM"],
                        help="How overlapping tile detections are merged (default: NMM)")
    parser.add_argument("--postprocess-match-threshold", type=float, default=0.5,
                        help="IoU/IoS threshold for merging tile detections (default: 0.5)")

    args = parser.parse_args()

    import cv2

    if args.weights.endswith(".engine"):
        detection_model = TRTDetectionModel(
            model_path=args.weights,
            confidence_threshold=args.conf,
            device=args.device,
            image_size=args.imgsz,
            half=args.half,
        )
    else:
        detection_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=args.weights,
            confidence_threshold=args.conf,
            device=args.device,
            image_size=args.imgsz,
            half=args.half,
        )

    if args.source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        cap = cv2.VideoCapture(args.source)
        

        frame_count = 0
        last_infer_time = None
        infer_fps = 0.0

        while True:
            
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            # Draw boxes and print detections
            label_counts = dict()
            Detections = dict()
            if frame_count % 1 == 0: #skip by 10 frames to speed up processing

                print(f"\n ------------------------------------------------- Frame {frame_count} ------------------------------------------------\n")
            # SAHI expects RGB; convert then draw on original BGR frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                now = time.perf_counter()
                if last_infer_time is not None:
                    infer_fps = 1.0 / (now - last_infer_time)
                last_infer_time = now

                # Downscale 4K frame to 1080p for faster processing
                orig_h, orig_w = frame_rgb.shape[:2]
                if frame_rgb.shape[1] > 1920:
                    frame_rgb = cv2.resize(frame_rgb, (1920, 1080), interpolation=cv2.INTER_LINEAR)
                scale_x = orig_w / frame_rgb.shape[1]
                scale_y = orig_h / frame_rgb.shape[0]

                result = get_sliced_prediction(
                    frame_rgb,
                    detection_model,
                    slice_height=args.slice_height,
                    slice_width=args.slice_width,
                    overlap_height_ratio=args.overlap_height_ratio,
                    overlap_width_ratio=args.overlap_width_ratio,
                    postprocess_type=args.postprocess_type,
                    postprocess_match_threshold=args.postprocess_match_threshold,
                    verbose=0,
                )


                for pred in result.object_prediction_list:
                    bbox = pred.bbox
                    x1 = int(bbox.minx * scale_x)
                    y1 = int(bbox.miny * scale_y)
                    x2 = int(bbox.maxx * scale_x)
                    y2 = int(bbox.maxy * scale_y)
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    W = x2 - x1
                    H = y2 - y1
                    conf = pred.score.value
                    label = pred.category.name.lower()
                    class_id = pred.category.id
                    label_counts[label] = label_counts.get(label, 0) + 1
                    label_id = f"{label} {label_counts[label]}"
                    Detections[label_id] = (cx, cy, W, H, float(f'{conf:.2f}'))
                    # Set color: door (id 0) blue, window (id 1) green
                    if class_id == 0:
                        color = (255, 0, 0)  # Blue for door
                    elif class_id == 1:
                        color = (0, 255, 0)  # Green for window
                    else:
                        color = (0, 255, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.8
                    thickness = 2
                    text = f"{conf:.2f}"
                    (_, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
                    cv2.putText(frame, text, (x1, y1 - 10 if y1 - 10 > 10 else y1 + th + 10), font, font_scale, color, thickness, cv2.LINE_AA)
                print("\n")
                print("Detections:", Detections)

            # Draw inference FPS in top-right (uses last computed value on skipped frames)
            fps_text = f"FPS: {infer_fps:.1f}"
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
       


if __name__ == "__main__":
    main()
