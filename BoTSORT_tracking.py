#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from collections import defaultdict

import cv2
from ultralytics import YOLO

CLASS_COLORS = {
    "door":   (255, 0, 0),    # blue (BGR)
    "window": (0, 255, 0),    # green (BGR)
}


def local_id(cls_name, global_id, class_id_map, class_id_counter):
    if global_id not in class_id_map[cls_name]:
        class_id_counter[cls_name] += 1
        class_id_map[cls_name][global_id] = class_id_counter[cls_name]
    return class_id_map[cls_name][global_id]


def draw_fps(frame, fps):
    fps_text = f"FPS: {fps:.1f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(fps_text, font, 2.0, 3)
    margin = 20
    tx = frame.shape[1] - tw - margin
    ty = th + margin
    cv2.rectangle(frame, (tx - 6, margin - 6), (tx + tw + 6, ty + 6), (0, 0, 0), -1)
    cv2.putText(frame, fps_text, (tx, ty), font, 2.0, (0, 255, 255), 3, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BoTSORT / ByteTrack tracking with YOLO"
    )
    parser.add_argument("--weights", required=True,
                        help="Path to model (.pt or .engine)")
    parser.add_argument("--source", required=True,
                        help="Path to video file")
    parser.add_argument("--classes", nargs="+", default=None,
                        help="Class prompts for YOLO-World .pt models (e.g. --classes door window). "
                             "Skipped for standard YOLO and .engine models.")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Inference image size (default: 640)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (default: 0.25)")
    parser.add_argument("--device", default="cpu",
                        help="Inference device: 0 (CUDA), cpu, mps (default: cpu)")
    parser.add_argument("--tracker", default="botsort.yaml",
                        choices=["botsort.yaml", "bytetrack.yaml"],
                        help="Tracker config (default: botsort.yaml)")
    parser.add_argument("--half", action="store_true",
                        help="FP16 inference — requires GPU")
    parser.add_argument("--display-size", type=int, nargs=2, metavar=("W", "H"), default=None,
                        help="Resize display window (e.g. --display-size 1280 720)")
    args = parser.parse_args()

    model = YOLO(args.weights)
    if args.classes and not args.weights.endswith(".engine"):
        model.set_classes(args.classes)

    # Per-class ID remapping: {cls_name: {global_id: local_id}}
    class_id_map     = defaultdict(dict)
    class_id_counter = defaultdict(int)

    frame_times = []
    t_last = time.perf_counter()

    for result in model.track(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        half=args.half,
        tracker=args.tracker,
        stream=True,
    ):
        t_now = time.perf_counter()
        elapsed = t_now - t_last
        t_last = t_now

        frame = result.orig_img.copy()

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_name = model.names[int(box.cls)]
            global_track_id = int(box.id) if box.id is not None else -1
            track_id = local_id(cls_name, global_track_id, class_id_map, class_id_counter) \
                       if global_track_id != -1 else -1
            conf  = float(box.conf)
            color = CLASS_COLORS.get(cls_name, (0, 255, 255))
            label = f"ID:{track_id} {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)
            cv2.putText(frame, label, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, thickness=2)

        frame_times.append(elapsed)
        draw_fps(frame, 1.0 / elapsed)

        display_frame = cv2.resize(frame, tuple(args.display_size)) \
                        if args.display_size else frame
        cv2.imshow("Tracking", display_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()

    if frame_times:
        steady = frame_times[1:]
        if steady:
            avg_fps = len(steady) / sum(steady)
            print(f"Average FPS: {avg_fps:.2f}  ({len(steady)} frames, warmup excluded)")
        else:
            print(f"Average FPS: {1.0 / frame_times[0]:.2f}  (1 frame only)")


if __name__ == "__main__":
    main()
