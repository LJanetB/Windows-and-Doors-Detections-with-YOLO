#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import time

import cv2
import numpy as np
import torch
import torchvision.ops as tv_ops
from ultralytics import YOLO, YOLOWorld


_FIXED_COLORS: dict[str, tuple[int, int, int]] = {
    "door":   (255, 0,   0),   # blue  (BGR)
    "window": (0,   255, 0),   # green (BGR)
}

def make_colors(class_names: dict[int, str]) -> list[tuple[int, int, int]]:
    rng = random.Random(42)
    colors = []
    for i in sorted(class_names.keys()):
        name = class_names[i].lower()
        if name in _FIXED_COLORS:
            colors.append(_FIXED_COLORS[name])
        else:
            colors.append((rng.randint(50, 255), rng.randint(50, 255), rng.randint(50, 255)))
    return colors


def apply_nms(boxes: np.ndarray, scores: np.ndarray, cls_ids: np.ndarray,
              iou_thresh: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(boxes) == 0:
        return boxes, scores, cls_ids
    keep = tv_ops.nms(
        torch.from_numpy(boxes).float(),
        torch.from_numpy(scores).float(),
        iou_thresh,
    ).numpy()
    return boxes[keep], scores[keep], cls_ids[keep]


def draw_boxes(frame: np.ndarray, boxes: np.ndarray, scores: np.ndarray,
               cls_ids: np.ndarray, names: dict, colors: list) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    for box, score, cls_id in zip(boxes, scores, cls_ids):
        x1, y1, x2, y2 = map(int, box)
        ci = int(cls_id)
        color = colors[ci % len(colors)]
        text = f"{names.get(ci, ci)} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, font, 0.8, 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, text,
                    (x1, y1 - 10 if y1 - 10 > 10 else y1 + th + 10),
                    font, 0.8, color, 2, cv2.LINE_AA)


def draw_fps(frame: np.ndarray, fps: float) -> None:
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
        description="YOLO-World inference (no tiling) — supports .pt and .engine"
    )
    parser.add_argument("--weights", required=True,
                        help="Path to model (.pt for YOLO-World, .engine for TensorRT)")
    parser.add_argument("--source", required=True,
                        help="Video file path")
    parser.add_argument("--classes", nargs="+", default=None,
                        help="Class prompts for .pt YOLO-World (e.g. --classes door window). "
                             "Skipped for .engine — classes are baked in at export time.")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Inference image size (must match .engine export size)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="NMS IoU threshold")
    parser.add_argument("--device", default="0",
                        help="CUDA device index or 'cpu'")
    parser.add_argument("--half", action="store_true",
                        help="FP16 inference — required for .engine exported with half=True, "
                             "optional for .pt on GPU")
    parser.add_argument("--display-size", type=int, nargs=2, metavar=("W", "H"), default=None,
                        help="Resize display window (e.g. --display-size 1280 720)")
    args = parser.parse_args()

    is_engine = args.weights.endswith(".engine")

    if is_engine:
        model = YOLO(args.weights, task="detect")
    else:
        model = YOLOWorld(args.weights)
        if args.classes:
            model.set_classes(args.classes)

    class_names: dict[int, str] = {int(k): v for k, v in model.names.items()}
    colors = make_colors(class_names)

    if not args.source.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        print("Only video sources are supported.")
        return

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.source}")

    frame_times: list[float] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()

        results = model.predict(
            source=frame,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            half=args.half,
            verbose=False,
            show=False,
            save=False,
        )
        res = results[0]

        boxes   = res.boxes.xyxy.cpu().numpy()
        scores  = res.boxes.conf.cpu().numpy()
        cls_ids = res.boxes.cls.cpu().numpy()

        # Cross-class NMS — removes overlapping boxes regardless of class label
        boxes, scores, cls_ids = apply_nms(boxes, scores, cls_ids, args.iou)

        draw_boxes(frame, boxes, scores, cls_ids, class_names, colors)

        elapsed = time.perf_counter() - t0
        frame_times.append(elapsed)
        draw_fps(frame, 1.0 / elapsed)

        display_frame = frame
        if args.display_size is not None:
            w, h = args.display_size
            display_frame = cv2.resize(frame, (w, h))

        cv2.imshow("YOLO-World", display_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if frame_times:
        steady = frame_times[1:]  # skip first frame (CUDA/TRT warmup — slower on any model)
        if steady:
            avg_fps = len(steady) / sum(steady)
            print(f"Average FPS: {avg_fps:.2f}  ({len(steady)} frames, warmup excluded)")
        else:
            print(f"Average FPS: {1.0 / frame_times[0]:.2f}  (1 frame only)")


if __name__ == "__main__":
    main()
