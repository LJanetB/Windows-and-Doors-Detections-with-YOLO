#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.ops as tv_ops
from ultralytics import YOLO, YOLOWorld

# Make deep_sort importable from the sibling directory
_DEEP_SORT_DIR = Path(__file__).parent / "deep_sort"
sys.path.insert(0, str(_DEEP_SORT_DIR))

from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker


# ── Appearance feature extraction ────────────────────────────────────────────

def _build_tf_encoder(model_path: str):
    """Return a TF-based re-ID encoder (requires TensorFlow + mars-small128.pb)."""
    from tools.generate_detections import create_box_encoder  # noqa: PLC0415
    return create_box_encoder(model_path, batch_size=32)


def _simple_encoder(image: np.ndarray, boxes_tlwh: np.ndarray) -> np.ndarray:
    """
    Lightweight appearance descriptor: per-channel HSV histogram on the crop.
    Returns (N, 96) float32 features — no TF required.
    """
    features = []
    h_img, w_img = image.shape[:2]
    for box in boxes_tlwh:
        x, y, w, h = box
        x1, y1 = max(0, int(x)), max(0, int(y))
        x2, y2 = min(w_img - 1, int(x + w)), min(h_img - 1, int(y + h))
        if x2 <= x1 or y2 <= y1:
            features.append(np.zeros(96, dtype=np.float32))
            continue
        crop = image[y1:y2, x1:x2]
        crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([crop_hsv], [0], None, [32], [0, 180]).flatten()
        hist_s = cv2.calcHist([crop_hsv], [1], None, [32], [0, 256]).flatten()
        hist_v = cv2.calcHist([crop_hsv], [2], None, [32], [0, 256]).flatten()
        feat = np.concatenate([hist_h, hist_s, hist_v]).astype(np.float32)
        norm = np.linalg.norm(feat)
        features.append(feat / norm if norm > 0 else feat)
    return np.array(features, dtype=np.float32)


# ── Colour helpers ────────────────────────────────────────────────────────────

_FIXED_COLORS: dict[str, tuple[int, int, int]] = {
    "door":   (255, 0,   0),
    "window": (0,   255, 0),
}

def make_class_colors(class_names: dict[int, str]) -> list[tuple[int, int, int]]:
    rng = random.Random(42)
    colors = []
    for i in sorted(class_names.keys()):
        name = class_names[i].lower()
        if name in _FIXED_COLORS:
            colors.append(_FIXED_COLORS[name])
        else:
            None
    return colors

def track_color(track_id: int) -> tuple[int, int, int]:
    rng = random.Random(track_id * 137 + 42)
    return (rng.randint(50, 255), rng.randint(50, 255), rng.randint(50, 255))


# ── Drawing helpers ───────────────────────────────────────────────────────────

def draw_track(frame: np.ndarray, tlwh: np.ndarray, track_id: int,
               cls_name: str, color: tuple[int, int, int]) -> None:
    x, y, w, h = map(int, tlwh)
    x2, y2 = x + w, y + h
    font = cv2.FONT_HERSHEY_SIMPLEX
    label = f"ID{track_id} {cls_name}"
    (tw, th), _ = cv2.getTextSize(label, font, 0.7, 2)
    cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
    cv2.rectangle(frame, (x, y - th - 8), (x + tw + 4, y), color, -1)
    cv2.putText(frame, label, (x + 2, y - 4), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def draw_fps(frame: np.ndarray, fps: float) -> None:
    fps_text = f"FPS: {fps:.1f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(fps_text, font, 2.0, 3)
    margin = 20
    tx = frame.shape[1] - tw - margin
    ty = th + margin
    cv2.rectangle(frame, (tx - 6, margin - 6), (tx + tw + 6, ty + 6), (0, 0, 0), -1)
    cv2.putText(frame, fps_text, (tx, ty), font, 2.0, (0, 255, 255), 3, cv2.LINE_AA)


# ── NMS ───────────────────────────────────────────────────────────────────────

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


# ── Coordinate conversion ─────────────────────────────────────────────────────

def xyxy_to_tlwh(boxes_xyxy: np.ndarray) -> np.ndarray:
    """Convert (x1, y1, x2, y2) → (x, y, w, h)."""
    tlwh = boxes_xyxy.copy()
    tlwh[:, 2] = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
    tlwh[:, 3] = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
    return tlwh


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="YOLO-World + DeepSORT tracking — supports .pt and .engine"
    )
    # Detection args (mirrors yolo_world_infer.py)
    parser.add_argument("--weights", required=True,
                        help="Path to model (.pt for YOLO-World, .engine for TensorRT)")
    parser.add_argument("--source", required=True,
                        help="Video file path")
    parser.add_argument("--classes", nargs="+", default=None,
                        help="Class prompts for .pt YOLO-World (e.g. --classes door window). "
                             "Skipped for .engine.")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="NMS IoU threshold")
    parser.add_argument("--device", default="0",
                        help="CUDA device index or 'cpu'")
    parser.add_argument("--half", action="store_true",
                        help="FP16 inference")
    parser.add_argument("--display-size", type=int, nargs=2, metavar=("W", "H"), default=None,
                        help="Resize display window (e.g. --display-size 1280 720)")
    parser.add_argument("--save", default=None, metavar="OUT.mp4",
                        help="Save annotated video to this path")

    # DeepSORT args
    parser.add_argument("--reid-model", default=None, metavar="PATH",
                        help="Path to mars-small128.pb re-ID model for DeepSORT. "
                             "If omitted, a built-in HSV-histogram descriptor is used instead.")
    parser.add_argument("--max-cosine-distance", type=float, default=0.4,
                        help="Cosine distance gating threshold for appearance matching (default 0.4)")
    parser.add_argument("--nn-budget", type=int, default=100,
                        help="Max appearance descriptors kept per track (default 100)")
    parser.add_argument("--max-age", type=int, default=30,
                        help="Frames a track survives without a detection (default 30)")
    parser.add_argument("--n-init", type=int, default=3,
                        help="Detections required to confirm a new track (default 3)")
    parser.add_argument("--max-iou-distance", type=float, default=0.7,
                        help="IoU gating threshold for unconfirmed-track matching (default 0.7)")

    args = parser.parse_args()

    # ── Load YOLO-World model ─────────────────────────────────────────────────
    is_engine = args.weights.endswith(".engine")
    if is_engine:
        model = YOLO(args.weights, task="detect")
    else:
        model = YOLOWorld(args.weights)
        if args.classes:
            model.set_classes(args.classes)

    class_names: dict[int, str] = {int(k): v for k, v in model.names.items()}
    class_colors = make_class_colors(class_names)

    # ── Build appearance encoder ──────────────────────────────────────────────
    if args.reid_model:
        print(f"Using TF re-ID encoder: {args.reid_model}")
        encoder = _build_tf_encoder(args.reid_model)
        use_tf = True
    else:
        print("No re-ID model provided — using built-in HSV-histogram descriptor.")
        use_tf = False

    # ── Build DeepSORT tracker ────────────────────────────────────────────────
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", args.max_cosine_distance, args.nn_budget
    )
    tracker = Tracker(metric, max_iou_distance=args.max_iou_distance,
                      max_age=args.max_age, n_init=args.n_init)

    # ── Open video ────────────────────────────────────────────────────────────
    if not args.source.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        print("Only video sources are supported.")
        return

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.source}")

    writer = None
    if args.save:
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, fps_src, (w, h))

    frame_times: list[float] = []

    # ── Track→class mapping (updated each frame for confirmed tracks) ─────────
    track_class: dict[int, int] = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()

        # 1. Detect
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

        boxes_xyxy = res.boxes.xyxy.cpu().numpy()
        scores     = res.boxes.conf.cpu().numpy()
        cls_ids    = res.boxes.cls.cpu().numpy().astype(int)

        # Cross-class NMS (same as infer script)
        boxes_xyxy, scores, cls_ids = apply_nms(boxes_xyxy, scores, cls_ids, args.iou)

        # 2. Build DeepSORT Detection objects
        boxes_tlwh = xyxy_to_tlwh(boxes_xyxy) if len(boxes_xyxy) else np.empty((0, 4))

        if len(boxes_tlwh):
            if use_tf:
                features = encoder(frame, boxes_tlwh)
            else:
                features = _simple_encoder(frame, boxes_tlwh)
        else:
            features = np.empty((0, 96), dtype=np.float32)

        detections = [
            Detection(tlwh, score, feat)
            for tlwh, score, feat in zip(boxes_tlwh, scores, features)
        ]

        # 3. Update tracker
        tracker.predict()
        tracker.update(detections)

        # 4. Map detection index → class id for confirmed tracks
        #    DeepSORT doesn't store class, so we propagate it via IoU overlap
        #    between current raw detections and each confirmed track's bbox.
        if len(boxes_xyxy):
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                t_tlwh = track.to_tlwh()
                t_x2 = t_tlwh[0] + t_tlwh[2]
                t_y2 = t_tlwh[1] + t_tlwh[3]
                t_box = np.array([t_tlwh[0], t_tlwh[1], t_x2, t_y2])

                # Find best-overlapping detection
                inter_x1 = np.maximum(boxes_xyxy[:, 0], t_box[0])
                inter_y1 = np.maximum(boxes_xyxy[:, 1], t_box[1])
                inter_x2 = np.minimum(boxes_xyxy[:, 2], t_box[2])
                inter_y2 = np.minimum(boxes_xyxy[:, 3], t_box[3])
                inter_w  = np.maximum(0, inter_x2 - inter_x1)
                inter_h  = np.maximum(0, inter_y2 - inter_y1)
                inter    = inter_w * inter_h
                if inter.max() > 0:
                    best_idx = int(np.argmax(inter))
                    track_class[track.track_id] = int(cls_ids[best_idx])

        # 5. Draw confirmed tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            tlwh   = track.to_tlwh()
            tid    = track.track_id
            ci     = track_class.get(tid, 0)
            cname  = class_names.get(ci, str(ci))
            color  = track_color(tid)
            draw_track(frame, tlwh, tid, cname, color)

        elapsed = time.perf_counter() - t0
        frame_times.append(elapsed)
        draw_fps(frame, 1.0 / elapsed)

        if writer:
            writer.write(frame)

        display_frame = frame
        if args.display_size is not None:
            dw, dh = args.display_size
            display_frame = cv2.resize(frame, (dw, dh))

        cv2.imshow("YOLO-World + DeepSORT", display_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
        print(f"Saved to: {args.save}")
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
