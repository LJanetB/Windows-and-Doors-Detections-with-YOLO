#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.ops as tv_ops
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Tiling helpers
# ---------------------------------------------------------------------------

def get_tiles(h, w, tile_w, tile_h, overlap):
    """Return list of (x1, y1, x2, y2) tile rects covering the full image."""
    step_x = max(1, int(tile_w * (1 - overlap)))
    step_y = max(1, int(tile_h * (1 - overlap)))
    tiles = []
    y = 0
    while True:
        y2 = min(y + tile_h, h)
        y1 = max(0, y2 - tile_h)
        x = 0
        while True:
            x2 = min(x + tile_w, w)
            x1 = max(0, x2 - tile_w)
            tiles.append((x1, y1, x2, y2))
            if x2 == w:
                break
            x += step_x
        if y2 == h:
            break
        y += step_y
    return tiles


def tiled_predict(model, image, tile_w, tile_h, overlap,
                  conf_thresh, iou_thresh, device, half):
    """Run tiled inference and return merged (boxes_xyxy, scores, cls_ids)."""
    h, w = image.shape[:2]
    tiles = get_tiles(h, w, tile_w, tile_h, overlap)

    all_boxes, all_scores, all_cls = [], [], []

    for (x1, y1, x2, y2) in tiles:
        results = model.predict(
            image[y1:y2, x1:x2],
            conf=conf_thresh,
            verbose=False,
            half=half,
            device=device,
        )
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            continue
        boxes  = r.boxes.xyxy.cpu().numpy().copy()
        scores = r.boxes.conf.cpu().numpy()
        cls    = r.boxes.cls.cpu().numpy()

        boxes[:, [0, 2]] += x1
        boxes[:, [1, 3]] += y1

        all_boxes.append(boxes)
        all_scores.append(scores)
        all_cls.append(cls)

    if not all_boxes:
        return np.empty((0, 4)), np.empty(0), np.empty(0)

    all_boxes  = np.concatenate(all_boxes)
    all_scores = np.concatenate(all_scores)
    all_cls    = np.concatenate(all_cls)

    # Per-class NMS to remove duplicate detections across tile boundaries
    keep = []
    for cls_id in np.unique(all_cls):
        mask = all_cls == cls_id
        idx  = np.where(mask)[0]
        kept = tv_ops.nms(
            torch.from_numpy(all_boxes[mask]).float(),
            torch.from_numpy(all_scores[mask]).float(),
            iou_thresh,
        ).numpy()
        keep.extend(idx[kept])

    return all_boxes[keep], all_scores[keep], all_cls[keep]


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

_COLORS = [(0, 0, 255), (0, 255, 0)]   # blue / green (BGR); extend for more classes


def draw(image, boxes, scores, cls_ids, display_names):
    for box, score, cls in zip(boxes, scores, cls_ids):
        x1, y1, x2, y2 = map(int, box)
        ci    = int(cls)
        color = _COLORS[ci % len(_COLORS)]
        label = f"{display_names[ci] if ci < len(display_names) else ci} {score:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(image, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(image, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return image


# ---------------------------------------------------------------------------
# Image / video entry points
# ---------------------------------------------------------------------------

def to_display(image, display_size):
    if display_size is None:
        return image
    return cv2.resize(image, display_size, interpolation=cv2.INTER_LINEAR)


def run_image(model, path, args, display_names):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    boxes, scores, cls_ids = tiled_predict(
        model, image,
        args.tile_width, args.tile_height, args.overlap,
        args.conf, args.iou, args.device, args.half,
    )
    print(f"Detected {len(boxes)} object(s)")
    draw(image, boxes, scores, cls_ids, display_names)
    display_size = tuple(args.display_size) if args.display_size else None
    cv2.imshow("Tiled YOLO-World", to_display(image, display_size))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_video(model, path, args, display_names):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")

    display_size = tuple(args.display_size) if args.display_size else None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        boxes, scores, cls_ids = tiled_predict(
            model, frame,
            args.tile_width, args.tile_height, args.overlap,
            args.conf, args.iou, args.device, args.half,
        )
        infer_ms = (time.perf_counter() - t0) * 1000

        draw(frame, boxes, scores, cls_ids, display_names)

        infer_text = f"Infer: {infer_ms:.1f} ms"
        cv2.putText(frame, infer_text, (28, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 7)       # black outline
        cv2.putText(frame, infer_text, (28, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 4)   # yellow fill
        cv2.imshow("Tiled YOLO-World", to_display(frame, display_size))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tiled YOLO-World inference — supports .pt and .engine"
    )
    parser.add_argument("--weights", required=True,
                        help="Path to model (.pt for YOLO-World, .engine for TensorRT)")
    parser.add_argument("--source", required=True,
                        help="Path to image or video file")
    parser.add_argument("--classes", nargs="+", default=["a door", "a window"],
                        help="YOLO-World text prompts (e.g. --classes 'a door' 'a window'). "
                             "Skipped for .engine — classes are baked in at export time.")
    parser.add_argument("--display-names", nargs="+", default=None,
                        help="Short labels shown on boxes (default: same as --classes). "
                             "E.g. --display-names door window")
    parser.add_argument("--tile-width", type=int, default=672,
                        help="Tile width in pixels (default: 672)")
    parser.add_argument("--tile-height", type=int, default=672,
                        help="Tile height in pixels (default: 672)")
    parser.add_argument("--overlap", type=float, default=0.30,
                        help="Tile overlap fraction 0.0-0.5 (default: 0.30)")
    parser.add_argument("--conf", type=float, default=0.20,
                        help="Confidence threshold (default: 0.20)")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="NMS IoU threshold (default: 0.45)")
    parser.add_argument("--device", default="cpu",
                        help="Inference device: 0 (CUDA), cpu, mps (default: cpu)")
    parser.add_argument("--half", action="store_true",
                        help="FP16 inference — requires GPU")
    parser.add_argument("--display-size", type=int, nargs=2, metavar=("W", "H"), default=None,
                        help="Resize display window (e.g. --display-size 1320 800)")
    args = parser.parse_args()

    display_names = args.display_names if args.display_names else args.classes

    model = YOLO(args.weights)
    if not args.weights.endswith(".engine"):
        model.set_classes(args.classes)
        if args.device not in ("cpu", "mps"):
            model.to(args.device)

    p = Path(args.source)
    if not p.exists():
        raise FileNotFoundError(f"Input not found: {p}")

    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}:
        run_image(model, p, args, display_names)
    else:
        run_video(model, p, args, display_names)


if __name__ == "__main__":
    main()
