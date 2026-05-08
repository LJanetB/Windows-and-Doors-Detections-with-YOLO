#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Path to YOLO .pt weights")
    parser.add_argument("--source", required=True, help="Path to input image")
    parser.add_argument("--conf", type=float, default=0.10, help="Confidence threshold")
    parser.add_argument("--device", default="cuda:0", help="cuda:0 or cpu")
    parser.add_argument("--imgsz", type=int, default=640, help="Model inference size per tile")
    parser.add_argument("--slice_height", type=int, default=512)
    parser.add_argument("--slice_width", type=int, default=512)
    parser.add_argument("--overlap_height_ratio", type=float, default=0.20)
    parser.add_argument("--overlap_width_ratio", type=float, default=0.20)
    parser.add_argument("--output_dir", default="runs/detect/sahi_predict/")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=args.weights,
        confidence_threshold=args.conf,
        device=args.device,
        image_size=args.imgsz,
    )

    result = get_sliced_prediction(
        args.source,
        detection_model,
        slice_height=args.slice_height,
        slice_width=args.slice_width,
        overlap_height_ratio=args.overlap_height_ratio,
        overlap_width_ratio=args.overlap_width_ratio,
    )

    print(f"Total detections: {len(result.object_prediction_list)}")

    # Color per class: door=green, window=blue
    CLASS_COLORS = {"door": (0, 255, 0), "window": (255, 0, 0)}
    DEFAULT_COLOR = (0, 255, 255)

    # Draw bounding boxes with class-specific colors (no text)
    import cv2
    img = cv2.imread(args.source)
    for pred in result.object_prediction_list:
        bbox = pred.bbox
        x1, y1, x2, y2 = int(bbox.minx), int(bbox.miny), int(bbox.maxx), int(bbox.maxy)
        cls_name = pred.category.name.lower()
        color = CLASS_COLORS.get(cls_name, DEFAULT_COLOR)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    out_path = output_dir / f"{Path(args.source).stem}.jpg"
    cv2.imwrite(str(out_path), img)
    print(f"Saved result to: {out_path}")


if __name__ == "__main__":
    main()
