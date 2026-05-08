#!/usr/bin/env python3
"""
Benchmark YOLO-World on Dataset_yoloF (door / window).
Reports per-class and mean Precision, Recall, mAP@50, mAP@50:95, and FPS.

Run from the project root:
    python benchmark_yolo_world.py
"""

import time
import cv2
import torch
from pathlib import Path
from ultralytics import YOLO, YOLOWorld

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_PATH = "tensorrt/Yolov8s-worldv2.engine"                      # swap to yolov8l-world.pt if desired
DATA_YAML  = "Dataset_yoloF/yolo_doors_windows.yaml"
CLASSES    = ["door", "window"]
SPLIT      = "val"          # "val" or "test"
CONF       = 0.001          # low threshold — ultralytics sweeps it for mAP
IOU        = 0.60
BATCH      = 1
IMG_SIZE   = 640
DEVICE     = 0 if torch.cuda.is_available() else "cpu"
HALF       = torch.cuda.is_available()

FPS_WARMUP = 20             # frames discarded before timing starts
FPS_FRAMES = 100            # frames used for the FPS measurement
# ───────────────────────────────────────────────────────────────────────────────


def load_fps_images(img_dir: Path, n: int) -> list:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = [p for p in sorted(img_dir.iterdir()) if p.suffix.lower() in exts][:n]
    frames = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is not None:
            frames.append(img)
    return frames


def measure_fps(model, val_img_dir: Path) -> tuple[float, float]:
    """Return (fps, latency_ms) measured on single-image inference."""
    total_needed = FPS_WARMUP + FPS_FRAMES
    frames = load_fps_images(val_img_dir, total_needed)

    if len(frames) < FPS_WARMUP + 1:
        print(f"  [warn] Only {len(frames)} images found for FPS test.")

    for f in frames[:FPS_WARMUP]:
        model.predict(f, conf=0.25, verbose=False, half=HALF)

    timed = frames[FPS_WARMUP:]
    if not timed:
        return 0.0, 0.0

    t0 = time.perf_counter()
    for f in timed:
        model.predict(f, conf=0.25, verbose=False, half=HALF)
    elapsed = time.perf_counter() - t0

    fps = len(timed) / elapsed
    latency_ms = elapsed / len(timed) * 1000
    return fps, latency_ms


def fmt_row(label: str, mean_val: float, per_class: list[float]) -> str:
    parts = [f"  {label:<22}  {mean_val:>8.4f}"]
    for v in per_class:
        parts.append(f"  {v:>10.4f}")
    return "".join(parts)


def main():
    sep  = "=" * 64
    dash = "-" * 64

    print(f"\n{sep}")
    print(f"  YOLO-World Benchmark")
    print(f"  Model  : {MODEL_PATH}")
    print(f"  Dataset: {DATA_YAML}  [{SPLIT} split]")
    print(f"  Device : {'CUDA' if DEVICE != 'cpu' else 'CPU'}  |  half={HALF}")
    print(f"{sep}\n")

    # Engine files have classes baked in — use plain YOLO to load them
    if Path(MODEL_PATH).suffix == ".engine":
        model = YOLO(MODEL_PATH)
        imgsz = IMG_SIZE
    else:
        model = YOLOWorld(MODEL_PATH)
        model.set_classes(CLASSES)       #Only needed for .pt models, .engine files have classes baked in
        imgsz = IMG_SIZE

    # ── Precision / Recall / mAP ──────────────────────────────────────────────
    print("Running validation …")
    metrics = model.val(
        data=DATA_YAML,
        split=SPLIT,
        conf=CONF,
        iou=IOU,
        batch=BATCH,
        imgsz=imgsz,
        device=DEVICE,
        half=HALF,
        verbose=False,
    )

    mp       = metrics.box.mp     # mean precision
    mr       = metrics.box.mr     # mean recall
    map50    = metrics.box.map50  # mAP @ IoU=0.50
    map5095  = metrics.box.map    # mAP @ IoU=0.50:0.95

    p_cls    = metrics.box.p      # per-class precision
    r_cls    = metrics.box.r      # per-class recall
    ap50_cls = metrics.box.ap50   # per-class AP@50
    ap_cls   = metrics.box.ap     # per-class AP@50:95

    # ── FPS ───────────────────────────────────────────────────────────────────
    print("Measuring FPS …")
    val_img_dir = Path("Dataset_yoloF/door_window/images") / SPLIT
    fps, latency_ms = measure_fps(model, val_img_dir)

    # ── Report ────────────────────────────────────────────────────────────────
    header_cls = "".join(f"  {c:>10}" for c in CLASSES)
    print(f"\n{dash}")
    print(f"  Results — [{SPLIT}] split   (conf threshold swept for mAP)")
    print(f"{dash}")
    print(f"  {'Metric':<22}  {'Mean':>8}{header_cls}")
    print(f"  {'-'*60}")
    print(fmt_row("Precision",     mp,      p_cls))
    print(fmt_row("Recall",        mr,      r_cls))
    print(fmt_row("mAP @ 0.50",   map50,   ap50_cls))
    print(fmt_row("mAP @ 0.50:95", map5095, ap_cls))
    print(f"{dash}")
    if fps > 0:
        print(f"  FPS (single-image)    :  {fps:>7.1f} fps")
        print(f"  Latency (single-image):  {latency_ms:>7.1f} ms")
    else:
        print("  FPS measurement skipped — no images found.")
    print(f"{dash}\n")


if __name__ == "__main__":
    main()
