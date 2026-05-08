import time
import cv2
import numpy as np
import torch
import torchvision.ops as tv_ops
from pathlib import Path
from ultralytics import YOLO

# --- Config ---
INPUT_PATH   = "video1.mp4"   # image or video file
TILE_WIDTH   = 672             # tile width fed to the model — match engine imgsz to avoid rescaling
TILE_HEIGHT  = 672             # tile height fed to the model
OVERLAP      = 0.30            # fraction of tile that overlaps with neighbours
CONF_THRESH  = 0.20            # minimum detection confidence
IOU_THRESH   = 0.45            # NMS IoU threshold across tiles

DEVICE = 0 if torch.cuda.is_available() else "cpu"
HALF   = torch.cuda.is_available()   # FP16 on GPU, FP32 on CPU

CLASSES       = ["a door", "a window"]
DISPLAY_NAMES = ["door", "window"]
COLORS        = [(0, 0, 255), (0, 255, 0)]   # blue / green

DISPLAY_SIZE  = (1320, 800)    # (width, height) of the output window; set to None to keep original size


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


def tiled_predict(model, image):
    """Run tiled inference and return merged (boxes_xyxy, scores, cls_ids)."""
    h, w = image.shape[:2]
    tiles = get_tiles(h, w, TILE_WIDTH, TILE_HEIGHT, OVERLAP)

    all_boxes, all_scores, all_cls = [], [], []

    for (x1, y1, x2, y2) in tiles:
        results = model.predict(image[y1:y2, x1:x2], conf=CONF_THRESH, verbose=False, half=HALF, device=DEVICE)
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

    # per-class NMS to remove duplicate detections across tile boundaries
    keep = []
    for cls_id in np.unique(all_cls):
        mask = all_cls == cls_id
        idx  = np.where(mask)[0]
        kept = tv_ops.nms(
            torch.from_numpy(all_boxes[mask]).float(),
            torch.from_numpy(all_scores[mask]).float(),
            IOU_THRESH,
        ).numpy()
        keep.extend(idx[kept])

    return all_boxes[keep], all_scores[keep], all_cls[keep]


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw(image, boxes, scores, cls_ids):
    for box, score, cls in zip(boxes, scores, cls_ids):
        x1, y1, x2, y2 = map(int, box)
        ci    = int(cls)
        color = COLORS[ci % len(COLORS)]
        label = f"{DISPLAY_NAMES[ci]} {score:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(image, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(image, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return image


# ---------------------------------------------------------------------------
# Image / video entry points
# ---------------------------------------------------------------------------

def to_display(image):
    if DISPLAY_SIZE is None:
        return image
    return cv2.resize(image, DISPLAY_SIZE, interpolation=cv2.INTER_LINEAR)


def run_image(model, path):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    boxes, scores, cls_ids = tiled_predict(model, image)
    print(f"Detected {len(boxes)} object(s)")
    draw(image, boxes, scores, cls_ids)
    cv2.imshow("Tiled YOLO-World", to_display(image))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_video(model, path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        boxes, scores, cls_ids = tiled_predict(model, frame)
        infer_ms = (time.perf_counter() - t0) * 1000

        draw(frame, boxes, scores, cls_ids)

        infer_text = f"Infer: {infer_ms:.1f} ms"
        cv2.putText(frame, infer_text, (28, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 7)       # black outline
        cv2.putText(frame, infer_text, (28, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 4)   # yellow fill
        cv2.imshow("Tiled YOLO-World", to_display(frame))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    model_path = "yolov8s-world.engine"
    model = YOLO(model_path)
    # set_classes only works on .pt YOLO-World models; engine files have classes baked in
    if not model_path.endswith(".engine"):
        model.set_classes(CLASSES)
        model.to(DEVICE)

    p = Path(INPUT_PATH)
    if not p.exists():
        raise FileNotFoundError(f"Input not found: {p}")

    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}:
        run_image(model, p)
    else:
        run_video(model, p)


if __name__ == "__main__":
    main()
