import time

import cv2
from collections import defaultdict
from ultralytics import YOLO

model_path = "/home/user/Desktop/Yolo_Inference/tensorrt/Yolov8s-worldv2.engine"
video_path = "/home/user/Desktop/Yolo_Inference/videos/video1_reversed.mp4"


CLASS_COLORS = {
    "door": (255, 0, 0),   # blue (BGR)
    "window": (0, 255, 0),      # green (BGR)
}
DISPLAY_SIZE = (1280, 720)  # (width, height)

model = YOLO(model_path)

# Per-class ID remapping: {cls_name: {global_id: local_id}}
class_id_map = defaultdict(dict)
class_id_counter = defaultdict(int)

def local_id(cls_name, global_id):
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

#-------------------------------------------- Tracking loop --------------------------------------------#
frame_times = []
t_last = time.perf_counter()

for result in model.track(source=video_path,
                          imgsz=640,
                          conf=0.25,
                          device="0",
                          tracker= "bytetrack.yaml", #bytetrack.yaml
                          stream=True):
    # Measure from previous frame boundary — captures inference + draw time
    t_now = time.perf_counter()
    elapsed = t_now - t_last
    t_last = t_now

    frame = result.orig_img.copy()

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_name = model.names[int(box.cls)]
        global_track_id = int(box.id) if box.id is not None else -1
        track_id = local_id(cls_name, global_track_id) if global_track_id != -1 else -1
        conf = float(box.conf)
        color = CLASS_COLORS.get(cls_name, (0, 255, 255))
        label = f"ID:{track_id} {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)
        cv2.putText(frame, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, thickness=2)

    frame_times.append(elapsed)
    draw_fps(frame, 1.0 / elapsed)

    cv2.imshow("Tracking", cv2.resize(frame, DISPLAY_SIZE))
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
