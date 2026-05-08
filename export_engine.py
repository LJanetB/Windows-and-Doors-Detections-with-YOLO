from ultralytics import YOLO

CLASSES = ["door", "window"]

model = YOLO("pt/Yolov8s-worldv2.pt")

CLASSES = ["door", "window"]
model.set_classes(CLASSES)
model.export(format="engine", half=True, imgsz=640, device=0, simplify=True)
