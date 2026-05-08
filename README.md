''' Yolo-world inference '''

# Fine-tuned .pt (classes already in model, no --classes needed)
python world_infer_video.py --weights pt/Yolov8s-worldv2.pt --source video1.mp4 --half

# .pt with explicit class prompts (open-vocab pretrained)
python world_infer_video.py --weights pt/Yolov8s-worldv2.pt --source video1.mp4 --classes door window --half

# TensorRT engine (--classes ignored, baked-in names used)
python world_infer_video.py --weights tensorrt/Yolov8s-worldv2.engine --source video1.mp4 --half


''' Other Yolo models inferencing '''

# .pt model (FP32)
python infer_video.py --weights Yolov8s-worldv2.pt --source video1.mp4

# .pt model (FP16)
python infer_video.py --weights Yolov8s-worldv2.pt --source video1.mp4 --half

# .engine model (exported with half=True — must pass --half)
python infer_video.py --weights tensorrt/Yolov8s-worldv2.engine --source video1.mp4 --half
