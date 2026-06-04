"""
车牌检测器 — YOLOv8n 微调
"""
import torch
from ultralytics import YOLO
from pathlib import Path


class PlateDetector:
    """车牌检测，基于 YOLOv8n"""

    def __init__(self, model_path: str = None):
        if model_path and Path(model_path).exists():
            self.model = YOLO(model_path)
        else:
            self.model = YOLO('yolov8n.pt')

    def train(self, data_yaml: str, epochs: int = 100, imgsz: int = 640, **kwargs):
        """训练检测模型"""
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=16,
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            cos_lr=True,
            close_mosaic=10,
            augment=True,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10.0,
            translate=0.1,
            scale=0.5,
            shear=2.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            **kwargs,
        )
        return results

    def detect(self, image):
        """检测单张图片中的车牌位置"""
        results = self.model(image, verbose=False)
        plates = []
        for r in results:
            boxes = r.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    plates.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': round(conf, 4),
                    })
        return plates

    def export_onnx(self, output_path: str = "plate_detector.onnx"):
        """导出 ONNX 格式"""
        self.model.export(format='onnx', imgsz=640, simplify=True)
        print(f"✅ 检测模型已导出到 {output_path}")
