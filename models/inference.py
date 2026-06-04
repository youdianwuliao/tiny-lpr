"""
端到端车牌识别推理 — ONNX Runtime
支持 CPU / GPU (CUDA / TensorRT)
"""
import numpy as np
import cv2
import json
from pathlib import Path
from typing import Optional


class LPRInference:
    """端到端车牌识别推理器"""

    def __init__(
        self,
        detector_path: str = "models/plate_detector.onnx",
        recognizer_path: str = "models/plate_recognizer.onnx",
        char_map_path: str = "models/plate_recognizer.json",
        device: str = "cpu",
        confidence_threshold: float = 0.5,
    ):
        """
        Args:
            detector_path: 检测器 ONNX 路径
            recognizer_path: 识别器 ONNX 路径
            char_map_path: 字符映射 JSON
            device: 'cpu' / 'cuda'
            confidence_threshold: 检测置信度阈值
        """
        self.conf_threshold = confidence_threshold
        self.device = device

        # 加载 ONNX Runtime
        import onnxruntime as ort

        providers = {
            'cpu': ['CPUExecutionProvider'],
            'cuda': ['CUDAExecutionProvider', 'CPUExecutionProvider'],
            'tensorrt': ['TensorrtExecutionProvider', 'CUDAExecutionProvider'],
        }.get(device, ['CPUExecutionProvider'])

        self.detector = ort.InferenceSession(detector_path, providers=providers)
        self.recognizer = ort.InferenceSession(recognizer_path, providers=providers)

        # 加载字符映射
        char_map_path = Path(char_map_path)
        if char_map_path.exists():
            with open(char_map_path, encoding='utf-8') as f:
                meta = json.load(f)
            self.char_list = meta['char_list']
        else:
            # 默认字符集
            from scripts.train_recognizer import build_char_map
            _, self.char_list = build_char_map()

    def __call__(self, image) -> list:
        """
        识别图片中的车牌

        Args:
            image: 图片路径 (str) 或 numpy 数组 (H, W, 3) BGR
        Returns:
            [{'plate': '京A12345', 'bbox': [x1,y1,x2,y2], 'confidence': 0.95}, ...]
        """
        if isinstance(image, str):
            img = cv2.imread(image)
        else:
            img = image

        if img is None:
            return []

        # 1. 检测车牌
        plates = self._detect(img)
        if not plates:
            return []

        # 2. 识别每个车牌
        results = []
        for plate in plates:
            x1, y1, x2, y2 = plate['bbox']
            crop = img[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            # 预处理
            crop_input = self._preprocess(crop)
            if crop_input is None:
                continue

            # 推理
            ort_inputs = {self.recognizer.get_inputs()[0].name: crop_input}
            ort_output = self.recognizer.run(None, ort_inputs)[0]  # (1, T, 68)

            # CTC 解码
            plate_number = self._ctc_decode(ort_output[0])

            results.append({
                'plate': plate_number,
                'bbox': plate['bbox'],
                'confidence': plate['confidence'],
            })

        return results

    def _detect(self, img: np.ndarray) -> list:
        """检测车牌位置"""
        # YOLO 预处理
        h, w = img.shape[:2]
        input_img = cv2.resize(img, (640, 640))
        input_img = input_img.transpose(2, 0, 1)  # HWC → CHW
        input_img = np.expand_dims(input_img, 0).astype(np.float32) / 255.0

        ort_inputs = {self.detector.get_inputs()[0].name: input_img}
        det_output = self.detector.run(None, ort_inputs)[0]  # (1, 84, 8400)

        # 解析 YOLOv8 输出
        # 简化版：提取第一个 batch 的检测结果
        det_output = det_output[0]  # (84, 8400)
        det_output = det_output.transpose(1, 0)  # (8400, 84)

        plates = []
        for det in det_output:
            # YOLOv8 格式: [x,y,w,h, obj_conf, class_scores...]
            boxes = det[:4]
            scores = det[4:]
            conf = float(scores.max())
            
            if conf < self.conf_threshold:
                continue

            # 转换回原图坐标
            cx, cy, bw, bh = boxes
            cx = (cx / 640) * w
            cy = (cy / 640) * h
            bw = (bw / 640) * w
            bh = (bh / 640) * h

            x1 = int(cx - bw / 2)
            y1 = int(cy - bh / 2)
            x2 = int(cx + bw / 2)
            y2 = int(cy + bh / 2)

            plates.append({
                'bbox': [max(0, x1), max(0, y1), min(w, x2), min(h, y2)],
                'confidence': round(conf, 4),
            })

        return plates

    def _preprocess(self, crop: np.ndarray) -> Optional[np.ndarray]:
        """预处理车牌裁剪区域"""
        try:
            crop = cv2.resize(crop, (160, 48))
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop = crop.astype(np.float32) / 255.0
            crop = (crop - 0.5) / 0.5
            crop = crop.transpose(2, 0, 1)  # HWC → CHW
            crop = np.expand_dims(crop, 0)   # add batch
            return crop
        except:
            return None

    def _ctc_decode(self, log_probs: np.ndarray) -> str:
        """CTC 贪心解码"""
        indices = np.argmax(log_probs, axis=-1)  # (T,)
        merged = []
        prev = -1
        for idx in indices:
            if idx != prev and idx != 0:  # 跳过 blank 和重复
                if idx < len(self.char_list):
                    merged.append(self.char_list[idx])
            prev = idx
        return ''.join(merged)


# 便捷函数
def recognize_plate(image_path: str) -> list:
    """快速识别车牌"""
    lpr = LPRInference()
    return lpr(image_path)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        lpr = LPRInference()
        results = lpr(sys.argv[1])
        for r in results:
            print(f"🚗 车牌: {r['plate']} | 置信度: {r['confidence']} | 位置: {r['bbox']}")
    else:
        print("用法: python models/inference.py <图片路径>")