# TinyLPR 部署指南

训练完成后，如何把模型用到实际场景中。

---

## 一、部署概览

```
训练好的模型 → ONNX 导出 → 选择部署方式
                              ├── 服务器 (API 调用)
                              ├── 手机 App (本地推理)
                              ├── 嵌入式设备 (Jetson/树莓派)
                              └── Docker 容器
```

---

## 二、服务器部署（最简单）

### 2.1 FastAPI 服务

```bash
# 安装依赖
pip install fastapi uvicorn onnxruntime opencv-python pillow

# 确认模型文件存在
ls models/plate_detector.onnx models/plate_recognizer.onnx models/plate_recognizer.json

# 启动服务
python app/main.py
# → http://localhost:8000
```

### 2.2 API 调用方式

```python
# 客户端调用
import requests

url = "http://your-server:8000/api/recognize"
files = {'file': open('car.jpg', 'rb')}
resp = requests.post(url, files=files)
print(resp.json())
# → {'results': [{'plate': '京A12345', 'bbox': [100,200,300,260], 'confidence': 0.98}]}
```

### 2.3 Docker 部署

```dockerfile
# Dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python", "app/main.py"]
```

```bash
# 构建 + 运行
docker build -t tiny-lpr .
docker run -p 8000:8000 tiny-lpr
```

### 2.4 生产环境（nginx + gunicorn）

```bash
# 用 gunicorn + uvicorn workers
pip install gunicorn
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

---

## 三、手机 App 部署

### 3.1 Android 端

#### 路线 A：ONNX Runtime Mobile（推荐，最简单）

```kotlin
// build.gradle.kts
dependencies {
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.17.0")
}
```

```kotlin
// PlateRecognizer.kt
import ai.onnxruntime.*

class PlateRecognizer(context: Context) {
    private val detector: OrtSession
    private val recognizer: OrtSession

    init {
        val env = OrtEnvironment.getEnvironment()
        // 模型放在 assets 目录
        val detectorBytes = context.assets.open("plate_detector.onnx").readBytes()
        val recognizerBytes = context.assets.open("plate_recognizer.onnx").readBytes()
        detector = env.createSession(detectorBytes)
        recognizer = env.createSession(recognizerBytes)
    }

    fun recognize(bitmap: Bitmap): List<PlateResult> {
        // 1. 转 tensor
        val input = bitmapToTensor(bitmap, 640, 640)
        
        // 2. 检测
        val detOutput = detector.run(mapOf("images" to input))
        
        // 3. 裁剪 + 识别
        val plates = mutableListOf<PlateResult>()
        for (box in parseDetections(detOutput)) {
            val crop = cropPlate(bitmap, box)
            val cropTensor = bitmapToTensor(crop, 160, 48)
            val recOutput = recognizer.run(mapOf("input" to cropTensor))
            val plate = ctcDecode(recOutput)
            plates.add(PlateResult(plate, box))
        }
        return plates
    }
}
```

#### 路线 B：NCNN（更轻量，适合低端机）

```bash
# 1. ONNX → NCNN
git clone https://github.com/Tencent/ncnn
cd ncnn && mkdir build && cd build && cmake .. && make

# 转换
./tools/onnx/onnx2ncnn plate_detector.onnx plate_detector.param plate_detector.bin
./tools/onnx/onnx2ncnn plate_recognizer.onnx plate_recognizer.param plate_recognizer.bin

# 2. 优化
ncnnoptimize plate_detector.param plate_detector.bin \
             plate_detector_opt.param plate_detector_opt.bin 0
```

### 3.2 iOS 端

#### 路线 A：ONNX → CoreML（推荐）

```bash
# 安装 coremltools
pip install coremltools

# 转换
python -c "
import coremltools as ct
import onnx

# 检测器
onnx_model = onnx.load('plate_detector.onnx')
coreml_model = ct.convert(onnx_model, minimum_deployment_target=ct.target.iOS15)
coreml_model.save('PlateDetector.mlmodel')

# 识别器
onnx_model = onnx.load('plate_recognizer.onnx')
coreml_model = ct.convert(onnx_model, minimum_deployment_target=ct.target.iOS15)
coreml_model.save('PlateRecognizer.mlmodel')
"
```

```swift
// PlateRecognizer.swift
import CoreML
import Vision
import UIKit

class PlateRecognizer {
    private let detector: VNCoreMLModel
    private let recognizer: PlateRecognizerML
    
    init() throws {
        let detModel = try VNCoreMLModel(for: PlateDetector().model)
        self.detector = detModel
        self.recognizer = try PlateRecognizerML()
    }
    
    func recognize(_ image: UIImage, completion: @escaping ([PlateResult]) -> Void) {
        guard let cgImage = image.cgImage else { return }
        
        let request = VNCoreMLRequest(model: detector) { request, error in
            guard let results = request.results as? [VNRecognizedObjectObservation],
                  !results.isEmpty else {
                completion([])
                return
            }
            
            var plates: [PlateResult] = []
            for observation in results {
                let bbox = observation.boundingBox
                // 裁剪车牌区域
                let cropRect = VNImageRectForNormalizedRect(bbox, 
                    Int(image.size.width), Int(image.size.height))
                guard let cropCG = cgImage.cropping(to: cropRect) else { continue }
                let cropImg = UIImage(cgImage: cropCG)
                
                // 识别
                if let plate = try? self.recognizer.prediction(image: cropImg) {
                    plates.append(PlateResult(plate: plate.output, bbox: bbox))
                }
            }
            completion(plates)
        }
        
        let handler = VNImageRequestHandler(cgImage: cgImage)
        try? handler.perform([request])
    }
}
```

#### 路线 B：ONNX Runtime Mobile（iOS）

```swift
// Podfile
pod 'onnxruntime-mobile-c'
```

---

## 四、模型转换速查表

| 源格式 | 目标格式 | 命令 | 适用平台 |
|--------|---------|------|---------|
| .pt | .onnx | `python scripts/04_export.py` | 通用 |
| .onnx | .mlmodel | `coremltools` | iOS |
| .onnx | .param/.bin | `onnx2ncnn` | Android (NCNN) |
| .onnx | .tflite | `onnx2tf` | Android (TFLite) |
| .onnx | .engine | `trtexec` | Jetson / GPU |
| .onnx | .wasm | `onnxruntime-web` | 浏览器 |

---

## 五、浏览器端（纯前端，无需服务器）

```bash
# 用 ONNX Runtime Web
npm install onnxruntime-web
```

```javascript
// 浏览器端推理
import * as ort from 'onnxruntime-web';

const detector = await ort.InferenceSession.create('plate_detector.onnx');
const recognizer = await ort.InferenceSession.create('plate_recognizer.onnx');

async function recognizePlate(imageElement) {
    // 1. Canvas → Tensor
    const canvas = document.createElement('canvas');
    canvas.width = 640;
    canvas.height = 640;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(imageElement, 0, 0, 640, 640);
    const imageData = ctx.getImageData(0, 0, 640, 640);
    const input = new ort.Tensor('float32', preprocess(imageData), [1, 3, 640, 640]);
    
    // 2. 检测
    const detOutput = await detector.run({ images: input });
    
    // 3. 识别（同上逻辑）
    // ...
}
```

---

## 六、推荐方案总结

| 场景 | 推荐方案 | 延迟 | 难度 |
|------|---------|------|------|
| 公网 API | FastAPI + Docker | ~200ms | ⭐ |
| Android App | ONNX Runtime Mobile | ~50ms | ⭐⭐ |
| iOS App | CoreML | ~30ms | ⭐⭐ |
| 浏览器 | ONNX Runtime Web | ~100ms | ⭐⭐ |
| 嵌入式 | NCNN / TensorRT | ~20ms | ⭐⭐⭐ |
| 离线批量 | Python + ONNX Runtime | ~50ms/张 | ⭐ |

---

## 七、常见问题

**Q: 模型太大，手机装不下？**
A: 用 NCNN 量化 → int8，< 3MB

**Q: 识别速度慢？**
A: 检测器用 320x320 输入，识别器用 FP16

**Q: 没有 GPU 能跑吗？**
A: 能，ONNX Runtime CPU 版，单张 ~50ms

**Q: 怎么处理视频流？**
A: 每 5 帧检测一次，中间帧用追踪（ByteTrack）