# Java LPR 子项目

纯 Java 车牌识别，**仅需 onnxruntime 一个 AI 依赖**，零 DJL、零 OpenCV。

JDK 自带的 BufferedImage 搞定所有图片预处理，YOLOv8 后处理 + NMS + CTC 解码全部自实现。

## 目录结构

```
java-lpr/
├── pom.xml
├── README.md
└── src/
    ├── main/java/com/tinylpr/
    │   ├── LprApplication.java      ← Spring Boot 启动
    │   ├── config/LprConfig.java    ← 模型加载配置
    │   ├── controller/LprController.java ← REST API
    │   └── core/
    │       ├── LprEngine.java       ← 🔥 核心引擎（ONNX Runtime）
    │       └── PlateResult.java     ← 结果模型
    └── test/java/com/tinylpr/
        └── LprTest.java             ← 命令行测试
```

## 依赖

```xml
<!-- 唯一 AI 依赖 -->
<dependency>
    <groupId>com.microsoft.onnxruntime</groupId>
    <artifactId>onnxruntime</artifactId>
    <version>1.17.0</version>
</dependency>
```

## 运行

```bash
# 1. 确认模型文件
ls ../models/plate_detector.onnx ../models/plate_recognizer.onnx

# 2. 启动
mvn spring-boot:run

# 3. 测试
curl -X POST -F "file=@car.jpg" http://localhost:8080/api/lpr/recognize
```

## API

```json
// POST /api/lpr/recognize?file=图片
{
  "success": true,
  "count": 1,
  "results": [
    {
      "plate": "京A12345",
      "confidence": 0.98,
      "bbox": [100, 200, 300, 260]
    }
  ]
}
```

## 配置

```yaml
# application.yml
lpr:
  detector-path: models/plate_detector.onnx
  recognizer-path: models/plate_recognizer.onnx
  confidence-threshold: 0.5
```

## 核心类说明

### LprEngine.java

`recognize(byte[] imageBytes)` → `List<PlateResult>`

内部流程：
```
图片字节流 → BufferedImage → resize(640) → float[] → ONNX检测
                                                         ↓
                                               YOLOv8输出解析 + NMS
                                                         ↓
                                          裁剪车牌 → resize(160,48) → ONNX识别
                                                         ↓
                                                    CTC贪心解码 → 车牌号
```

### 为什么不用 DJL？

DJL 的 YOLOv8 Translator 需要自定义实现，复杂且容易出错。
直接用 ONNX Runtime Java API 更简单、更可控、依赖更少。