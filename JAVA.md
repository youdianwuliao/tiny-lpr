# Java 服务器集成指南

TinyLPR 模型在 Java 服务器中的三种使用方式。

---

## 方案对比

| 方案 | 延迟 | 部署 | 维护 | 推荐场景 |
|------|------|------|------|---------|
| ONNX Runtime Java | ~50ms | 中等 | 中等 | 高性能、纯 Java |
| DJL (Deep Java Library) | ~50ms | 简单 | 简单 | ⭐ 推荐，API 最友好 |
| Python 微服务 | ~100ms | 简单 | 简单 | 快速上线、已有 Python |

---

## 方案一：DJL（推荐 ⭐）

AWS 出品，API 友好，内置 ONNX Runtime 引擎。

### 1. 添加依赖

```xml
<!-- pom.xml -->
<dependency>
    <groupId>ai.djl</groupId>
    <artifactId>api</artifactId>
    <version>0.29.0</version>
</dependency>
<dependency>
    <groupId>ai.djl.onnxruntime</groupId>
    <artifactId>onnxruntime-engine</artifactId>
    <version>0.29.0</version>
</dependency>
<dependency>
    <groupId>ai.djl.opencv</groupId>
    <artifactId>opencv</artifactId>
    <version>0.29.0</version>
</dependency>
```

### 2. 识别器封装

```java
package com.yourcompany.lpr;

import ai.djl.*;
import ai.djl.inference.*;
import ai.djl.modality.cv.*;
import ai.djl.modality.cv.output.*;
import ai.djl.modality.cv.transform.*;
import ai.djl.ndarray.*;
import ai.djl.ndarray.types.*;
import ai.djl.repository.zoo.*;
import ai.djl.translate.*;
import ai.djl.onnxruntime.engine.*;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import java.nio.file.*;
import java.util.*;

public class LPRRecognizer implements AutoCloseable {

    private final Predictor<Image, DetectedObjects> detector;
    private final Predictor<Image, String> recognizer;

    public LPRRecognizer(String detectorPath, String recognizerPath) throws Exception {
        // 检测器
        Criteria<Image, DetectedObjects> detCriteria = Criteria.builder()
            .setTypes(Image.class, DetectedObjects.class)
            .optModelPath(Paths.get(detectorPath))
            .optEngine("OnnxRuntime")
            .optTranslator(new YoloV8Translator())
            .build();
        this.detector = detCriteria.loadModel().newPredictor();

        // 识别器
        Criteria<Image, String> recCriteria = Criteria.builder()
            .setTypes(Image.class, String.class)
            .optModelPath(Paths.get(recognizerPath))
            .optEngine("OnnxRuntime")
            .optTranslator(new PlateRecognitionTranslator())
            .build();
        this.recognizer = recCriteria.loadModel().newPredictor();
    }

    public List<PlateResult> recognize(byte[] imageBytes) throws Exception {
        Image img = ImageFactory.getInstance().fromInputStream(
            new java.io.ByteArrayInputStream(imageBytes));

        // 1. 检测车牌
        DetectedObjects detections = detector.predict(img);
        List<PlateResult> results = new ArrayList<>();

        for (DetectedObjects.DetectedObject obj : detections.items()) {
            BoundingBox bbox = obj.getBoundingBox();
            Rectangle rect = bbox.getBounds();

            // 2. 裁剪车牌区域
            Image crop = img.getSubImage(
                (int) rect.getX(), (int) rect.getY(),
                (int) rect.getWidth(), (int) rect.getHeight()
            );

            // 3. 识别
            String plate = recognizer.predict(crop);
            results.add(new PlateResult(plate, rect, obj.getProbability()));
        }

        return results;
    }

    @Override
    public void close() {
        detector.close();
        recognizer.close();
    }

    // 数据类
    public static class PlateResult {
        public String plate;
        public Rectangle bbox;
        public double confidence;

        public PlateResult(String plate, Rectangle bbox, double confidence) {
            this.plate = plate;
            this.bbox = bbox;
            this.confidence = confidence;
        }
    }
}
```

### 3. Spring Boot Controller

```java
package com.yourcompany.lpr.controller;

import com.yourcompany.lpr.LPRRecognizer;
import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.*;

@RestController
@RequestMapping("/api/lpr")
public class LPRController {

    @Value("${lpr.detector-path:models/plate_detector.onnx}")
    private String detectorPath;

    @Value("${lpr.recognizer-path:models/plate_recognizer.onnx}")
    private String recognizerPath;

    private LPRRecognizer recognizer;

    @PostConstruct
    public void init() throws Exception {
        System.out.println("🚗 加载车牌识别模型...");
        this.recognizer = new LPRRecognizer(detectorPath, recognizerPath);
        System.out.println("✅ 模型加载完成");
    }

    @PreDestroy
    public void destroy() {
        if (recognizer != null) recognizer.close();
    }

    @PostMapping("/recognize")
    public Map<String, Object> recognize(@RequestParam("file") MultipartFile file) {
        try {
            List<LPRRecognizer.PlateResult> plates = recognizer.recognize(file.getBytes());

            List<Map<String, Object>> resultList = new ArrayList<>();
            for (LPRRecognizer.PlateResult p : plates) {
                Map<String, Object> item = new LinkedHashMap<>();
                item.put("plate", p.plate);
                item.put("confidence", p.confidence);
                item.put("bbox", Arrays.asList(
                    (int) p.bbox.getX(), (int) p.bbox.getY(),
                    (int) (p.bbox.getX() + p.bbox.getWidth()),
                    (int) (p.bbox.getY() + p.bbox.getHeight())
                ));
                resultList.add(item);
            }

            Map<String, Object> response = new LinkedHashMap<>();
            response.put("success", true);
            response.put("results", resultList);
            return response;
        } catch (Exception e) {
            Map<String, Object> error = new LinkedHashMap<>();
            error.put("success", false);
            error.put("error", e.getMessage());
            return error;
        }
    }

    @GetMapping("/health")
    public Map<String, String> health() {
        return Map.of("status", "ok", "model", "loaded");
    }
}
```

### 4. 配置文件

```yaml
# application.yml
lpr:
  detector-path: models/plate_detector.onnx
  recognizer-path: models/plate_recognizer.onnx

spring:
  servlet:
    multipart:
      max-file-size: 10MB
```

---

## 方案二：ONNX Runtime Java

直接使用 ONNX Runtime 的 Java API，更底层但更灵活。

### 1. 添加依赖

```xml
<dependency>
    <groupId>com.microsoft.onnxruntime</groupId>
    <artifactId>onnxruntime</artifactId>
    <version>1.17.0</version>
</dependency>
```

### 2. 核心代码

```java
package com.yourcompany.lpr;

import ai.onnxruntime.*;
import java.nio.FloatBuffer;
import java.util.*;

public class OnnxLPRRecognizer implements AutoCloseable {

    private final OrtEnvironment env;
    private final OrtSession detectorSession;
    private final OrtSession recognizerSession;

    public OnnxLPRRecognizer(String detectorPath, String recognizerPath) throws Exception {
        this.env = OrtEnvironment.getEnvironment();
        this.detectorSession = env.createSession(detectorPath, new OrtSession.SessionOptions());
        this.recognizerSession = env.createSession(recognizerPath, new OrtSession.SessionOptions());
    }

    public List<PlateResult> recognize(float[][][][] imageData) throws Exception {
        // imageData: (1, 3, 640, 640) normalized
        OnnxTensor inputTensor = OnnxTensor.createTensor(env, imageData);

        // 检测
        OrtSession.Result detResult = detectorSession.run(
            Collections.singletonMap("images", inputTensor));

        // 解析检测结果
        float[][] detOutput = (float[][]) detResult.get(0).getValue();
        List<PlateResult> results = new ArrayList<>();

        for (float[] det : detOutput) {
            float confidence = det[4];
            if (confidence < 0.5) continue;

            float cx = det[0], cy = det[1], w = det[2], h = det[3];
            int x1 = (int) (cx - w / 2);
            int y1 = (int) (cy - h / 2);
            int x2 = (int) (cx + w / 2);
            int y2 = (int) (cy + h / 2);

            // 裁剪 + 识别（需要单独的图像处理逻辑）
            String plate = recognizePlate(imageData, x1, y1, x2, y2);
            results.add(new PlateResult(plate, x1, y1, x2, y2, confidence));
        }

        return results;
    }

    private String recognizePlate(float[][][][] image, int x1, int y1, int x2, int y2) throws Exception {
        // 裁剪、resize 到 (1, 3, 48, 160)、归一化后推理
        // ... CTC 解码
        return "京A12345"; // 简化示例
    }

    @Override
    public void close() {
        detectorSession.close();
        recognizerSession.close();
        env.close();
    }
}
```

---

## 方案三：Python 微服务（最简单）

不改 Java 代码，部署一个 Python 边车服务。

### 1. Python 服务

```bash
# 在服务器上启动 Python 服务
pip install fastapi uvicorn onnxruntime opencv-python
cd tiny-lpr
python app/main.py
# 监听 127.0.0.1:8000（仅本地访问）
```

### 2. Java 调用

```java
// 不需要任何 AI 依赖，纯 HTTP 调用
package com.yourcompany.lpr;

import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.http.*;
import org.springframework.web.client.RestTemplate;
import org.springframework.core.io.*;

import java.util.*;

@RestController
public class LPRProxyController {

    // Python 服务地址
    private static final String LPR_SERVICE = "http://127.0.0.1:8000";
    private final RestTemplate rest = new RestTemplate();

    @PostMapping("/api/recognize")
    public Map<String, Object> recognize(@RequestParam("file") MultipartFile file) {
        try {
            // 直接把文件转发给 Python 服务
            MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
            body.add("file", new ByteArrayResource(file.getBytes()) {
                @Override
                public String getFilename() { return file.getOriginalFilename(); }
            });

            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.MULTIPART_FORM_DATA);

            ResponseEntity<Map> response = rest.exchange(
                LPR_SERVICE + "/api/recognize",
                HttpMethod.POST,
                new HttpEntity<>(body, headers),
                Map.class
            );

            return response.getBody();
        } catch (Exception e) {
            return Map.of("error", e.getMessage());
        }
    }
}
```

### 3. Docker Compose 一键部署

```yaml
# docker-compose.yml
version: '3.8'
services:
  java-app:
    build: .
    ports:
      - "8080:8080"
    depends_on:
      - lpr-service

  lpr-service:
    image: tiny-lpr:latest
    ports:
      - "8000:8000"
```

---

## 总结

| 你的 Java 框架 | 推荐方案 | 理由 |
|--------------|---------|------|
| Spring Boot 新项目 | **DJL** | 纯 Java，一个依赖搞定 |
| Spring Boot 旧项目 | **Python 微服务** | 不改 Java 代码，零风险 |
| 已有 Python 环境 | **Python 微服务** | 最快，10 分钟部署 |
| 追求极致性能 | **ONNX Runtime Java** | 无跨进程开销 |
| 不想折腾 | **Python 微服务** | 最省事 |

### 快速决策

```
有 Python 环境? → 方案三（Python 微服务），10 分钟搞定
纯 Java 项目?   → 方案一（DJL），一个 Maven 依赖的事
```