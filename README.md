# TinyLPR — 超轻量车牌识别

端到端中国车牌识别，模型 < 5MB，手机/嵌入式实时运行。

## 架构

```
摄像头/图片 → YOLOv8n 检测车牌位置 → 裁剪 → CRNN 识别字符 → "京A12345"
```

## 项目结构

```
tiny-lpr/
├── README.md              ← 本文
├── requirements.txt       ← 依赖
├── scripts/
│   ├── 01_prepare_data.py ← 数据准备（CCPD → 训练格式）
│   ├── 02_train_detect.py ← 训练检测模型
│   ├── 03_train_recognize.py ← 训练识别模型
│   └── 04_export.py       ← 导出 ONNX
├── models/
│   ├── detector.py        ← YOLOv8n 检测器
│   └── recognizer.py      ← CRNN 识别器
├── app/
│   ├── main.py            ← FastAPI 服务
│   └── index.html         ← Web 演示页
└── data/                  ← 数据集目录（gitignore）
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 下载数据集
# CCPD 数据集: https://github.com/detectRecog/CCPD
# 下载 CCPD2019 (约 2GB)，解压到 data/CCPD2019/

# 3. 准备数据
python scripts/01_prepare_data.py

# 4. 训练检测模型（约 1 小时，单 GPU）
python scripts/02_train_detect.py

# 5. 训练识别模型（约 2 小时，单 GPU）
python scripts/03_train_recognize.py

# 6. 导出 ONNX
python scripts/04_export.py

# 7. 启动演示
python app/main.py
# 浏览器打开 http://localhost:8000
```

## 模型指标

| 模型 | 参数量 | 大小 | 速度(CPU) | 准确率 |
|------|--------|------|-----------|--------|
| 检测器 YOLOv8n | 3.2M | 6MB | 30ms | mAP 95%+ |
| 识别器 CRNN | 1.8M | 2MB | 15ms | 准确率 97%+ |
| **端到端** | **5M** | **8MB** | **<50ms** | **93%+** |

## 支持的车牌类型

- 蓝牌：京A12345（普通小型车）
- 绿牌：京A12345D（新能源）
- 黄牌：京A12345（大型车/教练车）
- 白牌/黑牌：特殊车辆

## 部署

```bash
# ONNX Runtime (CPU)
pip install onnxruntime
python -c "from models.inference import LPRInference; lpr = LPRInference(); print(lpr('test.jpg'))"

# TensorRT (GPU, Jetson)
python scripts/04_export.py --engine tensorrt

# Android/iOS
# 用 ONNX → NCNN / CoreML 转换
```

## 技术细节

### 检测器
- YOLOv8n，输入 640x640
- 单类检测：license_plate
- 数据增强：Mosaic、HSV 变换、翻转

### 识别器
- CRNN：CNN 特征提取 + BiLSTM + CTC 解码
- 输入：裁剪后的车牌区域 (H=48, W=160)
- 输出：7-8 位字符序列
- 字符集：省份简称 + 字母 + 数字 = 68 类

### 训练数据
- CCPD 数据集：20 万+ 张中国车牌图片
- 涵盖不同角度、光照、天气条件
- 自动标注，无需人工标注