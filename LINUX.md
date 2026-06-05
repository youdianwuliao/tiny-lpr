# TinyLPR Linux 本地运行指南

## 环境要求

- Python 3.10+
- pip
- 8GB+ 内存（推理够用，训练需要 GPU）

---

## 一、快速体验（Demo 模式，无需模型）

```bash
# 1. 克隆
git clone https://github.com/youdianwuliao/tiny-lpr.git
cd tiny-lpr

# 2. 安装依赖
pip install fastapi uvicorn opencv-python pillow python-multipart

# 3. 启动（无模型自动进入 Demo 模式）
python app/main.py

# 4. 浏览器打开
# http://localhost:8000
# 上传任意图片，会返回模拟结果
```

---

## 二、完整流程（训练 + 推理）

### Step 1：安装全部依赖

```bash
cd tiny-lpr

# 基础依赖
pip install fastapi uvicorn opencv-python pillow python-multipart

# 训练依赖（需要 GPU）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics onnx onnxruntime onnxsim tqdm matplotlib

# 如果只有 CPU，装 CPU 版 PyTorch
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Step 2：下载数据集

```bash
# CCPD 中国车牌数据集（约 2GB）
# 下载地址：https://github.com/detectRecog/CCPD
# 选 CCPD2019 或 CCPD2020

mkdir -p data/CCPD2019
cd data/CCPD2019

# 方式一：直接下载（如果有直链）
# wget https://example.com/CCPD2019.tar.gz
# tar -xzf CCPD2019.tar.gz

# 方式二：从百度网盘下载后上传到服务器
# 网盘链接见 https://github.com/detectRecog/CCPD

cd ../..
```

### Step 3：准备数据

```bash
# 将 CCPD 图片转为训练格式
python scripts/01_prepare_data.py \
    --ccpd_dir data/CCPD2019 \
    --output_dir data/processed

# 输出：
#   data/processed/images/train/   ← 训练图片
#   data/processed/images/val/     ← 验证图片
#   data/processed/crops/train/    ← 裁剪的车牌
#   data/processed/plate.yaml      ← YOLO 配置
#   data/processed/char_map.json   ← 字符映射
```

### Step 4：训练检测器

```bash
# 训练 YOLOv8n（单 GPU 约 1 小时）
python scripts/02_train_detect.py \
    --data data/processed/plate.yaml \
    --output models \
    --epochs 100

# 输出：models/plate_detector.pt
```

### Step 5：训练识别器

```bash
# 训练 CRNN（单 GPU 约 2 小时）
python scripts/03_train_recognize.py \
    --data data/processed \
    --output models \
    --epochs 50 \
    --batch_size 64

# 输出：models/plate_recognizer.pt
```

### Step 6：导出 ONNX

```bash
# PyTorch → ONNX（CPU 也能推理）
python scripts/04_export.py \
    --detector models/plate_detector.pt \
    --recognizer models/plate_recognizer.pt \
    --output_dir models

# 输出：
#   models/plate_detector.onnx
#   models/plate_recognizer.onnx
#   models/plate_recognizer.json
```

### Step 7：启动服务

```bash
python app/main.py

# 浏览器打开 http://localhost:8000
# 上传车牌图片即可识别
```

---

## 三、命令行测试

```bash
# 单张图片测试
python models/inference.py test.jpg

# 输出：
# 🚗 车牌: 京A12345 | 置信度: 0.98 | 位置: [100, 200, 300, 260]
```

---

## 四、API 调用

```bash
# 上传图片识别
curl -X POST -F "file=@car.jpg" http://localhost:8000/api/recognize

# 返回：
# {
#   "results": [
#     {"plate": "京A12345", "confidence": 0.98, "bbox": [100, 200, 300, 260]}
#   ]
# }
```

---

## 五、生产部署

```bash
# 后台运行
nohup python app/main.py > lpr.log 2>&1 &

# 或用 gunicorn（多 worker）
pip install gunicorn
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000

# 或用 systemd
sudo tee /etc/systemd/system/tiny-lpr.service << 'EOF'
[Unit]
Description=TinyLPR Service
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/path/to/tiny-lpr
ExecStart=/usr/bin/python3 app/main.py
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now tiny-lpr
```

---

## 六、常见问题

**Q: 没有 GPU 能训练吗？**
A: 能，但很慢。YOLOv8n 在 CPU 上训练约 10-20 小时。建议用 GPU 或直接下载预训练模型。

**Q: 没有 CCPD 数据集怎么办？**
A: 先用 Demo 模式体验。或者自己标注 100-200 张车牌图片也能训练出可用模型。

**Q: ONNX 推理需要 GPU 吗？**
A: 不需要。`onnxruntime` 默认 CPU 推理，单张图片约 50ms。

**Q: 端口被占用？**
A: 改端口：`python app/main.py` 里修改 `port=8000` 为其他端口。