# TinyLPR Linux 本地运行指南

## 环境要求

- Python 3.10+
- pip
- 8GB+ 内存（推理够用，训练需要 GPU）

---

## 一、快速体验（Demo 模式，无需模型）

```bash
# 克隆
git clone https://github.com/youdianwuliao/tiny-lpr.git
cd tiny-lpr

# 创建虚拟环境（推荐）
python3 -m venv venv
source venv/bin/activate

# 如果提示没有 venv 模块，先安装
# sudo apt install python3.12-venv

# 安装依赖
pip install fastapi uvicorn opencv-python pillow python-multipart

# 启动（无模型自动进入 Demo 模式）
python app/main.py

# 浏览器打开 http://localhost:8000
```

> 💡 实际运行过的命令，已验证可用

---

## 二、完整流程（训练 + 推理）

### Step 1：安装全部依赖

```bash
cd tiny-lpr

# 基础依赖
pip install fastapi uvicorn opencv-python pillow python-multipart

# 训练依赖（GTX 1050 Ti 用 CUDA 11.8）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics onnx onnxruntime-gpu onnxsim tqdm matplotlib

# 如果只有 CPU，装 CPU 版 PyTorch
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

> ⚠️ GTX 1050 Ti 专属注意：
> - 显存只有 4GB，训练时 batch_size 必须设为 8 以下
> - 用 CUDA 11.8（不要用 CUDA 12，1050 Ti 不支持）
> - 推理装 `onnxruntime-gpu` 替代 `onnxruntime`（GPU 推理快 5x）

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

### Step 4：训练检测器（GTX 1050 Ti 专属参数）

```bash
# 4GB 显存版参数
python scripts/02_train_detect.py \
    --data data/processed/plate.yaml \
    --output models \
    --epochs 100 \
    --imgsz 320                    # ⚡ 降低分辨率省显存

# 如果还 OOM，手动改 batch_size
# 编辑 scripts/02_train_detect.py，把 batch=16 改成 batch=4
```

> 💡 1050 Ti 训练 YOLOv8n 约 1.5-2 小时（比 3060 慢一倍，但能跑）

### Step 5：训练识别器

```bash
# CRNN 很轻量，4GB 显存够用
python scripts/03_train_recognize.py \
    --data data/processed \
    --output models \
    --epochs 50 \
    --batch_size 32

# 输出：models/plate_recognizer.pt
# 预计时间：约 1-2 小时
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

### Step 7：GPU 推理 + 启动

```bash
# 安装 GPU 版 ONNX Runtime
pip install onnxruntime-gpu

# 验证 GPU 是否被识别
python -c "import onnxruntime; print(onnxruntime.get_available_providers())"
# 应该输出包含 'CUDAExecutionProvider'

# 启动服务
python app/main.py
# → http://localhost:8000
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

## 六、不同显卡适配

### RTX 4060 Laptop (8GB) — 推荐

```bash
# CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install ultralytics onnx onnxruntime-gpu onnxsim tqdm matplotlib

# 训练参数 — 全默认，不用降配置
python scripts/02_train_detect.py --data data/processed/plate.yaml --epochs 100
python scripts/03_train_recognize.py --data data/processed --epochs 50
# 预计时间：检测器~40分钟，识别器~30分钟
```

### GTX 1050 Ti (4GB) — 降配置

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 训练必须降参数
python scripts/02_train_detect.py --imgsz 320 --epochs 100
# 如果 OOM：编辑脚本把 batch=16 改成 batch=4
python scripts/03_train_recognize.py --batch_size 32 --epochs 50
# 预计时间：检测器~2-3小时，识别器~1-2小时
```

### 无 GPU (CPU only)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# 能跑但很慢，检测器约10-20小时，不推荐训练，建议直接下载预训练模型
```

### 显卡速查

| 显卡 | 显存 | CUDA | batch | imgsz | 训练速度 |
|------|------|------|-------|-------|---------|
| RTX 4060 | 8GB | 12.x | 16 | 640 | ⭐⭐⭐ |
| RTX 3060 | 12GB | 11.x | 16 | 640 | ⭐⭐⭐ |
| RTX 2060 | 6GB | 11.x | 8 | 640 | ⭐⭐ |
| GTX 1050 Ti | 4GB | 11.x | 4 | 320 | ⭐ |
| CPU | — | — | 4 | 320 | 🐌 |

---

## 七、常见问题

**Q: 没有 GPU 能训练吗？**
A: 能，但很慢。YOLOv8n 在 CPU 上训练约 10-20 小时。建议用 GPU 或直接下载预训练模型。

**Q: 没有 CCPD 数据集怎么办？**
A: 先用 Demo 模式体验。或者自己标注 100-200 张车牌图片也能训练出可用模型。

**Q: ONNX 推理需要 GPU 吗？**
A: 不需要。`onnxruntime` 默认 CPU 推理，单张图片约 50ms。

**Q: 端口被占用？**
A: 改端口：`python app/main.py` 里修改 `port=8000` 为其他端口。