"""
Web 演示服务 — FastAPI
提供图片上传 + 车牌识别 API
"""
import sys
import io
import os
from pathlib import Path

import numpy as np
from PIL import Image
import cv2

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent))

app = FastAPI(title="TinyLPR — 车牌识别演示")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# 延迟加载模型
_lpr = None

def get_lpr():
    global _lpr
    if _lpr is None:
        try:
            from models.inference import LPRInference
            detector_path = "models/plate_detector.onnx"
            recognizer_path = "models/plate_recognizer.onnx"
            char_map_path = "models/plate_recognizer.json"

            if not Path(detector_path).exists():
                print("⚠️ 检测器模型未找到，使用 Demo 模式")

            _lpr = LPRInference(
                detector_path=detector_path,
                recognizer_path=recognizer_path,
                char_map_path=char_map_path,
            ) if Path(detector_path).exists() else None
        except Exception as e:
            print(f"⚠️ 模型加载失败: {e}")
            _lpr = None
    return _lpr


@app.get("/")
async def index():
    html_path = Path(__file__).parent / "index.html"
    with open(html_path, encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.post("/api/recognize")
async def recognize(file: UploadFile = File(...)):
    """识别上传图片中的车牌"""
    try:
        # 读取图片
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse({"error": "无法解析图片"}, status_code=400)

        lpr = get_lpr()
        if lpr is None:
            # Demo 模式：返回模拟结果
            import random
            provinces = ['京','津','沪','渝','冀','晋','辽','吉','黑','苏','浙','皖','闽','赣','鲁','豫','鄂','湘','粤','桂','琼','川','贵','云','陕','甘']
            letters = 'ABCDEFGHJKLMNPQRSTUVWXYZ'
            p = random.choice(provinces)
            l = random.choice(letters)
            n = ''.join(str(random.randint(0,9)) for _ in range(5))
            return {
                "results": [{
                    "plate": f"{p}{l}{n}",
                    "bbox": [100, 200, 300, 260],
                    "confidence": round(random.uniform(0.85, 0.99), 4),
                }],
                "demo": True,
                "message": "Demo 模式 — 训练模型后可获得真实结果",
            }

        results = lpr(img)
        return {"results": results, "demo": False}

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/health")
async def health():
    return {"status": "ok", "model_loaded": get_lpr() is not None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")