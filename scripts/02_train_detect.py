"""
训练车牌检测模型（YOLOv8n）
"""
import argparse
from pathlib import Path
from ultralytics import YOLO


def train_detector(data_yaml: str, output_dir: str = "models", epochs: int = 100, imgsz: int = 640):
    """
    训练 YOLOv8n 车牌检测器

    Args:
        data_yaml: YOLO 格式数据配置文件
        output_dir: 模型输出目录
        epochs: 训练轮数
        imgsz: 输入图片尺寸
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载预训练模型
    model = YOLO('yolov8n.pt')

    print(f"🚀 开始训练检测器...")
    print(f"   数据配置: {data_yaml}")
    print(f"   训练轮数: {epochs}")
    print(f"   图片尺寸: {imgsz}")

    # 训练
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=16,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
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
        flipud=0.0,
        fliplr=0.5,
        project=str(output_dir),
        name='detect_train',
        exist_ok=True,
        verbose=True,
    )

    # 保存最佳模型
    best_model = output_dir / 'detect_train' / 'weights' / 'best.pt'
    final_model = output_dir / 'plate_detector.pt'

    import shutil
    if best_model.exists():
        shutil.copy2(best_model, final_model)
        print(f"\n✅ 检测模型已保存到 {final_model}")

    # 评估
    metrics = model.val()
    print(f"\n📊 检测模型评估:")
    print(f"   mAP50: {metrics.box.map50:.4f}")
    print(f"   mAP50-95: {metrics.box.map:.4f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/processed/plate.yaml', help='YOLO 数据配置')
    parser.add_argument('--output', type=str, default='models', help='输出目录')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--imgsz', type=int, default=640, help='图片尺寸')
    args = parser.parse_args()
    train_detector(args.data, args.output, args.epochs, args.imgsz)