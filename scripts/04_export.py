"""
模型导出：PyTorch → ONNX（支持量化、TensorRT）
"""
import torch
import json
import argparse
from pathlib import Path

from models.recognizer import TinyLPR


def export_detector(model_path: str, output_path: str, simplify: bool = True):
    """导出 YOLOv8 检测器为 ONNX"""
    from ultralytics import YOLO

    model = YOLO(model_path)
    model.export(format='onnx', imgsz=640, simplify=simplify, opset=12)

    # 移动文件
    onnx_file = Path(model_path).with_suffix('.onnx')
    if onnx_file.exists():
        import shutil
        shutil.move(str(onnx_file), output_path)
        print(f"✅ 检测器 ONNX: {output_path}")


def export_recognizer(model_path: str, output_path: str, simplify: bool = True):
    """导出识别器为 ONNX"""
    # 加载模型
    checkpoint = torch.load(model_path, map_location='cpu')
    num_classes = checkpoint['num_classes']

    model = TinyLPR(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 导出
    dummy_input = torch.randn(1, 3, 48, 160)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch'},
            'output': {0: 'batch'},
        },
        opset_version=12,
        do_constant_folding=True,
    )

    print(f"✅ 识别器 ONNX: {output_path}")

    # ONNX Simplifier
    if simplify:
        try:
            import onnx
            from onnxsim import simplify as onnx_simplify

            onnx_model = onnx.load(output_path)
            model_simp, check = onnx_simplify(onnx_model)
            if check:
                onnx.save(model_simp, output_path)
                print(f"   ONNX 简化完成")
            else:
                print(f"   ⚠️ ONNX 简化失败，使用原始模型")
        except ImportError:
            print(f"   ⚠️ onnxsim 未安装，跳过简化")

    # 保存字符映射
    meta_path = Path(output_path).with_suffix('.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump({
            'char_list': checkpoint['char_list'],
            'char_map': checkpoint['char_map'],
        }, f, ensure_ascii=False)
    print(f"   字符映射: {meta_path}")


def export_tensorrt(onnx_path: str, engine_path: str, fp16: bool = True):
    """ONNX → TensorRT 引擎（需要 TensorRT 环境）"""
    try:
        import tensorrt as trt

        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)

        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(f"TensorRT Error: {parser.get_error(i)}")
                return

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

        if fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("   使用 FP16 精度")

        engine = builder.build_serialized_network(network, config)
        with open(engine_path, 'wb') as f:
            f.write(engine)

        print(f"✅ TensorRT 引擎: {engine_path}")

    except ImportError:
        print("⚠️ TensorRT 未安装，跳过")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--detector', type=str, default='models/plate_detector.pt', help='检测器路径')
    parser.add_argument('--recognizer', type=str, default='models/plate_recognizer.pt', help='识别器路径')
    parser.add_argument('--output_dir', type=str, default='models', help='输出目录')
    parser.add_argument('--engine', type=str, choices=['onnx', 'tensorrt'], default='onnx', help='导出格式')
    parser.add_argument('--no_simplify', action='store_true', help='不简化 ONNX')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 导出检测器
    detector_pt = Path(args.detector)
    if detector_pt.exists():
        export_detector(
            str(detector_pt),
            str(output_dir / 'plate_detector.onnx'),
            simplify=not args.no_simplify,
        )
    else:
        print(f"⚠️ 检测器未找到: {detector_pt}")

    # 导出识别器
    recognizer_pt = Path(args.recognizer)
    if recognizer_pt.exists():
        export_recognizer(
            str(recognizer_pt),
            str(output_dir / 'plate_recognizer.onnx'),
            simplify=not args.no_simplify,
        )

        # TensorRT
        if args.engine == 'tensorrt':
            export_tensorrt(
                str(output_dir / 'plate_recognizer.onnx'),
                str(output_dir / 'plate_recognizer.engine'),
            )
    else:
        print(f"⚠️ 识别器未找到: {recognizer_pt}")

    print("\n📦 导出完成！")
    print(f"   检测器: {output_dir}/plate_detector.onnx")
    print(f"   识别器: {output_dir}/plate_recognizer.onnx")