"""
数据准备：CCPD 数据集 → 训练格式
CCPD 数据集文件名即标注信息，格式：
  025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg
  面积-角度_中心点&对角点-四个角点-亮度_模糊度_...-车牌号
"""
import os
import cv2
import json
import shutil
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict


# 省份简称映射
PROVINCES = {
    "京": 0, "津": 1, "冀": 2, "晋": 3, "蒙": 4,
    "辽": 5, "吉": 6, "黑": 7, "沪": 8, "苏": 9,
    "浙": 10, "皖": 11, "闽": 12, "赣": 13, "鲁": 14,
    "豫": 15, "鄂": 16, "湘": 17, "粤": 18, "桂": 19,
    "琼": 20, "渝": 21, "川": 22, "贵": 23, "云": 24,
    "藏": 25, "陕": 26, "甘": 27, "青": 28, "宁": 29,
    "新": 30,
}

# 字母数字映射（31-65）
LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"
DIGITS = "0123456789"

# 完整字符集: blank(0) + 省份(1-31) + 字母(32-55) + 数字(56-65) + 特殊(66-67)
def build_char_map():
    chars = ['-']  # blank
    for p in PROVINCES:
        chars.append(p)
    for l in LETTERS:
        chars.append(l)
    for d in DIGITS:
        chars.append(d)
    chars.append('D')  # 新能源尾号（已在字母中，但确保）
    chars.append('F')  # 新能源尾号
    return {c: i for i, c in enumerate(chars)}, chars


def parse_ccpd_filename(filename: str) -> dict:
    """解析 CCPD 文件名获取标注信息"""
    parts = filename.replace('.jpg', '').split('-')
    if len(parts) < 6:
        return None

    # 车牌号（最后一段）
    plate_str = parts[-1]
    # 车牌号格式: 省份_字母数字，如 皖_A12345
    plate_parts = plate_str.split('_')
    if len(plate_parts) >= 2:
        province = plate_parts[0]
        number = plate_parts[1]
        plate_number = province + number
    else:
        plate_number = plate_str

    # 边界框（第二段: 中心点&对角点）
    bbox_str = parts[1]
    try:
        center_diag = bbox_str.split('&')
        center = center_diag[0].split('_')
        diag = center_diag[1].split('_')
        x_center, y_center = int(center[0]), int(center[1])
        w, h = int(diag[0]), int(diag[1])
        x1 = x_center - w // 2
        y1 = y_center - h // 2
        x2 = x_center + w // 2
        y2 = y_center + h // 2
    except:
        return None

    return {
        'filename': filename,
        'plate_number': plate_number,
        'bbox': [max(0, x1), max(0, y1), x2, y2],
    }


def prepare_data(ccpd_dir: str, output_dir: str, val_ratio: float = 0.1):
    """
    将 CCPD 数据集转换为训练格式

    Args:
        ccpd_dir: CCPD 数据集目录（含所有 .jpg 文件）
        output_dir: 输出目录
        val_ratio: 验证集比例
    """
    ccpd_dir = Path(ccpd_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建目录结构
    for split in ['train', 'val']:
        (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'crops' / split).mkdir(parents=True, exist_ok=True)

    # 解析所有文件（支持 .jpg/.png/.jpeg）
    all_files = []
    for ext in ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG']:
        all_files.extend(ccpd_dir.glob(f'**/{ext}'))
    print(f"找到 {len(all_files)} 张图片")
    if len(all_files) == 0:
        print(f"⚠️ 未找到图片文件！请检查路径: {ccpd_dir.absolute()}")
        print(f"   目录是否存在: {ccpd_dir.exists()}")
        if ccpd_dir.exists():
            contents = list(ccpd_dir.iterdir())[:5]
            print(f"   目录内容: {[c.name for c in contents]}")
        return

    # 检测是否是 CCPD2020 目录结构（有 train/val 子目录）
    has_splits = (ccpd_dir / 'train').exists() or (ccpd_dir / 'Train').exists()

    annotations = []
    for f in tqdm(all_files, desc="解析标注"):
        ann = parse_ccpd_filename(f.name)
        if ann:
            ann['filepath'] = str(f)
            annotations.append(ann)

    print(f"有效标注: {len(annotations)} 张")

    # 划分训练/验证集
    if has_splits:
        # CCPD2020 已分好 train/val，直接按目录分
        print("检测到 CCPD2020 目录结构，按 train/val 目录划分")
        train_anns = [a for a in annotations if '/train/' in a['filepath'].lower()]
        val_anns = [a for a in annotations if '/val/' in a['filepath'].lower() or '/test/' in a['filepath'].lower()]
    else:
        # CCPD2019 平铺目录，随机划分
        split_idx = int(len(annotations) * (1 - val_ratio))
        train_anns = annotations[:split_idx]
        val_anns = annotations[split_idx:]

    # 处理数据
    char_map, char_list = build_char_map()

    for split, anns in [('train', train_anns), ('val', val_anns)]:
        print(f"\n处理 {split} 集 ({len(anns)} 张)...")

        # YOLO 检测标注
        yolo_labels = []

        for ann in tqdm(anns, desc=f"  {split}"):
            # 复制/链接图片
            src = ann['filepath']
            dst_img = output_dir / 'images' / split / ann['filename']
            if not dst_img.exists():
                shutil.copy2(src, dst_img)

            # 读取图片获取尺寸
            img = cv2.imread(str(dst_img))
            if img is None:
                continue
            h, w = img.shape[:2]

            # YOLO 格式: class x_center y_center width height (归一化)
            x1, y1, x2, y2 = ann['bbox']
            x_center = ((x1 + x2) / 2) / w
            y_center = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h

            # 裁剪车牌区域
            crop = img[max(0, y1):y2, max(0, x1):x2]
            if crop.size > 0:
                crop = cv2.resize(crop, (160, 48))
                cv2.imwrite(
                    str(output_dir / 'crops' / split / ann['filename']),
                    crop
                )

            # 写入 YOLO 标注
            label_file = output_dir / 'labels' / split / ann['filename'].replace('.jpg', '.txt')
            with open(label_file, 'w') as f:
                f.write(f"0 {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n")

            # 识别标注（字符索引序列）
            plate_chars = []
            for c in ann['plate_number']:
                if c in char_map:
                    plate_chars.append(char_map[c])
                else:
                    plate_chars.append(0)  # unknown → blank

            yolo_labels.append({
                'image': str(dst_img),
                'crop': str(output_dir / 'crops' / split / ann['filename']),
                'plate': ann['plate_number'],
                'chars': plate_chars,
            })

        # 保存标注 JSON
        with open(output_dir / f'{split}_labels.json', 'w', encoding='utf-8') as f:
            json.dump(yolo_labels, f, ensure_ascii=False, indent=2)

    # 保存字符映射
    with open(output_dir / 'char_map.json', 'w', encoding='utf-8') as f:
        json.dump({'char_to_idx': char_map, 'idx_to_char': char_list}, f, ensure_ascii=False)

    # 生成 YOLO 配置文件
    yaml_content = f"""path: {output_dir.absolute()}
train: images/train
val: images/val

nc: 1
names: ['license_plate']
"""
    with open(output_dir / 'plate.yaml', 'w') as f:
        f.write(yaml_content)

    print(f"\n✅ 数据准备完成！")
    print(f"   训练集: {len(train_anns)} 张")
    print(f"   验证集: {len(val_anns)} 张")
    print(f"   字符集: {len(char_list)} 类")
    print(f"   YOLO 配置: {output_dir / 'plate.yaml'}")
    print(f"   字符映射: {output_dir / 'char_map.json'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ccpd_dir', type=str, required=True, help='CCPD 数据集目录')
    parser.add_argument('--output_dir', type=str, default='data/processed', help='输出目录')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='验证集比例')
    args = parser.parse_args()
    prepare_data(args.ccpd_dir, args.output_dir, args.val_ratio)
