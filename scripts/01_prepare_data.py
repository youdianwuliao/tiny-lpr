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
    """解析 CCPD 文件名获取标注信息
    
    CCPD2019: 025-95_113-...-37-15.jpg  (最后两段=省份编码_车牌号)
    CCPD2020: 003607...-117-16.jpg      (同样格式)
    """
    name = filename.rsplit('.', 1)[0]  # 去掉扩展名
    parts = name.split('-')
    if len(parts) < 6:
        return None

    # 车牌号：倒数两段是 省份编码-车牌号
    province_code = parts[-2]
    plate_code = parts[-1]
    
    # 尝试解码为可读车牌号
    plate_number = decode_plate(province_code, plate_code)

    # 边界框（第三段，格式: x1&y1_x2&y2）
    bbox_str = parts[2]
    try:
        coords = bbox_str.replace('&', '_').split('_')
        x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
    except (ValueError, IndexError):
        return None

    return {
        'filename': filename,
        'plate_number': plate_number,
        'bbox': [max(0, x1), max(0, y1), x2, y2],
    }


# CCPD 省份映射（数值索引 → 省份简称）
CCPD_PROVINCES = [
    "皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
    "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
    "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新"
]
CCPD_LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"  # 24个（不含I/O）
CCPD_CHARS = CCPD_LETTERS + "0123456789"   # 34个


def decode_plate(province_code: str, plate_code: str) -> str:
    """解码 CCPD 车牌号
    
    蓝牌: province * 24 * 34^5 + letter * 34^5 + 5个字符
    """
    try:
        pc = int(province_code)
        pn = int(plate_code)
    except ValueError:
        return f"{province_code}_{plate_code}"
    
    # 省份
    if 0 <= pc < len(CCPD_PROVINCES):
        province = CCPD_PROVINCES[pc]
        # 解码车牌号
        if pn >= 24 * (34 ** 5):
            return decode_plate_n(pn, province, 6)
        else:
            return decode_plate_n(pn, province, 5)
    else:
        # 省份编码不在标准范围（如 CCPD2020 绿牌），直接用编码值
        # 编码为纯数字字符串用于训练
        return f"{pc:03d}{pn:04d}"


def decode_plate_n(pn: int, province: str, n_digits: int) -> str:
    """解码 n 位车牌号"""
    result = province
    
    # 第二位：字母
    letter_base = 34 ** n_digits
    letter_idx = pn // letter_base
    pn = pn % letter_base
    if 0 <= letter_idx < len(CCPD_LETTERS):
        result += CCPD_LETTERS[letter_idx]
    else:
        result += "?"
    
    # 剩余位
    for i in range(n_digits - 1, -1, -1):
        char_base = 34 ** i
        char_idx = pn // char_base if char_base > 0 else pn
        pn = pn % char_base if char_base > 0 else 0
        if 0 <= char_idx < len(CCPD_CHARS):
            result += CCPD_CHARS[char_idx]
        else:
            result += "?"
    
    return result


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
    parse_errors = 0
    for f in tqdm(all_files, desc="解析标注"):
        ann = parse_ccpd_filename(f.name)
        if ann:
            ann['filepath'] = str(f)
            annotations.append(ann)
        else:
            parse_errors += 1
            if parse_errors <= 3:
                print(f"\n  ⚠️ 解析失败 #{parse_errors}: {f.name[:80]}")

    print(f"有效标注: {len(annotations)} 张 (失败: {parse_errors})")

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
