"""调试 CCPD 文件名解析"""
import sys
sys.path.insert(0, '.')
from scripts.01_prepare_data import parse_ccpd_filename

# 老板那边的几个样本
samples = [
    "00360785590278-91_265-311&485_406&524-406&524_313&520_311&485_402&489-0_0_3_24_28_24_31_33-117-16.jpg",
    "00373372395833-90_96-276&514_387&548-387&548_276&547_276&516_384&514-0_0_3_26_25_31_33_32-157-19.jpg",
    "00378472222222-90_268-291&442_400&477-396&476_292&477_291&442_400&442-0_0_3_24_25_30_31_31-125-83.jpg",
]

for f in samples:
    result = parse_ccpd_filename(f)
    if result:
        print(f"✅ {f[:40]}... → plate={result['plate_number']} bbox={result['bbox']}")
    else:
        # 手动调试
        name = f.rsplit('.', 1)[0]
        parts = name.split('-')
        print(f"❌ {f[:40]}... → parts={len(parts)}")
        for i, p in enumerate(parts):
            print(f"    [{i}] {p}")
        # 试解析 bbox
        if len(parts) >= 3:
            bbox_str = parts[2]
            try:
                coords = bbox_str.replace('&', '_').split('_')
                print(f"    bbox coords: {coords}")
                x1,y1,x2,y2 = int(coords[0]),int(coords[1]),int(coords[2]),int(coords[3])
                print(f"    bbox ok: ({x1},{y1})-({x2},{y2})")
            except Exception as e:
                print(f"    bbox error: {e}")
