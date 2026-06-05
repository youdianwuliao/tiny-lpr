#!/bin/bash
# CCPD 数据集一键下载脚本
# 在 Linux 服务器上运行：bash download_ccpd.sh
set -e

echo "========================================="
echo "  CCPD 车牌数据集下载"
echo "========================================="
echo ""

DATA_DIR="data/CCPD2019"
mkdir -p "$DATA_DIR"

# ===== 方式一：百度网盘（推荐，国内快）=====
download_baidu() {
    echo "📦 方式一：百度网盘"
    echo ""
    echo "  1. 浏览器打开: https://pan.baidu.com/s/1i5AOjAbtkwb17Zy-NQGqkw"
    echo "     提取码: hm0u"
    echo ""
    echo "  2. 下载 CCPD2019.tar.xz（约 2GB）"
    echo ""
    echo "  3. 上传到服务器并解压："
    echo "     scp CCPD2019.tar.xz user@server:$PWD/$DATA_DIR/"
    echo "     cd $DATA_DIR && tar -xf CCPD2019.tar.xz"
    echo ""
}

# ===== 方式二：gdown（需要 VPN/代理）=====
download_gdown() {
    echo "📦 方式二：Google Drive + gdown"
    echo ""

    # 检查 gdown
    if ! python3 -c "import gdown" 2>/dev/null; then
        echo "   安装 gdown..."
        pip install gdown
    fi

    cd "$DATA_DIR"

    # 下载
    echo "   下载 CCPD2019.tar.xz（约 2GB，需要代理）..."
    gdown "https://drive.google.com/uc?id=1rdEsCUcIUaYOVRkx5IMTRNA7PcGMmSgc"

    # 解压
    echo "   解压中..."
    tar -xf CCPD2019.tar.xz

    echo "   ✅ 完成！"
    cd - > /dev/null
}

# ===== 方式三：用代理下载 =====
download_proxy() {
    local PROXY="${1:-http://127.0.0.1:7890}"
    echo "📦 方式三：代理下载（代理: $PROXY）"
    echo ""

    cd "$DATA_DIR"

    # 用 curl + 代理下载 Google Drive
    # 获取文件 ID
    FILEID="1rdEsCUcIUaYOVRkx5IMTRNA7PcGMmSgc"
    
    echo "   获取下载链接..."
    CONFIRM=$(curl -s -x "$PROXY" -c /tmp/cookies.txt "https://docs.google.com/uc?export=download&id=$FILEID" | \
        grep -o 'confirm=[^&]*' | head -1)
    
    echo "   下载中（约 2GB）..."
    curl -L -x "$PROXY" -b /tmp/cookies.txt \
        "https://docs.google.com/uc?export=download&$CONFIRM&id=$FILEID" \
        -o CCPD2019.tar.xz

    echo "   解压中..."
    tar -xf CCPD2019.tar.xz
    rm -f /tmp/cookies.txt

    echo "   ✅ 完成！"
    cd - > /dev/null
}

# ===== 菜单 =====
echo "请选择下载方式："
echo "  1) 百度网盘（国内推荐）"
echo "  2) gdown（需要 VPN）"
echo "  3) 代理下载（需要本地代理）"
echo "  q) 退出"
echo ""
read -p "输入选项 [1-3]: " choice

case $choice in
    1) download_baidu ;;
    2) download_gdown ;;
    3)
        read -p "代理地址 [http://127.0.0.1:7890]: " proxy
        proxy=${proxy:-http://127.0.0.1:7890}
        download_proxy "$proxy"
        ;;
    *) echo "已取消" ;;
esac

echo ""
echo "========================================="
echo "  下载完成后运行："
echo "  python scripts/01_prepare_data.py --ccpd_dir $DATA_DIR --output_dir data/processed"
echo "========================================="