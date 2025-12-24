#!/bin/bash
# 构建 TensorRT Engine 文件
# 用于将 ONNX 模型转换为 TensorRT Engine 以获得最佳性能

set -e

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== TensorRT Engine 构建脚本 ===${NC}"

# 配置
CACHE_DIR="/root/.cache/openocr"
WORKSPACE="/workspace"

# 确保 cache 目录存在
mkdir -p "$CACHE_DIR"

# 检查 ONNX 模型是否存在
echo -e "${YELLOW}检查 ONNX 模型...${NC}"

if [ ! -f "$CACHE_DIR/openocr_rec_model.onnx" ]; then
    echo -e "${RED}错误: OCR 识别 ONNX 模型不存在: $CACHE_DIR/openocr_rec_model.onnx${NC}"
    exit 1
fi

if [ ! -f "$CACHE_DIR/openocr_det_model.onnx" ]; then
    echo -e "${RED}错误: OCR 检测 ONNX 模型不存在: $CACHE_DIR/openocr_det_model.onnx${NC}"
    exit 1
fi

echo -e "${GREEN}ONNX 模型检查通过${NC}"

# 1. 构建 OCR 识别模型 Engine
echo -e "${GREEN}=== 构建 OCR 识别模型 TensorRT Engine ===${NC}"
python3 - << 'EOF'
import sys
sys.path.insert(0, '/workspace/openocr/tools')

from infer.trt_engine import build_engine_from_onnx
import os

# OCR 识别模型
onnx_path = '/root/.cache/openocr/openocr_rec_model.onnx'
engine_path = '/root/.cache/openocr/openocr_rec_model.trt.engine'

# 动态形状配置: [batch, channel, height, width]
# 根据实际使用情况调整
min_shapes = {'input': [1, 3, 48, 10]}
opt_shapes = {'input': [16, 3, 48, 320]}
max_shapes = {'input': [64, 3, 48, 640]}

print(f'Building OCR Recognition Engine...')
build_engine_from_onnx(
    onnx_path,
    engine_path,
    fp16=True,
    min_shapes=min_shapes,
    opt_shapes=opt_shapes,
    max_shapes=max_shapes
)
print(f'OCR Recognition Engine saved to: {engine_path}')
EOF

# 2. 构建 OCR 检测模型 Engine
echo -e "${GREEN}=== 构建 OCR 检测模型 TensorRT Engine ===${NC}"
python3 - << 'EOF'
import sys
sys.path.insert(0, '/workspace/openocr/tools')

from infer.trt_engine import build_engine_from_onnx
import os

# OCR 检测模型
onnx_path = '/root/.cache/openocr/openocr_det_model.onnx'
engine_path = '/root/.cache/openocr/openocr_det_model.trt.engine'

# 检测模型通常固定输入尺寸
# 使用 netron 查看实际输入形状
min_shapes = {'input': [1, 3, 960, 960]}
opt_shapes = {'input': [1, 3, 960, 960]}
max_shapes = {'input': [1, 3, 960, 960]}

print(f'Building OCR Detection Engine...')
build_engine_from_onnx(
    onnx_path,
    engine_path,
    fp16=True,
    min_shapes=min_shapes,
    opt_shapes=opt_shapes,
    max_shapes=max_shapes
)
print(f'OCR Detection Engine saved to: {engine_path}')
EOF

echo -e "${GREEN}=== TensorRT Engine 构建完成 ===${NC}"
echo -e "${GREEN}识别模型: $CACHE_DIR/openocr_rec_model.trt.engine${NC}"
echo -e "${GREEN}检测模型: $CACHE_DIR/openocr_det_model.trt.engine${NC}"
echo ""
echo -e "${YELLOW}现在可以在 detect_crop_ocr.py 中使用 backend='tensorrt' 了${NC}"
