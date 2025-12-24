#!/usr/bin/env python3
"""
构建 OCR TensorRT Engine 文件
用于将 OCR 的 ONNX 模型转换为 TensorRT Engine 以获得最佳性能
适用于 Jetson Orin Nano 平台
"""

import os
import sys
from pathlib import Path

# 添加 openocr/tools 到路径
sys.path.insert(0, str(Path(__file__).parent / 'openocr' / 'tools'))

from infer.trt_engine import build_engine_from_onnx


def main():
    print("=" * 60)
    print("OCR TensorRT Engine 构建脚本")
    print("=" * 60)

    # 配置路径
    BASE_DIR = Path(__file__).parent
    MODELS_DIR = BASE_DIR / 'models' / 'ocr'

    # ONNX 模型路径
    DET_ONNX = MODELS_DIR / 'openocr_det_model.onnx'
    REC_ONNX = MODELS_DIR / 'openocr_rec_model.onnx'

    # 输出 Engine 路径
    DET_ENGINE = MODELS_DIR / 'openocr_det_model.trt.engine'
    REC_ENGINE = MODELS_DIR / 'openocr_rec_model.trt.engine'

    # 验证 ONNX 文件存在
    if not DET_ONNX.exists():
        print(f"[错误] OCR 检测 ONNX 模型不存在: {DET_ONNX}")
        sys.exit(1)

    if not REC_ONNX.exists():
        print(f"[错误] OCR 识别 ONNX 模型不存在: {REC_ONNX}")
        sys.exit(1)

    print(f"\n[检查] ONNX 模型文件存在")
    print(f"  - 检测模型: {DET_ONNX}")
    print(f"  - 识别模型: {REC_ONNX}")

    # ==================== 构建 OCR 识别模型 Engine ====================
    print("\n" + "=" * 60)
    print("步骤 1/2: 构建 OCR 识别模型 TensorRT Engine")
    print("=" * 60)

    # OCR 识别模型动态形状配置
    # 输入格式: [batch, channels, height, width]
    # - batch: 支持批量推理
    # - channels: 固定为 3 (RGB)
    # - height: 固定为 48 (RepSVTR 模型要求)
    # - width: 可变，根据文本长度变化
    min_shapes_rec = {'input': [1, 3, 48, 10]}    # 最小: batch=1, width=10
    opt_shapes_rec = {'input': [16, 3, 48, 320]}  # 优化: batch=16, width=320
    max_shapes_rec = {'input': [64, 3, 48, 640]}  # 最大: batch=64, width=640

    print(f"  输入形状配置:")
    print(f"    - 最小: {min_shapes_rec['input']}")
    print(f"    - 优化: {opt_shapes_rec['input']}")
    print(f"    - 最大: {max_shapes_rec['input']}")
    print(f"  FP16 模式: True")

    try:
        build_engine_from_onnx(
            str(REC_ONNX),
            str(REC_ENGINE),
            fp16=True,
            min_shapes=min_shapes_rec,
            opt_shapes=opt_shapes_rec,
            max_shapes=max_shapes_rec
        )
        print(f"[成功] OCR 识别 Engine 已保存: {REC_ENGINE}")
    except Exception as e:
        print(f"[失败] OCR 识别 Engine 构建失败: {e}")
        sys.exit(1)

    # ==================== 构建 OCR 检测模型 Engine ====================
    print("\n" + "=" * 60)
    print("步骤 2/2: 构建 OCR 检测模型 TensorRT Engine")
    print("=" * 60)

    # OCR 检测模型动态形状配置
    # 输入格式: [batch, channels, height, width]
    # - batch: 固定为 1
    # - channels: 固定为 3 (RGB)
    # - height/width: 可变，根据图像尺寸变化
    min_shapes_det = {'input': [1, 3, 64, 64]}      # 最小: 64x64
    opt_shapes_det = {'input': [1, 3, 960, 960]}    # 优化: 960x960
    max_shapes_det = {'input': [1, 3, 1280, 1280]}  # 最大: 1280x1280

    print(f"  输入形状配置:")
    print(f"    - 最小: {min_shapes_det['input']}")
    print(f"    - 优化: {opt_shapes_det['input']}")
    print(f"    - 最大: {max_shapes_det['input']}")
    print(f"  FP16 模式: True")

    try:
        build_engine_from_onnx(
            str(DET_ONNX),
            str(DET_ENGINE),
            fp16=True,
            min_shapes=min_shapes_det,
            opt_shapes=opt_shapes_det,
            max_shapes=max_shapes_det
        )
        print(f"[成功] OCR 检测 Engine 已保存: {DET_ENGINE}")
    except Exception as e:
        print(f"[失败] OCR 检测 Engine 构建失败: {e}")
        sys.exit(1)

    # ==================== 验证生成的文件 ====================
    print("\n" + "=" * 60)
    print("验证生成的 Engine 文件")
    print("=" * 60)

    engines = [
        ("OCR 检测模型", DET_ENGINE),
        ("OCR 识别模型", REC_ENGINE)
    ]

    all_exist = True
    for name, path in engines:
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  ✓ {name}: {path} ({size_mb:.2f} MB)")
        else:
            print(f"  ✗ {name}: {path} (未找到)")
            all_exist = False

    print("\n" + "=" * 60)
    if all_exist:
        print("[完成] 所有 TensorRT Engine 构建成功!")
        print("=" * 60)
        print("\n下一步:")
        print("  1. 运行推理测试:")
        print(f"     python3 {BASE_DIR / 'detect_crop_ocr.py'} <图像路径>")
        print("  2. YOLO 和 OCR 都将使用 TensorRT 进行高速推理")
    else:
        print("[错误] 部分 Engine 文件未生成，请检查错误信息")
        sys.exit(1)


if __name__ == '__main__':
    main()
