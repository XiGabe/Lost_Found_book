#!/usr/bin/env python3
"""
简单OCR演示脚本
演示如何使用批量OCR处理器

使用方法:
python simple_ocr_demo.py
"""

import os
from pathlib import Path

def demo_batch_ocr():
    """演示批量OCR处理"""

    print("🚀 批量OCR处理演示")
    print("="*50)

    # 导入批量处理器
    from batch_ocr_processor import BatchOCRProcessor

    # 设置输入和输出目录
    input_directory = "./data/dataset1"  # 你的图像目录
    output_directory = "./demo_ocr_results"

    # 检查输入目录是否存在
    if not os.path.exists(input_directory):
        print(f"❌ 输入目录不存在: {input_directory}")
        print("💡 请确保图像目录存在，或修改 input_directory 变量")
        return

    print(f"📁 输入目录: {input_directory}")
    print(f"📁 输出目录: {output_directory}")
    print()

    # 创建处理器
    print("🔧 初始化OCR处理器...")
    processor = BatchOCRProcessor(confidence_threshold=0.0)  # 0.0表示不过滤任何结果

    try:
        # 执行批量处理
        summary = processor.process_directory(input_directory, output_directory)

        # 打印结果
        processor.print_summary(summary)

        # 显示输出文件
        if summary.output_file:
            print(f"\n📄 详细结果文件: {summary.output_file}")

            # 简化版本的输出文件路径
            output_path = Path(summary.output_file)
            simple_output = output_path.parent / f"ocr_simple_{output_path.stem.split('_')[-1]}.json"
            print(f"📄 简化结果文件: {simple_output}")

    except Exception as e:
        print(f"❌ 处理失败: {e}")
        return

    print("\n🎉 演示完成!")

def demo_single_image():
    """演示单张图像OCR处理"""

    print("\n" + "="*50)
    print("🔍 单张图像OCR演示")
    print("="*50)

    from batch_ocr_processor import BatchOCRProcessor

    # 设置单张图像路径
    image_path = "./data/dataset1/8.png"  # 你的图像文件

    if not os.path.exists(image_path):
        print(f"❌ 图像文件不存在: {image_path}")
        print("💡 请确保图像文件存在，或修改 image_path 变量")
        return

    print(f"📖 处理图像: {image_path}")

    # 创建处理器
    processor = BatchOCRProcessor()

    try:
        # 处理单张图像
        result = processor.process_single_image(image_path)

        # 显示结果
        print(f"\n✅ 处理完成!")
        print(f"   状态: {result.status}")
        print(f"   识别文本: {result.reconstructed_text}")
        print(f"   置信度: {result.confidence:.3f}")
        print(f"   处理时间: {result.processing_time:.2f}秒")
        print(f"   文字列表: {result.transcriptions}")

        if result.error_message:
            print(f"   错误信息: {result.error_message}")

    except Exception as e:
        print(f"❌ 处理失败: {e}")

if __name__ == "__main__":
    # 运行演示
    demo_batch_ocr()
    demo_single_image()