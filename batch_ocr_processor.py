#!/usr/bin/env python3
"""
批量OCR处理器
支持对整个文件夹的图像进行OCR处理，并输出统一格式的结果

Author: Hongxi Chen
Date: 2025-11-04
"""

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import logging
from datetime import datetime

# OpenOCR导入 (根据你的实际安装方式)
try:
    from openocr import OpenOCR
except ImportError:
    print("请安装OpenOCR: pip install openocr")
    exit(1)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class OCRResult:
    """OCR结果数据结构"""
    image_path: str
    image_name: str
    ocr_data: List[Dict]  # 原始OCR数据
    transcriptions: List[str]  # 提取的文字列表
    reconstructed_text: str  # 重构的完整文本
    confidence: float  # 平均置信度
    processing_time: float  # 处理时间(秒)
    timestamp: str  # 处理时间戳
    status: str  # "success", "error"
    error_message: Optional[str] = None

@dataclass
class BatchOCRSummary:
    """批量OCR处理摘要"""
    total_images: int
    successful: int
    failed: int
    total_time: float
    average_time_per_image: float
    output_file: str

class BatchOCRProcessor:
    """批量OCR处理器"""

    def __init__(self, confidence_threshold: float = 0.0):
        """
        初始化批量OCR处理器

        Args:
            confidence_threshold: 置信度阈值，低于此值的结果将被过滤
        """
        self.confidence_threshold = confidence_threshold
        self.engine = OpenOCR()
        self.results: List[OCRResult] = []

    def process_single_image(self, image_path: str) -> OCRResult:
        """
        处理单个图像文件

        Args:
            image_path: 图像文件路径

        Returns:
            OCRResult: OCR处理结果
        """
        start_time = time.time()
        timestamp = datetime.now().isoformat()
        image_name = Path(image_path).name

        try:
            logger.info(f"处理图像: {image_name}")

            # 执行OCR
            ocr_data, elapse = self.engine(image_path)

            # 提取transcriptions
            transcriptions = []
            total_confidence = 0.0

            # OpenOCR返回的是字符串格式，需要解析
            if isinstance(ocr_data, list) and len(ocr_data) > 0:
                # ocr_data[0] 包含实际的OCR结果字符串
                ocr_string = ocr_data[0]
                if isinstance(ocr_string, str) and '\t' in ocr_string:
                    # 解析类似: "8.png\t[{\"transcription\": \"OLIN\", \"score\": 0.998}, ...]"
                    try:
                        import json as json_module
                        # 提取JSON部分
                        json_part = ocr_string.split('\t', 1)[1]
                        ocr_results = json_module.loads(json_part)

                        if isinstance(ocr_results, list):
                            for item in ocr_results:
                                if isinstance(item, dict) and 'transcription' in item:
                                    text = item['transcription']
                                    confidence = item.get('score', item.get('confidence', 1.0))

                                    # 应用置信度过滤
                                    if confidence >= self.confidence_threshold:
                                        transcriptions.append(text)
                                        total_confidence += confidence

                    except (json_module.JSONDecodeError, IndexError, KeyError) as e:
                        logger.warning(f"解析OCR数据失败: {e}")
                        # 如果解析失败，尝试简单的文本提取
                        pass

            # 计算平均置信度
            avg_confidence = total_confidence / len(transcriptions) if transcriptions else 0.0

            # 重构文本
            reconstructed_text = ' '.join(transcriptions)

            processing_time = time.time() - start_time

            result = OCRResult(
                image_path=image_path,
                image_name=image_name,
                ocr_data=ocr_data,
                transcriptions=transcriptions,
                reconstructed_text=reconstructed_text,
                confidence=avg_confidence,
                processing_time=processing_time,
                timestamp=timestamp,
                status="success"
            )

            logger.info(f"✅ 完成: {image_name}, 耗时: {processing_time:.2f}s, 置信度: {avg_confidence:.3f}")
            return result

        except Exception as e:
            processing_time = time.time() - start_time
            error_message = str(e)

            logger.error(f"❌ 失败: {image_name}, 错误: {error_message}")

            return OCRResult(
                image_path=image_path,
                image_name=image_name,
                ocr_data=[],
                transcriptions=[],
                reconstructed_text="",
                confidence=0.0,
                processing_time=processing_time,
                timestamp=timestamp,
                status="error",
                error_message=error_message
            )

    def process_directory(self, input_dir: str, output_dir: str = "./ocr_results") -> BatchOCRSummary:
        """
        处理整个目录的图像文件

        Args:
            input_dir: 输入图像目录
            output_dir: 输出结果目录

        Returns:
            BatchOCRSummary: 批量处理摘要
        """
        start_time = time.time()

        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 获取所有图像文件
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"输入目录不存在: {input_dir}")

        # 支持的图像格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))

        total_images = len(image_files)
        if total_images == 0:
            logger.warning(f"在目录 {input_dir} 中没有找到图像文件")
            return BatchOCRSummary(
                total_images=0,
                successful=0,
                failed=0,
                total_time=0.0,
                average_time_per_image=0.0,
                output_file=""
            )

        logger.info(f"📁 找到 {total_images} 个图像文件，开始批量处理...")

        # 处理每个图像
        successful = 0
        failed = 0
        self.results = []

        for i, image_file in enumerate(image_files, 1):
            logger.info(f"📖 进度: {i}/{total_images}")

            result = self.process_single_image(str(image_file))
            self.results.append(result)

            if result.status == "success":
                successful += 1
            else:
                failed += 1

        # 计算总处理时间
        total_time = time.time() - start_time
        avg_time = total_time / total_images if total_images > 0 else 0.0

        # 保存结果
        output_file = self.save_results(output_path)

        # 创建摘要
        summary = BatchOCRSummary(
            total_images=total_images,
            successful=successful,
            failed=failed,
            total_time=total_time,
            average_time_per_image=avg_time,
            output_file=output_file
        )

        logger.info(f"🎉 批量处理完成!")
        logger.info(f"   总计: {total_images} | 成功: {successful} | 失败: {failed}")
        logger.info(f"   总耗时: {total_time:.2f}s | 平均: {avg_time:.2f}s/图")
        logger.info(f"   结果已保存到: {output_file}")

        return summary

    def save_results(self, output_path: Path) -> str:
        """
        保存OCR结果到JSON文件

        Args:
            output_path: 输出目录路径

        Returns:
            str: 输出文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_path / f"ocr_results_{timestamp}.json"

        # 准备输出数据
        output_data = {
            "metadata": {
                "total_images": len(self.results),
                "processing_time": datetime.now().isoformat(),
                "confidence_threshold": self.confidence_threshold,
                "processor_version": "1.0"
            },
            "results": [asdict(result) for result in self.results]
        }

        # 保存到文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        # 同时保存简化版本（只包含关键信息）
        simple_output_file = output_path / f"ocr_simple_{timestamp}.json"
        simple_data = {
            "results": [
                {
                    "image_name": result.image_name,
                    "text": result.reconstructed_text,
                    "confidence": result.confidence,
                    "status": result.status
                }
                for result in self.results
            ]
        }

        with open(simple_output_file, 'w', encoding='utf-8') as f:
            json.dump(simple_data, f, indent=2, ensure_ascii=False)

        # 生成CSV格式的报告
        csv_file = output_path / f"ocr_report_{timestamp}.csv"
        with open(csv_file, 'w', encoding='utf-8') as f:
            f.write("image_name,transcription,confidence,processing_time,status\n")
            for result in self.results:
                transcriptions = "|".join(result.transcriptions)
                f.write(f"{result.image_name},{transcriptions},{result.confidence:.3f},"
                       f"{result.processing_time:.3f},{result.status}\n")

        return str(output_file)

    def print_summary(self, summary: BatchOCRSummary):
        """
        打印处理摘要

        Args:
            summary: 批量处理摘要
        """
        print("\n" + "="*60)
        print("📊 批量OCR处理报告")
        print("="*60)
        print(f"📁 处理图像数量: {summary.total_images}")
        print(f"✅ 成功处理: {summary.successful}")
        print(f"❌ 处理失败: {summary.failed}")
        print(f"⏱️  总处理时间: {summary.total_time:.2f}秒")
        print(f"📈 平均处理时间: {summary.average_time_per_image:.2f}秒/图")
        print(f"📄 结果文件: {summary.output_file}")
        print("="*60)

        # 显示成功处理的图像示例
        successful_results = [r for r in self.results if r.status == "success"]
        if successful_results:
            print("\n📖 处理示例 (前5个成功的结果):")
            for i, result in enumerate(successful_results[:5]):
                print(f"\n{i+1}. {result.image_name}")
                print(f"   识别文本: {result.reconstructed_text}")
                print(f"   置信度: {result.confidence:.3f}")
                print(f"   处理时间: {result.processing_time:.2f}s")

def main():
    """
    主函数 - 命令行接口
    """
    import argparse

    parser = argparse.ArgumentParser(description="批量OCR处理器")
    parser.add_argument("--input", "-i", required=True,
                       help="输入图像目录路径")
    parser.add_argument("--output", "-o", default="./ocr_results",
                       help="输出结果目录路径")
    parser.add_argument("--confidence", "-c", type=float, default=0.0,
                       help="置信度阈值 (0.0-1.0)")

    args = parser.parse_args()

    try:
        # 创建处理器
        processor = BatchOCRProcessor(confidence_threshold=args.confidence)

        # 执行批量处理
        summary = processor.process_directory(args.input, args.output)

        # 打印摘要
        processor.print_summary(summary)

    except Exception as e:
        logger.error(f"批量处理失败: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())