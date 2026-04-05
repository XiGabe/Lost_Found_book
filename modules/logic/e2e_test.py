"""
端到端测试：YOLO检测 -> OCR识别 -> LSTM比较 -> 结果可视化

使用方法:
    python modules/logic/e2e_test.py data/images/test_image.jpg
    python modules/logic/e2e_test.py --batch data/images/*.jpg
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict

import cv2
import numpy as np
import torch

# 添加父目录到路径以导入现有模块（兼容性）
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from modules.vision import YOLODetector, OCRWrapper, DetectionResult, VisionConfig
from modules.logic.inference import load_model, compare_lcc


# ==================== 数据类 ====================

@dataclass
class PairComparisonResult:
    """单对比较结果"""
    text_a: str
    text_b: str
    idx_a: int
    idx_b: int
    label: int  # 0: In_Order, 1: Duplicate, 2: Out_of_Order
    label_name: str
    confidence: float
    probabilities: Dict[str, float]

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class E2ETestResult:
    """端到端测试结果"""
    image_path: str
    image_name: str
    processing_time: float

    # OCR统计
    num_detections: int
    num_successful_ocr: int
    ocr_texts: List[str]

    # 比较统计
    num_pairs: int
    num_out_of_order: int
    num_in_order: int
    num_duplicates: int

    # 详细结果
    pair_results: List[PairComparisonResult]
    out_of_order_pairs: List[PairComparisonResult]

    # 状态
    status: str  # 'success', 'partial_success', 'failed'
    error_message: Optional[str] = None

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['pair_results'] = [r.to_dict() for r in self.pair_results]
        d['out_of_order_pairs'] = [r.to_dict() for r in self.out_of_order_pairs]
        return d

    def get_summary(self) -> str:
        """生成人类可读的摘要"""
        lines = [
            "="*60,
            f"[端到端测试结果] {self.image_name}",
            "="*60,
            f"检测数量: {self.num_detections}",
            f"成功OCR: {self.num_successful_ocr}/{self.num_detections}",
            f"",
            f"比较统计:",
            f"  总对数: {self.num_pairs}",
            f"  ✓ 正确顺序: {self.num_in_order}",
            f"  ✗ 顺序错误: {self.num_out_of_order}",
            f"  ≈ 重复: {self.num_duplicates}",
            f"",
            f"处理时间: {self.processing_time:.3f}秒",
            f"状态: {self.status}",
        ]

        if self.out_of_order_pairs:
            lines.append("")
            lines.append("顺序错误的书本对:")
            for pair in self.out_of_order_pairs:
                lines.append(f"  [{pair.idx_a}, {pair.idx_b}] {pair.label_name}")
                lines.append(f"    A: {pair.text_a}")
                lines.append(f"    B: {pair.text_b}")
                lines.append(f"    置信度: {pair.confidence:.2%}")

        lines.append("="*60)
        return "\n".join(lines)


# ==================== 连续对生成器 ====================

class SequentialPairGenerator:
    """从OCR结果生成连续的书本对"""

    def generate_pairs(
        self,
        ocr_results: List[DetectionResult]
    ) -> List[Tuple[str, str, int, int]]:
        """
        从OCR结果生成连续对

        Args:
            ocr_results: OCR检测结果列表（已按x1坐标从左到右排序）

        Returns:
            List of (text_a, text_b, idx_a, idx_b) 元组
            例如：[("QA76.5", "QA76.8", 0, 1), ("QA76.8", "QA77.1", 1, 2), ...]
        """
        # 过滤掉空的OCR结果
        valid_results = [
            r for r in ocr_results
            if r.text and r.text.strip()
        ]

        if len(valid_results) < 2:
            return []

        # 生成连续对：(0,1), (1,2), (2,3), ...
        pairs = []
        for i in range(len(valid_results) - 1):
            text_a = valid_results[i].text
            text_b = valid_results[i+1].text
            idx_a = valid_results[i].idx
            idx_b = valid_results[i+1].idx

            pairs.append((text_a, text_b, idx_a, idx_b))

        return pairs


# ==================== 结果可视化器 ====================

class ResultVisualizer:
    """结果可视化 - 在原图上绘制颜色编码的标注"""

    # 颜色定义 (BGR格式，OpenCV使用)
    COLOR_IN_ORDER = (0, 255, 0)      # 绿色
    COLOR_OUT_OF_ORDER = (0, 0, 255)  # 红色
    COLOR_DUPLICATE = (0, 255, 255)   # 黄色
    COLOR_FAILED_OCR = (255, 0, 0)    # 蓝色

    def visualize(
        self,
        image: np.ndarray,
        ocr_results: List[DetectionResult],
        pair_results: List[PairComparisonResult],
        output_path: str
    ):
        """
        创建带标注的可视化结果

        Args:
            image: 原始图像
            ocr_results: OCR检测结果
            pair_results: 比较结果列表
            output_path: 输出图片路径
        """
        # 创建图像副本
        vis_image = image.copy()

        # 创建idx到颜色的映射
        idx_to_label = {}  # {idx: (label_name, color)}

        for pair in pair_results:
            if pair.label == 0:  # In_Order
                color = self.COLOR_IN_ORDER
            elif pair.label == 1:  # Duplicate
                color = self.COLOR_DUPLICATE
            else:  # Out_of_Order
                color = self.COLOR_OUT_OF_ORDER

            idx_to_label[pair.idx_a] = (pair.label_name, color)
            idx_to_label[pair.idx_b] = (pair.label_name, color)

        # 绘制每个检测框
        for result in ocr_results:
            x1, y1, x2, y2 = result.bbox

            # 获取颜色
            if result.idx in idx_to_label:
                label_name, color = idx_to_label[result.idx]
            else:
                color = self.COLOR_FAILED_OCR
                label_name = "Failed_OCR"

            # 绘制边界框（粗线）
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 3)

            # 绘制文字背景
            text = f"[{result.idx}] {result.text}"
            (text_w, text_h), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )

            # 确保文字不会超出图像边界
            text_y = max(y1 - 10, text_h + 10)
            cv2.rectangle(
                vis_image,
                (x1, text_y - text_h - 5),
                (x1 + text_w + 10, text_y + 5),
                color,
                -1
            )

            # 绘制文字
            cv2.putText(
                vis_image,
                text,
                (x1 + 5, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        # 绘制图例
        self._draw_legend(vis_image, pair_results)

        # 保存图像
        cv2.imwrite(output_path, vis_image)
        print(f"可视化已保存: {output_path}")

    def _draw_legend(
        self,
        image: np.ndarray,
        pair_results: List[PairComparisonResult]
    ):
        """在图像上绘制图例和统计信息"""
        h, w = image.shape[:2]

        # 统计
        num_in_order = sum(1 for p in pair_results if p.label == 0)
        num_duplicates = sum(1 for p in pair_results if p.label == 1)
        num_out_of_order = sum(1 for p in pair_results if p.label == 2)

        # 图例背景
        legend_x = 10
        legend_y = h - 120
        legend_w = 400
        legend_h = 110

        # 半透明背景
        overlay = image.copy()
        cv2.rectangle(
            overlay,
            (legend_x, legend_y),
            (legend_x + legend_w, legend_y + legend_h),
            (0, 0, 0),
            -1
        )
        cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)

        # 绘制图例项
        y_offset = legend_y + 25

        # In_Order
        cv2.rectangle(image, (legend_x + 10, y_offset - 10),
                     (legend_x + 30, y_offset + 10), self.COLOR_IN_ORDER, -1)
        cv2.putText(image, f"正确顺序 ({num_in_order})",
                   (legend_x + 40, y_offset + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        y_offset += 25

        # Out_of_Order
        cv2.rectangle(image, (legend_x + 10, y_offset - 10),
                     (legend_x + 30, y_offset + 10), self.COLOR_OUT_OF_ORDER, -1)
        cv2.putText(image, f"顺序错误 ({num_out_of_order})",
                   (legend_x + 40, y_offset + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        y_offset += 25

        # Duplicate
        cv2.rectangle(image, (legend_x + 10, y_offset - 10),
                     (legend_x + 30, y_offset + 10), self.COLOR_DUPLICATE, -1)
        cv2.putText(image, f"重复 ({num_duplicates})",
                   (legend_x + 40, y_offset + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


# ==================== 端到端测试流水线 ====================

class E2EPipeline:
    """端到端测试流水线：YOLO -> OCR -> LSTM -> 结果"""

    def __init__(
        self,
        vision_config: VisionConfig = None,
        comparator_checkpoint: str = None,
        output_dir: str = None
    ):
        """
        初始化端到端流水线

        Args:
            vision_config: 视觉模块配置
            comparator_checkpoint: LSTM比较器模型路径
            output_dir: 输出目录路径
        """
        self.vision_config = vision_config or VisionConfig()
        self.comparator_checkpoint = comparator_checkpoint or 'weights/comparator.pth'

        # 设置输出目录
        if output_dir is None:
            output_dir = Path(__file__).parent.parent.parent / 'output' / 'e2e_results'
        self.output_dir = Path(output_dir)
        self.json_dir = self.output_dir / 'json'
        self.vis_dir = self.output_dir / 'visualizations'

        # 创建输出目录
        self.json_dir.mkdir(parents=True, exist_ok=True)
        self.vis_dir.mkdir(parents=True, exist_ok=True)

        # 初始化组件
        print("[初始化] 加载YOLO检测器...")
        self.yolo_detector = YOLODetector(
            model_path=self.vision_config.YOLO_MODEL_PATH,
            fallback_path=self.vision_config.YOLO_FALLBACK_PATH,
            conf_threshold=self.vision_config.YOLO_CONF_THRESHOLD,
            iou_threshold=self.vision_config.YOLO_IOU_THRESHOLD,
            img_size=self.vision_config.YOLO_IMG_SIZE,
            device=self.vision_config.YOLO_DEVICE,
            target_class_id=self.vision_config.TARGET_CLASS_ID
        )

        print("[初始化] 加载OCR引擎...")
        self.ocr_engine = OCRWrapper(
            backend=self.vision_config.OCR_BACKEND,
            device=self.vision_config.OCR_DEVICE,
            trt_det_model_path=self.vision_config.OCR_TRT_DET_PATH,
            trt_rec_model_path=self.vision_config.OCR_TRT_REC_PATH,
            warmup=True
        )

        print("[初始化] 加载LSTM比较器...")
        self.comparator_model, self.comparator_tokenizer, self.device = load_model(
            self.comparator_checkpoint
        )

        # 初始化辅助组件
        self.pair_generator = SequentialPairGenerator()
        self.visualizer = ResultVisualizer()

        print("[初始化] 完成\n")

    def process_single_image(
        self,
        image_path: str,
        save_json: bool = True,
        save_visualization: bool = True
    ) -> E2ETestResult:
        """
        处理单张图片的完整端到端流程

        Args:
            image_path: 输入图片路径
            save_json: 是否保存JSON结果
            save_visualization: 是否保存可视化结果

        Returns:
            E2ETestResult对象
        """
        start_time = time.time()

        # 验证输入
        if not os.path.exists(image_path):
            result = E2ETestResult(
                image_path=image_path,
                image_name=os.path.basename(image_path),
                processing_time=0,
                num_detections=0,
                num_successful_ocr=0,
                ocr_texts=[],
                num_pairs=0,
                num_out_of_order=0,
                num_in_order=0,
                num_duplicates=0,
                pair_results=[],
                out_of_order_pairs=[],
                status='failed',
                error_message=f'图片不存在: {image_path}'
            )
            return result

        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            result = E2ETestResult(
                image_path=image_path,
                image_name=os.path.basename(image_path),
                processing_time=0,
                num_detections=0,
                num_successful_ocr=0,
                ocr_texts=[],
                num_pairs=0,
                num_out_of_order=0,
                num_in_order=0,
                num_duplicates=0,
                pair_results=[],
                out_of_order_pairs=[],
                status='failed',
                error_message=f'无法读取图片: {image_path}'
            )
            return result

        image_name = Path(image_path).stem

        print(f"\n{'='*60}")
        print(f"[STEP 1] YOLO检测 - {image_name}")
        print(f"{'='*60}")

        # Step 1: YOLO检测
        detections = self.yolo_detector.detect_labels(image)
        print(f"检测到 {len(detections)} 个标签")

        if len(detections) == 0:
            result = E2ETestResult(
                image_path=image_path,
                image_name=os.path.basename(image_path),
                processing_time=time.time() - start_time,
                num_detections=0,
                num_successful_ocr=0,
                ocr_texts=[],
                num_pairs=0,
                num_out_of_order=0,
                num_in_order=0,
                num_duplicates=0,
                pair_results=[],
                out_of_order_pairs=[],
                status='success',
                error_message='未检测到任何标签'
            )
            return result

        print(f"\n{'='*60}")
        print(f"[STEP 2] OCR识别")
        print(f"{'='*60}")

        # Step 2: 裁剪标签区域
        crops = self.yolo_detector.crop_labels(image, detections)

        # Step 3: OCR识别
        ocr_texts = self.ocr_engine.recognize_batch(crops, verbose=False)

        # Step 4: 创建DetectionResult对象并打印结果
        ocr_results = []
        for idx, (det, text) in enumerate(zip(detections, ocr_texts)):
            bbox = det['bbox']
            conf = det['conf']
            x1, y1, x2, y2 = bbox

            print(f"  [{idx+1}/{len(detections)}] bbox=({x1},{y1},{x2},{y2}), conf={conf:.3f} -> OCR: '{text}'")

            result = DetectionResult(
                idx=idx,
                bbox=bbox,
                text=text,
                confidence=conf
            )
            ocr_results.append(result)

        # 统计成功的OCR
        successful_ocr = [r for r in ocr_results if r.text and r.text.strip()]
        ocr_texts = [r.text for r in successful_ocr]

        print(f"\n{'='*60}")
        print(f"[STEP 3] 生成连续对")
        print(f"{'='*60}")

        # Step 2: 生成连续对
        pairs = self.pair_generator.generate_pairs(ocr_results)
        print(f"生成 {len(pairs)} 个连续对")

        if len(pairs) == 0:
            result = E2ETestResult(
                image_path=image_path,
                image_name=os.path.basename(image_path),
                processing_time=time.time() - start_time,
                num_detections=len(ocr_results),
                num_successful_ocr=len(successful_ocr),
                ocr_texts=ocr_texts,
                num_pairs=0,
                num_out_of_order=0,
                num_in_order=0,
                num_duplicates=0,
                pair_results=[],
                out_of_order_pairs=[],
                status='success',
                error_message='检测数量不足，无法生成比较对'
            )
            return result

        print(f"\n{'='*60}")
        print(f"[STEP 4] LSTM比较")
        print(f"{'='*60}")

        # Step 3: LSTM批量比较
        pair_results = []
        out_of_order_pairs = []

        for text_a, text_b, idx_a, idx_b in pairs:
            result = compare_lcc(
                text_a, text_b,
                model=self.comparator_model,
                tokenizer=self.comparator_tokenizer,
                device=self.device
            )

            pair_result = PairComparisonResult(
                text_a=text_a,
                text_b=text_b,
                idx_a=idx_a,
                idx_b=idx_b,
                label=result['label'],
                label_name=result['label_name'],
                confidence=result['confidence'],
                probabilities=result['probabilities']
            )
            pair_results.append(pair_result)

            # 记录乱序对
            if result['label'] == 1:  # Out_of_Order
                out_of_order_pairs.append(pair_result)

            print(f"  [{idx_a}, {idx_b}] {text_a} vs {text_b} -> {result['label_name']} ({result['confidence']:.2%})")

        # 统计
        num_in_order = sum(1 for p in pair_results if p.label == 0)
        num_duplicates = sum(1 for p in pair_results if p.label == 1)
        num_out_of_order = sum(1 for p in pair_results if p.label == 2)

        # 创建结果对象
        processing_time = time.time() - start_time

        result = E2ETestResult(
            image_path=image_path,
            image_name=os.path.basename(image_path),
            processing_time=processing_time,
            num_detections=len(ocr_results),
            num_successful_ocr=len(successful_ocr),
            ocr_texts=ocr_texts,
            num_pairs=len(pair_results),
            num_out_of_order=num_out_of_order,
            num_in_order=num_in_order,
            num_duplicates=num_duplicates,
            pair_results=pair_results,
            out_of_order_pairs=out_of_order_pairs,
            status='success'
        )

        # 保存结果
        if save_json:
            json_path = self.json_dir / f"{image_name}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
            print(f"\nJSON已保存: {json_path}")

        if save_visualization:
            vis_path = self.vis_dir / f"{image_name}.jpg"
            self.visualizer.visualize(image, ocr_results, pair_results, str(vis_path))

        return result

    def process_image_bytes(self, image_bytes: bytes, **kwargs) -> E2ETestResult:
        """
        处理字节格式的图片数据

        Args:
            image_bytes: 图片字节数据 (JPEG/PNG 等)
            **kwargs: 传递给 process_single_image 的其他参数

        Returns:
            E2ETestResult 对象
        """
        import cv2
        import numpy as np
        from pathlib import Path
        import uuid

        # 将字节解码为 numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            result = E2ETestResult(
                image_path="bytes_input",
                image_name="bytes_input",
                processing_time=0,
                num_detections=0,
                num_successful_ocr=0,
                ocr_texts=[],
                num_pairs=0,
                num_out_of_order=0,
                num_in_order=0,
                num_duplicates=0,
                pair_results=[],
                out_of_order_pairs=[],
                status='failed',
                error_message='无法解码图片字节数据'
            )
            return result

        # 临时保存到文件，用于兼容现有的 process_single_image
        temp_dir = Path(__file__).parent.parent.parent / 'output' / 'temp'
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / f"temp_{uuid.uuid4().hex}.jpg"
        cv2.imwrite(str(temp_path), image)

        try:
            return self.process_single_image(str(temp_path), **kwargs)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def process_batch(
        self,
        image_paths: List[str],
        save_json: bool = True,
        save_visualization: bool = True
    ) -> List[E2ETestResult]:
        """
        批量处理多张图片

        Args:
            image_paths: 图片路径列表
            save_json: 是否保存JSON结果
            save_visualization: 是否保存可视化结果

        Returns:
            E2ETestResult对象列表
        """
        results = []

        for i, image_path in enumerate(image_paths):
            print(f"\n\n{'#'*60}")
            print(f"处理进度: [{i+1}/{len(image_paths)}] {image_path}")
            print(f"{'#'*60}")

            result = self.process_single_image(
                image_path,
                save_json=save_json,
                save_visualization=save_visualization
            )

            results.append(result)

            # 打印摘要
            print(result.get_summary())

        return results


# ==================== CLI接口 ====================

def main():
    parser = argparse.ArgumentParser(
        description='端到端测试：YOLO检测 -> OCR识别 -> LSTM比较 -> 可视化'
    )

    parser.add_argument(
        'image_path',
        type=str,
        nargs='?',
        help='输入图片路径（或使用--batch指定多张图片）'
    )
    parser.add_argument(
        '--batch',
        type=str,
        nargs='+',
        help='批量处理多张图片'
    )
    parser.add_argument(
        '--comparator',
        type=str,
        default='weights/comparator.pth',
        help='LSTM比较器模型路径'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='输出目录路径（默认：output/e2e_results/）'
    )
    parser.add_argument(
        '--no-json',
        action='store_true',
        help='不保存JSON结果'
    )
    parser.add_argument(
        '--no-visualization',
        action='store_true',
        help='不生成可视化结果'
    )

    args = parser.parse_args()

    # 检查输入
    if args.batch:
        image_paths = args.batch
    elif args.image_path:
        image_paths = [args.image_path]
    else:
        parser.print_help()
        print("\n错误：必须指定 image_path 或 --batch")
        sys.exit(1)

    # 创建流水线
    pipeline = E2EPipeline(
        comparator_checkpoint=args.comparator,
        output_dir=args.output_dir
    )

    # 处理
    if len(image_paths) == 1:
        result = pipeline.process_single_image(
            image_paths[0],
            save_json=not args.no_json,
            save_visualization=not args.no_visualization
        )
        print(result.get_summary())
    else:
        results = pipeline.process_batch(
            image_paths,
            save_json=not args.no_json,
            save_visualization=not args.no_visualization
        )

        # 打印总体统计
        print("\n\n" + "="*60)
        print("批量处理完成")
        print("="*60)
        print(f"总图片数: {len(results)}")
        print(f"成功: {sum(1 for r in results if r.status == 'success')}")
        print(f"失败: {sum(1 for r in results if r.status == 'failed')}")

        total_out_of_order = sum(r.num_out_of_order for r in results)
        print(f"总乱序对数: {total_out_of_order}")


if __name__ == '__main__':
    main()
