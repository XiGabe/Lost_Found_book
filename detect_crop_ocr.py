"""
完整流水线实现: YOLO 检测 -> 裁剪 -> OCR -> 输出 JSON + 可视化
"""
import os
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO
from openocr import OpenOCR


# ==================== 配置 ====================
class Config:
    """流水线配置"""

    # 模型目录
    MODELS_DIR = Path(__file__).parent / 'models'

    # YOLO 模型路径
    YOLO_MODEL_PATH = str(MODELS_DIR / 'yolo/best.engine')
    YOLO_FALLBACK_PATH = str(MODELS_DIR / 'yolo/best.pt')

    # YOLO 检测参数
    YOLO_IMG_SIZE = 640
    YOLO_CONF_THRESHOLD = 0.25
    YOLO_IOU_THRESHOLD = 0.45
    YOLO_DEVICE = 0

    # 目标检测类别 (call_label)
    TARGET_CLASS_ID = 1
    TARGET_CLASS_NAME = 'call_label'

    # OCR 参数 - 启用 TensorRT 加速
    # 可选: 'torch', 'onnx', 'tensorrt'
    # 'tensorrt' 需要 .engine 文件，性能最优
    OCR_BACKEND = 'tensorrt'
    OCR_DEVICE = 'gpu'
    OCR_TRT_REC_PATH = str(MODELS_DIR / 'ocr/openocr_rec_model.trt.engine')
    OCR_TRT_DET_PATH = str(MODELS_DIR / 'ocr/openocr_det_model.trt.engine')

    # 输出路径
    OUTPUT_DIR = Path('output')
    JSON_DIR = OUTPUT_DIR / 'json'
# =============================================


@dataclass
class DetectionResult:
    """单个检测结果的数据类"""
    idx: int                    # 标签索引 ID
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) 绝对像素坐标
    text: str                   # OCR 识别的文本
    confidence: float           # 检测置信度

    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'id': self.idx,
            'bbox': {
                'x1': int(self.bbox[0]),
                'y1': int(self.bbox[1]),
                'x2': int(self.bbox[2]),
                'y2': int(self.bbox[3])
            },
            'text': self.text,
            'confidence': float(self.confidence)
        }


class DetectCropOCRPipeline:
    """完整流水线处理器"""

    def __init__(self, config: Config = None):
        """
        初始化流水线

        Args:
            config: 配置对象，默认使用 Config()
        """
        self.config = config or Config()

        # 创建输出目录
        self._create_directories()

        # 初始化模型
        self.yolo_model = None
        self.ocr_engine = None
        self._load_models()

    def _create_directories(self):
        """创建输出目录"""
        self.config.JSON_DIR.mkdir(parents=True, exist_ok=True)

    def _load_models(self):
        """加载 YOLO 和 OCR 模型"""
        # 加载 YOLO 模型
        yolo_path = self.config.YOLO_MODEL_PATH
        if not os.path.exists(yolo_path):
            yolo_path = self.config.YOLO_FALLBACK_PATH
            if not os.path.exists(yolo_path):
                raise FileNotFoundError(f"YOLO 模型不存在: {yolo_path}")

        print(f"[加载 YOLO] {yolo_path}")
        self.yolo_model = YOLO(yolo_path, task='segment')

        # 加载 OCR 引擎（TensorRT/ONNX 加速）
        print(f"[加载 OCR] OpenOCR (backend={self.config.OCR_BACKEND}, device={self.config.OCR_DEVICE})")

        # 根据 backend 准备模型路径
        ocr_kwargs = {'backend': self.config.OCR_BACKEND, 'device': self.config.OCR_DEVICE}

        if self.config.OCR_BACKEND == 'tensorrt':
            ocr_kwargs['trt_det_model_path'] = self.config.OCR_TRT_DET_PATH
            ocr_kwargs['trt_rec_model_path'] = self.config.OCR_TRT_REC_PATH
            print(f"  - TRT Detection: {self.config.OCR_TRT_DET_PATH}")
            print(f"  - TRT Recognition: {self.config.OCR_TRT_REC_PATH}")

        self.ocr_engine = OpenOCR(**ocr_kwargs)

        # 预热 OCR 模型（消除首帧延迟）
        print("[预热 OCR] 运行一次推理以初始化模型...")
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        try:
            _ = self.ocr_engine(img_numpy=dummy_img)
            print("[预热 OCR] 完成")
        except Exception as e:
            print(f"[预热 OCR] 警告: {e}")

    def process_image(self, image_path: str, output_name: str = None) -> Dict:
        """
        处理单张图片的完整流程

        Args:
            image_path: 输入图片路径
            output_name: 输出文件名前缀，默认使用图片原名

        Returns:
            包含检测结果的字典
        """
        start_time = time.time()

        # 验证输入
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片不存在: {image_path}")

        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")

        # 设置输出名称
        if output_name is None:
            output_name = Path(image_path).stem

        print(f"\n{'='*60}")
        print(f"[STEP 1] YOLO 检测 - {image_path}")
        print(f"{'='*60}")

        # 步骤 1: YOLO 检测
        detection_results = self._detect_labels(image)
        print(f"检测到 {len(detection_results)} 个 call_label")

        if len(detection_results) == 0:
            print("[WARNING] 未检测到任何 call_label，跳过 OCR 处理")
            return self._create_empty_result(image, image_path, output_name, start_time)

        print(f"\n{'='*60}")
        print(f"[STEP 2] OCR 识别")
        print(f"{'='*60}")

        # 步骤 2: 裁剪 + OCR 识别
        detection_results = self._ocr_recognition(
            image, detection_results, output_name
        )

        print(f"\n{'='*60}")
        print(f"[STEP 3] 生成输出")
        print(f"{'='*60}")

        # 步骤 3: 生成 JSON
        result_dict = self._generate_outputs(
            detection_results, image_path, output_name, start_time
        )

        # 打印摘要
        self._print_summary(detection_results, start_time)

        return result_dict

    def _detect_labels(self, image: np.ndarray) -> List[Dict]:
        """
        使用 YOLO 检测所有 call_label

        Returns:
            检测框列表，每个元素包含 {'bbox': (x1,y1,x2,y2), 'conf': float}
        """
        results = self.yolo_model.predict(
            source=image,
            imgsz=self.config.YOLO_IMG_SIZE,
            conf=self.config.YOLO_CONF_THRESHOLD,
            iou=self.config.YOLO_IOU_THRESHOLD,
            device=self.config.YOLO_DEVICE,
            save=False,
            verbose=False
        )

        # 提取结果 (第一张图)
        result = results[0]

        # 提取 call_label 类别 (class_id = 1)
        detections = []

        if result.boxes is not None:
            boxes = result.boxes

            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())

                # 只处理 call_label (class_id = 1)
                if cls_id == self.config.TARGET_CLASS_ID:
                    # 获取边界框坐标 (xyxy 格式)
                    bbox = boxes.xyxy[i].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = bbox

                    # 边界检查
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(image.shape[1], x2)
                    y2 = min(image.shape[0], y2)

                    if x2 > x1 and y2 > y1:  # 有效框
                        detections.append({
                            'bbox': (x1, y1, x2, y2),
                            'conf': conf
                        })

        # 按 x1 坐标从左到右排序
        detections.sort(key=lambda d: d['bbox'][0])

        return detections

    def _ocr_recognition(
        self,
        image: np.ndarray,
        detections: List[Dict],
        output_name: str
    ) -> List[DetectionResult]:
        """
        批量 OCR 识别（优化版）：收集所有裁剪图，一次性调用 OCR

        Args:
            image: 原图
            detections: 检测框列表
            output_name: 输出文件名前缀

        Returns:
            DetectionResult 对象列表
        """
        if not detections:
            return []

        print(f"  批量 OCR 处理 {len(detections)} 个标签...")

        # 1. 收集所有裁剪图
        crop_images = []
        for idx, det in enumerate(detections):
            bbox = det['bbox']
            x1, y1, x2, y2 = bbox
            crop_img = image[y1:y2, x1:x2]
            crop_images.append(crop_img)

        # 2. 批量 OCR 调用（ONNX/TensorRT 加速）
        try:
            ocr_results, _ = self.ocr_engine(img_numpy=crop_images)
        except Exception as e:
            print(f"      [ERROR] 批量 OCR 失败: {e}")
            # 回退到空结果
            ocr_results = [[] for _ in range(len(detections))]

        # 3. 解析批量结果并创建 DetectionResult
        results = []
        for idx, (det, ocr_result) in enumerate(zip(detections, ocr_results)):
            bbox = det['bbox']
            conf = det['conf']
            x1, y1, x2, y2 = bbox

            # 解析 OCR 结果 (OpenOCR 返回格式: [{'transcription': '...', ...}, ...])
            text_parts = []
            if isinstance(ocr_result, list):
                for item in ocr_result:
                    if isinstance(item, dict) and 'transcription' in item:
                        text_parts.append(item['transcription'])

            text = ' '.join(text_parts).strip()

            print(f"  [{idx+1}/{len(detections)}] bbox=({x1},{y1},{x2},{y2}), conf={conf:.3f} -> OCR: '{text}'")

            # 创建结果对象
            result = DetectionResult(
                idx=idx,
                bbox=bbox,
                text=text,
                confidence=conf
            )
            results.append(result)

        return results

    def _generate_outputs(
        self,
        detection_results: List[DetectionResult],
        image_path: str,
        output_name: str,
        start_time: float
    ) -> Dict:
        """
        生成 JSON 输出
        """
        processing_time = time.time() - start_time

        # 构建结果字典
        result_dict = {
            'image_path': image_path,
            'image_name': os.path.basename(image_path),
            'processing_time': f"{processing_time:.3f}",
            'num_detections': len(detection_results),
            'detections': [r.to_dict() for r in detection_results]
        }

        # 保存 JSON 文件
        json_path = self.config.JSON_DIR / f"{output_name}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=2)
        print(f"JSON 保存: {json_path}")

        return result_dict

    def _create_empty_result(
        self,
        image: np.ndarray,
        image_path: str,
        output_name: str,
        start_time: float
    ) -> Dict:
        """创建空结果 (无检测到目标时)"""
        processing_time = time.time() - start_time

        result_dict = {
            'image_path': image_path,
            'image_name': os.path.basename(image_path),
            'processing_time': f"{processing_time:.3f}",
            'num_detections': 0,
            'detections': []
        }

        # 保存 JSON
        json_path = self.config.JSON_DIR / f"{output_name}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=2)

        return result_dict

    def _print_summary(self, results: List[DetectionResult], start_time: float):
        """打印处理摘要"""
        processing_time = time.time() - start_time

        print(f"\n{'='*60}")
        print(f"[摘要] 处理完成")
        print(f"{'='*60}")
        print(f"  检测数量:    {len(results)}")
        print(f"  处理耗时:    {processing_time:.3f} 秒")
        print(f"  平均每个:    {processing_time/max(len(results),1):.3f} 秒")
        print(f"{'='*60}\n")


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description='YOLO 检测 -> 裁剪 -> OCR 流水线'
    )
    parser.add_argument(
        'image_path',
        type=str,
        help='输入图片路径'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='输出文件名前缀 (默认使用图片原名)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='YOLO 模型路径 (覆盖配置文件)'
    )

    args = parser.parse_args()

    # 创建配置
    config = Config()
    if args.model:
        config.YOLO_MODEL_PATH = args.model

    # 运行流水线
    pipeline = DetectCropOCRPipeline(config)
    result = pipeline.process_image(args.image_path, args.output)

    print(f"\n[完成] 结果已保存到: {config.OUTPUT_DIR}")


if __name__ == '__main__':
    main()
