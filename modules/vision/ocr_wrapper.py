"""
OCR封装模块 - 使用OpenOCR识别裁剪的书标图像
"""
import os
import sys
from pathlib import Path
from typing import List, Dict

import numpy as np

# 添加external目录到路径以导入OpenOCR
_external_path = Path(__file__).parent.parent / 'external'
if str(_external_path) not in sys.path:
    sys.path.insert(0, str(_external_path))

from openocr import OpenOCR


class OCRWrapper:
    """OCR引擎封装 - 批量识别裁剪的标签图像"""

    def __init__(
        self,
        backend: str = 'tensorrt',
        device: str = 'gpu',
        trt_det_model_path: str = None,
        trt_rec_model_path: str = None,
        warmup: bool = True
    ):
        """
        初始化OCR引擎

        Args:
            backend: 后端类型 ('tensorrt', 'onnx', 'torch')
            device: 设备类型 ('gpu' 或 'cpu')
            trt_det_model_path: TensorRT检测模型路径
            trt_rec_model_path: TensorRT识别模型路径
            warmup: 是否预热模型（消除首帧延迟）
        """
        self.backend = backend
        self.device = device

        # 准备OCR模型路径
        ocr_kwargs = {'backend': backend, 'device': device}

        if backend == 'tensorrt':
            if trt_det_model_path:
                ocr_kwargs['trt_det_model_path'] = trt_det_model_path
            if trt_rec_model_path:
                ocr_kwargs['trt_rec_model_path'] = trt_rec_model_path

            if trt_det_model_path and trt_rec_model_path:
                print(f"  - TRT Detection: {trt_det_model_path}")
                print(f"  - TRT Recognition: {trt_rec_model_path}")

        # 加载OCR引擎
        print(f"[加载 OCR] OpenOCR (backend={backend}, device={device})")
        self.ocr_engine = OpenOCR(**ocr_kwargs)

        # 预热模型
        if warmup:
            self._warmup()

    def _warmup(self):
        """预热OCR模型（消除首帧延迟）"""
        print("[预热 OCR] 运行一次推理以初始化模型...")
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        try:
            _ = self.ocr_engine(img_numpy=dummy_img)
            print("[预热 OCR] 完成")
        except Exception as e:
            print(f"[预热 OCR] 警告: {e}")

    def recognize_batch(
        self,
        images: List[np.ndarray],
        verbose: bool = True
    ) -> List[str]:
        """
        批量OCR识别

        Args:
            images: 裁剪的标签图像列表
            verbose: 是否打印详细输出

        Returns:
            识别的文本列表
        """
        if not images:
            return []

        if verbose:
            print(f"  批量 OCR 处理 {len(images)} 个标签...")

        # 批量OCR调用
        try:
            ocr_results, _ = self.ocr_engine(img_numpy=images)
        except Exception as e:
            print(f"      [ERROR] 批量 OCR 失败: {e}")
            # 返回空结果
            return ['' for _ in range(len(images))]

        # 解析结果
        texts = []
        for idx, ocr_result in enumerate(ocr_results):
            # 解析 OCR 结果
            # OpenOCR 返回格式: [{'transcription': '...', ...}, ...]
            text_parts = []
            if isinstance(ocr_result, list):
                for item in ocr_result:
                    if isinstance(item, dict) and 'transcription' in item:
                        text_parts.append(item['transcription'])

            text = ' '.join(text_parts).strip()
            texts.append(text)

            if verbose:
                print(f"  [{idx+1}/{len(images)}] -> OCR: '{text}'")

        return texts

    def recognize_single(self, image: np.ndarray) -> str:
        """
        单张图像OCR识别

        Args:
            image: 单张裁剪的标签图像

        Returns:
            识别的文本
        """
        texts = self.recognize_batch([image], verbose=False)
        return texts[0] if texts else ''
