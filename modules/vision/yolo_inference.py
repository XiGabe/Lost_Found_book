"""
YOLO检测模块 - 从书架图片中检测书标标签
"""
import os
from typing import List, Dict, Tuple
from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO


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


class YOLODetector:
    """YOLO检测器 - 检测书架上的call_label标签"""

    def __init__(
        self,
        model_path: str = None,
        fallback_path: str = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        img_size: int = 640,
        device: int = 0,
        target_class_id: int = 1  # call_label类别
    ):
        """
        初始化YOLO检测器

        Args:
            model_path: YOLO模型路径（.engine或.pt文件）
            fallback_path: 备用模型路径
            conf_threshold: 置信度阈值
            iou_threshold: IOU阈值
            img_size: 输入图像尺寸
            device: GPU设备ID
            target_class_id: 目标检测类别ID（1=call_label）
        """
        self.model_path = model_path
        self.fallback_path = fallback_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self.device = device
        self.target_class_id = target_class_id

        # 加载模型
        self._load_model()

    def _load_model(self):
        """加载YOLO模型"""
        yolo_path = self.model_path
        if yolo_path is None or not os.path.exists(yolo_path):
            yolo_path = self.fallback_path
            if yolo_path is None or not os.path.exists(yolo_path):
                raise FileNotFoundError(f"YOLO 模型不存在: {yolo_path}")

        print(f"[加载 YOLO] {yolo_path}")
        self.model = YOLO(yolo_path, task='segment')

    def detect_labels(self, image: np.ndarray) -> List[Dict]:
        """
        使用YOLO检测所有call_label标签

        Args:
            image: 输入图像（BGR格式）

        Returns:
            检测框列表，每个元素包含 {'bbox': (x1,y1,x2,y2), 'conf': float}
            bbox已按x1坐标从左到右排序
        """
        results = self.model.predict(
            source=image,
            imgsz=self.img_size,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
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
                if cls_id == self.target_class_id:
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

    def crop_labels(
        self,
        image: np.ndarray,
        detections: List[Dict]
    ) -> List[np.ndarray]:
        """
        根据检测框裁剪图像区域

        Args:
            image: 原始图像
            detections: 检测结果列表

        Returns:
            裁剪后的图像列表
        """
        crops = []
        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = bbox
            crop_img = image[y1:y2, x1:x2]
            crops.append(crop_img)
        return crops
