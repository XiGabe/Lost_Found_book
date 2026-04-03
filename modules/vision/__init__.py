"""
视觉模块 - YOLO检测和OCR识别
"""
from pathlib import Path

# ==================== 配置 ====================
class VisionConfig:
    """视觉模块配置"""

    # 模型目录
    MODELS_DIR = Path(__file__).parent.parent.parent / 'weights'

    # YOLO 模型路径
    YOLO_MODEL_PATH = str(MODELS_DIR / 'yolo/best.pt')
    YOLO_FALLBACK_PATH = str(MODELS_DIR / 'yolo/best.engine')

    # YOLO 检测参数
    YOLO_IMG_SIZE = 640
    YOLO_CONF_THRESHOLD = 0.25
    YOLO_IOU_THRESHOLD = 0.45
    YOLO_DEVICE = 0

    # 目标检测类别 (call_label)
    TARGET_CLASS_ID = 0
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


# 导出主要类
from .yolo_inference import YOLODetector, DetectionResult
from .ocr_wrapper import OCRWrapper

__all__ = [
    'VisionConfig',
    'YOLODetector',
    'OCRWrapper',
    'DetectionResult'
]
