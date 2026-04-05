"""
视觉处理流水线 - 供机器人控制系统调用

使用方法:
    from modules.vision_pipeline import VisionPipeline

    vision = VisionPipeline()
    result = vision.check(image)  # image 可以是 numpy array 或 bytes
"""

from modules.logic.e2e_test import E2EPipeline, E2ETestResult


class VisionPipeline:
    """
    简化的视觉处理接口

    封装 E2EPipeline，提供简洁的 API 给机器人控制系统调用
    """

    def __init__(self, weights_dir: str = 'weights'):
        """
        初始化视觉处理流水线

        Args:
            weights_dir: 模型权重目录路径
        """
        self.pipeline = E2EPipeline()
        self._ready = True

    def process(self, image) -> E2ETestResult:
        """
        处理图片

        Args:
            image: numpy array (BGR格式 from OpenCV) 或 bytes (JPEG/PNG)

        Returns:
            E2ETestResult 对象，包含完整处理结果
        """
        if isinstance(image, bytes):
            return self.pipeline.process_image_bytes(image)
        else:
            # 假设是 numpy array，转换为 bytes
            import cv2
            _, encoded = cv2.imencode('.jpg', image)
            return self.pipeline.process_image_bytes(encoded.tobytes())

    def check(self, image) -> dict:
        """
        快速检查接口，返回简单字典

        推荐用于机器人控制系统的集成

        Args:
            image: numpy array (BGR格式 from OpenCV) 或 bytes

        Returns:
            dict:
                - has_issues: bool, 是否存在问题
                - num_out_of_order: int, 错位书本数量
                - num_duplicates: int, 重复书本数量
                - out_of_order_pairs: list, 错位对详情
        """
        result = self.process(image)
        return {
            'has_issues': result.num_out_of_order > 0 or result.num_duplicates > 0,
            'num_out_of_order': result.num_out_of_order,
            'num_duplicates': result.num_duplicates,
            'num_in_order': result.num_in_order,
            'ocr_texts': result.ocr_texts,
            'processing_time': result.processing_time,
            'out_of_order_pairs': [
                {
                    'idx_a': p.idx_a,
                    'idx_b': p.idx_b,
                    'text_a': p.text_a,
                    'text_b': p.text_b,
                    'confidence': p.confidence
                }
                for p in result.out_of_order_pairs
            ],
            'duplicate_pairs': [
                {
                    'idx_a': p.idx_a,
                    'idx_b': p.idx_b,
                    'text_a': p.text_a,
                    'text_b': p.text_b,
                    'confidence': p.confidence
                }
                for p in result.pair_results if p.label == 1  # Duplicate
            ]
        }


__all__ = ['VisionPipeline', 'E2ETestResult']
