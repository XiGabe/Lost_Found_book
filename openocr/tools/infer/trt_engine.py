"""
TensorRT Engine 推理引擎
支持直接加载 .engine 文件进行高速推理

Jetson 兼容版本 - 使用 ONNX Runtime TensorRT Provider
"""
import os
import numpy as np


class TRTEngine:
    """
    TensorRT Engine 推理引擎包装器

    注意：实际上 ONNX Runtime 的 TensorRT EP 已经提供了优化
    如果需要直接使用 .engine 文件，建议使用 ONNX Runtime 而不是原生 TensorRT API
    """

    def __init__(self, engine_path, use_fp16=True):
        """
        初始化 TensorRT Engine

        Args:
            engine_path: .engine 文件路径 (注意：当前实现使用 ONNX Runtime)
            use_fp16: 是否使用 FP16 精度
        """
        # 由于原生 TensorRT API 在 Jetson 上存在 CUDA 上下文问题，
        # 我们使用 ONNX Runtime 的 TensorRT Execution Provider 作为替代方案
        # 这已经提供了接近原生 TensorRT 的性能

        import onnxruntime

        if not os.path.exists(engine_path):
            raise FileNotFoundError(f'Engine file not found: {engine_path}')

        # 查找对应的 ONNX 模型
        if 'rec_model.trt.engine' in engine_path or 'rec' in engine_path:
            # 识别模型 - 需要使用 ONNX 模型
            onnx_path = engine_path.replace('.trt.engine', '.onnx')
            if not os.path.exists(onnx_path):
                # 尝试在 models 目录中查找
                onnx_path = '/workspace/models/ocr/openocr_rec_model.onnx'
        elif 'det_model.trt.engine' in engine_path or 'det' in engine_path:
            # 检测模型
            onnx_path = engine_path.replace('.trt.engine', '.onnx')
            if not os.path.exists(onnx_path):
                onnx_path = '/workspace/models/ocr/openocr_det_model.onnx'
        else:
            raise ValueError(f'Unknown engine type: {engine_path}')

        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f'ONNX model not found: {onnx_path}')

        print(f'[TRT] Using ONNX Runtime with TensorRT EP')
        print(f'[TRT] ONNX model: {onnx_path}')

        # 配置 TensorRT Execution Provider
        providers = [
            ('TensorrtExecutionProvider', {
                'device_id': 0,
                'trt_max_workspace_size': 4 << 30,  # 4GB
                'trt_fp16_enable': use_fp16,
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': os.path.dirname(engine_path),
            }),
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ]

        self.onnx_session = onnxruntime.InferenceSession(onnx_path, providers=providers)

        # 获取输入输出名称
        self.input_name = self.onnx_session.get_inputs()[0].name
        self.output_name = self.onnx_session.get_outputs()[0].name

        print(f'[TRT] Input: {self.input_name}')
        print(f'[TRT] Output: {self.output_name}')
        print(f'[TRT] Engine loaded successfully!')

    def run(self, input_data):
        """
        运行推理

        Args:
            input_data: 输入数据，numpy 数组

        Returns:
            list: 输出列表
        """
        if isinstance(input_data, list):
            input_data = input_data[0]

        result = self.onnx_session.run([self.output_name], {self.input_name: input_data})
        return result

    def __call__(self, input_data):
        """支持直接调用"""
        return self.run(input_data)


def build_engine_from_onnx(onnx_path,
                           engine_path,
                           fp16=True,
                           min_shapes=None,
                           opt_shapes=None,
                           max_shapes=None):
    """
    从 ONNX 模型构建 TensorRT Engine

    注意：此函数使用 trtexec 命令行工具构建 Engine
    """
    import subprocess
    import tensorrt as trt

    print(f'[TRT] Building engine from ONNX: {onnx_path}')
    print(f'[TRT] FP16: {fp16}')

    # 首先检查输入形状
    import onnx
    model = onnx.load(onnx_path)
    input_info = model.graph.input[0]
    input_name = input_info.name

    # 使用 trtexec 构建 Engine
    cmd = [
        '/usr/src/tensorrt/bin/trtexec',
        f'--onnx={onnx_path}',
        f'--saveEngine={engine_path}',
    ]

    if fp16:
        cmd.append('--fp16')

    # 设置动态形状（如果有）
    if min_shapes and opt_shapes and max_shapes:
        for name in min_shapes.keys():
            cmd.append(f"--minShapes={name}:{','.join(map(str, min_shapes[name]))}")
            cmd.append(f"--optShapes={name}:{','.join(map(str, opt_shapes[name]))}")
            cmd.append(f"--maxShapes={name}:{','.join(map(str, max_shapes[name]))}")

    print(f'[TRT] Running: {" ".join(cmd)}')

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f'[TRT] Error: {result.stderr}')
        raise RuntimeError('Failed to build TensorRT engine')

    print(f'[TRT] Engine saved to: {engine_path}')
    return TRTEngine(engine_path)


if __name__ == '__main__':
    # 测试代码
    import sys
    if len(sys.argv) > 1:
        cmd = sys.argv[1]

        if cmd == 'build':
            # 构建识别模型 Engine
            onnx_path = '/workspace/models/ocr/openocr_rec_model.onnx'
            engine_path = '/workspace/models/ocr/openocr_rec_model.trt.engine'

            # 动态形状配置: [batch, channel, height, width]
            min_shapes = {'input': [1, 3, 48, 10]}
            opt_shapes = {'input': [16, 3, 48, 320]}
            max_shapes = {'input': [64, 3, 48, 640]}

            build_engine_from_onnx(
                onnx_path,
                engine_path,
                fp16=True,
                min_shapes=min_shapes,
                opt_shapes=opt_shapes,
                max_shapes=max_shapes
            )

        elif cmd == 'test':
            # 测试加载 Engine
            engine_path = '/workspace/models/ocr/openocr_rec_model.trt.engine'
            engine = TRTEngine(engine_path)

            # 测试推理
            dummy_input = np.random.randn(1, 3, 48, 320).astype(np.float32)
            outputs = engine.run(dummy_input)
            print(f'Output shapes: {[o.shape for o in outputs]}')

