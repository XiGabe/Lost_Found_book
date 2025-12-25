"""
TensorRT Engine 推理引擎 (PyTorch 版 - 兼容 TensorRT 10+)
包含构建工具函数，支持直接加载 .engine 文件进行高速推理
"""
import os
import tensorrt as trt
import torch
import numpy as np
import subprocess

class TRTEngine:
    """
    使用 PyTorch 进行显存管理的 TensorRT 引擎
    """

    def __init__(self, engine_path):
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f'Engine file not found: {engine_path}')

        print(f'[TRT] Loading Engine (Torch-based): {engine_path}')
        
        # 1. 初始化 Logger 和 Runtime
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)

        # 2. 反序列化 Engine
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        
        if not self.engine:
            raise RuntimeError("Failed to deserialize TensorRT engine")

        # 3. 创建执行上下文
        self.context = self.engine.create_execution_context()

        # 4. 分析输入输出张量
        self.io_tensors = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            dtype = self.engine.get_tensor_dtype(name)
            
            # 映射 TRT 类型到 Torch 类型
            torch_dtype = self._trt_to_torch_dtype(dtype)
            
            self.io_tensors.append({
                'name': name,
                'mode': mode,
                'dtype': torch_dtype,
                'index': i
            })
            
            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name

        print(f'[TRT] Engine loaded successfully!')

    def _trt_to_torch_dtype(self, trt_dtype):
        mapping = {
            trt.DataType.FLOAT: torch.float32,
            trt.DataType.HALF: torch.float16,
            trt.DataType.INT32: torch.int32,
            trt.DataType.INT8: torch.int8,
            trt.DataType.BOOL: torch.bool,
        }
        return mapping.get(trt_dtype, torch.float32)

    def __call__(self, input_data):
        """
        推理入口
        Args:
            input_data: numpy array 或 torch.Tensor
        """
        # 1. 准备输入 Tensor (确保在 GPU 上)
        if isinstance(input_data, list):
            input_data = input_data[0]
            
        if isinstance(input_data, np.ndarray):
            input_tensor = torch.from_numpy(input_data).cuda()
        elif isinstance(input_data, torch.Tensor):
            input_tensor = input_data.cuda()
        else:
            raise TypeError(f"Unsupported input type: {type(input_data)}")

        # 确保输入连续
        if not input_tensor.is_contiguous():
            input_tensor = input_tensor.contiguous()

        # 2. 设置动态输入形状
        input_shape = tuple(input_tensor.shape)
        self.context.set_input_shape(self.input_name, input_shape)
        
        # 3. 绑定输入地址
        self.context.set_tensor_address(self.input_name, input_tensor.data_ptr())

        # 4. 准备输出 Tensor
        outputs = []
        for tensor_info in self.io_tensors:
            if tensor_info['mode'] == trt.TensorIOMode.OUTPUT:
                name = tensor_info['name']
                # 获取推理后的输出形状
                out_shape = self.context.get_tensor_shape(name)
                
                # 分配输出显存
                output_tensor = torch.empty(tuple(out_shape), dtype=tensor_info['dtype'], device='cuda')
                
                # 绑定输出地址
                self.context.set_tensor_address(name, output_tensor.data_ptr())
                outputs.append(output_tensor)

        # 5. 执行推理 (异步)
        stream = torch.cuda.current_stream().cuda_stream
        self.context.execute_async_v3(stream_handle=stream)
        
        # 6. 返回结果 (转回 CPU numpy)
        return [o.cpu().numpy() for o in outputs]


def build_engine_from_onnx(onnx_path,
                           engine_path,
                           fp16=True,
                           min_shapes=None,
                           opt_shapes=None,
                           max_shapes=None):
    """
    调用 trtexec 从 ONNX 构建 TensorRT Engine
    """
    print(f'[TRT] Building engine from ONNX: {onnx_path}')
    print(f'[TRT] FP16: {fp16}')
    
    # 尝试找到 trtexec
    trtexec_path = 'trtexec'
    if os.path.exists('/usr/src/tensorrt/bin/trtexec'):
        trtexec_path = '/usr/src/tensorrt/bin/trtexec'

    cmd = [
        trtexec_path,
        f'--onnx={onnx_path}',
        f'--saveEngine={engine_path}',
    ]

    if fp16:
        cmd.append('--fp16')

    # 设置动态形状
    if min_shapes and opt_shapes and max_shapes:
        for name in min_shapes.keys():
            cmd.append(f"--minShapes={name}:{','.join(map(str, min_shapes[name]))}")
            cmd.append(f"--optShapes={name}:{','.join(map(str, opt_shapes[name]))}")
            cmd.append(f"--maxShapes={name}:{','.join(map(str, max_shapes[name]))}")

    print(f'[TRT] Running: {" ".join(cmd)}')

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f'[TRT] Error Output:\n{result.stderr}')
        raise RuntimeError('Failed to build TensorRT engine via trtexec')
    
    print('[TRT] Build success.')
    print(f'[TRT] Engine saved to: {engine_path}')
    
    # 返回 Engine 实例
    return TRTEngine(engine_path)