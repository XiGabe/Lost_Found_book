# 模型文件说明

本项目使用的所有深度学习模型文件存放在此目录。

## 目录结构

```
models/
├── yolo/                    # YOLO 目标检测模型
│   ├── best.engine         # TensorRT Engine (推荐，最快)
│   ├── best.onnx           # ONNX 模型
│   └── best.pt             # PyTorch 模型
│
└── ocr/                     # OpenOCR 文字识别模型
    ├── openocr_rec_model.trt.engine      # 识别 TensorRT Engine (推荐)
    ├── openocr_det_model.trt.engine      # 检测 TensorRT Engine (推荐)
    ├── openocr_rec_model.onnx            # 识别 ONNX 模型
    ├── openocr_det_model.onnx            # 检测 ONNX 模型
    ├── openocr_repsvtr_ch.pth            # 识别 PyTorch 模型 (mobile)
    └── openocr_det_repvit_ch.pth         # 检测 PyTorch 模型
```

## 模型说明

### YOLO 模型
- **用途**: 检测书籍上的 call_label 标签
- **输入**: 图像 (建议 640x640)
- **输出**: 标签位置和类别
- **推荐**: `best.engine` (TensorRT 加速)

### OCR 模型
- **用途**: 识别标签上的文字内容
- **检测模型**: 定位文字区域
- **识别模型**: 识别具体文字
- **推荐**: `*_trt.engine` (TensorRT 加速)

## 性能对比

| 后端 | 速度 | 显存占用 | 备注 |
|------|------|----------|------|
| TensorRT Engine | ⭐⭐⭐⭐⭐ | 低 | 推荐 |
| ONNX | ⭐⭐⭐ | 中 | 兼容性好 |
| PyTorch | ⭐⭐ | 高 | 灵活性高 |

## 模型来源

- **OpenOCR**: https://github.com/Topdu/OpenOCR
- **YOLO**: 自定义训练模型

## 更新日期

2024-12-24
