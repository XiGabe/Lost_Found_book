# 批量OCR处理器使用指南

## 📖 概述

这个批量OCR处理器基于OpenOCR，可以批量处理整个文件夹的图像文件，并输出统一格式的OCR结果。

## 🚀 快速开始

### 1. 基本使用

```python
from batch_ocr_processor import BatchOCRProcessor

# 创建处理器
processor = BatchOCRProcessor(confidence_threshold=0.0)

# 批量处理
summary = processor.process_directory("./images", "./results")

# 打印结果
processor.print_summary(summary)
```

### 2. 命令行使用

```bash
# 基本用法
python batch_ocr_processor.py --input ./images --output ./results

# 带置信度阈值
python batch_ocr_processor.py --input ./images --output ./results --confidence 0.5
```

### 3. 快速演示

```bash
# 运行演示脚本
python simple_ocr_demo.py
```

## 📋 输出格式

### JSON格式 (详细版)
```json
{
  "metadata": {
    "total_images": 10,
    "processing_time": "2025-11-04T15:30:00",
    "confidence_threshold": 0.0,
    "processor_version": "1.0"
  },
  "results": [
    {
      "image_path": "./images/1.png",
      "image_name": "1.png",
      "ocr_data": [...],  // 原始OpenOCR输出
      "transcriptions": ["OLIN", "BV", "4208", ".G7", "T44X", "1995"],
      "reconstructed_text": "OLIN BV 4208 .G7 T44X 1995",
      "confidence": 0.980,
      "processing_time": 0.15,
      "timestamp": "2025-11-04T15:30:01",
      "status": "success"
    }
  ]
}
```

### CSV格式 (报告版)
```csv
image_name,transcription,confidence,processing_time,status
1.png,OLIN|BV|4208|.G7|T44X|1995,0.980,0.150,success
2.png,OLIN|BV|4208|.G7|W32|2002,0.996,0.120,success
```

## ⚙️ 配置选项

### 置信度阈值
```python
# 只保留置信度 >= 0.8 的结果
processor = BatchOCRProcessor(confidence_threshold=0.8)
```

### 支持的图像格式
- JPG/JPEG
- PNG
- BMP
- TIFF/TIF

## 📊 结果数据结构

### OCRResult
```python
@dataclass
class OCRResult:
    image_path: str          # 完整路径
    image_name: str          # 文件名
    ocr_data: List[Dict]     # 原始OCR数据
    transcriptions: List[str] # 提取的文字列表
    reconstructed_text: str   # 重构的完整文本
    confidence: float         # 平均置信度
    processing_time: float    # 处理时间(秒)
    timestamp: str           # 处理时间戳
    status: str              # "success" 或 "error"
    error_message: Optional[str]  # 错误信息
```

### BatchOCRSummary
```python
@dataclass
class BatchOCRSummary:
    total_images: int           # 总图像数
    successful: int             # 成功处理数
    failed: int                 # 失败数
    total_time: float           # 总处理时间
    average_time_per_image: float # 平均处理时间
    output_file: str            # 输出文件路径
```

## 🔧 高级用法

### 自定义处理
```python
# 处理单张图像
result = processor.process_single_image("./image.jpg")

if result.status == "success":
    print(f"识别结果: {result.reconstructed_text}")
    print(f"置信度: {result.confidence}")
else:
    print(f"处理失败: {result.error_message}")
```

### 过滤结果
```python
# 处理完成后过滤
successful_results = [r for r in processor.results if r.status == "success"]
high_confidence_results = [r for r in successful_results if r.confidence > 0.9]
```

## 📁 输出文件说明

批量处理会生成多个输出文件：

1. **`ocr_results_YYYYMMDD_HHMMSS.json`** - 完整的详细结果
2. **`ocr_simple_YYYYMMDD_HHMMSS.json`** - 简化版结果（只包含关键信息）
3. **`ocr_report_YYYYMMDD_HHMMSS.csv`** - CSV格式的报告

## ⚠️ 注意事项

### 性能优化
- 大量图像处理建议分批进行
- GPU环境可以显著提升处理速度
- 考虑图像分辨率对处理速度的影响

### 错误处理
- 损坏的图像文件会被跳过，不会中断整个处理过程
- 所有错误都会记录在日志和结果文件中
- 建议检查失败图像的格式和完整性

### 内存管理
- 处理大量图像时注意内存使用
- 可以通过分批处理来控制内存占用

## 🔍 故障排除

### 常见问题

1. **OpenOCR导入失败**
   ```bash
   pip install openocr
   ```

2. **处理速度慢**
   - 检查图像分辨率
   - 考虑使用GPU加速
   - 减少同时处理的图像数量

3. **识别准确率低**
   - 检查图像质量
   - 调整置信度阈值
   - 考虑图像预处理

### 日志查看
处理器使用Python logging模块，可以通过以下方式查看详细日志：

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📞 技术支持

如有问题，请检查：
1. OpenOCR是否正确安装
2. 图像文件格式是否支持
3. 输入路径是否正确
4. 置信度阈值设置是否合理

---

**开发者**: Hongxi Chen
**版本**: 1.0
**更新日期**: 2025-11-04