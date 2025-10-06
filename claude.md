# Lost Book Robot - 智能图书馆书籍错位检测系统

## 项目概述

Lost Book Robot是一个用于自动检测图书馆书籍错位的智能机器人系统。该系统通过计算机视觉和OCR技术识别书脊上的Call Number，结合深度学习模型验证书籍是否按照LC（Library of Congress）分类规则正确排列。

## 系统架构 (重构版)

### 整体架构
```
图像采集 → 书脊分割 → 单书OCR → 特征提取 → 神经网络验证 → 结果输出
```

### 核心模块分工

#### 🔧 硬件层面 (机器人团队负责)
- **移动平台控制**：机器人沿书架移动和定位
- **图像采集系统**：相机参数控制、图像质量优化
- **照明系统**：确保拍摄环境的一致性
- **硬件集成**：将OCR系统整合到机器人平台

#### 🧠 算法层面 (当前专注)
1. **书脊分割模块**：从整张书架图像中精确分割出每个书脊
2. **OCR处理模块**：对单个书脊图像进行文字识别
3. **特征提取模块**：提取transcription、空间坐标、置信度等特征
4. **神经网络验证模块**：判断书籍位置正确性

## 当前开发策略

### 🎯 开发优先级

#### **阶段1：端到端图像处理 pipeline** (当前重点)
```
目标：实现从原始图像到分割后单本书脊的完整流程

核心任务：
1. 书脊分割算法 (OpenCV)
   - 边缘检测与轮廓提取
   - 透视变换与图像矫正
   - 书脊边界精确识别

2. 单书脊OCR优化
   - 图像预处理 (去噪、增强)
   - OCR引擎参数调优
   - 连续拍摄的数据关联

输出：标准化的单本书脊数据
```

#### **阶段2：神经网络训练系统** (后续专注)
```
目标：训练智能位置验证模型

核心任务：
1. 训练数据构建
   - 图书馆管理系统数据对接
   - 特征工程与标准化
   - 正负样本采集

2. 模型开发
   - 网络架构设计
   - 训练pipeline搭建
   - 模型评估与优化

输出：高精度的位置验证模型
```

## 技术实现细节

### 📋 数据流设计

#### 输入数据格式
```json
{
  "image_id": "shelf_001_001",
  "timestamp": "2025-10-06T16:30:00Z",
  "position_data": {
    "robot_position": {"x": 1.5, "y": 3.2, "z": 0.0},
    "camera_angle": 0,
    "shelf_id": "A3-2"
  },
  "image_data": "base64_encoded_image"
}
```

#### 书脊分割输出
```json
{
  "spine_id": "spine_001",
  "original_image": "shelf_001_001.png",
  "spine_image": "spine_001.png",
  "bounding_box": {
    "x": 120, "y": 50, "width": 80, "height": 300
  },
  "confidence": 0.95,
  "position_in_shelf": 3
}
```

#### OCR处理输出
```json
{
  "spine_id": "spine_001",
  "transcriptions": [
    {"text": "OLIN", "confidence": 0.998, "position": [23, 7]},
    {"text": "BV", "confidence": 0.984, "position": [26, 48]},
    {"text": "4208", "confidence": 0.998, "position": [30, 86]},
    {"text": ".G7", "confidence": 0.994, "position": [32, 126]},
    {"text": "T44X", "confidence": 0.907, "position": [36, 168]},
    {"text": "1995", "confidence": 0.999, "position": [38, 204]}
  ],
  "reconstructed_call_number": "OLIN BV 4208 .G7 T44X 1995",
  "spatial_center": [82.0, 127.0],
  "overall_confidence": 0.980
}
```

### 🛠️ 核心算法设计

#### 书脊分割算法
```python
class SpineSegmentator:
    def __init__(self):
        self.edge_detector = CannyEdgeDetector()
        self.contour_finder = ContourFinder()
        self.perspective_transformer = PerspectiveTransformer()

    def extract_spines(self, shelf_image):
        # 1. 预处理
        enhanced = self.preprocess_image(shelf_image)

        # 2. 边缘检测
        edges = self.edge_detector.detect(enhanced)

        # 3. 轮廓提取
        contours = self.contour_finder.find(edges)

        # 4. 书脊识别与分割
        spines = self.identify_spine_contours(contours)

        # 5. 透视矫正
        corrected_spines = []
        for spine in spines:
            corrected = self.perspective_transformer.correct(spine)
            corrected_spines.append(corrected)

        return corrected_spines
```

#### 连续图像关联算法
```python
class SequenceProcessor:
    def __init__(self):
        self.feature_matcher = FeatureMatcher()
        self.position_tracker = PositionTracker()

    def process_sequence(self, image_sequence):
        tracked_books = []

        for i, image in enumerate(image_sequence):
            # 1. 分割当前图像的书脊
            current_spines = self.segment_spines(image)

            # 2. 与前一帧进行特征匹配
            if i > 0:
                matched_pairs = self.feature_matcher.match(
                    previous_spines, current_spines
                )

                # 3. 更新位置追踪
                tracked_books = self.position_tracker.update(
                    tracked_books, matched_pairs
                )

            previous_spines = current_spines

        return tracked_books
```

## 数据管理策略

### 📊 训练数据收集

#### 数据来源
1. **图书馆管理系统**：获取准确的书籍位置信息
2. **机器人扫描数据**：收集真实场景的图像和OCR结果
3. **人工标注数据**：专家验证的错位/正确样本

#### 数据格式标准
```json
{
  "sample_id": "train_001",
  "library_data": {
    "call_number": "BV 4208 .G7 T44X 1995",
    "correct_position": {"aisle": "A", "shelf": 3, "position": 5},
    "title": "Book Title",
    "author": "Author Name"
  },
  "scan_data": {
    "image_path": "scans/A3/shelf3/001.png",
    "spine_image": "spines/spine_001.png",
    "ocr_result": {...},
    "extracted_features": {
      "transcription": "OLIN BV 4208 .G7 T44X 1995",
      "spatial_center": [82.0, 127.0],
      "confidence": 0.980,
      "position_in_shelf": 5
    }
  },
  "label": {
    "is_correct": true,
    "actual_position": 5,
    "should_be_position": 5,
    "confidence_score": 0.95
  }
}
```

### 🔄 模型训练流程

#### 特征工程
```python
def extract_training_features(ocr_result, library_data):
    features = {
        # 空间特征
        "relative_x": ocr_result["spatial_center"][0] / image_width,
        "relative_y": ocr_result["spatial_center"][1] / image_height,
        "position_ratio": ocr_result["position_in_shelf"] / total_books,

        # OCR置信度特征
        "avg_confidence": ocr_result["overall_confidence"],
        "min_confidence": min([t["confidence"] for t in ocr_result["transcriptions"]]),
        "confidence_variance": calculate_confidence_variance(ocr_result),

        # 文本特征
        "call_number_length": len(ocr_result["reconstructed_call_number"]),
        "text_embedding": get_text_embedding(ocr_result["reconstructed_call_number"]),

        # 位置匹配特征
        "position_match": 1 if ocr_result["position_in_shelf"] == library_data["correct_position"]["position"] else 0,
        "position_offset": abs(ocr_result["position_in_shelf"] - library_data["correct_position"]["position"])
    }

    return features
```

## 开发工具和框架

### 💻 核心技术栈
- **图像处理**: OpenCV, Scikit-image
- **OCR引擎**: OpenOCR
- **深度学习**: PyTorch, torchvision
- **数据处理**: NumPy, Pandas
- **版本控制**: Git

### 🧪 测试框架
```python
class TestFramework:
    def __init__(self):
        self.image_tests = ImageTestSuite()
        self.ocr_tests = OCRTestSuite()
        self.model_tests = ModelTestSuite()

    def run_comprehensive_tests(self):
        # 1. 书脊分割准确性测试
        segmentation_accuracy = self.image_tests.test_segmentation()

        # 2. OCR识别准确性测试
        ocr_accuracy = self.ocr_tests.test_recognition()

        # 3. 连续处理稳定性测试
        sequence_stability = self.image_tests.test_sequence_processing()

        # 4. 端到端性能测试
        e2e_performance = self.test_end_to_end()

        return {
            "segmentation_accuracy": segmentation_accuracy,
            "ocr_accuracy": ocr_accuracy,
            "sequence_stability": sequence_stability,
            "e2e_performance": e2e_performance
        }
```

## 项目里程碑

### 🎯 Phase 1: 图像处理 Pipeline (当前)
- [x] 基础OCR引擎集成
- [x] 批量图像处理框架
- [ ] 书脊分割算法实现
- [ ] 连续图像数据关联
- [ ] 性能优化和测试

### 🧠 Phase 2: 神经网络开发 (后续)
- [ ] 训练数据收集和标注
- [ ] 特征提取系统
- [ ] 神经网络模型设计
- [ ] 模型训练和验证
- [ ] 模型部署和集成

### 🤖 Phase 3: 机器人集成 (机器人团队)
- [ ] 硬件接口开发
- [ ] 实时处理优化
- [ ] 现场测试和调试
- [ ] 系统集成和交付

## 性能指标

### 📈 图像处理指标
- **书脊分割准确率**: > 95%
- **分割处理速度**: < 0.5秒/图像
- **OCR识别准确率**: > 90%
- **连续追踪稳定性**: > 98%

### 🎯 神经网络指标
- **位置判断准确率**: > 95%
- **错位检测召回率**: > 90%
- **假阳性率**: < 5%
- **推理速度**: < 10ms/书籍

## 风险评估与应对

### ⚠️ 技术风险
1. **书脊分割难度**
   - 风险：书籍密集排列、光照不均
   - 应对：多算法融合、自适应预处理

2. **OCR识别错误**
   - 风险：字体模糊、特殊字符
   - 应对：多模型集成、后处理纠错

3. **数据质量**
   - 风险：训练数据不足
   - 应对：数据增强、合成数据生成

## 📁 项目文件结构

```
OCR/
├── claude.md                              # 项目文档（本文件）
├── batch_ocr_processor.py                 # 批量OCR处理器
├── dataset5_integration.py                # 数据聚合工具
├── data/
│   └── dataset1/                          # 测试数据集
├── ocr_results/                           # OCR输出结果
├── e2e_results/                           # 端到端测试结果
├── modules/                               # 核心模块
│   ├── spine_segmentation/               # 书脊分割模块
│   ├── ocr_processing/                   # OCR处理模块
│   ├── feature_extraction/               # 特征提取模块
│   └── neural_network/                   # 神经网络模块
├── tests/                                # 测试代码
├── utils/                                # 工具函数
└── docs/                                 # 文档
```

## 📞 联系信息

**项目负责人**: Hongxi Chen
**技术方向**: 计算机视觉 + 深度学习
**当前专注**: 书脊分割与OCR优化
**协作方式**: 提供标准化数据接口，机器人团队负责硬件集成

---

**文档版本**: 2.0
**创建日期**: 2025-10-06
**最后更新**: 2025-10-06
**维护者**: Hongxi Chen

## 🎯 当前状态
- **🔄 进行中**: 书脊分割算法开发
- **✅ 已完成**: 基础OCR处理框架
- **⏳ 待开始**: 神经网络训练系统