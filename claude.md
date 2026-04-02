# Lost Book Robot - 智能图书馆书籍错位检测系统 (V2.0)

## 📖 项目概述

Lost Book Robot 是一个用于自动检测图书馆书籍错位的智能机器人系统。本项目采用**端到端（End-to-End）**的视觉处理方案，利用 Jetson Orin Nano 的边缘计算能力，通过"单帧全量计算"策略，实时判断书架上书籍的排序是否符合美国国会图书馆分类法（LC Classification）。

**核心技术革新 (V2.0)**：

1. **视觉前端**：使用 **YOLOv11 目标检测** 检测书标标签，配合OpenOCR进行文字识别。
2. **排序核心**：采用 **Bi-LSTM 孪生网络 (Siamese Network)** 直接比较两个OCR字符串，具备极强的OCR容错能力。
3. **逻辑推断**：滑动窗口遍历相邻书籍对，自动检测排序错误。

---

## 🏗 系统架构

### 整体数据流

```mermaid
graph TD
    Camera[相机采集] --> YOLO[YOLOv11 目标检测]

    subgraph "视觉预处理"
    YOLO -->|Class: Call Label| CropLabel[裁剪标签图]
    end

    subgraph "文字识别"
    CropLabel --> OpenOCR[OpenOCR TensorRT]
    end

    subgraph "逻辑验证 (Jetson Orin)"
    OpenOCR -->|OCR String List| SequentialPair[生成连续对]
    SequentialPair -->|Pair (A, B)| BiLSTM[Bi-LSTM 比较器]

    BiLSTM -->|Out_of_Order| Alarm[错位报警]
    BiLSTM -->|Duplicate| Warning[重复警告]
    BiLSTM -->|In_Order| Pass[通过]
    end

    subgraph "结果输出"
    BiLSTM --> JSON[JSON报告]
    BiLSTM --> Visualize[可视化标注]
    end
```

---

## 🧠 核心算法详解

### 1. 视觉感知：YOLOv11 目标检测

**目标**：从书架图片中检测索书号标签区域。

* **模型**：YOLOv11 (Object Detection)
* **检测类别**：
  1. `call_label`：索书号标签贴纸

* **部署优化**：
  * TensorRT引擎（`.engine`格式）- 性能最优
  * ONNX格式（`.onnx`）- 跨平台兼容
  * PyTorch格式（`.pt`）- 作为fallback

### 2. 文字识别：OpenOCR

**目标**：将图像像素转化为文本字符串。

* **引擎**：OpenOCR (TensorRT加速)
* **性能**：
  * 准确率 > 95%
  * 速度 ~0.05s/图
  * 支持批量处理

* **策略**：容忍OCR噪声（如`1`变`I`，`B`变`8`），交由下游神经网络处理。

### 3. 排序验证：Siamese Bi-LSTM

**目标**：判断两个OCR字符串的排序关系。

* **架构**：Siamese Bi-LSTM (孪生双向长短期记忆网络)
* **输入**：字符级Tokenizer (Char-level)
* **Backbone**：共享权重的2层Bi-LSTM
* **Head**：拼接特征向量，通过MLP分类

* **输出类别**：
  * `0: In_Order` (顺序正确：A < B)
  * `1: Duplicate` (重复：A = B)
  * `2: Out_of_Order` (错位：A > B) —— **核心报警触发条件**

* **优势**：
  * 自动学习LC分类法的层级权重（字母 > 数字 > 小数）
  * 自动忽略无效前缀
  * 极强的鲁棒性，能处理模糊、缺损字符

---

## 📊 数据策略

### 1. YOLO 训练数据 (真实数据)

* **来源**：真实拍摄的书架照片
* **标注方式**：使用CVAT或LabelImg
* **标注类别**：`call_label` (索书号标签)
* **规模**：约100张图像

### 2. Bi-LSTM 训练数据 (合成数据)

* **来源**：**完全合成 (Synthetic Generation)**
* **生成脚本**：`modules/logic/data_gen.py`
  * **规则生成**：批量生成符合LC规则的正确排序对子
  * **噪声注入**：字符替换、前缀干扰、格式干扰
  * **负样本构造**：翻转正确对子生成Out_of_Order样本

* **规模**：已生成300,000+对样本

---

## 🗓 开发路线图 (Roadmap)

### Phase 1: 视觉前端构建 ✅

* ✅ **数据采集**：拍摄书架照片
* ✅ **数据标注**：完成标签标注
* ✅ **YOLO训练**：训练YOLOv8模型并部署
* ✅ **OCR集成**：集成OpenOCR引擎

### Phase 2: 神经网络核心开发 ✅

* ✅ **数据生成器**：编写`data_gen.py`，生成OCR噪声样本
* ✅ **模型搭建**：实现Bi-LSTM Siamese Network
* ✅ **模型训练**：在合成数据集上训练，验证准确率95.5%
* ✅ **端到端测试**：将OCR输出接入网络，测试真实图片

### Phase 3: 系统集成 ✅

* ✅ **滑动窗口逻辑**：实现单帧内的连续对比较
* ✅ **结果输出**：生成JSON报告和可视化标注
* ✅ **批量处理**：支持批量图片处理
* ⏳ **数据库接口**：待实现
* ⏳ **Jetson部署**：待优化

---

## 📁 当前文件结构

```
LostBookRobot/
├── data/
│   ├── raw_images/              # 原始书架照片 (36K+ 张, 447MB)
│   ├── yolo_dataset/            # YOLO标注数据 (预留)
│   └── synthetic_pairs/         # LSTM训练数据
│       └── lcc_training_data.csv # 300K+ 对样本
│
├── modules/                     # 核心模块目录
│   ├── __init__.py
│   ├── external/                # 外部依赖库
│   │   └── openocr/             # OpenOCR库 (TensorRT加速)
│   ├── vision/                  # 视觉模块
│   │   ├── yolo_inference.py    # YOLO检测器
│   │   ├── ocr_wrapper.py       # OCR封装
│   │   └── __init__.py
│   └── logic/                   # 逻辑模块
│       ├── comparator.py        # Bi-LSTM网络
│       ├── e2e_test.py          # 端到端测试 ⭐
│       ├── train_lstm.py        # 训练脚本
│       ├── inference.py         # 推理接口
│       ├── data_gen.py          # 数据生成
│       ├── evaluate.py          # 模型评估
│       ├── dataset.py           # 数据集类
│       ├── tokenizer.py         # 字符级Tokenizer
│       ├── utils.py             # 工具函数
│       └── benchmark_batch.py   # 批量测试
│
├── weights/                     # 模型权重 (47MB)
│   ├── yolo/
│   │   ├── best.engine          # YOLO TensorRT (使用中)
│   │   └── best.pt              # YOLO PyTorch (备用)
│   ├── ocr/
│   │   ├── openocr_det_model.trt.engine    # OCR检测 TensorRT
│   │   └── openocr_rec_model.trt.engine    # OCR识别 TensorRT
│   └── comparator.pth           # Bi-LSTM模型 (10.8MB)
│
├── output/
│   └── e2e_results/             # 端到端测试结果
│       ├── json/                # JSON报告
│       └── visualizations/      # 可视化标注图片
│
├── .gitignore                   # Git忽略配置
├── claude.md                    # 项目文档 (本文件)
├── requirements.txt             # Python依赖
├── run_yolo.txt                 # Docker开发脚本参考
└── README.md                    # 项目说明
```

---

## 🚀 使用方法

### 端到端测试 (推荐)

```bash
# 单张图片测试
python -m modules.logic.e2e_test data/raw_images/test_image.jpg

# 批量测试
python -m modules.logic.e2e_test --batch data/raw_images/*.jpg

# 查看结果
JSON: output/e2e_results/json/{image_name}.json
可视化: output/e2e_results/visualizations/{image_name}.jpg
```

### 模型训练

```bash
# 训练Bi-LSTM模型
python -m modules.logic.train_lstm \
    --data data/synthetic_pairs/lcc_training_data.csv \
    --epochs 30 \
    --batch-size 64

# 评估模型
python -m modules.logic.evaluate \
    --checkpoint weights/comparator.pth \
    --data data/synthetic_pairs/lcc_training_data.csv
```

### 单对推理

```bash
# 比较两个索书号
python -m modules.logic.inference \
    --text-a "QA76.5 .C64" \
    --text-b "QA76.6 .A12" \
    --checkpoint weights/comparator.pth
```

---

## 📈 性能指标

### 测试结果（404张图片）

* **成功率**：100% (404/404)
* **平均速度**：0.92秒/张
* **吞吐量**：1.09张/秒
* **总耗时**：6分11秒

### 模型性能

* **YOLO检测**：TensorRT加速，实时检测
* **OCR识别**：准确率 > 95%
* **LSTM比较**：准确率 95.5%，推理毫秒级

---

## 🔧 配置说明

### 模型更新

**训练新LSTM模型后**：
```bash
# 直接替换即可
cp modules/outputs/run_xxx/best_comparator.pth weights/comparator.pth
```

**训练新YOLO模型后**：
需要导出3种格式（`.pt`, `.onnx`, `.engine`），参考Ultralytics文档。

### Docker开发环境

项目提供了基于Docker的开发环境配置（用于Jetson平台），参考 `run_yolo.txt`：

```bash
# 使用预配置的Ultralytics Docker镜像
sudo docker run -it \
    --ipc=host \
    --runtime=nvidia \
    --gpus all \
    -v /home/$USER/Documents/Lost_Found_book:/workspace \
    -e DISPLAY=$DISPLAY \
    ultralytics/ultralytics:latest-jetson-jetpack6
```
```bash
# 如何进入启动和进入docker
sudo docker start yolo_dev
sudo docker exec -it yolo_dev /bin/bash
```
---

## 📝 待办事项

- [ ] 实现数据库接口（Gap-Filling逻辑）
- [ ] Jetson Orin Nano部署优化
- [ ] 实时视频流处理
- [ ] Web界面开发
- [ ] 训练新的yolo模型
- [ ] 重构生成数据的逻辑/训练新的LSTM模型

---

**文档维护者**: Hongxi Chen 
**最后更新**: 2025-12-25
**版本**: V2.0 (模块化重构)
**状态**: Phase 2完成，系统已可用