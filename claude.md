# Lost Book Robot - 智能图书馆书籍错位检测系统 (V2.0)

## 📖 项目概述

Lost Book Robot 是一个用于自动检测图书馆书籍错位的智能机器人系统。本项目采用**端到端（End-to-End）**的视觉处理方案，利用 Jetson Orin Nano 的边缘计算能力，通过“单帧全量计算”策略，实时判断书架上书籍的排序是否符合美国国会图书馆分类法（LC Classification）。

**核心技术革新 (V2.0)**：

1. **视觉前端**：使用 **YOLOv8 实例分割** 同时检测“书脊”和“索书号标签”，解决无标签书籍漏检问题。
2. **排序核心**：放弃传统规则匹配，采用 **Bi-LSTM 孪生网络 (Siamese Network)** 直接比较两本书的 OCR 字符串，具备极强的 OCR 容错能力。
3. **逻辑推断**：引入“数据库夹击验证（Gap-Filling）”策略，通过左右邻居推断无法识别的“幽灵书”。

---

## 🏗 系统架构

### 整体数据流

```mermaid
graph TD
    Camera[相机采集] --> YOLO[YOLOv8 实例分割]
    
    subgraph "视觉预处理"
    YOLO -->|Class: Book Spine| CropSpine[裁剪书脊图]
    YOLO -->|Class: Call Label| CropLabel[裁剪标签图]
    end
    
    subgraph "文字识别"
    CropLabel --> OpenOCR[OpenOCR 引擎]
    CropSpine -->|无标签时| OpenOCR
    end
    
    subgraph "逻辑验证 (Jetson Orin)"
    OpenOCR -->|OCR String List| SlidingWindow[滑动窗口配对]
    SlidingWindow -->|Pair (A, B)| BiLSTM[Bi-LSTM 比较器]
    
    BiLSTM -->|Out_of_Order| Alarm[错位报警]
    BiLSTM -->|Duplicate| Ignore[忽略]
    BiLSTM -->|In_Order| Pass[通过]
    end
    
    subgraph "异常处理"
    CropSpine -->|无文字/无标签| DBCheck[数据库邻居推断]
    end

```

---

## 🧠 核心算法详解

### 1. 视觉感知：YOLOv8 双层分割

**目标**：从复杂背景中提取出物理书脊，并定位索书号区域。

* **模型**：YOLOv8-Seg (Instance Segmentation)
* **检测类别 (Classes)**：
1. `book_spine` (书脊)：覆盖整本书侧面（使用多边形标注以适应倾斜）。
2. `call_label` (标签)：覆盖白色索书号贴纸。


* **后处理逻辑**：
* 计算包含关系：如果 `call_label` 的中心点位于某 `book_spine` 的 Mask 内，则判定为该书的标签。
* **有标书**：裁剪 `call_label` 区域送入 OCR。
* **无标书**：裁剪 `book_spine` 底部 1/4 区域（针对印制文字）或全书脊（用于数据库指纹匹配）。



### 2. 文字识别：OpenOCR

**目标**：将图像像素转化为文本字符串。

* **引擎**：OpenOCR (已验证，准确率 > 95%，速度 ~0.05s/图)。
* **策略**：不修改模型，直接调用。容忍 OCR 产生的噪声（如 `1` 变 `I`, `B` 变 `8`），交由下游神经网络处理。

### 3. 排序验证：DeepSort-Comparator (Bi-LSTM)

**目标**：在不进行复杂规则解析的情况下，判断两个 OCR 字符串的排序关系。

* **架构**：**Siamese Bi-LSTM (孪生双向长短期记忆网络)**
* **输入**：字符级 Tokenizer (Char-level)，保留 `|` 分隔符和 OCR 噪声。
* **Backbone**：共享权重的 2层 Bi-LSTM，提取序列特征。
* **Head**：将两个特征向量拼接，通过 MLP 分类。


* **输出类别**：
* `0: In_Order` (顺序正确：A < B)
* `1: Out_of_Order` (错位：A > B) —— **核心报警触发条件**
* `2: Duplicate` (重复/重影：A ≈ B)


* **优势**：
* 自动学习 LC 分类法的层级权重（字母 > 数字 > 小数）。
* 自动忽略无效前缀（如 `OLIN`, `REF`）。
* 极强的鲁棒性，能处理模糊、缺损字符。



### 4. 异常推断：数据库夹击法 (Gap-Filling)

**目标**：处理 OCR 失败或完全无特征的“幽灵书”。

* **触发条件**：YOLO 检测到 `book_spine` 但无法提取有效文本。
* **逻辑**：
1. **锚定**：识别左邻居 A 和右邻居 C。
2. **查询**：访问图书馆数据库，查询 A 和 C 之间应有的书籍 B。
3. **验证**：
* **数量验证**：视觉检测到的无标书数量 == 数据库记录数量？
* **内容验证**：对无标书脊进行全图 OCR，匹配数据库中的 Title/Author 关键词。


4. **决策**：如果验证通过，则认为位置正确，不报警。



---

## 📊 数据策略

### 1. YOLO 训练数据 (真实数据)

* **来源**：真实拍摄的书架视频/照片。
* **标注方式**：
* 使用 CVAT 或 LabelImg。
* **必须使用多边形 (Polygon)** 标注书脊，防止重叠干扰。
* **双类标注**：每本书同时画 `book_spine` 和 `call_label`（如果有）。


* **规模**：约 500-1000 张图像。

### 2. Bi-LSTM 训练数据 (合成数据)

* **来源**：**完全合成 (Synthetic Generation)**，无需人工标注真实对子。
* **生成脚本 (`data_gen.py`)**：
1. **规则生成**：利用 Python 脚本批量生成符合 LC 规则的正确排序对子。
2. **噪声注入 (关键)**：
* **字符替换**：随机将 `1`->`I`, `0`->`O`, `8`->`B`。
* **前缀干扰**：随机添加/删除 `OLIN`, `REF` 等前缀。
* **格式干扰**：随机删除 `.`，随机替换换行符 `|` 为空格。


3. **负样本构造**：翻转正确对子生成 `Out_of_Order` 样本。


* **规模**：可生成 10万+ 对样本，确保模型充分收敛。

---

## 🗓 开发路线图 (Roadmap)

### Phase 1: 视觉前端构建 (当前重点)

* [/] **数据采集**：拍摄书架视频，截取关键帧。
* [/] **数据标注**：完成 `spine` + `label` 的多边形标注。
* [/] **YOLO 训练**：训练 YOLOv8-Seg(nano) 模型，并部署推理脚本。
* [ ] **裁图流水线**：实现 `detect -> crop -> ocr` 的完整 Python 流程。

### Phase 2: 神经网络核心开发

* [ ] **数据生成器**：编写 `data_gen.py`，生成带有 OCR 噪声的 LC 索书号对子。
* [ ] **模型搭建**：使用 PyTorch 实现 Bi-LSTM Siamese Network。
* [ ] **模型训练**：在合成数据集上训练，验证准确率 > 98%。
* [ ] **端到端测试**：将 OCR 输出接入网络，测试真实图片的排序判断。

### Phase 3: 系统集成与推断逻辑

* [ ] **滑动窗口逻辑**：实现单帧内的 `check(A,B), check(B,C)...` 循环。
* [ ] **数据库接口**：模拟一个简单的 Library DB（JSON/SQLite）。
* [ ] **推断模块**：实现“无标书”的数据库查验逻辑。
* [ ] **Jetson 部署**：在 Orin Nano 上进行性能优化（TensorRT 可选）。

---

## 📁 建议文件结构

```
LostBookRobot/
├── data/
│   ├── raw_images/           # 原始书架照片
│   ├── yolo_dataset/         # 标注好的 YOLO 数据
│   └── synthetic_pairs/      # 生成的 LSTM 训练数据
├── modules/
│   ├── vision/
│   │   ├── yolo_inference.py # YOLO 推理与裁剪
│   │   └── ocr_wrapper.py    # OpenOCR 调用封装
│   ├── logic/
│   │   ├── comparator.py     # Bi-LSTM 网络定义
│   │   ├── train_lstm.py     # 网络训练脚本
│   │   └── data_gen.py       # 合成数据生成器 
│   └── database/
│       └── library_db.py     # 模拟数据库接口
├── weights/
│   ├── best_yolo.pt          # 训练好的 YOLO 模型
│   └── comparator.pth        # 训练好的 LSTM 模型
├── main_pipeline.py          # 主程序：串联整个流程
└── requirements.txt

```

---

**文档维护者**: Hongxi Chen
**最后更新**: 2025-12-22
**状态**: 方案重构完成，进入 Phase 1 开发。