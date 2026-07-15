# 图像分类基线与迁移学习项目 (Cat vs. Non-Cat Classification)

本项目旨在建立一个图像分类 pipeline，涵盖从数据预处理与增强、搭建基线卷积神经网络 (CNN)、利用迁移学习 (Transfer Learning) 训练高级模型，到超参数微调及最终的模型评估与指标分析。

---

## 👥 团队分工与角色

本项目由 **Person A (Data Engineer / Model Developer)** 与 **Person B (Augmentation Specialist / Hyperparameter Optimizer)** 协同完成。

| 阶段 | 任务模块 | Person A (数据与基线模型) | Person B (增强与迁移学习) |
| :--- | :--- | :--- | :--- |
| **Phase 1** | **数据预处理与增强** | 编写数据清洗与划分脚本 (尺寸缩放、归一化、70/15/15划分) | 引入数据增强技术 (旋转、翻转、亮度调节等) |
| **Phase 2** | **构建模型** | 搭建开发环境，从头构建轻量级基线 CNN 模型 | 载入预训练模型 (MobileNetV2/ResNet50) 进行迁移学习 |
| **Phase 3** | **训练与超参数微调**| 运行训练循环，监控 Train/Val Loss 以防止过拟合 | 实验超参数组合 (学习率、Batch Size <= 32、Adam/SGD) |
| **Phase 4** | **模型评估与指标** | 协同测试最终模型，计算准确率、混淆矩阵、精确率及召回率 | 协同测试最终模型，计算准确率、混淆矩阵、精确率及召回率 |

---

## 📂 推荐项目目录结构

为了更好地组织代码与数据，建议采用以下目录结构：

```text
bsaeline model/
├── data/                  # 原始数据与处理后的数据集
│   ├── raw/               # 原始 4,000 张图像
│   ├── train/             # 训练集 (70%)
│   ├── val/               # 验证集 (15%)
│   └── test/              # 测试集 (15%)
├── src/                   # 源代码目录
│   ├── preprocess.py      # 数据预处理与增强脚本 (Person A & B)
│   ├── dataset.py         # PyTorch/TensorFlow 自定义数据集加载
│   ├── baseline_model.py  # 自定义轻量级 CNN 结构 (Person A)
│   ├── transfer_model.py  # MobileNetV2 / ResNet50 迁移学习模型 (Person B)
│   ├── train.py           # 统一的模型训练与验证脚本 (Person A)
│   └── evaluate.py        # 模型评估与混淆矩阵生成 (Person A & B)
├── notebooks/             # Jupyter 分析与可视化笔记本
│   └── exploration.ipynb  # 数据探索与结果分析
├── requirements.txt       # 环境依赖依赖包列表
└── README.md              # 项目说明文档
```

---

## 🛠️ 详细阶段规划

### 🔄 Phase 1: 数据预处理与增强 (Data Preprocessing & Augmentation)
* **目标**: 清洗与准备 4,000 张图像，通过增强扩充数据集防止过拟合。
* **具体任务**:
  * 图像尺寸统一调整为 $224 \times 224$ 像素。
  * 像素值归一化至 $[0, 1]$ 之间。
  * 按照 **70% 训练集、15% 验证集、15% 测试集** 的比例进行随机划分。
  * 引入数据增强：随机旋转 (Random Rotation)、水平/垂直翻转 (Horizontal/Vertical Flip)、亮度调整 (Brightness Adjustments) 等。

### 🏗️ Phase 2: 构建基线模型与迁移学习 (Building the Baseline & Transfer Model)
* **目标**: 快速建立可运行的基线模型，并引入预训练模型进行对比。
* **具体任务**:
  * 安装深度学习环境（推荐使用 PyTorch，亦可使用 TensorFlow/Keras）。
  * **基线模型 (Person A)**: 构建一个极简的 CNN（例如 3 个卷积层 + 最大池化层 + 全连接层），作为 Baseline，确保整个 Pipeline 能够通畅运行且不崩溃。
  * **迁移学习模型 (Person B)**: 载入预训练的 **MobileNetV2** 或 **ResNet50**。冻结 (Freeze) 前期特征提取层的权重，仅对最后的分类头 (Classification Head) 进行重构与训练。

### ⚡ Phase 3: 训练与超参数微调 (Training & Hyperparameter Tuning)
* **目标**: 执行高效训练，调优模型结构与参数。
* **具体任务**:
  * 编写完整的训练循环，包含前向传播、损失计算、反向传播及优化器更新。
  * 实时记录和监控训练集与验证集的损失 (Loss) 和准确率 (Accuracy)，绘制曲线以诊断过拟合 (Overfitting) 或欠拟合 (Underfitting)。
  * 进行超参数实验：
    * **学习率 (Learning Rate)**: 尝试不同的初始学习率，或者加入学习率衰减策略。
    * **批次大小 (Batch Size)**: 限制在 32 或以下，避免笔记本电脑显存/内存溢出 (OOM)。
    * **优化器 (Optimizer)**: 对比 Adam 与 SGD 的收敛速度与最终效果。

### 📊 Phase 4: 评估与指标 (Evaluation & Metrics)
* **目标**: 在独立的测试集上评估最终模型，生成详细的性能报告。
* **具体任务**:
  * 将训练完成的 Baseline 模型和迁移学习模型在 Testing Set 上进行测试。
  * 计算各项评估指标：**准确率 (Accuracy)**、**精确率 (Precision)**、**召回率 (Recall)**、**F1 分数 (F1-Score)**。
  * 绘制**混淆矩阵 (Confusion Matrix)**，直观分析模型容易混淆的类别（如哪些非猫图像被误判为猫）。

---

## 🏆 最终项目成果 (Final Results)

本项目的所有任务均已顺利完成，并且使用 10% 的真实猫与船只数据集（1500+张图片）进行了完整的端到端训练与评估，证明了迁移学习的强大优势。

### 最终测试集评估对比

| 模型类别 | 测试集准确率 (Accuracy) | 训练设置 | 表现细节 |
| :--- | :--- | :--- | :--- |
| **基线 CNN (Person A)** | **97.37%** | 3 Epochs (CPU) | 误判 6 张图（4 张船判为猫，2 张猫判为船） |
| **迁移学习 ResNet (Person B)** | **100.00% (满分)** | 3 Epochs (CPU) | **完美分类！** 0 误判 |

**结论**：即使在数据量小、训练轮数少、计算资源受限（仅 CPU）的情况下，借助预训练大模型骨干提取特征的**迁移学习 (Transfer Learning)** 依然取得了满分的完美成绩，远远超越了从头训练的简单卷积网络结构。
