# Vision Transformer (ViT) 图像分类项目

这是一个基于Vision Transformer (ViT) 的图像分类项目，结合了CNN特征提取器和Transformer架构，用于进行图像二分类任务。

## 项目结构

```
.
├── model.py          # 模型定义文件，包含ViT和相关组件
├── train.py         # 训练脚本
├── dataset.py       # 数据集加载和预处理
├── requirements.txt # 项目依赖
├── checkpoints/     # 模型检查点保存目录
└── results/         # 训练结果和可视化保存目录
```

## 主要特性

- 结合CNN特征提取器和Vision Transformer的混合架构
- 支持图像分类任务
- 提供完整的训练和评估流程
- 包含丰富的可视化功能：
  - 训练曲线绘制
  - 混淆矩阵分析
  - ROC曲线分析
  - 样本预测结果可视化

## 环境要求

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
Pillow>=8.0.0
tqdm>=4.65.0
einops>=0.6.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.0.0
```

## 使用说明

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 训练模型：
```bash
python train.py
```

训练脚本支持多个命令行参数，可以通过 `python train.py --help` 查看详细说明。

3. 训练过程中会自动保存：
- 模型检查点到 `checkpoints/` 目录
- 训练曲线图到 `results/` 目录
- 评估结果和可视化到 `results/` 目录

## 模型架构

该项目实现了一个混合架构的Vision Transformer：
- CNN特征提取器用于初步特征提取
- Transformer编码器进行特征序列建模
- 最终通过全连接层输出分类结果

主要组件包括：
- 多头自注意力机制
- 前馈神经网络
- Layer Normalization
- Position Embedding

## 可视化功能

项目提供了多种可视化工具：
1. 训练/验证损失和准确率曲线
2. 混淆矩阵
3. ROC曲线
4. 样本预测结果可视化 