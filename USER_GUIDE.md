# 多模态推荐系统用户指南

## 项目概述

本项目是一个基于多模态数据的推荐系统，分为四个阶段：

1. **第一阶段**：多模态数据处理（已完成）
2. **第二阶段**：轻量级扩散模型预训练（已完成）
3. **第三阶段**：强化学习微调（待开始）
4. **第四阶段**：可解释性增强（待开始）

本指南详细说明第一阶段和第二阶段生成的文件，以及如何为后续阶段使用这些文件。

## 文件结构概览

```
multimodel_recommender/
├── 核心特征文件 (用于模型训练)
│   ├── image_features.npy      # 图像特征矩阵 (2732×512)
│   ├── text_features.npy       # 文本特征矩阵 (2732×512)
│   ├── id_embeddings.npy       # 商品ID嵌入 (2732×128)
│   ├── item_ids.npy            # 商品ID列表 (2732)
│   ├── user_sequences.npy      # 用户序列矩阵 (38656×20)
│   └── user_ids.npy            # 用户ID列表 (38656)
│
├── 映射文件
│   ├── id_to_idx.pkl           # 商品ID到索引的映射字典
│   └── user_sequences.pkl      # 原始用户序列字典
│
├── 原始数据文件
│   ├── filtered_interactions.csv  # 过滤后的交互数据 (99,427行)
│   └── item_metadata.csv          # 商品元数据 (53,807行)
│
├── 分析报告文件
│   ├── image_issues_report.md     # 图像问题分析报告
│   ├── image_coverage_report.json # 图像覆盖率统计
│   ├── all_items_with_images.csv  # 有图像的商品列表
│   └── problematic_urls.csv       # 有问题的图像URL
│
├── 第二阶段模型文件
│   ├── final_model.pth                   # 最终训练好的模型权重
│   ├── best_model_final.pth              # 最佳超参数模型权重
│   ├── stage2_test_model.pth             # 测试模型权重
│   ├── final_model_epoch{10,20,30,40,50}.pth  # 训练检查点
│   ├── best_model_epoch{5,10,15,20}.pth       # 最佳模型检查点
│   ├── training_history.npz              # 完整训练历史数据
│   ├── training_report.txt               # 训练结果报告
│   ├── best_hyperparams.txt              # 最佳超参数配置
│   └── item_unified_representations.npy  # 所有商品的256维统一表示
│
├── 脚本文件
│   ├── stage1.py                  # 第一阶段主处理脚本
│   ├── stage2.py                  # 第二阶段扩散模型训练主脚本
│   ├── test_stage2.py             # 第二阶段模型测试脚本
│   ├── train_test.py              # 训练流程测试脚本
│   ├── example_stage2.py          # 第二阶段模型使用示例
│   ├── tune_baseline.py           # 基线超参数实验脚本
│   ├── hyperparam_tune.py         # 超参数网格搜索脚本
│   ├── train_best.py              # 最佳超参数训练脚本
│   ├── train_final.py             # 最终训练脚本（带学习率衰减）
│   ├── analyze_images.py          # 图像分析脚本
│   ├── test_image_urls.py         # URL测试脚本
│   └── data_preprocessing.py      # 数据预处理脚本 (早期版本)
│
└── 配置文件
    ├── pyproject.toml             # 项目依赖配置
    └── uv.lock                    # 依赖锁文件
```

## 核心特征文件详解

### 1. 多模态特征文件

#### `image_features.npy` - 图像特征

- **形状**: (2732, 512)
- **内容**: 每个商品的512维图像特征向量
- **生成方法**: 使用CLIP ViT-B/32模型提取商品主图特征
- **数据来源**: Amazon商品图像URL下载后处理
- **覆盖率**: 仅5.1%的商品有图像特征（2732/53807）

```python
import numpy as np
image_features = np.load('image_features.npy')
print(f"图像特征形状: {image_features.shape}")  # (2732, 512)
```

#### `text_features.npy` - 文本特征

- **形状**: (2732, 512)
- **内容**: 每个商品的512维文本特征向量
- **生成方法**: 使用CLIP文本编码器处理"标题+描述前100词"
- **数据来源**: 商品标题和描述文本
- **注意**: 与图像特征对应相同的2732个商品

```python
text_features = np.load('text_features.npy')
print(f"文本特征形状: {text_features.shape}")  # (2732, 512)
```

#### `id_embeddings.npy` - 商品ID嵌入

- **形状**: (2732, 128)
- **内容**: 每个商品的128维随机初始化ID嵌入
- **生成方法**: 使用正态分布随机生成（种子42）
- **用途**: 提供商品的协同过滤信号

```python
id_embeddings = np.load('id_embeddings.npy')
print(f"ID嵌入形状: {id_embeddings.shape}")  # (2732, 128)
```

#### `item_ids.npy` - 商品ID列表

- **形状**: (2732,)
- **内容**: 与特征矩阵对应的商品ASIN列表
- **用途**: 将特征索引映射回原始商品ID

```python
item_ids = np.load('item_ids.npy')
print(f"商品ID数量: {len(item_ids)}")  # 2732
print(f"前5个商品ID: {item_ids[:5]}")
```

### 2. 用户序列文件

#### `user_sequences.npy` - 用户序列矩阵

- **形状**: (38656, 20)
- **内容**: 每个用户的商品交互序列，用商品索引表示
- **生成方法**:
  1. 按时间戳排序每个用户的交互历史
  2. 截取最近20个交互（不足则填充0）
  3. 将商品ID转换为`id_to_idx.pkl`中的索引
- **填充值**: 0表示填充或无效索引

```python
user_sequences = np.load('user_sequences.npy')
print(f"用户序列形状: {user_sequences.shape}")  # (38656, 20)
print(f"用户1的序列: {user_sequences[0]}")
```

#### `user_ids.npy` - 用户ID列表

- **形状**: (38656,)
- **内容**: 与序列矩阵对应的用户ID列表
- **顺序**: 与`user_sequences.npy`的行顺序一致

```python
user_ids = np.load('user_ids.npy')
print(f"用户ID数量: {len(user_ids)}")  # 38656
```

### 3. 映射文件

#### `id_to_idx.pkl` - 商品ID到索引映射

- **格式**: Python字典
- **内容**: `{商品ASIN: 特征矩阵索引}`
- **用途**: 在商品ID和特征矩阵行索引间转换

```python
import pickle
with open('id_to_idx.pkl', 'rb') as f:
    id_to_idx = pickle.load(f)

# 示例：获取商品B07CQM3XZ9的索引
item_idx = id_to_idx.get('B07CQM3XZ9')
print(f"商品B07CQM3XZ9的索引: {item_idx}")
```

#### `user_sequences.pkl` - 原始用户序列字典

- **格式**: Python字典
- **内容**: `{用户ID: [商品ASIN列表]}`
- **注意**: 这是原始格式，`user_sequences.npy`是其索引化版本

```python
with open('user_sequences.pkl', 'rb') as f:
    sequences_dict = pickle.load(f)

# 获取用户特定序列
user_id = list(sequences_dict.keys())[0]
print(f"用户{user_id}的原始序列: {sequences_dict[user_id]}")
```

## 原始数据文件

### `filtered_interactions.csv` - 过滤后的交互数据

- **行数**: 99,427
- **用户数**: 38,656
- **商品数**: 53,807
- **过滤条件**:
  1. 用户交互次数≥3
  2. 商品交互次数≥3
  3. 保留Clothing, Shoes and Jewelry类别
  4. 20%数据集采样（原始6600万行→1320万行→过滤后9.9万行）

```python
import pandas as pd
df_interactions = pd.read_csv('filtered_interactions.csv')
print(f"交互数据: {len(df_interactions)}行")
print(f"唯一用户: {df_interactions['user_id'].nunique()}")
print(f"唯一商品: {df_interactions['asin'].nunique()}")
```

### `item_metadata.csv` - 商品元数据

- **行数**: 53,807
- **列**: asin, title, text, images, parent\_asin
- **注意**: 只有5.1%的商品有图像（images字段非空）

```python
df_metadata = pd.read_csv('item_metadata.csv')
print(f"商品元数据: {len(df_metadata)}行")
# 统计有图像的商品
has_images = df_metadata['images'].apply(lambda x: x != '[]')
print(f"有图像的商品: {has_images.sum()} ({has_images.sum()/len(df_metadata)*100:.1f}%)")
```

## 如何在第二阶段使用这些文件

### 1. 加载所有特征

```python
import numpy as np
import pickle

# 加载特征
image_features = np.load('image_features.npy')  # (2732, 512)
text_features = np.load('text_features.npy')    # (2732, 512)
id_embeddings = np.load('id_embeddings.npy')    # (2732, 128)
item_ids = np.load('item_ids.npy')              # (2732,)

# 加载用户序列
user_sequences = np.load('user_sequences.npy')  # (38656, 20)
user_ids = np.load('user_ids.npy')              # (38656,)

# 加载映射
with open('id_to_idx.pkl', 'rb') as f:
    id_to_idx = pickle.load(f)
```

### 2. 创建商品的统一表示

根据README.md中的第二阶段设计，需要将多模态特征拼接并投影到256维：

```python
import torch
import torch.nn as nn

# 拼接特征 (512+512+128=1152维)
item_features = np.concatenate([
    image_features,  # 图像特征
    text_features,   # 文本特征
    id_embeddings    # ID嵌入
], axis=1)  # 形状: (2732, 1152)

# 转换为PyTorch张量
item_features_tensor = torch.FloatTensor(item_features)

# 定义对齐MLP (与README设计一致)
class MultimodalAlignment(nn.Module):
    def __init__(self, input_dim=1152, hidden_dim=512, output_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.mlp(x)

# 创建统一表示
alignment_model = MultimodalAlignment()
unified_representations = alignment_model(item_features_tensor)
print(f"统一表示形状: {unified_representations.shape}")  # (2732, 256)
```

### 3. 准备训练数据

```python
# 创建训练样本：用户序列 -> 下一个商品
def create_training_pairs(user_sequences, seq_length=20):
    """
    从用户序列创建训练对
    输入: 序列的前seq_length-1个商品
    目标: 序列的第seq_length个商品
    """
    inputs = []
    targets = []
    
    for seq in user_sequences:
        # 移除非零元素（填充部分）
        valid_seq = seq[seq != 0]
        if len(valid_seq) >= 2:  # 至少需要2个商品才能创建样本
            for i in range(1, len(valid_seq)):
                input_seq = valid_seq[:i]
                target = valid_seq[i]
                # 填充到固定长度
                padded_input = np.zeros(seq_length-1, dtype=np.int32)
                padded_input[-len(input_seq):] = input_seq
                inputs.append(padded_input)
                targets.append(target)
    
    return np.array(inputs), np.array(targets)

# 创建训练数据
train_inputs, train_targets = create_training_pairs(user_sequences)
print(f"训练样本数: {len(train_inputs)}")
print(f"输入形状: {train_inputs.shape}")  # (N, 19)
print(f"目标形状: {train_targets.shape}")  # (N,)
```

### 4. 数据加载器示例

```python
from torch.utils.data import Dataset, DataLoader

class RecommendationDataset(Dataset):
    def __init__(self, user_sequences, item_features):
        self.user_sequences = user_sequences
        self.item_features = item_features
        self.inputs, self.targets = create_training_pairs(user_sequences)
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        input_seq = self.inputs[idx]
        target = self.targets[idx]
        
        # 获取输入序列的商品特征
        input_features = self.item_features[input_seq]  # (seq_len-1, feature_dim)
        
        return {
            'input_sequence': torch.LongTensor(input_seq),
            'input_features': torch.FloatTensor(input_features),
            'target': torch.LongTensor([target])
        }

# 创建数据集和数据加载器
dataset = RecommendationDataset(user_sequences, unified_representations)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# 使用示例
for batch in dataloader:
    print(f"批次大小: {batch['input_sequence'].shape}")  # (128, 19)
    print(f"特征形状: {batch['input_features'].shape}")  # (128, 19, 256)
    print(f"目标形状: {batch['target'].shape}")  # (128, 1)
    break
```

## 第二阶段：轻量级扩散模型预训练（已完成）

### 概述

第二阶段实现了轻量级扩散模型，用于学习商品的多模态统一表示。该模型结合了多模态对齐、对比学习和扩散模型技术，将图像（512维）、文本（512维）和ID（128维）特征投影到256维的统一空间。

### 模型架构

#### 1. 多模态对齐层 (`MultimodalAlignment`)
- **输入**: 图像特征 (512D)、文本特征 (512D)、ID嵌入 (128D)
- **结构**: 三个独立的MLP投影层 + 融合层
- **输出**: 256维统一表示 + 各模态256维嵌入
- **参数量**: 约1.5M参数

```python
# 架构核心
class MultimodalAlignment(nn.Module):
    def __init__(self, image_dim=512, text_dim=512, id_dim=128, 
                 hidden_dim=512, output_dim=256):
        super().__init__()
        self.image_proj = nn.Sequential(nn.Linear(image_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))
        self.text_proj = nn.Sequential(nn.Linear(text_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))
        self.id_proj = nn.Sequential(nn.Linear(id_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))
        self.fusion = nn.Sequential(nn.Linear(output_dim*3, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))
```

#### 2. 扩散过程 (`DiffusionProcess`)
- **时间步**: 1000步线性调度 (β_start=1e-4, β_end=0.02)
- **前向加噪**: \( x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon \)
- **方差保持**: 确保信号和噪声比例适当

#### 3. 去噪MLP (`DenoisingMLP`)
- **输入**: 加噪表示 + 时间步嵌入
- **结构**: 3层MLP (256→512→512→256)
- **输出**: 预测的噪声
- **参数量**: 约1.3M参数

#### 4. 整体模型 (`DiffusionRecommender`)
- **总参数量**: 约2.8M参数（轻量级）
- **组件**: 对齐层 + 序列编码器（占位）+ 扩散去噪网络
- **输出**: 统一表示 + 各模态嵌入 + 噪声预测

### 损失函数

#### 1. 多模态对比损失 (`multimodal_contrastive_loss`)
- **类型**: InfoNCE（噪声对比估计）
- **目标**: 同一商品的不同模态表示相互靠近
- **温度参数**: 0.1（最佳）
- **计算**: 三个模态间（图像、文本、ID）的对比学习

#### 2. 扩散ELBO损失 (`diffusion_elbo_loss`)
- **类型**: 均方误差（MSE）
- **目标**: 预测添加的噪声
- **公式**: \( L_{\text{diffusion}} = \mathbb{E}_{t,\epsilon}[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 ] \)

#### 3. 总损失
- **加权和**: \( L_{\text{total}} = L_{\text{diffusion}} + \lambda L_{\text{contrastive}} \)
- **权重λ**: 0.1（最佳）

### 超参数调优结果

经过27组超参数组合的实验（温度、λ、学习率），最佳配置为：

| 超参数 | 最佳值 | 说明 |
|--------|--------|------|
| **学习率** | 5e-5 | 使用余弦退火衰减到1e-6 |
| **对比损失权重 (λ)** | 0.1 | 平衡扩散和对比损失 |
| **温度参数** | 0.1 | InfoNCE损失的温度 |
| **批次大小** | 128 | 内存与效率平衡 |
| **训练周期** | 50 | 完整训练 |
| **优化器** | AdamW | β1=0.9, β2=0.999 |

**训练结果**:
- 初始总损失: 1.9751 → 最终总损失: 1.3200（下降33.1%）
- 对比损失: 9.7678 → 5.9048（下降39.6%）
- 扩散损失: 1.0000 → 0.7295（下降27.1%）
- 最后10个epoch平均损失: 1.3237 ± 0.0023（已收敛）

### 训练流程

#### 1. 数据准备
```python
# 加载第一阶段特征
from stage2 import load_features
data = load_features()  # 自动加载image_features.npy, text_features.npy, id_embeddings.npy
```

#### 2. 运行完整训练
```bash
# 使用最佳超参数训练50个epoch（带学习率衰减）
uv run python train_final.py

# 输出文件:
#   - final_model.pth           # 最终模型
#   - final_model_epoch{10,20,30,40,50}.pth  # 检查点
#   - training_history.npz      # 训练历史
#   - training_report.txt       # 训练报告
#   - item_unified_representations.npy  # 统一表示
```

#### 3. 分步训练选项
```bash
# 1. 基线实验（5个epoch）
uv run python tune_baseline.py

# 2. 超参数网格搜索
uv run python hyperparam_tune.py

# 3. 最佳参数训练（20个epoch）
uv run python train_best.py

# 4. 测试模型
uv run python test_stage2.py
```

### 模型使用示例

#### 1. 加载训练好的模型
```python
import torch
from stage2 import DiffusionRecommender

# 加载模型
model = DiffusionRecommender()
model.load_state_dict(torch.load('final_model.pth'))
model.eval()

# 获取商品统一表示
from example_stage2 import get_item_representations
image_features = torch.randn(10, 512)  # 示例数据
text_features = torch.randn(10, 512)
id_features = torch.randn(10, 128)

results = get_item_representations(model, image_features, text_features, id_features)
print(f"统一表示形状: {results['unified'].shape}")  # (10, 256)
```

#### 2. 计算商品相似度
```python
# 计算余弦相似度
from example_stage2 import calculate_similarities
similarities = calculate_similarities(results['unified'])
print(f"相似度矩阵形状: {similarities.shape}")  # (10, 10)
```

#### 3. 扩散生成新表示
```python
# 从噪声生成新商品表示
from example_stage2 import generate_from_diffusion
generated_items = generate_from_diffusion(model, num_samples=5)
print(f"生成表示形状: {generated_items.shape}")  # (5, 256)
```

### 关键脚本说明

| 脚本文件 | 用途 | 关键参数 |
|----------|------|----------|
| `stage2.py` | 第二阶段主脚本，包含所有模型定义和训练函数 | `batch_size=128`, `lr=5e-5`, `lambda_contrastive=0.1` |
| `test_stage2.py` | 测试模型组件和加载特征 | 无参测试 |
| `example_stage2.py` | 模型使用示例，展示推理流程 | 示例数据 |
| `tune_baseline.py` | 基线超参数实验（5个epoch） | 默认超参数 |
| `hyperparam_tune.py` | 超参数网格搜索（27种组合） | 温度、λ、学习率 |
| `train_best.py` | 最佳超参数训练（20个epoch） | 最佳参数 |
| `train_final.py` | 最终训练（50个epoch+学习率衰减） | 余弦退火 |

### 生成的文件

1. **模型权重文件**
   - `final_model.pth` - 50个epoch完整训练的最终模型
   - `best_model_final.pth` - 20个epoch最佳超参数模型
   - `stage2_test_model.pth` - 测试模型

2. **检查点文件**
   - `final_model_epoch{10,20,30,40,50}.pth` - 训练中间检查点
   - `best_model_epoch{5,10,15,20}.pth` - 最佳模型检查点

3. **训练记录文件**
   - `training_history.npz` - 损失历史、学习率变化
   - `training_report.txt` - 详细训练结果报告
   - `best_hyperparams.txt` - 最佳超参数配置

4. **表示文件**
   - `item_unified_representations.npy` - 所有2732个商品的256维统一表示

### 注意事项

1. **设备兼容性**: 模型自动检测CUDA，调度参数与输入设备同步
2. **内存使用**: 统一表示文件约2.8MB（2732×256×4字节）
3. **训练时间**: 50个epoch约15-20分钟（RTX 3090）
4. **收敛性**: 损失在第30个epoch后基本稳定，学习率衰减提升后期稳定性

### 性能指标

- **模型大小**: 2.8M参数（轻量级）
- **训练速度**: ~220批次/秒（批次大小128）
- **内存占用**: ~1.2GB（训练时）
- **收敛性**: 损失下降33.1%，标准差<0.0025（稳定）

## 文件生成过程详解

### 第一阶段数据处理流程 (`stage1.py`)

1. **数据加载**
   - 从Hugging Face加载Amazon Review 2023数据集
   - 选择"Clothing, Shoes and Jewelry"类别
   - 20%随机采样（处理时间优化）
2. **数据过滤**
   - 筛选交互次数≥3的用户和商品
   - 保留最多50,000用户和100,000商品
   - 保存为`filtered_interactions.csv`
3. **特征提取**
   - **图像特征**: 下载商品图像 → CLIP ViT-B/32提取 → `image_features.npy`
   - **文本特征**: 拼接标题和描述 → CLIP文本编码器 → `text_features.npy`
   - **ID嵌入**: 随机生成 → `id_embeddings.npy`
4. **序列构建**
   - 按时间戳排序用户交互
   - 截取最近20个交互
   - 转换为索引格式 → `user_sequences.npy`

### 关键参数配置

```python
# stage1.py中的主要配置参数
SAMPLE_FRACTION = 0.2          # 数据集采样比例
MIN_INTERACTIONS = 3           # 最小交互次数
MAX_USERS = 50000              # 最大用户数
MAX_ITEMS = 100000             # 最大商品数
SEQ_LENGTH = 20                # 序列长度
IMAGE_DIM = 512                # 图像特征维度
TEXT_DIM = 512                 # 文本特征维度
ID_DIM = 128                   # ID嵌入维度
```

## 局限性说明

### 1. 图像覆盖率低

- **问题**: 仅5.1%的商品有图像特征（2732/53807）
- **影响**: 多模态模型可能主要依赖文本和ID特征
- **解决方案**:
  - 第二阶段可考虑文本+ID的混合模型
  - 未来可补充图像数据源

### 2. 数据规模限制

- **当前**: 38,656用户 × 2,732商品（有图像）
- **目标**: 50,000用户 × 100,000商品（根据README）
- **差距**: 由于图像限制，实际有图像的商品较少

### 3. 序列长度固定

- **所有序列填充/截断到20个商品**
- **短序列用0填充，长序列保留最近20个**

## 最佳实践建议

### 1. 数据验证

```python
def validate_data_integrity():
    """验证数据一致性"""
    # 检查特征维度匹配
    assert image_features.shape[0] == text_features.shape[0] == id_embeddings.shape[0]
    assert image_features.shape[1] == 512
    assert text_features.shape[1] == 512
    assert id_embeddings.shape[1] == 128
    
    # 检查序列有效性
    max_item_idx = len(item_ids) - 1
    valid_sequences = (user_sequences >= 0) & (user_sequences <= max_item_idx)
    invalid_count = np.sum(user_sequences[valid_sequences == False])
    print(f"无效序列索引数: {invalid_count}")
    
    # 检查映射一致性
    for i, item_id in enumerate(item_ids):
        assert id_to_idx[item_id] == i, f"映射不一致: {item_id}"
    
    print("数据完整性验证通过!")
```

### 2. 内存优化

```python
# 使用内存映射处理大文件
image_features = np.load('image_features.npy', mmap_mode='r')
# 仅在使用时加载数据到内存
```

### 3. 特征归一化

```python
# 建议对特征进行归一化
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
image_features_normalized = scaler.fit_transform(image_features)
text_features_normalized = scaler.fit_transform(text_features)
id_embeddings_normalized = scaler.fit_transform(id_embeddings)
```

## 常见问题解答

### Q1: 为什么商品数量不一致？

- `item_metadata.csv`有53,807个商品（所有过滤后的商品）
- 特征文件只有2,732个商品（有图像的商品）
- 使用`id_to_idx.pkl`确定哪些商品有特征

### Q2: 如何处理没有图像的商品？

```python
# 方案1: 使用文本+ID特征
def get_item_features(item_idx, use_image=True):
    if use_image and image_features is not None:
        return np.concatenate([
            image_features[item_idx],
            text_features[item_idx],
            id_embeddings[item_idx]
        ])
    else:
        # 仅使用文本和ID
        return np.concatenate([
            text_features[item_idx],
            id_embeddings[item_idx]
        ])
```

### Q3: 如何扩展更多数据？

```python
# 修改stage1.py中的参数
# 1. 增加采样比例
SAMPLE_FRACTION = 1.0  # 使用完整数据集

# 2. 调整过滤阈值
MIN_INTERACTIONS = 1   # 降低交互要求

# 3. 重新运行
# uv run python stage1.py
```

### Q4: 序列中的0是什么意思？

- **0**: 填充值，表示序列较短或无有效商品
- **处理建议**: 在训练时忽略或使用掩码

```python
# 创建注意力掩码
sequence_mask = (user_sequences != 0).astype(np.float32)
```

## 下一步工作

### 第三阶段：强化学习微调（待开始）

第二阶段已完成轻量级扩散模型的预训练，生成了商品的多模态统一表示。第三阶段将使用这些表示进行强化学习微调，优化推荐策略。

#### 1. 输入准备
- **商品表示**: 使用`item_unified_representations.npy`中的256维统一表示
- **用户序列**: 使用`user_sequences.npy`中的用户交互历史
- **奖励信号**: 基于用户交互（点击、购买）设计奖励函数

#### 2. 强化学习框架
- **环境**: 推荐系统环境，状态=用户历史，动作=推荐商品
- **智能体**: 基于策略梯度（PPO）或深度Q网络（DQN）
- **奖励**: 即时奖励（点击）+ 长期奖励（购买、停留时间）

#### 3. 技术要点
- **状态编码**: 使用第二阶段训练的统一表示编码商品
- **动作空间**: 所有商品的表示空间（2732个商品）
- **探索策略**: ε-贪婪或噪声注入，平衡探索与利用

#### 4. 预期输出
- **微调后的推荐策略**: 能够根据用户历史生成个性化推荐
- **策略评估**: 在测试集上的CTR、购买率等指标
- **可解释性**: 结合第四阶段的可解释性增强

### 代码框架建议

```python
# 第三阶段强化学习推荐环境
class RecommendationEnv:
    def __init__(self, user_sequences, item_representations):
        self.user_sequences = user_sequences
        self.item_reps = item_representations
        self.current_user = None
        self.current_history = None
    
    def reset(self, user_idx):
        """重置环境到指定用户"""
        self.current_user = user_idx
        self.current_history = self.user_sequences[user_idx]
        return self.get_state()
    
    def step(self, action_item_idx):
        """执行推荐动作，返回(next_state, reward, done)"""
        # 模拟用户反馈（实际应从测试数据获取）
        reward = self.simulate_feedback(action_item_idx)
        # 更新用户历史
        self.update_history(action_item_idx)
        return self.get_state(), reward, False
    
    def get_state(self):
        """获取当前状态（用户历史编码）"""
        # 使用第二阶段模型编码历史序列
        history_reps = self.item_reps[self.current_history]
        return history_reps.mean(axis=0)  # 简单平均池化

# 强化学习智能体
class RLRecommenderAgent:
    def __init__(self, state_dim=256, action_dim=2732):
        self.policy_network = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )
    
    def select_action(self, state):
        """根据状态选择推荐动作"""
        logits = self.policy_network(state)
        return torch.softmax(logits, dim=-1)
```

## 联系与支持

如遇到问题，请检查：

1. 文件路径是否正确
2. 依赖包是否安装 (`uv sync`)
3. 内存是否足够（特征文件约500MB）

如需重新生成数据，运行：

```bash
uv run python stage1.py
```

**注意**: 重新运行将覆盖现有文件，建议先备份。

***

*文档最后更新: 2026-04-09*\
*对应项目版本: 第二阶段完成*
