# 多模态推荐系统用户指南

## 项目概述

本项目是一个基于多模态数据的推荐系统，分为四个阶段：
1. **第一阶段**：多模态数据处理（已完成）
2. **第二阶段**：轻量级扩散模型预训练
3. **第三阶段**：强化学习微调
4. **第四阶段**：可解释性增强

本指南详细说明第一阶段生成的文件，以及如何为后续阶段使用这些文件。

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
├── 脚本文件
│   ├── stage1.py                  # 第一阶段主处理脚本
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
- **列**: asin, title, text, images, parent_asin
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

### 第二阶段：扩散模型预训练
1. **多模态对齐**: 使用MLP将1152维特征投影到256维
2. **对比学习**: InfoNCE损失对齐同一商品的不同模态
3. **扩散训练**: 在嵌入空间训练轻量级扩散模型

### 代码框架建议
```python
# 第二阶段模型框架
class DiffusionRecommender(nn.Module):
    def __init__(self, item_dim=256):
        super().__init__()
        # 1. 多模态对齐层
        self.alignment = MultimodalAlignment(1152, 512, 256)
        # 2. 扩散去噪网络
        self.denoiser = DenoisingMLP(256, 512, 256)
        # 3. 序列编码器
        self.sequence_encoder = SequenceEncoder(256, 512)
    
    def forward(self, user_sequence, item_features):
        # 实现扩散训练流程
        pass
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

---

*文档最后更新: 2026-03-31*  
*对应项目版本: 第一阶段完成*