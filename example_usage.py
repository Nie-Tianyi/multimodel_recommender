#!/usr/bin/env python3
"""
多模态推荐系统文件使用示例

这个脚本展示了如何加载和使用第一阶段生成的所有文件，
并为第二阶段（扩散模型训练）准备数据。

运行方式:
    uv run python example_usage.py
"""

import numpy as np
import pickle
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

def load_all_features():
    """加载所有特征文件"""
    print("=" * 60)
    print("加载特征文件...")
    print("=" * 60)
    
    # 1. 加载核心特征
    image_features = np.load('stage1/image_features.npy')
    text_features = np.load('stage1/text_features.npy')
    id_embeddings = np.load('stage1/id_embeddings.npy')
    item_ids = np.load('stage1/item_ids.npy')
    
    print(f"图像特征形状: {image_features.shape}")      # (2732, 512)
    print(f"文本特征形状: {text_features.shape}")      # (2732, 512)
    print(f"ID嵌入形状: {id_embeddings.shape}")        # (2732, 128)
    print(f"商品ID数量: {len(item_ids)}")              # 2732
    
    # 2. 加载用户序列
    user_sequences = np.load('stage1/user_sequences.npy')
    user_ids = np.load('stage1/user_ids.npy')
    
    print(f"用户序列形状: {user_sequences.shape}")      # (38656, 20)
    print(f"用户ID数量: {len(user_ids)}")              # 38656
    
    # 3. 加载映射文件
    with open('stage1/id_to_idx.pkl', 'rb') as f:
        id_to_idx = pickle.load(f)
    
    print(f"商品ID到索引映射数量: {len(id_to_idx)}")    # 2732
    
    # 4. 加载原始数据（可选）
    try:
        df_interactions = pd.read_csv('stage1/filtered_interactions.csv')
        df_metadata = pd.read_csv('stage1/item_metadata.csv')
        print(f"交互数据: {len(df_interactions)}行, {df_interactions['user_id'].nunique()}用户")
        print(f"商品元数据: {len(df_metadata)}行")
    except FileNotFoundError:
        print("原始数据文件未找到，跳过...")
    
    return {
        'image_features': image_features,
        'text_features': text_features,
        'id_embeddings': id_embeddings,
        'item_ids': item_ids,
        'user_sequences': user_sequences,
        'user_ids': user_ids,
        'id_to_idx': id_to_idx
    }

def create_unified_representations(features_dict):
    """创建商品的统一表示（第二阶段第一步）"""
    print("\n" + "=" * 60)
    print("创建商品统一表示...")
    print("=" * 60)
    
    # 拼接多模态特征 (512 + 512 + 128 = 1152)
    image_features = features_dict['image_features']
    text_features = features_dict['text_features']
    id_embeddings = features_dict['id_embeddings']
    
    item_features = np.concatenate([
        image_features,
        text_features,
        id_embeddings
    ], axis=1)
    
    print(f"拼接后特征形状: {item_features.shape}")  # (2732, 1152)
    
    # 转换为PyTorch张量
    item_features_tensor = torch.FloatTensor(item_features)
    
    # 定义多模态对齐MLP（与README设计一致）
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
    
    # 创建模型并生成统一表示
    alignment_model = MultimodalAlignment()
    unified_representations = alignment_model(item_features_tensor)
    
    print(f"统一表示形状: {unified_representations.shape}")  # (2732, 256)
    print(f"统一表示数据类型: {unified_representations.dtype}")
    
    return unified_representations.detach().numpy()

def prepare_training_data(user_sequences, seq_length=20):
    """准备训练数据：从序列创建输入-目标对"""
    print("\n" + "=" * 60)
    print("准备训练数据...")
    print("=" * 60)
    
    inputs = []
    targets = []
    
    for seq in user_sequences:
        # 移除非零元素（填充部分）
        valid_seq = seq[seq != 0]
        
        # 至少需要2个商品才能创建训练样本
        if len(valid_seq) >= 2:
            for i in range(1, len(valid_seq)):
                input_seq = valid_seq[:i]  # 历史序列
                target = valid_seq[i]      # 下一个商品
                
                # 填充到固定长度（seq_length-1）
                padded_input = np.zeros(seq_length-1, dtype=np.int32)
                if len(input_seq) > seq_length-1:
                    # 保留最近的部分
                    padded_input[:] = input_seq[-(seq_length-1):]
                else:
                    # 填充到左边
                    padded_input[-len(input_seq):] = input_seq
                
                inputs.append(padded_input)
                targets.append(target)
    
    inputs_array = np.array(inputs)
    targets_array = np.array(targets)
    
    print(f"生成的训练样本数: {len(inputs_array)}")
    print(f"输入序列形状: {inputs_array.shape}")    # (N, 19)
    print(f"目标形状: {targets_array.shape}")       # (N,)
    
    # 统计目标分布
    unique_targets, target_counts = np.unique(targets_array, return_counts=True)
    print(f"唯一目标商品数: {len(unique_targets)}")
    print(f"最热门10个商品的出现次数:")
    top_indices = np.argsort(-target_counts)[:10]
    for idx in top_indices:
        print(f"  商品索引 {unique_targets[idx]}: {target_counts[idx]}次")
    
    return inputs_array, targets_array

def create_dataset_and_dataloader(inputs, targets, item_features, batch_size=128):
    """创建PyTorch数据集和数据加载器"""
    print("\n" + "=" * 60)
    print("创建数据集和数据加载器...")
    print("=" * 60)
    
    class RecommendationDataset(Dataset):
        def __init__(self, inputs, targets, item_features):
            self.inputs = inputs
            self.targets = targets
            self.item_features = item_features
        
        def __len__(self):
            return len(self.inputs)
        
        def __getitem__(self, idx):
            input_seq = self.inputs[idx]
            target = self.targets[idx]
            
            # 获取输入序列的商品特征
            # 注意：输入序列可能包含0（填充），需要处理
            input_features = np.zeros((len(input_seq), self.item_features.shape[1]))
            for i, item_idx in enumerate(input_seq):
                if item_idx != 0:  # 非填充值
                    input_features[i] = self.item_features[item_idx]
            
            return {
                'input_sequence': torch.LongTensor(input_seq),
                'input_features': torch.FloatTensor(input_features),
                'target': torch.LongTensor([target])
            }
    
    # 创建数据集
    dataset = RecommendationDataset(inputs, targets, item_features)
    print(f"数据集大小: {len(dataset)}")
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0  # Windows上设置为0
    )
    
    # 测试一个批次
    for batch in dataloader:
        print(f"批次输入序列形状: {batch['input_sequence'].shape}")  # (batch_size, 19)
        print(f"批次输入特征形状: {batch['input_features'].shape}")  # (batch_size, 19, feature_dim)
        print(f"批次目标形状: {batch['target'].shape}")              # (batch_size, 1)
        break
    
    return dataset, dataloader

def validate_data_integrity(features_dict):
    """验证数据完整性"""
    print("\n" + "=" * 60)
    print("验证数据完整性...")
    print("=" * 60)
    
    # 检查特征维度一致性
    n_items = features_dict['image_features'].shape[0]
    assert features_dict['text_features'].shape[0] == n_items
    assert features_dict['id_embeddings'].shape[0] == n_items
    assert len(features_dict['item_ids']) == n_items
    
    print(f"✓ 特征维度一致: {n_items}个商品")
    
    # 检查用户序列维度
    n_users = features_dict['user_sequences'].shape[0]
    assert len(features_dict['user_ids']) == n_users
    print(f"✓ 用户维度一致: {n_users}个用户")
    
    # 检查映射一致性
    id_to_idx = features_dict['id_to_idx']
    item_ids = features_dict['item_ids']
    for i, item_id in enumerate(item_ids):
        assert id_to_idx[item_id] == i, f"映射不一致: {item_id}"
    print(f"✓ ID映射一致性验证通过")
    
    # 检查序列索引有效性
    user_sequences = features_dict['user_sequences']
    max_item_idx = n_items - 1
    
    # 统计有效和无效索引
    valid_mask = (user_sequences >= 0) & (user_sequences <= max_item_idx)
    zero_mask = (user_sequences == 0)
    
    valid_count = np.sum(valid_mask)
    zero_count = np.sum(zero_mask)
    invalid_count = np.sum(~valid_mask & ~zero_mask)
    
    print(f"✓ 序列索引统计:")
    print(f"  有效索引: {valid_count}")
    print(f"  填充零值: {zero_count}")
    print(f"  无效索引: {invalid_count}")
    
    if invalid_count > 0:
        print(f"⚠ 警告: 发现{invalid_count}个无效索引")
        # 找出无效索引
        invalid_indices = np.where(~valid_mask & ~zero_mask)
        print(f"  示例无效索引位置: {list(zip(invalid_indices[0][:5], invalid_indices[1][:5]))}")
    
    print("数据完整性验证完成!")

def main():
    """主函数"""
    print("多模态推荐系统 - 文件使用示例")
    print("=" * 60)
    
    try:
        # 步骤1: 加载所有文件
        features_dict = load_all_features()
        
        # 步骤2: 验证数据完整性
        validate_data_integrity(features_dict)
        
        # 步骤3: 创建商品统一表示（第二阶段）
        unified_features = create_unified_representations(features_dict)
        
        # 步骤4: 准备训练数据
        inputs, targets = prepare_training_data(features_dict['user_sequences'])
        
        # 步骤5: 创建数据集和数据加载器
        dataset, dataloader = create_dataset_and_dataloader(
            inputs, targets, unified_features, batch_size=32
        )
        
        # 步骤6: 展示数据使用示例
        print("\n" + "=" * 60)
        print("数据使用示例")
        print("=" * 60)
        
        # 示例1: 获取特定商品的特征
        item_idx = 100  # 示例商品索引
        print(f"示例商品索引 {item_idx} 的特征:")
        print(f"  商品ID: {features_dict['item_ids'][item_idx]}")
        print(f"  统一表示维度: {unified_features[item_idx].shape}")
        print(f"  统一表示前5维: {unified_features[item_idx][:5]}")
        
        # 示例2: 获取特定用户的序列
        user_idx = 500  # 示例用户索引
        print(f"\n示例用户索引 {user_idx} 的序列:")
        print(f"  用户ID: {features_dict['user_ids'][user_idx]}")
        seq = features_dict['user_sequences'][user_idx]
        valid_seq = seq[seq != 0]
        print(f"  有效序列长度: {len(valid_seq)}")
        print(f"  序列商品索引: {valid_seq}")
        
        # 将索引转换为商品ID
        item_ids = features_dict['item_ids']
        if len(valid_seq) > 0:
            item_id_list = [item_ids[idx] for idx in valid_seq if idx < len(item_ids)]
            print(f"  序列商品ID: {item_id_list[:5]}...")
        
        # 示例3: 创建训练批次
        print("\n训练批次示例:")
        batch_count = 0
        for batch in dataloader:
            if batch_count >= 2:  # 只展示2个批次
                break
            print(f"  批次{batch_count+1}:")
            print(f"    输入序列形状: {batch['input_sequence'].shape}")
            print(f"    输入特征形状: {batch['input_features'].shape}")
            print(f"    目标形状: {batch['target'].shape}")
            batch_count += 1
        
        print("\n" + "=" * 60)
        print("示例运行完成！")
        print("=" * 60)
        print("\n下一步建议:")
        print("1. 使用上述数据加载器训练扩散模型")
        print("2. 参考USER_GUIDE.md获取完整文档")
        print("3. 查看stage1.py了解数据生成过程")
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        print("\n请确保:")
        print("1. 已运行第一阶段生成所有文件")
        print("2. 文件位于正确目录")
        print("3. 已安装所有依赖 (uv sync)")

if __name__ == '__main__':
    main()