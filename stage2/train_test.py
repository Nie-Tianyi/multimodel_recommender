#!/usr/bin/env python3
"""
训练流程测试：运行一个epoch的简化训练
"""

import torch
import numpy as np
from stage2 import DiffusionRecommender, load_features, ItemDataset, train_epoch
from torch.utils.data import DataLoader

print("训练流程测试")
print("=" * 60)

# 加载数据
data = load_features()
image_features = data['image_features']
text_features = data['text_features']
id_features = data['id_features']

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 初始化模型
model = DiffusionRecommender().to(device)

# 创建数据集和数据加载器（使用较小的批次大小）
dataset = ItemDataset(image_features, text_features, id_features)
# 限制数据加载器只使用前128个样本，以便快速测试
subset_indices = list(range(min(128, len(dataset))))
from torch.utils.data import Subset
subset = Subset(dataset, subset_indices)
dataloader = DataLoader(subset, batch_size=32, shuffle=True, num_workers=0)

# 初始化优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

print(f"数据集大小: {len(dataset)}")
print(f"测试子集大小: {len(subset)}")
print(f"批次大小: 32")
print(f"开始训练一个epoch...")

# 运行一个训练epoch
avg_loss, avg_contrastive, avg_diffusion = train_epoch(
    model, dataloader, optimizer, device, lambda_contrastive=0.1
)

print(f"\n训练完成!")
print(f"平均总损失: {avg_loss:.4f}")
print(f"平均对比损失: {avg_contrastive:.4f}")
print(f"平均扩散损失: {avg_diffusion:.4f}")

# 保存测试模型
torch.save(model.state_dict(), 'stage2_test_model.pth')
print(f"测试模型已保存到 stage2_test_model.pth")

print("\n训练流程测试通过!")