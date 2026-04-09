#!/usr/bin/env python3
"""
快速测试第二阶段模型
"""

import torch
import numpy as np
from stage2 import DiffusionRecommender, load_features, ItemDataset

print("测试第二阶段模型初始化...")

# 加载数据
data = load_features()
print(f"图像特征形状: {data['image_features'].shape}")
print(f"文本特征形状: {data['text_features'].shape}")
print(f"ID特征形状: {data['id_features'].shape}")

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 初始化模型
model = DiffusionRecommender().to(device)
print("模型创建成功")

# 打印参数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"模型总参数量: {total_params:,}")
print(f"可训练参数量: {trainable_params:,}")

# 测试前向传播
batch_size = 4
image_feat = data['image_features'][:batch_size].to(device)
text_feat = data['text_features'][:batch_size].to(device)
id_feat = data['id_features'][:batch_size].to(device)

print(f"测试批次大小: {batch_size}")
outputs = model(None, image_feat, text_feat, id_feat)
print("前向传播成功!")
print(f"统一表示形状: {outputs['unified'].shape}")
print(f"图像嵌入形状: {outputs['image_emb'].shape}")
print(f"文本嵌入形状: {outputs['text_emb'].shape}")
print(f"ID嵌入形状: {outputs['id_emb'].shape}")

# 测试损失函数
from stage2 import multimodal_contrastive_loss, diffusion_elbo_loss
contrastive_loss = multimodal_contrastive_loss(
    outputs['image_emb'], outputs['text_emb'], outputs['id_emb']
)
print(f"对比损失: {contrastive_loss.item():.4f}")

# 测试扩散过程
t = model.diffusion.sample_timesteps(batch_size, device)
x_noisy, noise_true = model.diffusion.add_noise(outputs['unified'], t)
noise_pred = model.denoiser(x_noisy, t)
diffusion_loss = diffusion_elbo_loss(noise_pred, noise_true)
print(f"扩散损失: {diffusion_loss.item():.4f}")

# 测试数据集
dataset = ItemDataset(data['image_features'], data['text_features'], data['id_features'])
print(f"数据集大小: {len(dataset)}")
sample = dataset[0]
print(f"样本键: {list(sample.keys())}")

print("\n所有测试通过! 模型准备就绪。")