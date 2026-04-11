#!/usr/bin/env python3
"""
基线超参数实验：运行5个epoch观察训练情况
"""

import torch
import numpy as np
from stage2 import DiffusionRecommender, load_features, ItemDataset, train_epoch
from torch.utils.data import DataLoader

def run_experiment(batch_size=128, lr=1e-4, lambda_contrastive=0.1, num_epochs=5):
    """运行单个实验"""
    print(f"\n{'='*60}")
    print(f"实验配置: batch_size={batch_size}, lr={lr}, lambda={lambda_contrastive}")
    print(f"{'='*60}")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载数据
    data = load_features()
    image_features = data['image_features'].to(device)
    text_features = data['text_features'].to(device)
    id_features = data['id_features'].to(device)
    
    # 初始化模型
    model = DiffusionRecommender().to(device)
    
    # 创建数据集和数据加载器
    dataset = ItemDataset(image_features, text_features, id_features)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # 初始化优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # 训练循环
    history = {
        'total_loss': [],
        'contrastive_loss': [],
        'diffusion_loss': []
    }
    
    for epoch in range(num_epochs):
        avg_loss, avg_contrastive, avg_diffusion = train_epoch(
            model, dataloader, optimizer, device, lambda_contrastive
        )
        history['total_loss'].append(avg_loss)
        history['contrastive_loss'].append(avg_contrastive)
        history['diffusion_loss'].append(avg_diffusion)
        
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"总损失={avg_loss:.4f}, 对比损失={avg_contrastive:.4f}, 扩散损失={avg_diffusion:.4f}")
    
    # 保存模型
    torch.save(model.state_dict(), f'baseline_bs{batch_size}_lr{lr}_lam{lambda_contrastive}.pth')
    
    return history, model

def analyze_history(history):
    """分析训练历史"""
    print(f"\n训练历史分析:")
    print(f"最终总损失: {history['total_loss'][-1]:.4f}")
    print(f"最终对比损失: {history['contrastive_loss'][-1]:.4f}")
    print(f"最终扩散损失: {history['diffusion_loss'][-1]:.4f}")
    
    # 检查损失是否下降
    if len(history['total_loss']) > 1:
        loss_change = history['total_loss'][0] - history['total_loss'][-1]
        if loss_change > 0:
            print(f"总损失下降: {loss_change:.4f} (良好)")
        else:
            print(f"总损失上升: {-loss_change:.4f} (需要调整)")
    
    # 检查损失稳定性
    losses = history['total_loss']
    if len(losses) >= 3:
        last_three = losses[-3:]
        var = np.var(last_three)
        if var < 0.01:
            print(f"最后3个epoch损失稳定 (方差={var:.6f})")
        else:
            print(f"最后3个epoch损失波动较大 (方差={var:.6f})")

def main():
    print("基线超参数实验")
    print("=" * 60)
    
    # 运行基线实验
    history, model = run_experiment(
        batch_size=128,
        lr=1e-4,
        lambda_contrastive=0.1,
        num_epochs=5
    )
    
    # 分析结果
    analyze_history(history)
    
    print(f"\n基线实验完成!")
    print("根据结果调整超参数:")
    print("1. 如果损失不下降或波动大 → 降低学习率")
    print("2. 如果对比损失远大于扩散损失 → 降低lambda权重")
    print("3. 如果训练速度慢 → 增加批次大小")
    print("4. 如果内存不足 → 减小批次大小")

if __name__ == '__main__':
    main()