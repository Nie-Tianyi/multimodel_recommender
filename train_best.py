#!/usr/bin/env python3
"""
使用最佳超参数进行训练
"""

import torch
import numpy as np
from stage2 import DiffusionRecommender, load_features, ItemDataset, train_epoch
from torch.utils.data import DataLoader

def main():
    print("使用最佳超参数训练")
    print("=" * 60)
    
    # 最佳超参数（从调优结果中获取）
    best_params = {
        'batch_size': 128,
        'lr': 5e-5,
        'lambda_contrastive': 0.1,
        'temperature': 0.1,  # 对比损失温度
        'num_epochs': 20     # 先训练20个epoch观察
    }
    
    print("超参数配置:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    
    # 加载数据
    data = load_features()
    image_features = data['image_features'].to(device)
    text_features = data['text_features'].to(device)
    id_features = data['id_features'].to(device)
    
    # 初始化模型
    model = DiffusionRecommender().to(device)
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 创建数据集和数据加载器
    dataset = ItemDataset(image_features, text_features, id_features)
    dataloader = DataLoader(dataset, batch_size=best_params['batch_size'], 
                           shuffle=True, num_workers=0)
    
    # 初始化优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=best_params['lr'])
    
    # 训练历史记录
    history = {
        'total_loss': [],
        'contrastive_loss': [],
        'diffusion_loss': []
    }
    
    print(f"\n开始训练，共{best_params['num_epochs']}个epoch")
    print("=" * 60)
    
    # 训练循环
    for epoch in range(best_params['num_epochs']):
        # 注意：train_epoch使用默认温度0.1，与我们的配置一致
        avg_loss, avg_contrastive, avg_diffusion = train_epoch(
            model, dataloader, optimizer, device, best_params['lambda_contrastive']
        )
        
        history['total_loss'].append(avg_loss)
        history['contrastive_loss'].append(avg_contrastive)
        history['diffusion_loss'].append(avg_diffusion)
        
        print(f"Epoch {epoch+1}/{best_params['num_epochs']}: "
              f"总损失={avg_loss:.4f}, 对比损失={avg_contrastive:.4f}, 扩散损失={avg_diffusion:.4f}")
        
        # 每5个epoch保存一次检查点
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f'best_model_epoch{epoch+1}.pth'
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  检查点已保存到 {checkpoint_path}")
    
    # 保存最终模型
    torch.save(model.state_dict(), 'best_model_final.pth')
    print(f"\n最终模型已保存到 best_model_final.pth")
    
    # 分析训练结果
    print(f"\n训练结果分析:")
    print(f"初始总损失: {history['total_loss'][0]:.4f}")
    print(f"最终总损失: {history['total_loss'][-1]:.4f}")
    print(f"总损失下降: {history['total_loss'][0] - history['total_loss'][-1]:.4f}")
    
    # 检查损失趋势
    if len(history['total_loss']) >= 5:
        last_five = history['total_loss'][-5:]
        avg_last_five = np.mean(last_five)
        std_last_five = np.std(last_five)
        print(f"最后5个epoch平均损失: {avg_last_five:.4f} (±{std_last_five:.4f})")
        
        if std_last_five < 0.01:
            print("损失已收敛，训练稳定")
        else:
            print("损失仍有波动，可能需要更多epoch或调整学习率")
    
    # 绘制损失曲线（可选）
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(history['total_loss']) + 1)
        
        plt.subplot(2, 1, 1)
        plt.plot(epochs, history['total_loss'], 'b-', label='总损失')
        plt.plot(epochs, history['diffusion_loss'], 'r-', label='扩散损失')
        plt.xlabel('Epoch')
        plt.ylabel('损失')
        plt.title('训练损失曲线')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(epochs, history['contrastive_loss'], 'g-', label='对比损失')
        plt.xlabel('Epoch')
        plt.ylabel('对比损失')
        plt.title('对比损失曲线')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_losses.png', dpi=100)
        print(f"损失曲线已保存到 training_losses.png")
    except ImportError:
        print("未安装matplotlib，跳过绘制损失曲线")
    
    print(f"\n训练完成!")

if __name__ == '__main__':
    main()