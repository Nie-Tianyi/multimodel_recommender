#!/usr/bin/env python3
"""
最终训练：使用最佳超参数训练50个epoch，加入学习率衰减
"""

import torch
import numpy as np
from stage2 import DiffusionRecommender, load_features, ItemDataset, train_epoch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

def main():
    print("最终训练：50个epoch，带学习率衰减")
    print("=" * 60)
    
    # 最佳超参数
    best_params = {
        'batch_size': 128,
        'lr': 5e-5,
        'lambda_contrastive': 0.1,
        'num_epochs': 50,
        'lr_decay': True  # 使用余弦退火学习率衰减
    }
    
    print("超参数配置:")
    for key, value in best_params.items():
        if key != 'lr_decay':
            print(f"  {key}: {value}")
    print(f"  学习率衰减: {'是' if best_params['lr_decay'] else '否'}")
    
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
    
    # 创建数据集和数据加载器
    dataset = ItemDataset(image_features, text_features, id_features)
    dataloader = DataLoader(dataset, batch_size=best_params['batch_size'], 
                           shuffle=True, num_workers=0)
    
    # 初始化优化器和学习率调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=best_params['lr'])
    if best_params['lr_decay']:
        scheduler = CosineAnnealingLR(optimizer, T_max=best_params['num_epochs'], 
                                     eta_min=1e-6)
    else:
        scheduler = None
    
    # 训练历史记录
    history = {
        'total_loss': [],
        'contrastive_loss': [],
        'diffusion_loss': [],
        'learning_rate': []
    }
    
    print(f"\n开始训练，共{best_params['num_epochs']}个epoch")
    print("=" * 60)
    
    # 训练循环
    for epoch in range(best_params['num_epochs']):
        avg_loss, avg_contrastive, avg_diffusion = train_epoch(
            model, dataloader, optimizer, device, best_params['lambda_contrastive']
        )
        
        # 记录学习率
        current_lr = optimizer.param_groups[0]['lr']
        
        history['total_loss'].append(avg_loss)
        history['contrastive_loss'].append(avg_contrastive)
        history['diffusion_loss'].append(avg_diffusion)
        history['learning_rate'].append(current_lr)
        
        print(f"Epoch {epoch+1}/{best_params['num_epochs']}: "
              f"总损失={avg_loss:.4f}, 对比损失={avg_contrastive:.4f}, "
              f"扩散损失={avg_diffusion:.4f}, 学习率={current_lr:.2e}")
        
        # 更新学习率
        if scheduler is not None:
            scheduler.step()
        
        # 每10个epoch保存一次检查点
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f'final_model_epoch{epoch+1}.pth'
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  检查点已保存到 {checkpoint_path}")
    
    # 保存最终模型
    torch.save(model.state_dict(), 'final_model.pth')
    print(f"\n最终模型已保存到 final_model.pth")
    
    # 保存训练历史
    np.savez('training_history.npz',
             total_loss=np.array(history['total_loss']),
             contrastive_loss=np.array(history['contrastive_loss']),
             diffusion_loss=np.array(history['diffusion_loss']),
             learning_rate=np.array(history['learning_rate']))
    print(f"训练历史已保存到 training_history.npz")
    
    # 分析训练结果
    print(f"\n训练结果分析:")
    print(f"初始总损失: {history['total_loss'][0]:.4f}")
    print(f"最终总损失: {history['total_loss'][-1]:.4f}")
    print(f"总损失下降: {history['total_loss'][0] - history['total_loss'][-1]:.4f}")
    
    # 检查最后10个epoch的稳定性
    if len(history['total_loss']) >= 10:
        last_ten = history['total_loss'][-10:]
        avg_last_ten = np.mean(last_ten)
        std_last_ten = np.std(last_ten)
        print(f"最后10个epoch平均损失: {avg_last_ten:.4f} (±{std_last_ten:.4f})")
        
        if std_last_ten < 0.01:
            print("✓ 损失已收敛，训练稳定")
        elif std_last_ten < 0.05:
            print("⚠ 损失基本稳定，有轻微波动")
        else:
            print("✗ 损失波动较大，可能需要进一步调整")
    
    # 检查学习率衰减效果
    if best_params['lr_decay']:
        print(f"学习率从 {history['learning_rate'][0]:.2e} 下降到 {history['learning_rate'][-1]:.2e}")
    
    # 生成训练报告
    with open('training_report.txt', 'w') as f:
        f.write("最终训练报告\n")
        f.write("=" * 40 + "\n")
        f.write(f"超参数:\n")
        for key, value in best_params.items():
            f.write(f"  {key}: {value}\n")
        f.write(f"\n训练结果:\n")
        f.write(f"  初始总损失: {history['total_loss'][0]:.4f}\n")
        f.write(f"  最终总损失: {history['total_loss'][-1]:.4f}\n")
        f.write(f"  总损失下降: {history['total_loss'][0] - history['total_loss'][-1]:.4f}\n")
        f.write(f"  最终对比损失: {history['contrastive_loss'][-1]:.4f}\n")
        f.write(f"  最终扩散损失: {history['diffusion_loss'][-1]:.4f}\n")
        if len(history['total_loss']) >= 10:
            last_ten = history['total_loss'][-10:]
            f.write(f"  最后10个epoch平均损失: {np.mean(last_ten):.4f}\n")
            f.write(f"  最后10个epoch损失标准差: {np.std(last_ten):.4f}\n")
        f.write(f"\n模型文件:\n")
        f.write(f"  final_model.pth - 最终模型权重\n")
        f.write(f"  final_model_epoch10.pth - 第10个epoch检查点\n")
        f.write(f"  final_model_epoch20.pth - 第20个epoch检查点\n")
        f.write(f"  final_model_epoch30.pth - 第30个epoch检查点\n")
        f.write(f"  final_model_epoch40.pth - 第40个epoch检查点\n")
        f.write(f"  final_model_epoch50.pth - 第50个epoch检查点\n")
    
    print(f"\n训练报告已保存到 training_report.txt")
    print(f"训练完成!")

if __name__ == '__main__':
    main()