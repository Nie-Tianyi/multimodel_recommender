#!/usr/bin/env python3
"""
超参数调优：测试不同温度、lambda和学习率组合
"""

import torch
import numpy as np
from stage2 import DiffusionRecommender, load_features, ItemDataset, multimodal_contrastive_loss, diffusion_elbo_loss
from torch.utils.data import DataLoader
from tqdm import tqdm
import itertools

def train_with_params(batch_size=128, lr=1e-4, lambda_contrastive=0.1, temperature=0.1, num_epochs=3):
    """使用指定超参数训练"""
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
    
    history = {
        'total_loss': [],
        'contrastive_loss': [],
        'diffusion_loss': []
    }
    
    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_contrastive_loss = 0
        total_diffusion_loss = 0
        batch_count = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            # 获取批次数据
            image_feat = batch['image'].to(device)
            text_feat = batch['text'].to(device)
            id_feat = batch['id'].to(device)
            
            # 前向传播
            outputs = model(None, image_feat, text_feat, id_feat)
            unified = outputs['unified']
            img_emb = outputs['image_emb']
            txt_emb = outputs['text_emb']
            id_emb = outputs['id_emb']
            
            # 计算对比损失（使用指定温度）
            contrastive_loss = multimodal_contrastive_loss(img_emb, txt_emb, id_emb, temperature=temperature)
            
            # 扩散训练
            batch_size_actual = unified.shape[0]
            t = model.diffusion.sample_timesteps(batch_size_actual, device)
            x_noisy, noise_true = model.diffusion.add_noise(unified, t)
            noise_pred = model.denoiser(x_noisy, t)
            diffusion_loss = diffusion_elbo_loss(noise_pred, noise_true)
            
            # 总损失
            loss = diffusion_loss + lambda_contrastive * contrastive_loss
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 累加损失
            total_loss += loss.item()
            total_contrastive_loss += contrastive_loss.item()
            total_diffusion_loss += diffusion_loss.item()
            batch_count += 1
            
            pbar.set_postfix({
                'loss': loss.item(),
                'contrastive': contrastive_loss.item(),
                'diffusion': diffusion_loss.item()
            })
        
        # 计算平均损失
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        avg_contrastive = total_contrastive_loss / batch_count if batch_count > 0 else 0
        avg_diffusion = total_diffusion_loss / batch_count if batch_count > 0 else 0
        
        history['total_loss'].append(avg_loss)
        history['contrastive_loss'].append(avg_contrastive)
        history['diffusion_loss'].append(avg_diffusion)
        
        print(f"  Epoch {epoch+1}: 总损失={avg_loss:.4f}, 对比损失={avg_contrastive:.4f}, 扩散损失={avg_diffusion:.4f}")
    
    return history, model

def evaluate_params(history):
    """评估超参数组合"""
    final_total = history['total_loss'][-1]
    final_contrastive = history['contrastive_loss'][-1]
    final_diffusion = history['diffusion_loss'][-1]
    
    # 计算损失下降
    if len(history['total_loss']) > 1:
        loss_decline = history['total_loss'][0] - history['total_loss'][-1]
    else:
        loss_decline = 0
    
    # 计算损失稳定性（最后两个epoch的方差）
    if len(history['total_loss']) >= 2:
        stability = np.var(history['total_loss'][-2:])
    else:
        stability = 0
    
    # 评估指标：损失下降越大越好，最终损失越小越好，稳定性越高越好
    score = loss_decline * 10 - final_total - stability * 100
    
    return {
        'final_total': final_total,
        'final_contrastive': final_contrastive,
        'final_diffusion': final_diffusion,
        'loss_decline': loss_decline,
        'stability': stability,
        'score': score
    }

def main():
    print("超参数调优实验")
    print("=" * 60)
    
    # 定义超参数网格
    param_grid = {
        'temperature': [0.1, 0.5, 1.0],
        'lambda_contrastive': [0.1, 0.05, 0.01],
        'lr': [1e-4, 5e-5, 2e-4]
    }
    
    # 固定参数
    batch_size = 128
    num_epochs = 3
    
    results = []
    
    # 生成所有组合
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))
    
    print(f"总共 {len(combinations)} 种超参数组合")
    print(f"每种组合训练 {num_epochs} 个epoch")
    print("=" * 60)
    
    for i, combo in enumerate(combinations):
        temp, lam, lr = combo
        
        print(f"\n实验 {i+1}/{len(combinations)}")
        print(f"温度={temp}, lambda={lam}, 学习率={lr}")
        
        try:
            history, model = train_with_params(
                batch_size=batch_size,
                lr=lr,
                lambda_contrastive=lam,
                temperature=temp,
                num_epochs=num_epochs
            )
            
            metrics = evaluate_params(history)
            results.append({
                'temperature': temp,
                'lambda_contrastive': lam,
                'lr': lr,
                **metrics
            })
            
            print(f"  结果: 最终损失={metrics['final_total']:.4f}, "
                  f"损失下降={metrics['loss_decline']:.4f}, "
                  f"得分={metrics['score']:.4f}")
            
            # 保存模型
            model_path = f'tune_temp{temp}_lam{lam}_lr{lr}.pth'
            torch.save(model.state_dict(), model_path)
            
        except Exception as e:
            print(f"  错误: {e}")
            results.append({
                'temperature': temp,
                'lambda_contrastive': lam,
                'lr': lr,
                'error': str(e)
            })
    
    # 分析结果
    print(f"\n{'='*60}")
    print("超参数调优结果汇总")
    print("=" * 60)
    
    # 筛选有效结果
    valid_results = [r for r in results if 'score' in r]
    if valid_results:
        # 按得分排序
        sorted_results = sorted(valid_results, key=lambda x: x['score'], reverse=True)
        
        print("\n最佳超参数组合 (按得分排序):")
        for i, res in enumerate(sorted_results[:5]):
            print(f"{i+1}. 温度={res['temperature']}, lambda={res['lambda_contrastive']}, "
                  f"lr={res['lr']}")
            print(f"   最终损失={res['final_total']:.4f}, 对比损失={res['final_contrastive']:.4f}, "
                  f"扩散损失={res['final_diffusion']:.4f}")
            print(f"   损失下降={res['loss_decline']:.4f}, 稳定性={res['stability']:.6f}, "
                  f"得分={res['score']:.4f}")
            print()
        
        # 选择最佳组合
        best = sorted_results[0]
        print(f"\n推荐最佳超参数组合:")
        print(f"  温度: {best['temperature']}")
        print(f"  lambda_contrastive: {best['lambda_contrastive']}")
        print(f"  学习率: {best['lr']}")
        print(f"  批次大小: {batch_size}")
        
        # 保存最佳配置
        with open('best_hyperparams.txt', 'w') as f:
            f.write(f"temperature={best['temperature']}\n")
            f.write(f"lambda_contrastive={best['lambda_contrastive']}\n")
            f.write(f"lr={best['lr']}\n")
            f.write(f"batch_size={batch_size}\n")
            f.write(f"final_loss={best['final_total']:.4f}\n")
            f.write(f"score={best['score']:.4f}\n")
        
        print(f"\n最佳配置已保存到 best_hyperparams.txt")
    else:
        print("没有有效的实验结果")
    
    print(f"\n调优完成!")

if __name__ == '__main__':
    main()