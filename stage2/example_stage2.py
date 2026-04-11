#!/usr/bin/env python3
"""
第二阶段模型使用示例

展示如何：
1. 加载训练好的扩散推荐模型
2. 获取商品的统一表示
3. 使用扩散模型生成新的商品表示
"""

import torch
import numpy as np
from stage2 import DiffusionRecommender, load_features
import torch.nn.functional as F

def load_trained_model(model_path, device='cuda'):
    """加载训练好的模型"""
    model = DiffusionRecommender().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def get_item_representations(model, image_features, text_features, id_features, device='cuda'):
    """获取商品的统一表示和各模态嵌入"""
    with torch.no_grad():
        outputs = model(None, image_features, text_features, id_features)
    return outputs

def generate_from_diffusion(model, num_samples=5, device='cuda'):
    """使用扩散模型从噪声生成商品表示"""
    model.eval()
    with torch.no_grad():
        # 从标准正态分布采样噪声
        x_t = torch.randn(num_samples, 256).to(device)
        
        # 反向扩散过程（简化版：使用DDIM采样）
        # 这里为了示例，只展示单步生成
        # 实际应该迭代多个时间步
        t = torch.zeros(num_samples, dtype=torch.long, device=device)
        # 使用去噪网络预测噪声
        noise_pred = model.denoiser(x_t, t)
        # 粗略估计去噪后的表示
        x0_pred = x_t - noise_pred  # 简化
        
        return x0_pred

def main():
    print("第二阶段模型使用示例")
    print("=" * 60)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载数据
    print("\n1. 加载特征数据...")
    data = load_features()
    image_features = data['image_features'].to(device)
    text_features = data['text_features'].to(device)
    id_features = data['id_features'].to(device)
    
    # 加载训练好的模型（使用测试模型作为示例）
    print("\n2. 加载训练好的模型...")
    model_path = 'stage2_test_model.pth'  # 示例模型，实际应使用完整训练的模型
    try:
        model = load_trained_model(model_path, device)
        print(f"模型已加载: {model_path}")
    except FileNotFoundError:
        print(f"模型文件 {model_path} 未找到，使用随机初始化的模型")
        model = DiffusionRecommender().to(device)
        model.eval()
    
    # 示例1：获取商品表示
    print("\n3. 获取商品统一表示示例...")
    sample_indices = [0, 1, 2, 3, 4]  # 前5个商品
    sample_image = image_features[sample_indices]
    sample_text = text_features[sample_indices]
    sample_id = id_features[sample_indices]
    
    outputs = get_item_representations(model, sample_image, sample_text, sample_id, device)
    
    print(f"商品统一表示形状: {outputs['unified'].shape}")
    print(f"前3个商品统一表示的前5维:")
    for i in range(3):
        print(f"  商品{i}: {outputs['unified'][i][:5].cpu().numpy()}")
    
    # 示例2：计算商品相似度
    print("\n4. 计算商品相似度示例...")
    unified_reps = outputs['unified']
    # 归一化
    unified_norm = F.normalize(unified_reps, dim=1)
    similarity_matrix = torch.mm(unified_norm, unified_norm.T)
    print(f"相似度矩阵形状: {similarity_matrix.shape}")
    print(f"商品0与商品1的相似度: {similarity_matrix[0, 1].item():.4f}")
    print(f"商品0与商品2的相似度: {similarity_matrix[0, 2].item():.4f}")
    
    # 示例3：生成新表示
    print("\n5. 扩散生成示例...")
    generated = generate_from_diffusion(model, num_samples=3, device=device)
    print(f"生成的表示形状: {generated.shape}")
    print(f"生成表示示例（前5维）: {generated[0][:5].cpu().numpy()}")
    
    # 示例4：保存表示用于下游任务
    print("\n6. 保存所有商品统一表示...")
    # 分批处理以避免内存不足
    batch_size = 256
    all_unified = []
    n_items = len(image_features)
    
    with torch.no_grad():
        for i in range(0, n_items, batch_size):
            batch_image = image_features[i:i+batch_size]
            batch_text = text_features[i:i+batch_size]
            batch_id = id_features[i:i+batch_size]
            outputs_batch = model(None, batch_image, batch_text, batch_id)
            all_unified.append(outputs_batch['unified'].cpu())
    
    all_unified_tensor = torch.cat(all_unified, dim=0)
    np.save('item_unified_representations.npy', all_unified_tensor.numpy())
    print(f"统一表示已保存到 item_unified_representations.npy")
    print(f"表示形状: {all_unified_tensor.shape}")
    
    print("\n" + "=" * 60)
    print("示例完成！")
    print("\n下一步建议:")
    print("1. 使用完整训练后的模型替换测试模型")
    print("2. 将统一表示用于推荐任务（如计算用户-商品相似度）")
    print("3. 探索扩散生成用于数据增强")

if __name__ == '__main__':
    main()