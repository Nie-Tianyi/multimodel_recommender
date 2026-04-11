#!/usr/bin/env python3
"""
第二阶段：轻量级扩散模型预训练

实现多模态对齐和扩散模型训练，基于第一阶段生成的特征文件。

运行方式:
    uv run python stage2.py
"""

import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import faiss
from tqdm import tqdm
import random
import math
import os
import warnings
warnings.filterwarnings('ignore')

# ===================== 模型组件 =====================

class MultimodalAlignment(nn.Module):
    """多模态对齐层：将图像、文本、ID特征分别投影到统一空间"""
    def __init__(self, image_dim=512, text_dim=512, id_dim=128, hidden_dim=512, output_dim=256):
        super().__init__()
        # 三个模态的投影层
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        self.id_proj = nn.Sequential(
            nn.Linear(id_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        # 融合层：将三个投影后的表示融合为统一表示
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, image_feat, text_feat, id_feat):
        # 分别投影
        img_emb = self.image_proj(image_feat)  # [batch, output_dim]
        txt_emb = self.text_proj(text_feat)    # [batch, output_dim]
        id_emb = self.id_proj(id_feat)         # [batch, output_dim]
        
        # 融合（拼接后投影）
        fused = torch.cat([img_emb, txt_emb, id_emb], dim=-1)
        unified = self.fusion(fused)           # [batch, output_dim]
        
        return unified, img_emb, txt_emb, id_emb

class DenoisingMLP(nn.Module):
    """扩散去噪网络：三层MLP，输入为加噪表示和时间步"""
    def __init__(self, input_dim=256, hidden_dim=512, output_dim=256):
        super().__init__()
        # 时间步嵌入
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # 主网络
        self.net = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x, t):
        # x: [batch, input_dim], t: [batch] 整数时间步索引
        # 将时间步归一化到[0,1]范围
        t_norm = t.float() / self.time_embed[0].in_features  # 简单的归一化，实际应为num_timesteps
        # 注意：上面有误，应该使用扩散过程的总步数，但这里未传递。暂时使用常数
        t_norm = t.float() / 1000.0  # 总步数为1000
        t_norm = t_norm.unsqueeze(-1)  # [batch, 1]
        t_embed = self.time_embed(t_norm)
        x_concat = torch.cat([x, t_embed], dim=-1)
        return self.net(x_concat)

class SequenceEncoder(nn.Module):
    """序列编码器：将用户序列编码为用户表示"""
    def __init__(self, item_dim=256, user_dim=512):
        super().__init__()
        self.item_dim = item_dim
        self.user_dim = user_dim
        # 简单的注意力池化
        self.attention = nn.Sequential(
            nn.Linear(item_dim, user_dim),
            nn.Tanh(),
            nn.Linear(user_dim, 1)
        )
        self.output_proj = nn.Linear(item_dim, user_dim)
        
    def forward(self, seq_embeddings, mask=None):
        # seq_embeddings: [batch, seq_len, item_dim]
        # mask: [batch, seq_len] (1 for valid, 0 for padding)
        batch_size, seq_len, _ = seq_embeddings.shape
        
        # 注意力权重
        attn_scores = self.attention(seq_embeddings).squeeze(-1)  # [batch, seq_len]
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(attn_scores, dim=-1)  # [batch, seq_len]
        
        # 加权平均
        weighted = torch.bmm(attn_weights.unsqueeze(1), seq_embeddings).squeeze(1)  # [batch, item_dim]
        user_rep = self.output_proj(weighted)  # [batch, user_dim]
        return user_rep

class DiffusionProcess:
    """扩散过程管理器：处理噪声调度和采样"""
    def __init__(self, beta_start=1e-4, beta_end=0.02, num_timesteps=1000):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        # 线性beta调度
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
    def add_noise(self, x0, t):
        """前向扩散：x0加噪得到xt"""
        # x0: [batch, dim]
        # t: [batch] 时间步索引
        # 将调度参数移动到x0的设备上
        sqrt_alpha = self.sqrt_alphas_cumprod.to(x0.device)[t].unsqueeze(1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod.to(x0.device)[t].unsqueeze(1)
        noise = torch.randn_like(x0)
        xt = sqrt_alpha * x0 + sqrt_one_minus_alpha * noise
        return xt, noise
        
    def sample_timesteps(self, batch_size, device):
        """随机采样时间步"""
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device)

class DiffusionRecommender(nn.Module):
    """扩散推荐模型：整合多模态对齐、序列编码和扩散去噪"""
    def __init__(self, item_dim=256, user_dim=512, diff_steps=1000):
        super().__init__()
        # 多模态对齐层
        self.alignment = MultimodalAlignment(512, 512, 128, 512, item_dim)
        # 序列编码器
        self.sequence_encoder = SequenceEncoder(item_dim, user_dim)
        # 扩散去噪网络
        self.denoiser = DenoisingMLP(item_dim, 512, item_dim)
        # 扩散过程
        self.diffusion = DiffusionProcess(num_timesteps=diff_steps)
        
    def forward(self, user_sequence, image_features, text_features, id_features):
        # 1. 对齐多模态特征
        unified, img_emb, txt_emb, id_emb = self.alignment(
            image_features, text_features, id_features
        )  # unified: [batch_items, item_dim]
        
        # 2. 编码用户序列（这里简化，实际需要根据序列索引获取对应的商品表示）
        # 暂时返回统一表示和各模态嵌入
        return {
            'unified': unified,
            'image_emb': img_emb,
            'text_emb': txt_emb,
            'id_emb': id_emb
        }

# ===================== 损失函数 =====================

def multimodal_contrastive_loss(img_emb, txt_emb, id_emb, temperature=0.1):
    """多模态对比损失：同一商品的不同模态表示应靠近，不同商品的表示应推开"""
    # 输入: img_emb, txt_emb, id_emb 形状均为 [batch, dim]
    batch_size = img_emb.shape[0]
    device = img_emb.device
    
    # 归一化
    img_emb = F.normalize(img_emb, dim=1)
    txt_emb = F.normalize(txt_emb, dim=1)
    id_emb = F.normalize(id_emb, dim=1)
    
    # 拼接所有模态表示，用于计算相似度矩阵
    # 顺序: [img1, txt1, id1, img2, txt2, id2, ...]
    all_embeddings = torch.cat([img_emb, txt_emb, id_emb], dim=0)  # [3*batch, dim]
    
    # 相似度矩阵
    sim_matrix = torch.matmul(all_embeddings, all_embeddings.T) / temperature  # [3*batch, 3*batch]
    
    # 创建标签：每个样本的正样本是同一商品的其他模态
    # 对于第i个商品，其三个模态的索引为 i, i+batch, i+2*batch
    labels = torch.arange(0, 3*batch_size, device=device)
    # 将每个模态的标签映射到其商品索引（0,1,2,...）
    label_map = torch.arange(batch_size, device=device).repeat(3)
    # 对于每个样本，正样本是标签相同的其他样本
    pos_mask = (label_map.unsqueeze(1) == label_map.unsqueeze(0)).float()
    # 将自身置零
    pos_mask.fill_diagonal_(0)
    
    # 计算对比损失（InfoNCE）
    # 分子：与正样本的相似度
    pos_sim = torch.sum(sim_matrix * pos_mask, dim=1) / torch.sum(pos_mask, dim=1).clamp(min=1)
    # 分母：所有样本的相似度（包括正样本和负样本）
    denom = torch.logsumexp(sim_matrix, dim=1)
    loss = -torch.mean(pos_sim - denom)
    
    return loss

def diffusion_elbo_loss(noise_pred, noise_true):
    """扩散模型ELBO损失（简化版MSE）"""
    return F.mse_loss(noise_pred, noise_true)

# ===================== 数据加载 =====================

def load_features():
    """加载所有特征文件，返回分离的特征"""
    print("=" * 60)
    print("加载特征文件...")
    print("=" * 60)
    
    image_features = np.load('../stage1/image_features.npy')
    text_features = np.load('../stage1/text_features.npy')
    id_embeddings = np.load('../stage1/id_embeddings.npy')
    item_ids = np.load('../stage1/item_ids.npy')
    user_sequences = np.load('../stage1/user_sequences.npy')
    user_ids = np.load('../stage1/user_ids.npy')
    
    with open('../stage1/id_to_idx.pkl', 'rb') as f:
        id_to_idx = pickle.load(f)
    
    print(f"图像特征形状: {image_features.shape}")
    print(f"文本特征形状: {text_features.shape}")
    print(f"ID嵌入形状: {id_embeddings.shape}")
    print(f"商品ID数量: {len(item_ids)}")
    print(f"用户序列形状: {user_sequences.shape}")
    print(f"用户ID数量: {len(user_ids)}")
    
    # 转换为PyTorch张量
    image_tensor = torch.FloatTensor(image_features)
    text_tensor = torch.FloatTensor(text_features)
    id_tensor = torch.FloatTensor(id_embeddings)
    
    return {
        'image_features': image_tensor,
        'text_features': text_tensor,
        'id_features': id_tensor,
        'item_ids': item_ids,
        'user_sequences': user_sequences,
        'user_ids': user_ids,
        'id_to_idx': id_to_idx
    }

# ===================== 数据集 =====================

class ItemDataset(Dataset):
    """商品级别数据集，用于多模态对齐和扩散训练"""
    def __init__(self, image_features, text_features, id_features):
        self.image_features = image_features
        self.text_features = text_features
        self.id_features = id_features
        self.n_items = len(image_features)
        
    def __len__(self):
        return self.n_items
    
    def __getitem__(self, idx):
        return {
            'image': self.image_features[idx],
            'text': self.text_features[idx],
            'id': self.id_features[idx],
            'index': idx
        }

# ===================== 负采样 =====================

class NegativeSampler:
    """负样本采样器：简单、中等、困难负样本"""
    def __init__(self, item_features, item_ids, id_to_idx):
        self.item_features = item_features.numpy()  # [n_items, 1152]
        self.item_ids = item_ids
        self.id_to_idx = id_to_idx
        self.n_items = len(item_ids)
        
        # 构建Faiss索引用于相似度搜索
        self.index = faiss.IndexFlatL2(1152)
        self.index.add(self.item_features.astype(np.float32))
        
    def sample_simple(self, positive_idx, n_samples=10):
        """简单负样本：随机采样"""
        all_indices = list(range(self.n_items))
        all_indices.remove(positive_idx)
        return random.sample(all_indices, min(n_samples, len(all_indices)))
    
    def sample_medium(self, positive_idx, n_samples=10):
        """中等难度：特征相似但ID不同"""
        # 查询相似商品
        query = self.item_features[positive_idx:positive_idx+1].astype(np.float32)
        distances, indices = self.index.search(query, n_samples + 1)  # +1 可能包含自身
        # 过滤掉自身
        result = [idx for idx in indices[0] if idx != positive_idx][:n_samples]
        if len(result) < n_samples:
            # 不足则用随机样本补充
            extra = self.sample_simple(positive_idx, n_samples - len(result))
            result.extend(extra)
        return result
    
    def sample_hard(self, positive_idx, n_samples=10):
        """困难负样本：目前先复用中等难度，后续可扩展"""
        return self.sample_medium(positive_idx, n_samples)

# ===================== 训练循环 =====================

def train_epoch(model, dataloader, optimizer, device, lambda_contrastive=0.1):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_contrastive_loss = 0
    total_diffusion_loss = 0
    batch_count = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        # 获取批次数据
        image_feat = batch['image'].to(device)
        text_feat = batch['text'].to(device)
        id_feat = batch['id'].to(device)
        
        # 前向传播：获取统一表示和各模态嵌入
        outputs = model(None, image_feat, text_feat, id_feat)  # user_sequence传入None
        unified = outputs['unified']
        img_emb = outputs['image_emb']
        txt_emb = outputs['text_emb']
        id_emb = outputs['id_emb']
        
        # 计算对比损失
        contrastive_loss = multimodal_contrastive_loss(img_emb, txt_emb, id_emb)
        
        # 扩散训练
        batch_size = unified.shape[0]
        # 采样时间步
        t = model.diffusion.sample_timesteps(batch_size, device)
        # 添加噪声
        x_noisy, noise_true = model.diffusion.add_noise(unified, t)
        # 预测噪声
        noise_pred = model.denoiser(x_noisy, t)
        # 计算扩散损失
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
        
        # 更新进度条
        pbar.set_postfix({
            'loss': loss.item(),
            'contrastive': contrastive_loss.item(),
            'diffusion': diffusion_loss.item()
        })
    
    # 计算平均损失
    avg_loss = total_loss / batch_count if batch_count > 0 else 0
    avg_contrastive = total_contrastive_loss / batch_count if batch_count > 0 else 0
    avg_diffusion = total_diffusion_loss / batch_count if batch_count > 0 else 0
    
    return avg_loss, avg_contrastive, avg_diffusion

def main():
    print("第二阶段：轻量级扩散模型预训练")
    print("=" * 60)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
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
    
    # 创建数据集和数据加载器（商品级别）
    dataset = ItemDataset(image_features, text_features, id_features)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)
    
    # 初始化优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # 训练配置
    num_epochs = 50
    lambda_contrastive = 0.1  # 对比损失权重
    
    print(f"\n开始训练，共{num_epochs}个epoch")
    print(f"批次大小: 128, 学习率: 1e-4, 对比损失权重: {lambda_contrastive}")
    
    # 训练循环
    for epoch in range(num_epochs):
        total_loss, contrastive_loss, diffusion_loss = train_epoch(
            model, dataloader, optimizer, device, lambda_contrastive
        )
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"总损失={total_loss:.4f}, 对比损失={contrastive_loss:.4f}, 扩散损失={diffusion_loss:.4f}")
        
        # 每10个epoch保存一次检查点
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f'stage2_model_epoch{epoch+1}.pth'
            torch.save(model.state_dict(), checkpoint_path)
            print(f"检查点已保存到 {checkpoint_path}")
    
    # 保存最终模型
    torch.save(model.state_dict(), 'stage2_model_final.pth')
    print("\n训练完成！最终模型已保存到 stage2_model_final.pth")

if __name__ == '__main__':
    main()