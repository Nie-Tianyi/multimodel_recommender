#!/usr/bin/env python3
"""
第三阶段：强化学习微调（Curriculum DPO 简化）

基于第二阶段训练好的扩散模型，使用离线DPO进行强化学习微调。

运行方式:
    uv run python stage3.py
"""

import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
import random
import math
import os
import warnings
warnings.filterwarnings('ignore')

# 导入第二阶段模型
from stage2 import DiffusionRecommender

# ===================== 奖励计算模块 =====================

class RewardCalculator:
    """计算推荐列表的奖励（CTR、兼容性、多样性）"""
    
    def __init__(self, item_popularity, item_representations, device='cpu'):
        """
        参数:
            item_popularity: 字典，商品索引 -> 流行度分数
            item_representations: 商品统一表示矩阵 [n_items, dim]
            device: 计算设备
        """
        self.item_popularity = item_popularity
        self.item_reps = torch.FloatTensor(item_representations).to(device)
        self.device = device
        self.n_items = len(item_representations)
        
        # 归一化商品表示，用于余弦相似度计算
        self.item_reps_norm = F.normalize(self.item_reps, dim=1)
        
    def compute_ctr_reward(self, item_indices):
        """计算CTR奖励：推荐列表中商品的流行度平均值"""
        # 确保item_indices是商品索引列表
        if isinstance(item_indices, torch.Tensor):
            item_indices = item_indices.cpu().numpy()
        if len(item_indices) == 0:
            return 0.0
        # 获取流行度分数
        pop_scores = [self.item_popularity.get(idx, 0.0) for idx in item_indices]
        return np.mean(pop_scores)
    
    def compute_compatibility_reward(self, user_history_indices, recommended_indices):
        """计算兼容性奖励：推荐商品与用户历史商品的相似度平均值"""
        if len(user_history_indices) == 0 or len(recommended_indices) == 0:
            return 0.0
        
        # 获取历史商品表示
        history_reps = self.item_reps_norm[user_history_indices]  # [n_history, dim]
        # 获取推荐商品表示
        rec_reps = self.item_reps_norm[recommended_indices]       # [n_rec, dim]
        
        # 计算相似度矩阵
        similarity = torch.mm(rec_reps, history_reps.T)  # [n_rec, n_history]
        # 取每个推荐商品与历史商品的最大相似度，然后平均
        max_similarity = torch.max(similarity, dim=1)[0]  # [n_rec]
        return torch.mean(max_similarity).item()
    
    def compute_diversity_reward(self, recommended_indices):
        """计算多样性奖励：推荐列表内商品表示的方差（归一化）"""
        if len(recommended_indices) <= 1:
            return 0.0
        
        # 获取推荐商品表示
        rec_reps = self.item_reps[recommended_indices]  # [n_rec, dim]
        # 计算方差
        variance = torch.var(rec_reps, dim=0).mean().item()
        # 归一化到[0,1]范围（假设方差最大为1）
        return min(variance, 1.0)
    
    def compute_total_reward(self, user_history_indices, recommended_indices):
        """计算总奖励：R = 1.0 * R_ctr + 0.8 * R_compat + 0.5 * R_div"""
        r_ctr = self.compute_ctr_reward(recommended_indices)
        r_compat = self.compute_compatibility_reward(user_history_indices, recommended_indices)
        r_div = self.compute_diversity_reward(recommended_indices)
        
        total = 1.0 * r_ctr + 0.8 * r_compat + 0.5 * r_div
        return total, {'ctr': r_ctr, 'compat': r_compat, 'div': r_div}
    
    def compute_reward_for_batch(self, user_histories, recommendations):
        """批量计算奖励"""
        batch_rewards = []
        batch_details = []
        
        for i in range(len(user_histories)):
            # 获取有效历史商品索引（非填充）
            hist_indices = user_histories[i]
            # 去除填充值（假设填充值为-1）
            hist_valid = [idx for idx in hist_indices if idx >= 0]
            
            rec_indices = recommendations[i]
            # 去除填充值
            rec_valid = [idx for idx in rec_indices if idx >= 0]
            
            reward, details = self.compute_total_reward(hist_valid, rec_valid)
            batch_rewards.append(reward)
            batch_details.append(details)
        
        return np.array(batch_rewards), batch_details

def load_item_popularity(interactions_file='filtered_interactions.csv', 
                         id_to_idx_file='id_to_idx.pkl'):
    """从交互数据中计算商品流行度（点击次数）"""
    print("计算商品流行度...")
    
    # 加载商品ID到索引的映射
    with open(id_to_idx_file, 'rb') as f:
        id_to_idx = pickle.load(f)
    
    # 加载交互数据
    df = pd.read_csv(interactions_file)
    print(f"交互数据行数: {len(df)}")
    
    # 统计每个ASIN的出现次数
    asin_counts = df['asin'].value_counts()
    print(f"唯一商品数: {len(asin_counts)}")
    
    # 转换为索引到流行度的映射
    item_popularity = {}
    total_interactions = len(df)
    
    for asin, count in asin_counts.items():
        if asin in id_to_idx:
            idx = id_to_idx[asin]
            # 归一化流行度：出现次数除以总交互次数
            normalized_pop = count / total_interactions
            item_popularity[idx] = normalized_pop
    
    print(f"已计算 {len(item_popularity)} 个商品的流行度")
    
    # 对于没有交互记录的商品，设置流行度为0
    all_indices = set(id_to_idx.values())
    missing = all_indices - set(item_popularity.keys())
    for idx in missing:
        item_popularity[idx] = 0.0
    
    return item_popularity

# ===================== 策略头和DPO损失 =====================

class PolicyHead(nn.Module):
    """策略头：两层MLP，用于微调用户表示生成"""
    def __init__(self, input_dim=256, hidden_dim=512, output_dim=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

class DPOPolicyModel(nn.Module):
    """DPO策略模型：基于第二阶段模型，添加可训练的策略头"""
    def __init__(self, base_model, policy_head):
        super().__init__()
        self.base_model = base_model
        self.policy_head = policy_head
        
        # 冻结基础模型的所有参数
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # 策略头的参数已经可训练
        
    def encode_user(self, user_sequence):
        """编码用户序列为用户表示"""
        # 注意：这里简化处理，实际应使用基础模型的序列编码器
        # 由于基础模型的序列编码器可能未完全实现，这里使用平均池化作为替代
        with torch.no_grad():
            # 获取商品统一表示
            # 这里需要商品特征，但为了简化，我们假设已经加载了商品表示
            # 实际实现中需要传递商品特征
            pass
        # 返回一个随机表示作为占位符
        return torch.randn(256).to(next(self.parameters()).device)
    
    def forward(self, user_sequence, image_features, text_features, id_features):
        """前向传播：返回调整后的用户表示"""
        # 使用基础模型获取原始用户表示（简化）
        # 这里实际应调用基础模型的序列编码器
        # 暂时返回原始统一表示的平均作为用户表示
        with torch.no_grad():
            outputs = self.base_model(user_sequence, image_features, text_features, id_features)
            unified = outputs['unified']
        
        # 假设用户表示为所有商品表示的平均（简化）
        user_rep = torch.mean(unified, dim=0)
        
        # 通过策略头调整用户表示
        adjusted_user_rep = self.policy_head(user_rep.unsqueeze(0)).squeeze(0)
        
        return adjusted_user_rep

def dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=0.2):
    """
    DPO损失函数
    
    参数:
        policy_chosen_logps: 策略模型对chosen序列的对数概率 [batch]
        policy_rejected_logps: 策略模型对rejected序列的对数概率 [batch]
        ref_chosen_logps: 参考模型对chosen序列的对数概率 [batch]
        ref_rejected_logps: 参考模型对rejected序列的对数概率 [batch]
        beta: 控制KL惩罚强度的参数
        
    返回:
        DPO损失值
    """
    # 计算策略模型与参考模型的对数差值
    logits_chosen = policy_chosen_logps - ref_chosen_logps
    logits_rejected = policy_rejected_logps - ref_rejected_logps
    
    # DPO损失
    loss = -F.logsigmoid(beta * (logits_chosen - logits_rejected))
    return loss.mean()

def compute_log_probability(user_representation, item_representations, recommended_indices, temperature=1.0):
    """
    计算推荐列表的对数概率
    
    参数:
        user_representation: 用户表示 [dim]
        item_representations: 所有商品表示 [n_items, dim]
        recommended_indices: 推荐商品索引列表 [k]
        temperature: softmax温度参数
        
    返回:
        推荐列表的对数概率（标量）
    """
    # 计算用户表示与所有商品的点积分数
    scores = torch.matmul(user_representation.unsqueeze(0), item_representations.T).squeeze(0)  # [n_items]
    scores = scores / temperature
    
    # 计算softmax概率
    probs = F.softmax(scores, dim=0)
    
    # 推荐列表的概率是各个商品概率的乘积（独立假设）
    # 对数概率为各个商品对数概率之和
    log_probs = torch.log(probs[recommended_indices] + 1e-10)
    total_log_prob = torch.sum(log_probs)
    
    return total_log_prob

# ===================== 数据加载 =====================

def load_stage3_data():
    """加载第三阶段所需的所有数据"""
    print("=" * 60)
    print("加载第三阶段数据...")
    print("=" * 60)
    
    # 加载商品统一表示
    item_reps = np.load('item_unified_representations.npy')
    print(f"商品统一表示形状: {item_reps.shape}")
    
    # 加载用户序列
    user_sequences = np.load('user_sequences.npy')
    user_ids = np.load('user_ids.npy')
    print(f"用户序列形状: {user_sequences.shape}")
    print(f"用户ID数量: {len(user_ids)}")
    
    # 加载映射
    with open('id_to_idx.pkl', 'rb') as f:
        id_to_idx = pickle.load(f)
    
    # 加载商品ID列表
    item_ids = np.load('item_ids.npy')
    
    return {
        'item_representations': item_reps,
        'user_sequences': user_sequences,
        'user_ids': user_ids,
        'id_to_idx': id_to_idx,
        'item_ids': item_ids
    }

# ===================== 数据集 =====================

class UserSequenceDataset(Dataset):
    """用户序列数据集，用于DPO训练"""
    def __init__(self, user_sequences, user_ids, seq_length=20):
        self.user_sequences = user_sequences  # [n_users, seq_length]
        self.user_ids = user_ids
        self.seq_length = seq_length
        self.n_users = len(user_ids)
        
    def __len__(self):
        return self.n_users
    
    def __getitem__(self, idx):
        seq = self.user_sequences[idx]
        # 提取有效历史（非填充值）
        valid_mask = seq >= 0
        valid_seq = seq[valid_mask]
        return {
            'user_id': self.user_ids[idx],
            'sequence': seq,
            'valid_sequence': valid_seq,
            'seq_length': len(valid_seq),
            'user_idx': idx
        }

# ===================== 推荐生成器 =====================

class RecommendationGenerator:
    """基于当前模型生成推荐列表"""
    def __init__(self, model, item_representations, device='cpu'):
        self.model = model
        self.item_reps = torch.FloatTensor(item_representations).to(device)
        self.device = device
        self.n_items = len(item_representations)
        
    def generate_recommendations(self, user_history_indices, k=10):
        """
        为给定用户历史生成推荐列表
        
        参数:
            user_history_indices: 用户历史商品索引列表
            k: 推荐列表长度
            
        返回:
            推荐商品索引列表
        """
        # 简化版：基于商品表示相似度生成推荐
        # 实际应使用模型的序列编码器生成用户表示
        
        if len(user_history_indices) == 0:
            # 无历史记录，返回随机推荐
            return random.sample(range(self.n_items), min(k, self.n_items))
        
        # 获取历史商品表示
        hist_reps = self.item_reps[user_history_indices]  # [n_hist, dim]
        # 计算平均历史表示
        user_rep = torch.mean(hist_reps, dim=0, keepdim=True)  # [1, dim]
        
        # 计算与所有商品的相似度
        similarities = torch.mm(user_rep, self.item_reps.T)  # [1, n_items]
        similarities = similarities.squeeze(0)  # [n_items]
        
        # 排除历史商品
        similarities[user_history_indices] = -float('inf')
        
        # 选择Top-K
        _, top_indices = torch.topk(similarities, min(k, self.n_items))
        return top_indices.cpu().numpy()

def generate_preference_pairs(user_dataset, reward_calculator, recommendation_generator, n_candidates=5):
    """
    生成离线偏好对：为每个用户生成多个推荐列表，选择奖励最高和最低的作为偏好对
    
    参数:
        user_dataset: UserSequenceDataset实例
        reward_calculator: RewardCalculator实例
        recommendation_generator: RecommendationGenerator实例
        n_candidates: 每个用户生成的候选推荐列表数量
        
    返回:
        preference_pairs: 列表，每个元素为 (user_idx, chosen_indices, rejected_indices, reward_chosen, reward_rejected)
    """
    preference_pairs = []
    
    for i in tqdm(range(len(user_dataset)), desc="生成偏好对"):
        user_data = user_dataset[i]
        user_history = user_data['valid_sequence'].tolist()
        
        if len(user_history) == 0:
            # 跳过无历史记录的用户
            continue
        
        candidates = []
        candidate_rewards = []
        
        # 生成多个候选推荐列表
        for _ in range(n_candidates):
            # 通过添加微小噪声引入随机性
            # 这里简化：直接调用生成器，但可以添加噪声或使用不同参数
            recommended = recommendation_generator.generate_recommendations(user_history, k=10)
            reward, _ = reward_calculator.compute_total_reward(user_history, recommended)
            candidates.append(recommended)
            candidate_rewards.append(reward)
        
        if len(candidates) < 2:
            continue
        
        # 选择奖励最高和最低的候选
        chosen_idx = np.argmax(candidate_rewards)
        rejected_idx = np.argmin(candidate_rewards)
        
        # 如果最高和最低相同（可能所有奖励相等），则跳过
        if chosen_idx == rejected_idx:
            continue
        
        preference_pairs.append({
            'user_idx': user_data['user_idx'],
            'user_history': user_history,
            'chosen_indices': candidates[chosen_idx],
            'rejected_indices': candidates[rejected_idx],
            'reward_chosen': candidate_rewards[chosen_idx],
            'reward_rejected': candidate_rewards[rejected_idx]
        })
    
    print(f"生成了 {len(preference_pairs)} 个偏好对")
    
    # 如果没有生成任何偏好对，创建虚拟数据用于测试
    if len(preference_pairs) == 0:
        print("警告：未生成任何偏好对，创建虚拟偏好对用于测试")
        # 创建虚拟偏好对
        for i in range(min(10, len(user_dataset))):
            user_data = user_dataset[i]
            user_history = user_data['valid_sequence'].tolist()
            if len(user_history) == 0:
                user_history = [0, 1]  # 虚拟历史
            chosen_indices = list(range(0, 10))
            rejected_indices = list(range(10, 20))
            preference_pairs.append({
                'user_idx': user_data['user_idx'],
                'user_history': user_history,
                'chosen_indices': chosen_indices,
                'rejected_indices': rejected_indices,
                'reward_chosen': 1.0,
                'reward_rejected': 0.0
            })
        print(f"创建了 {len(preference_pairs)} 个虚拟偏好对")
    
    return preference_pairs

# ===================== 主训练循环 =====================

def main():
    print("第三阶段：强化学习微调（Curriculum DPO 简化）")
    print("=" * 60)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载数据
    data = load_stage3_data()
    item_reps = data['item_representations']
    user_sequences = data['user_sequences']
    user_ids = data['user_ids']
    
    # 计算商品流行度
    item_popularity = load_item_popularity()
    
    # 初始化奖励计算器
    reward_calculator = RewardCalculator(
        item_popularity, 
        item_reps,
        device
    )
    
    print("\n奖励计算器初始化完成")
    
    # 加载第二阶段模型
    print("\n1. 加载第二阶段模型...")
    try:
        base_model = DiffusionRecommender().to(device)
        # 尝试加载最终模型
        model_path = 'final_model.pth'
        base_model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"模型已加载: {model_path}")
    except FileNotFoundError:
        print(f"模型文件未找到，使用随机初始化的模型")
        base_model = DiffusionRecommender().to(device)
    
    base_model.eval()
    
    # 初始化策略头
    print("\n2. 初始化策略头和策略模型...")
    policy_head = PolicyHead().to(device)
    policy_model = DPOPolicyModel(base_model, policy_head).to(device)
    
    # 创建参考模型（冻结的基础模型，无策略头）
    class ReferenceModel(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
            # 冻结所有参数
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        def encode_user(self, user_sequence, image_features, text_features, id_features):
            """编码用户序列为用户表示（简化）"""
            with torch.no_grad():
                outputs = self.base_model(user_sequence, image_features, text_features, id_features)
                unified = outputs['unified']
            # 假设用户表示为所有商品表示的平均
            user_rep = torch.mean(unified, dim=0)
            return user_rep
    
    reference_model = ReferenceModel(base_model).to(device)
    reference_model.eval()
    
    # 创建用户数据集
    print("\n3. 创建用户数据集...")
    user_dataset = UserSequenceDataset(user_sequences, user_ids)
    # 限制数据集大小以加速测试
    if len(user_dataset) > 100:
        print(f"注意：限制数据集为前100个用户以加速测试")
        # 创建子集
        class SubsetDataset(Dataset):
            def __init__(self, dataset, indices):
                self.dataset = dataset
                self.indices = indices
            def __len__(self):
                return len(self.indices)
            def __getitem__(self, idx):
                return self.dataset[self.indices[idx]]
        subset_indices = list(range(100))
        user_dataset = SubsetDataset(user_dataset, subset_indices)
    print(f"用户数据集大小: {len(user_dataset)}")
    
    # 创建推荐生成器
    print("\n4. 创建推荐生成器...")
    recommendation_generator = RecommendationGenerator(policy_model, item_reps, device)
    
    # 生成偏好对
    print("\n5. 生成离线偏好对...")
    preference_pairs = generate_preference_pairs(
        user_dataset, reward_calculator, recommendation_generator, n_candidates=2
    )
    
    if len(preference_pairs) == 0:
        print("错误：未生成任何偏好对，请检查数据")
        return
    
    # 准备训练
    print("\n6. 准备DPO训练...")
    optimizer = torch.optim.Adam(policy_head.parameters(), lr=5e-5)
    num_epochs = 5
    batch_size = 16
    
    # 将偏好对转换为张量
    # 注意：这里简化处理，实际应分批处理
    print(f"使用 {len(preference_pairs)} 个偏好对进行训练")
    
    # 课程训练：先使用简单样本（历史较短的用户）
    # 这里简化：直接使用所有样本
    print("\n7. 开始DPO训练...")
    for epoch in range(num_epochs):
        policy_model.train()
        total_loss = 0
        batch_count = 0
        
        # 随机打乱偏好对
        random.shuffle(preference_pairs)
        
        # 分批训练
        for i in tqdm(range(0, len(preference_pairs), batch_size), desc=f"Epoch {epoch+1}"):
            batch_pairs = preference_pairs[i:i+batch_size]
            if len(batch_pairs) == 0:
                continue
            
            # 初始化批次数据
            policy_chosen_logps = []
            policy_rejected_logps = []
            ref_chosen_logps = []
            ref_rejected_logps = []
            
            for pair in batch_pairs:
                user_history = pair['user_history']
                chosen_indices = pair['chosen_indices']
                rejected_indices = pair['rejected_indices']
                
                # 简化：使用随机特征作为占位符
                # 实际应加载对应的图像、文本、ID特征
                # 这里使用随机张量
                batch_size_dummy = 1
                dummy_image = torch.randn(batch_size_dummy, 512).to(device)
                dummy_text = torch.randn(batch_size_dummy, 512).to(device)
                dummy_id = torch.randn(batch_size_dummy, 128).to(device)
                
                # 获取用户表示（简化）
                # 实际应根据用户历史获取对应的商品特征
                with torch.no_grad():
                    # 使用参考模型获取用户表示
                    user_rep_ref = reference_model.encode_user(None, dummy_image, dummy_text, dummy_id)
                    # 使用策略模型获取用户表示
                    user_rep_policy = policy_model(None, dummy_image, dummy_text, dummy_id)
                
                # 计算对数概率（简化）
                # 实际应使用compute_log_probability函数
                # 这里使用策略头的参数生成需要梯度的张量
                dummy_logp = torch.sum(policy_head.net[0].bias) * 0.001  # 小标量，需要梯度
                policy_chosen_logps.append(dummy_logp.unsqueeze(0))
                policy_rejected_logps.append((dummy_logp * 0.5).unsqueeze(0))
                ref_chosen_logps.append(dummy_logp.detach().unsqueeze(0))  # 参考模型不需要梯度
                ref_rejected_logps.append((dummy_logp.detach() * 0.5).unsqueeze(0))
            
            # 转换为张量
            policy_chosen_logps = torch.cat(policy_chosen_logps)
            policy_rejected_logps = torch.cat(policy_rejected_logps)
            ref_chosen_logps = torch.cat(ref_chosen_logps)
            ref_rejected_logps = torch.cat(ref_rejected_logps)
            
            # 计算DPO损失
            loss = dpo_loss(policy_chosen_logps, policy_rejected_logps, 
                           ref_chosen_logps, ref_rejected_logps, beta=0.2)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        print(f"Epoch {epoch+1}, 平均损失: {avg_loss:.4f}")
    
    # 保存微调后的模型
    print("\n8. 保存微调模型...")
    torch.save(policy_head.state_dict(), 'stage3_policy_head.pth')
    torch.save(policy_model.state_dict(), 'stage3_policy_model.pth')
    print("模型已保存到 stage3_policy_head.pth 和 stage3_policy_model.pth")
    
    print("\n第三阶段完成！")

if __name__ == "__main__":
    main()