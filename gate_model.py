import torch
import torch.nn as nn
import math

class SimpleMLP(nn.Module):
    """
    简单的MLP门控网络 - 基础版本
    """
    
    def __init__(self, input_dim, hidden_dim=256, dropout=0.1):
        super(SimpleMLP, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # 输出 0-1 分数
        )
        
    def forward(self, embeddings):
        """
        Args:
            embeddings: (batch_size, input_dim) - 预计算的embedding向量
        Returns:
            scores: (batch_size, 1) - 门控分数
        """
        return self.mlp(embeddings)


class AttentionGate(nn.Module):
    """
    基于注意力机制的门控网络
    使用多头自注意力来捕捉embedding中的关键信息
    """
    
    def __init__(self, input_dim, hidden_dim=256, num_heads=8, dropout=0.1):
        super(AttentionGate, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # 投影到合适的维度
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # 多头自注意力
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer Norm
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # 最终的分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, embeddings):
        """
        Args:
            embeddings: (batch_size, input_dim)
        Returns:
            scores: (batch_size, 1)
        """
        # 投影到hidden_dim
        x = self.input_proj(embeddings)  # (batch, hidden_dim)
        
        # 添加序列维度用于attention (batch, 1, hidden_dim)
        x = x.unsqueeze(1)
        
        # Self-attention with residual
        attn_out, _ = self.self_attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        # 移除序列维度
        x = x.squeeze(1)  # (batch, hidden_dim)
        
        # 分类
        scores = self.classifier(x)
        
        return scores


class ResidualBlock(nn.Module):
    """残差块"""
    
    def __init__(self, dim, dropout=0.1):
        super(ResidualBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        Args:
            x: (batch, dim)
        Returns:
            out: (batch, dim)
        """
        residual = x
        out = self.block(x)
        out = out + residual  # 残差连接
        out = self.relu(out)
        return out


class DeepResNetGate(nn.Module):
    """
    深层ResNet风格的门控网络
    使用残差连接允许网络变得更深，学习更复杂的模式
    """
    
    def __init__(self, input_dim, hidden_dim=256, num_blocks=4, dropout=0.1):
        super(DeepResNetGate, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 多个残差块
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) 
            for _ in range(num_blocks)
        ])
        
        # 输出头
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, embeddings):
        """
        Args:
            embeddings: (batch_size, input_dim)
        Returns:
            scores: (batch_size, 1)
        """
        # 投影到hidden_dim
        x = self.input_proj(embeddings)
        
        # 通过残差块
        for block in self.res_blocks:
            x = block(x)
        
        # 输出
        scores = self.output_head(x)
        
        return scores


class GateNetwork(nn.Module):
    """
    门控网络的统一包装器
    根据gate_type选择不同的架构
    
    支持三种架构：
    - mlp: 简单的多层感知机（默认）
    - attention: 基于多头自注意力机制
    - resnet: 深层ResNet风格（带残差连接）
    """
    
    def __init__(self, input_dim, gate_type="mlp", hidden_dim=256, 
                 num_heads=8, num_blocks=4, dropout=0.1):
        super(GateNetwork, self).__init__()
        
        self.gate_type = gate_type
        self.input_dim = input_dim
        
        if gate_type == "mlp":
            self.gate = SimpleMLP(input_dim, hidden_dim, dropout)
        elif gate_type == "attention":
            self.gate = AttentionGate(input_dim, hidden_dim, num_heads, dropout)
        elif gate_type == "resnet":
            self.gate = DeepResNetGate(input_dim, hidden_dim, num_blocks, dropout)
        else:
            raise ValueError(f"Unknown gate_type: {gate_type}. Choose from: mlp, attention, resnet")
        
        print(f"Initialized {gate_type.upper()} Gate (input_dim={input_dim}, hidden_dim={hidden_dim})")
        
    def forward(self, embeddings):
        """
        Args:
            embeddings: (batch_size, input_dim) - 预计算的embedding向量
        Returns:
            scores: (batch_size, 1) - 门控分数 [0, 1]
        """
        return self.gate(embeddings)