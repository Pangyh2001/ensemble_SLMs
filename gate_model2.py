import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    """
    简单的MLP门控网络 - 4层版本（更深，训练时间更长）
    
    结构：input_dim -> hidden_dim*2 -> hidden_dim -> hidden_dim//2 -> 1
    例如：768 -> 512 -> 256 -> 128 -> 1
    """
    
    def __init__(self, input_dim, hidden_dim=256, dropout=0.1):
        super(SimpleMLP, self).__init__()
        
        # 4层MLP，从hidden_dim的2倍开始
        self.mlp = nn.Sequential(
            # Layer 1: input -> hidden_dim * 2
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Layer 2: hidden_dim * 2 -> hidden_dim
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Layer 3: hidden_dim -> hidden_dim // 2
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Layer 4: hidden_dim // 2 -> 1
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


class GateNetwork(nn.Module):
    """
    门控网络的统一包装器（简化版，只保留SimpleMLP）
    """
    
    def __init__(self, input_dim, gate_type="mlp", hidden_dim=256, 
                 num_heads=8, num_blocks=4, dropout=0.1):
        super(GateNetwork, self).__init__()
        
        self.gate_type = gate_type
        self.input_dim = input_dim
        
        if gate_type == "mlp":
            self.gate = SimpleMLP(input_dim, hidden_dim, dropout)
        else:
            raise ValueError(f"Unknown gate_type: {gate_type}. Only 'mlp' is supported.")
        
    def forward(self, embeddings):
        """
        Args:
            embeddings: (batch_size, input_dim) - 预计算的embedding向量
        Returns:
            scores: (batch_size, 1) - 门控分数 [0, 1]
        """
        return self.gate(embeddings)