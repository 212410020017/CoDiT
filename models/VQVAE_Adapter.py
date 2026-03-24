import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizerEMA(nn.Module):
    """
    带 EMA 更新和死码复活 (Dead Code Restart) 的量化器
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=1.0, decay=0.99, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        self.decay = decay
        self.epsilon = epsilon
        
        # 注册 buffer，不参与梯度下降
        self.register_buffer('embeddings', torch.randn(num_embeddings, embedding_dim))
        self.register_buffer('cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('embed_avg', self.embeddings.clone())
        
    def forward(self, inputs, is_training=True):
        # inputs: [Batch, Sequence_Length, Channel/Dim]
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.reshape(-1, self.embedding_dim)
        
        # 计算距离
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embeddings**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embeddings.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize
        quantized = torch.matmul(encodings, self.embeddings).view(input_shape)
        
        # Training 阶段执行 EMA 更新和死码复活
        if self.training and is_training:
            with torch.no_grad():
                # 1. 计算统计量
                curr_cluster_size = torch.sum(encodings, 0)
                curr_embed_avg = torch.matmul(encodings.t(), flat_input)
                
                # 2. EMA 更新
                self.cluster_size.data.mul_(self.decay).add_(curr_cluster_size, alpha=1 - self.decay)
                self.embed_avg.data.mul_(self.decay).add_(curr_embed_avg, alpha=1 - self.decay)
                
                # 3. 死码复活 (Dead Code Restart)
                # 阈值：如果某个 Code 的 cluster_size < 1.0 (长期未被选中)
                dead_codes = self.cluster_size < 1.0
                num_dead = dead_codes.sum().item()
                if num_dead > 0:
                    n_vectors = flat_input.shape[0]
                    # 确保我们要复活的数量不会超过当前 batch 拥有的向量数
                    num_to_restart = min(num_dead, n_vectors)
                    
                    # 找到需要被替换的死码的实际索引
                    dead_indices = torch.where(dead_codes)[0][:num_to_restart]
                    
                    # 从当前输入中随机抽样来覆盖它们
                    rand_indices = torch.randperm(n_vectors, device=inputs.device)[:num_to_restart]
                    self.embed_avg.data[dead_indices] = flat_input[rand_indices]
                    self.cluster_size.data[dead_indices] = 1.0 # 重置计数
                    
                # 4. 归一化更新 Embeddings
                n = self.cluster_size.sum()
                cluster_size = (self.cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n
                embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
                self.embeddings.data.copy_(embed_normalized)
        
        # Loss 计算 (只有 Commitment Loss)
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self.commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        
        return quantized, loss, encoding_indices

class LinearVQVAE_Adapter(nn.Module):
    """
    双流适配器：Trend (Linear) + Detail (VQ)
    """
    def __init__(self, args):
        super(LinearVQVAE_Adapter, self).__init__()
        
        # 参数提取
        input_dim = 1 # 时序数据通常按通道独立处理，所以输入维度为 1
        hidden_dim = getattr(args, 'vq_hidden_dim', 64)
        n_embed = getattr(args, 'vq_n_embed', 128)
        beta = getattr(args, 'vq_beta', 1.0)
        decay = getattr(args, 'vq_decay', 0.99)
        
        # Path A: Linear Stream (Trend)
        self.linear_stream = nn.Linear(input_dim, input_dim)
        
        # Path B: VQ Stream (Detail)
        # Encoder: Conv1d [B, C, L] -> [B, Hidden, L]
        self.vq_encoder = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.GELU()
        )
        
        # Quantizer
        self.quantizer = VectorQuantizerEMA(num_embeddings=n_embed, 
                                            embedding_dim=hidden_dim, 
                                            commitment_cost=beta, 
                                            decay=decay)
        
        # Decoder: [B, Hidden, L] -> [B, C, L]
        self.vq_decoder = nn.Conv1d(in_channels=hidden_dim, out_channels=input_dim, kernel_size=1)
        
        # 融合系数 (初始较小，让 Linear 主导)
        # self.vq_scale = nn.Parameter(torch.tensor(0.1)) 
        # 找到 LinearVQVAE_Adapter 的 __init__，修改这行：
        self.vq_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        x shape: [Batch*Channels, Length, 1]
        """
        # --- Path A: Linear Stream ---
        trend = self.linear_stream(x)
        
        # --- Path B: VQ Stream ---
        # Conv1d 需要 [B, Dim, Length]
        x_permuted = x.permute(0, 2, 1) # [BC, 1, L]
        
        # Encoder
        z = self.vq_encoder(x_permuted) # [BC, Hidden, L]
        
        # 变换维度适应 Quantizer: [BC, L, Hidden]
        z = z.permute(0, 2, 1)
        
        # Quantization
        z_q, vq_loss, _ = self.quantizer(z, is_training=self.training)
        
        # Decoder
        z_q = z_q.permute(0, 2, 1) # [BC, Hidden, L]
        detail = self.vq_decoder(z_q) # [BC, 1, L]
        
        # 转回 [BC, L, 1]
        detail = detail.permute(0, 2, 1)
        
        # --- Fusion ---
        output = trend + self.vq_scale * detail
        
        return output, vq_loss