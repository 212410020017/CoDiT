import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import weight_norm

# 假设 RevIN 位于 models 目录下
from models.RevIN import RevIN
# 注意：已移除 CasualTRM 的引用

class VQConfig:
    def __init__(self, args):
        self.entropy_penalty = getattr(args, 'vq_entropy_penalty', 0.005)
        self.entropy_temp = getattr(args, 'vq_entropy_temp', 1.0)

# ==========================================
# 核心组件 (TCN & Quantize)
# ==========================================

# class Quantize(nn.Module):
#     def __init__(self, dim, n_embed, configs, beta=0.25, eps=1e-5):
#         super().__init__()
#         self.dim = dim
#         self.n_embed = n_embed
#         self.beta = beta
#         self.entropy_penalty = configs.entropy_penalty
#         self.entropy_temp = configs.entropy_temp
#         self.eps = eps

#         self.embedding = nn.Embedding(n_embed, dim)
#         nn.init.normal_(self.embedding.weight, mean=0.0, std=dim ** -0.5)
#         self.embedding_proj = nn.Linear(dim, dim)

#     def forward(self, input):
#         B, T, C = input.shape
#         flatten = input.reshape(-1, C)

#         # codebook projection
#         codebook = self.embedding_proj(self.embedding.weight)

#         # compute distance
#         d = torch.sum(flatten ** 2, dim=1, keepdim=True) + \
#             torch.sum(codebook ** 2, dim=1) - 2 * torch.matmul(flatten, codebook.t())

#         # soft assignment
#         logits = -d / self.entropy_temp
#         probs = F.softmax(logits, dim=-1)
#         soft_entropy = -torch.sum(probs * torch.log(probs + self.eps), dim=-1).mean()
#         max_entropy = np.log(self.n_embed)
#         norm_soft_entropy = soft_entropy / max_entropy
#         soft_entropy_loss = self.entropy_penalty * (1.0 - norm_soft_entropy)

#         # hard assignment
#         indices = torch.argmax(probs, dim=-1)
#         z_q = F.embedding(indices, codebook).view(B, T, C)

#         # commitment and embedding loss
#         diff_loss = F.mse_loss(z_q.detach(), input)
#         commit_loss = F.mse_loss(z_q, input.detach())
#         vq_loss = diff_loss + self.beta * commit_loss

#         # token usage entropy
#         with torch.no_grad():
#             one_hot = F.one_hot(indices, num_classes=self.n_embed).float()
#             avg_probs = one_hot.mean(dim=0) + self.eps
#             token_usage_entropy = -torch.sum(avg_probs * torch.log(avg_probs))
#             token_usage_max = torch.log(torch.tensor(self.n_embed, dtype=token_usage_entropy.dtype, device=token_usage_entropy.device))
#             norm_token_usage_entropy = token_usage_entropy / token_usage_max
#             token_entropy_loss = self.entropy_penalty * (1.0 - norm_token_usage_entropy)

#         total_loss = vq_loss + soft_entropy_loss + token_entropy_loss
#         z_q = input + (z_q - input).detach() # straight-through

#         return z_q, total_loss, indices

import torch
import torch.nn as nn
import torch.nn.functional as F

class Quantize(nn.Module):
    def __init__(self, dim, n_embed, configs, decay=0.99, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps
        self.beta = getattr(configs, 'vq_beta', 0.25) # 承诺损失权重

        # 初始化 Embeddings (作为 buffer 注册，不通过梯度更新)
        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        
        # EMA 统计量
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        # input: [B, T, C]
        B, T, C = input.shape
        flatten = input.reshape(-1, self.dim) # [N, C]

        # 1. 计算距离: (x - e)^2 = x^2 + e^2 - 2xe
        # [N, 1] + [1, K] - [N, K] = [N, K]
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )

        # 2. 寻找最近的 Code
        _, embed_ind = (-dist).max(1) # [N]
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype) # [N, K]
        
        # 3. EMA 更新 (仅在训练时)
        if self.training:
            # 统计每个 Code 被选中的次数
            embed_onehot_sum = embed_onehot.sum(0) # [K]
            # 统计每个 Code 对应的输入向量之和
            embed_sum = flatten.transpose(0, 1) @ embed_onehot # [C, K]

            # EMA 更新 cluster_size
            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            
            # EMA 更新 embed_avg
            self.embed_avg.data.mul_(self.decay).add_(
                embed_sum, alpha=1 - self.decay
            )

            # --- 死码复活 (Dead Code Restart) ---
            # 如果某个 Code 的使用率过低，将其重置为当前 Batch 中的随机输入
            # 这是降低 MSE 的关键，防止码本利用率不足
            n_vectors = flatten.shape[0]
            if n_vectors < self.n_embed:
                # 如果 Batch 太小，稍微平滑一下避免除零
                pass 
            else:
                # 阈值：平均使用率的 1/10
                usage = (self.cluster_size.view(1, self.n_embed) / self.cluster_size.sum())
                dead_codes = torch.where(usage < (1.0 / self.n_embed) * 0.1)[1]
                
                if len(dead_codes) > 0:
                    # 随机采样输入数据来替换死码
                    rand_idx = torch.randperm(n_vectors, device=input.device)[:len(dead_codes)]
                    self.embed_avg.data[:, dead_codes] = flatten[rand_idx].transpose(0, 1)
                    self.cluster_size.data[dead_codes] = 1.0 # 重置计数
                    
            # 归一化得到新的 Embeddings
            n = self.cluster_size.sum()
            #加平滑防止除零
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        # 4. 量化 (Quantize)
        embed_ind = embed_ind.view(*input.shape[:-1]) # [B, T]
        # self.embed 是 [C, K]，需要转置为 [K, C] 进行 lookup
        quantize = F.embedding(embed_ind, self.embed.transpose(0, 1)) # [B, T, C]

        # 5. 计算 Loss
        # EMA 更新机制下，Loss 只需要包含 Commitment Loss (Encoder 靠近 Codebook)
        # 也就是 || sg[e] - z ||^2
        loss = F.mse_loss(quantize.detach(), input) * self.beta

        # 6. 直通估计 (Straight-through estimator)
        # 前向传播用 quantize，反向传播梯度传给 input
        quantize = input + (quantize - input).detach()

        return quantize, loss, embed_ind

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, channel_in, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 1 ** i
            in_channels = channel_in if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class Encoder(nn.Module):
    def __init__(self, chan_indep, channel_in, hidden_dim, block_num=3, kernel_size=3, dropout=0.2):
        super().__init__()
        self.chan_indep = chan_indep
        self.TCN = TemporalConvNet(channel_in, [hidden_dim]*block_num, kernel_size=kernel_size, dropout=dropout)
        
    def forward(self, x):
        # x: [B, L, C] -> [B, C, L] for TCN
        x = x.permute(0, 2, 1)
        if self.chan_indep:   
            x = x.reshape(-1, x.shape[-1]).unsqueeze(1) # [B*C, 1, L]
        x = self.TCN(x)
        x = x.permute(0, 2, 1) # [B*C, L, D]
        return x

# ==========================================
# 新的轻量级 Decoder
# ==========================================

class LightweightDecoder(nn.Module):
    """
    替换了原有的 CasualTRM Decoder。
    仅使用转置卷积 (Transposed Conv) 和简单的卷积层来从离散 Token 重构原始时序。
    目的：提供梯度以更新 Codebook，保持计算极其轻量。
    """
    def __init__(self, enc_in, hidden_dim, patch_len):
        super().__init__()
        
        # 1. 上采样 (Upsampling) / 反Patching
        # 对应 Encoder 中的 feature_proj (Conv1d with stride=patch_len)
        # Input: [B, D, Num_Patches] -> Output: [B, D, Seq_Len]
        self.upsample = nn.ConvTranspose1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=patch_len,
            stride=patch_len
        )
        
        # 2. 映射回原始空间 (Projection to Raw Space)
        # 对应 Encoder 中的 TCN，这里用简单的 Conv 层近似逆变换
        # 不需要太复杂，因为真正的预测任务交给 Server 端的 LLM
        self.reconstruct = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim // 2, enc_in, kernel_size=3, padding=1)
        )

    def forward(self, x):
        """
        x: [B, Num_Patches, D]
        return: [B, Seq_Len, enc_in]
        """
        # 转置为 [B, D, Num_Patches] 以适配 ConvTranspose1d
        x = x.permute(0, 2, 1) 
        
        # 上采样: [B, D, Num_Patches] -> [B, D, Seq_Len]
        x = self.upsample(x)
        
        # 重构: [B, D, Seq_Len] -> [B, enc_in, Seq_Len]
        x = self.reconstruct(x)
        
        # 转回: [B, enc_in, Seq_Len] -> [B, Seq_Len, enc_in]
        x = x.permute(0, 2, 1)
        return x

# ==========================================
# 适配器类 (主入口)
# ==========================================

class VQVAE_Adapter(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        # 参数提取
        self.seq_len = args.seq_len
        self.d_model = args.d_model
        self.patch_len = args.patch_len
        self.device = args.device
        
        n_embed = getattr(args, 'vq_n_embed', 1024)
        block_num = getattr(args, 'vq_block_num', 2)
        kernel_size = getattr(args, 'vq_kernel_size', 3)
        dropout = getattr(args, 'vq_dropout', 0.1)
        
        # 配置
        self.chan_indep = True 
        enc_in = 1 
        self.revin = True 
        
        # 1. 编码器 (TCN)
        self.enc = Encoder(
            chan_indep=self.chan_indep,
            channel_in=enc_in, 
            hidden_dim=self.d_model, 
            block_num=block_num, 
            kernel_size=kernel_size, 
            dropout=dropout
        )
        
        # 2. Patching 投影层
        self.feature_proj = nn.Conv1d(
            self.d_model, 
            self.d_model, 
            kernel_size=self.patch_len, 
            stride=self.patch_len
        )
        
        # 3. 量化器
        vq_configs = VQConfig(args)
        self.quantize = Quantize(dim=self.d_model, n_embed=n_embed, configs=vq_configs)
        
        # 4. 解码器 (已替换为轻量级版本)
        self.dec = LightweightDecoder(
            enc_in=enc_in,
            hidden_dim=self.d_model,
            patch_len=self.patch_len
        )
        
        # 5. 归一化
        if self.revin:
            self.revin_layer = RevIN(enc_in, affine=True, subtract_last=False)

    def forward(self, x):
        # 1. 维度适配 [B_total, L, 1]
        if x.dim() == 3 and x.shape[-1] > 1:
            B, L, C = x.shape
            x = x.permute(0, 2, 1).contiguous().view(B * C, L, 1)
        elif x.dim() == 2:
            x = x.unsqueeze(-1)
            
        B_total, L, C = x.shape
        
        # 2. RevIN
        if self.revin:
            x_norm = self.revin_layer(x, 'norm')
        else:
            x_norm = x

        # 3. Encoder: [B, L, D]
        enc_out = self.enc(x_norm) 
        
        # 4. Patching: [B, D, Num_Patches] -> [B, Num_Patches, D]
        enc_out = enc_out.permute(0, 2, 1)
        patch_features = self.feature_proj(enc_out) 
        patch_features = patch_features.permute(0, 2, 1)
        
        # 5. Quantize: z_q [B, Num_Patches, D]
        z_q, vq_loss, indices = self.quantize(patch_features)
        
        # 6. Decoder (Lightweight)
        # 从 z_q 重构回 x_recon [B, Recon_Len, 1]
        dec_out = self.dec(z_q)
        
        # 截断/对齐
        recon_len = dec_out.shape[1]
        if recon_len > L:
            dec_out = dec_out[:, :L, :]
        elif recon_len < L:
            x_norm = x_norm[:, :recon_len, :]
            
        # 7. Loss
        recon_loss = F.mse_loss(dec_out, x_norm)
        total_loss = recon_loss + vq_loss
        
        return z_q, total_loss