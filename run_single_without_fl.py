import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import time
from copy import deepcopy

# 将当前目录添加到路径，确保能引用项目模块
sys.path.append(os.getcwd())

from models.model import ServerEncoder, ClientEncoder, ClientHead
from data_provider.data_factory import data_provider
from utils.logger import get_logger
from utils.metrics import metric

class Config:
    def __init__(self):
        # --- 基础环境配置 ---
        self.gpu = 1
        self.seed = 2036
        self.is_training = 1
        self.num_workers = 0
        
        # --- 数据集配置 (ETTh1) ---
        self.data_id = 'ETTh1'
        self.data_path = 'dataset/ETT-small/ETTh1.csv' 
        self.data_reader = 'ETTh'
        self.features = 'M'       
        self.target = 'OT'
        self.freq = 'h'
        self.seq_len = 96         
        self.label_len = 0        
        self.pred_len = 96        
        self.stride = 8          
        self.batch_size = 64
        self.percent = 100        
        
        # --- 模型配置 (GPT2 + UniTime) ---
        self.model_path = './gpt2_local' 
        self.lm_pretrain_model = 'gpt2'
        self.lm_layer_num = 6
        self.lm_ft_type = 'freeze' 
        
        self.dynamic_prompt = 1
        self.num_tokens = 100
        self.topk = 12
        self.n_heads = 8
        self.mask_rate = -1       
        self.patch_len = 16
        self.max_backcast_len = 96
        self.max_forecast_len = 720 
        self.d_model = 768        
        self.ts_embed_dropout = 0.1
        self.dec_head_dropout = 0.1
        
        # =========== [修改] VQ-VAE 参数 ===========
        self.use_vq = 1            # 开启 Adapter
        self.vq_n_embed = 63      # 小码本 (128/64)
        self.vq_hidden_dim = 1024    # VQ 内部维度 (Conv1d 的 channel)
        self.vq_beta = 1.0         # Commitment Cost
        self.vq_decay = 0.99       # EMA Decay
        # =========================================
        
        # --- 训练参数 ---
        self.train_epochs = 30     
        self.patience = 10         
        self.learning_rate = 1e-4  
        self.weight_decay = 1e-4  
        self.clip = 3.0         
        
        self.checkpoint = 'checkpoint_linear_vq_etth1'
        self.logger = None
        self.device = None

# train_one_batch, evaluate, test, main 函数逻辑大部分保持不变
# 因为新的 Adapter 已经封装在 ClientEncoder 内部，外部调用接口 (return vq_loss) 是兼容的。
# 下面复制并保留关键部分...

def train_one_batch(args, batch, models, optimizers, criterion):
    client_encoder, server_model, client_head = models
    opt_client, opt_server, opt_head = optimizers
    
    batch_x, batch_y = batch
    batch_x = batch_x.float().to(args.device)
    batch_y = batch_y.float().to(args.device)
    b, t, n = batch_x.shape
    mask = torch.ones((b, t, n)).to(args.device)
    
    # Forward
    client_encoder.train()
    opt_client.zero_grad()
    info = [args.data_id, args.seq_len, args.stride]
    
    # 接收 vq_loss
    embeds, mean, std, channels, vq_loss = client_encoder(info, batch_x, mask)
    embeds1 = embeds.clone().detach().requires_grad_(True)
    
    server_model.train()
    opt_server.zero_grad()
    x_enc = server_model(info, embeds1)
    x_enc1 = x_enc.clone().detach().requires_grad_(True)
    
    if client_head is None:
        client_head = ClientHead(args, x_enc.shape[1]).to(args.device)
        opt_head = torch.optim.AdamW(client_head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        optimizers[2] = opt_head 

    client_head.train()
    opt_head.zero_grad()
    outputs = client_head(x_enc1, mean, std, channels)
    f_dim = 0 
    outputs = outputs[:, args.max_backcast_len : args.max_backcast_len + args.pred_len, f_dim:]
    batch_y = batch_y[..., f_dim:] 
    
    task_loss = criterion(outputs, batch_y)
    
    # Backward
    task_loss.backward()
    x_enc_hat = x_enc1.grad.clone().detach()
    if args.clip != 0: torch.nn.utils.clip_grad_norm_(client_head.parameters(), args.clip)
    opt_head.step()
    
    x_enc.backward(x_enc_hat) 
    embeds_hat = embeds1.grad.clone().detach()
    if args.clip != 0: torch.nn.utils.clip_grad_norm_(server_model.parameters(), args.clip)
    opt_server.step()
    
    embeds.backward(embeds_hat, retain_graph=True) 
    
    # VQ Loss Backward (建议权重 1.0)
    if args.use_vq:
        (vq_loss * 1.0).backward()
    
    if args.clip != 0: torch.nn.utils.clip_grad_norm_(client_encoder.parameters(), args.clip)
    opt_client.step()
    
    current_vq_loss = vq_loss.item() if isinstance(vq_loss, torch.Tensor) else 0
    total_loss = task_loss.item() + current_vq_loss
    
    return total_loss, client_head

def evaluate(args, loader, models, criterion):
    client_encoder, server_model, client_head = models
    if client_head is None: return np.inf 
    client_encoder.eval()
    server_model.eval()
    client_head.eval()
    total_loss = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.float().to(args.device)
            batch_y = batch_y.float().to(args.device)
            b, t, n = batch_x.shape
            mask = torch.ones((b, t, n)).to(args.device)
            info = [args.data_id, args.seq_len, args.stride]
            
            embeds, mean, std, channels, _ = client_encoder(info, batch_x, mask)
            x_enc = server_model(info, embeds)
            outputs = client_head(x_enc, mean, std, channels)
            outputs = outputs[:, args.max_backcast_len : args.max_backcast_len + args.pred_len, :]
            loss = criterion(outputs, batch_y)
            total_loss.append(loss.item())
    return np.mean(total_loss)

def test(args, loader, models):
    client_encoder, server_model, client_head = models
    client_encoder.eval()
    server_model.eval()
    client_head.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.float().to(args.device)
            batch_y = batch_y.float().to(args.device)
            b, t, n = batch_x.shape
            mask = torch.ones((b, t, n)).to(args.device)
            info = [args.data_id, args.seq_len, args.stride]
            
            embeds, mean, std, channels, _ = client_encoder(info, batch_x, mask)
            x_enc = server_model(info, embeds)
            outputs = client_head(x_enc, mean, std, channels)
            outputs = outputs[:, args.max_backcast_len : args.max_backcast_len + args.pred_len, :]
            preds.append(outputs.detach().cpu().numpy())
            trues.append(batch_y.detach().cpu().numpy())
    preds = np.concatenate(preds, 0)
    trues = np.concatenate(trues, 0)
    mae, mse, rmse, mape, mspe = metric(preds, trues)
    return mse, mae

def main():
    args = Config()
    if torch.cuda.is_available(): args.device = torch.device(f'cuda:{args.gpu}')
    else: args.device = torch.device('cpu')
    if not os.path.exists(args.checkpoint): os.makedirs(args.checkpoint)
    logger = get_logger(args.checkpoint, __name__, 'train_log.txt')
    args.logger = logger
    
    _, train_loader = data_provider(args, 'train')
    _, valid_loader = data_provider(args, 'val')
    _, test_loader = data_provider(args, 'test')
    
    server_model = ServerEncoder(args).to(args.device)
    word_embeddings = server_model.backbone.get_input_embeddings().weight.clone().detach().requires_grad_(True)
    client_encoder = ClientEncoder(args, word_embeddings).to(args.device)
    client_head = None 
    
    opt_server = torch.optim.AdamW(server_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler_s = torch.optim.lr_scheduler.CosineAnnealingLR(opt_server, T_max=args.train_epochs, eta_min=1e-6)
    opt_client = torch.optim.AdamW(client_encoder.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler_c = torch.optim.lr_scheduler.CosineAnnealingLR(opt_client, T_max=args.train_epochs, eta_min=1e-6)
    opt_head = None; scheduler_h = None
    
    criterion = nn.MSELoss()
    best_valid_loss = float('inf')
    early_stop_count = 0
    
    for epoch in range(args.train_epochs):
        epoch_start = time.time()
        train_loss = []
        iter_count = 0
        for batch in train_loader:
            iter_count += 1
            current_models = [client_encoder, server_model, client_head]
            current_opts = [opt_client, opt_server, opt_head]
            loss_val, updated_head = train_one_batch(args, batch, current_models, current_opts, criterion)
            train_loss.append(loss_val)
            
            if client_head is None:
                client_head = updated_head
                opt_head = current_opts[2]
                scheduler_h = torch.optim.lr_scheduler.CosineAnnealingLR(opt_head, T_max=args.train_epochs, eta_min=1e-6)
            
            if (iter_count + 1) % 100 == 0:
                print(f"\tEpoch {epoch+1} | Batch {iter_count} | Loss: {loss_val:.4f}")

        avg_train_loss = np.mean(train_loss)
        avg_valid_loss = evaluate(args, valid_loader, [client_encoder, server_model, client_head], criterion)
        epoch_end = time.time()
        logger.info(f"Epoch: {epoch+1}, Cost: {epoch_end-epoch_start:.2f}s, Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}")
        
        scheduler_s.step(); scheduler_c.step()
        if scheduler_h: scheduler_h.step()
        
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            early_stop_count = 0
            logger.info(f"Validation loss decreased. Saving best model.")
            torch.save(client_encoder.state_dict(), os.path.join(args.checkpoint, 'best_client_enc.pth'))
            torch.save(server_model.state_dict(), os.path.join(args.checkpoint, 'best_server.pth'))
            torch.save(client_head.state_dict(), os.path.join(args.checkpoint, 'best_head.pth'))
        else:
            early_stop_count += 1
            if early_stop_count >= args.patience:
                logger.info("Early stopping triggered.")
                break

    client_encoder.load_state_dict(torch.load(os.path.join(args.checkpoint, 'best_client_enc.pth')))
    server_model.load_state_dict(torch.load(os.path.join(args.checkpoint, 'best_server.pth')))
    client_head.load_state_dict(torch.load(os.path.join(args.checkpoint, 'best_head.pth')))
    mse, mae = test(args, test_loader, [client_encoder, server_model, client_head])
    logger.info(f"Final Test Result -> MSE: {mse:.4f}, MAE: {mae:.4f}")

if __name__ == '__main__':
    main()