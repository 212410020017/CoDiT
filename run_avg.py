import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 强制限制只使用系统上的第 1 张物理卡
import random
import torch
import numpy as np
import argparse

from engines.engine_avg import Engine
from utils.logger import get_logger

# 建议根据你的显卡显存情况调整线程数
torch.set_num_threads(3)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FedGPTime with VQ-Adapter')

    # ==========================================
    # 基础配置 (Basic)
    # ==========================================
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--training_list', type=str, default='execute_list/train_all.csv', help='list of the training tasks')
    parser.add_argument('--inference_list', type=str, default='execute_list/inference_all.csv', help='list of the inference tasks')
    parser.add_argument('--eval_model_path', type=str, default='', help='pretrain model path for evaluation')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--seed', type=int, default=2036, help='random seed')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--label_len', type=int, default=0, help='label length')

    # ==========================================
    # 模型配置 (Model & LLM)
    # ==========================================
    parser.add_argument('--lm_pretrain_model', type=str, default='gpt2', help='pretrain model name')
    parser.add_argument('--lm_layer_num', type=int, default=6, help='language model layer number')
    parser.add_argument('--lm_ft_type', type=str, default='freeze', help='fine-tuning type, options:[freeze: all parameters freeze, fpt: only tune positional embeddings and layernorms, full: full parameters tuning]')
    parser.add_argument('--local_batches', type=str, default='set', help='stable or set')

    parser.add_argument('--dynamic_prompt', type=int, default=1, help='use dynamic prompt instruction')
    parser.add_argument('--num_tokens', type=int, default=100, help='number of tokens for dynamic prompt')
    parser.add_argument('--topk', type=int, default=12, help='num of word embedding selected')
    parser.add_argument('--n_heads', type=int, default=8, help='num heads for attention')
    parser.add_argument('--percent', type=float, default=100, help='ratio of training samples for few shot')

    parser.add_argument('--mask_rate', type=float, default=-1, help='masking rate')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--max_backcast_len', type=int, default=96, help='maximum backcast sequence length')
    parser.add_argument('--max_forecast_len', type=int, default=720, help='maximum forecast sequence length')
    parser.add_argument('--ts_embed_dropout', type=float, default=0.1, help='time series embedding dropout')
    parser.add_argument('--dec_head_dropout', type=float, default=0.1, help='decoder head dropout')

    # ==========================================
    # [新增] VQ-VAE Adapter 配置
    # ==========================================
    parser.add_argument('--use_vq', type=int, default=1, help='Whether to use Linear+VQ Adapter (1=True, 0=False)')
    parser.add_argument('--vq_n_embed', type=int, default=128, help='Codebook size (e.g., 64, 128)')
    parser.add_argument('--vq_hidden_dim', type=int, default=64, help='Hidden dimension inside VQ Adapter')
    parser.add_argument('--vq_beta', type=float, default=1.0, help='Commitment cost for VQ')
    parser.add_argument('--vq_decay', type=float, default=0.99, help='EMA decay rate')

    # ==========================================
    # 优化与训练配置 (Optimization)
    # ==========================================
    parser.add_argument('--num_workers', type=int, default=8, help='data loader num workers')
    parser.add_argument('--train_epochs', type=int, default=50, help='training epochs') # 建议保持30
    parser.add_argument('--patience', type=int, default=10, help='early stop patience') # 建议保持10
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='optimizer learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay') # 也可以尝试 1e-3 以增强泛化
    parser.add_argument('--clip', type=int, default=3.0, help='gradient clipping') # 对应单机脚本的设置

    args = parser.parse_args()

    # set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # set logger
    # 在 checkpoint 名字中加入 VQ 标识，方便区分实验
    vq_tag = f"_vq{args.vq_n_embed}" if args.use_vq else "_no_vq"
    args.checkpoint = 'checkpoint_{}_{}_{}_{}_{}{}'.format(
        args.lm_pretrain_model.lower(), 
        args.lm_ft_type, 
        args.training_list.split('/')[1].split('.')[0], 
        args.lm_layer_num, 
        args.max_backcast_len,
        vq_tag
    )

    if not os.path.exists(args.checkpoint):
        os.makedirs(args.checkpoint)
    
    logger = get_logger(args.checkpoint, __name__, 'record_s' + str(args.seed) + '.log')
    logger.info(args)
    args.logger = logger

    # set engine
    engine = Engine(args)
    
    if args.is_training:
        engine.train()
    else:
        engine.test()