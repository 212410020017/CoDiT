import os
import torch
import warnings
import numpy as np
from copy import deepcopy
from models.model import ClientEncoder, ClientHead
warnings.filterwarnings('ignore')

from utils.metrics import metric

class Engine_Forecasting(object):
    # def __init__(self, args, model, client_encoder, client_head=None):
    def __init__(self, args, model, client_encoder, client_head=None, trained_data_ids=None):
        self.args = args
        self.data_id = args.data_id + '_' + str(args.seq_len) + '_'
        self.info = [self.data_id, args.seq_len, args.stride]
        self.criterion = torch.nn.MSELoss()
        self.train_loaders = None
        self.valid_loaders = None
        self.test_loaders = []
        self.train_batches = 0
        self.train_iter = None
        self.seen_batches = 0
        self.word_embeddings = model.backbone.get_input_embeddings().weight.clone().detach().requires_grad_(True)

        if client_encoder is None:
            self.client_encoder = ClientEncoder(args, self.word_embeddings).to(args.device)
        else:
            self.client_encoder = client_encoder

        if client_head is None:
            self.client_head = None
        else:
            self.client_head = client_head
            self.optimizer_client_head = torch.optim.AdamW(self.client_head.parameters(), lr=self.args.learning_rate,
                                                           weight_decay=self.args.weight_decay)
            self.scheduler_c2 = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_client_head, T_max=50,
                                                                           eta_min=1e-6)
            self._print_trainable_parameters(self.client_head)

        self._print_trainable_parameters(self.client_encoder)

        self.optimizer_client_encoder = torch.optim.AdamW(self.client_encoder.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        self.optimizer_server = torch.optim.AdamW(model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        self.scheduler_c1 = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_client_encoder, T_max=50, eta_min=1e-6)
        self.scheduler_s = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_server, T_max=50, eta_min=1e-6)

        self.trained_data_ids = trained_data_ids or set()  # ★ 新增
        # 判断本客户端是否属于少样本域（参与训练但不是全样本域）
        # 注意：0样本域不会进入 train_split，这个 flag 只对少样本有效
        self.is_fewshot_domain = (self.args.data_id not in self.trained_data_ids) \
                                   if self.trained_data_ids else False  # ★ 新增

    def _print_trainable_parameters(self, model):
        freeze = 0
        trainable = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable += param.nelement()
            else:
                freeze += param.nelement()
        self.args.logger.info('Trainable Params: {}, All Params: {}, Percent: {}'.format(
                              trainable, freeze + trainable, trainable / (freeze + trainable)))

    def _client_train_head(self, x_enc, batch_x, batch_y, mean, std, channels):
        self.client_head.train()
        self.optimizer_client_head.zero_grad()
        outputs = self.client_head(x_enc, mean, std, channels)
        f_dim = -1 if self.args.features == 'MS' else 0
        if self.args.max_backcast_len == 0:
            outputs = outputs[:, :self.args.pred_len, f_dim:]
            batch_y = batch_y[..., f_dim:]
        elif self.args.max_forecast_len == 0:
            outputs = outputs[:, self.args.max_backcast_len - self.args.seq_len:, f_dim:]
            batch_y = batch_x[..., f_dim:]
        else:
            outputs = outputs[:, self.args.max_backcast_len - self.args.seq_len:
                                 self.args.max_backcast_len + self.args.pred_len, f_dim:]
            batch_y = torch.cat((batch_x, batch_y), dim=1)  # bs, seq_len+pred_len, channels
            batch_y = batch_y[..., f_dim:]

        loss = self.criterion(outputs, batch_y)
        loss.backward()
        x_enc_hat = x_enc.grad.clone().detach()

        if self.args.clip != 0:
            torch.nn.utils.clip_grad_norm_(self.client_head.parameters(), self.args.clip)
        self.optimizer_client_head.step()

        return x_enc_hat, loss.item()

    def train_batch_split(self, model, embed_state):
        if self.seen_batches % self.train_batches == 0:
            self.train_iter = iter(self.train_loaders)
        batch = next(self.train_iter)
        self.client_encoder.train()
        model.train()

        batch_x, batch_y = batch
        batch_x = batch_x.float().to(self.args.device) 
        batch_y = batch_y.float().to(self.args.device) 

        b, t, n = batch_x.shape
        mask = torch.rand((b, t, n)).to(self.args.device)
        mask[mask < self.args.mask_rate] = 0
        mask[mask >= self.args.mask_rate] = 1
        inp = batch_x.masked_fill(mask == 0, 0)

        # 1. Client embedding forward (解包 5 个返回值)
        self.optimizer_client_encoder.zero_grad()
        embeds, mean, std, channels, vq_loss = self.client_encoder(self.info, inp, mask)
        embeds1 = embeds.clone().detach().requires_grad_(True)

        # 2. Server encoder forward
        self.optimizer_server.zero_grad()
        x_enc = model(self.info, embeds1)
        x_enc1 = x_enc.clone().detach().requires_grad_(True)

        # 3. 动态初始化 Client Head
        if self.client_head is None:
            self.client_head = ClientHead(self.args, x_enc.shape[1]).to(self.args.device)
            self.optimizer_client_head = torch.optim.AdamW(self.client_head.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
            self.scheduler_c2 = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_client_head, T_max=50, eta_min=1e-6)
            self._print_trainable_parameters(self.client_head)

        # 4. Client head training & Backward (Task Loss)
        self.client_head.train()
        x_enc_hat, loss = self._client_train_head(x_enc1, batch_x, batch_y, mean, std, channels)

        # 5. Server encoder backward
        x_enc.backward(x_enc_hat)
        embeds_hat = embeds1.grad.clone().detach()
        if self.args.clip != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip)
        self.optimizer_server.step()

        # 6. Client embed backward (增加 VQ Loss 的反向传播)
        # 注意必须 retain_graph=True，因为后面还要反向传播 vq_loss
        embeds.backward(embeds_hat, retain_graph=True)
        if getattr(self.args, 'use_vq', False):
            (vq_loss * 1.0).backward()
            
        if self.args.clip != 0:
            torch.nn.utils.clip_grad_norm_(self.client_encoder.parameters(), self.args.clip)
        self.optimizer_client_encoder.step()

        self.seen_batches = (self.seen_batches + 1) % self.train_batches

        # 返回包含 VQ 更新后的 client_encoder_state
        # return loss, deepcopy(self.client_encoder.state_dict())
        return loss

    def train_split(self, model, set_batches, embed_state):
        # 取消注释这行可以让你看到当前是哪个数据集在训练
        # print(self.data_id) 
        ALL_DATASETS = {'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 
                        'Electricity', 'Weather', 'Exchange', 'Illness'}
        is_fullshot_experiment = (self.trained_data_ids >= ALL_DATASETS)  # 全样本实验
        
        # 只有非全样本实验才启用码本隔离（对全样本流程零影响！）
        if not is_fullshot_experiment:
            local_state = self.client_encoder.state_dict()
            for key in embed_state.keys():
                if 'vq' in key.lower() or 'quantize' in key.lower() or 'codebook' in key.lower():
                    embed_state[key] = local_state[key]
        # ==========================================
        # 核心修改：基于数据量的自适应联邦特征隔离
        # ==========================================
        # 仅在少样本情况下，隔离本地的 VQ 码本防止被大数据集冲刷成噪声
        # if self.args.percent < 100:
        #     local_state = self.client_encoder.state_dict()
        #     for key in embed_state.keys():
        #         # 只要参数名包含 vq/quantize/embed/codebook，就拒绝全局覆盖，保留本地专属
        #         if 'vq' in key.lower() or 'quantize' in key.lower() or 'codebook' in key.lower():
        #             embed_state[key] = local_state[key]
        # ==========================================
        
        # 将处理好的 embed_state 加载到当前客户端
        self.client_encoder.load_state_dict(embed_state)
        
        epoch_loss = []
        batch = 0
        
        while batch < set_batches:
            # 兼容处理：不管 train_batch_split 返回一个值还是两个值，都只取第一个(loss)
            out = self.train_batch_split(model, embed_state)
            if isinstance(out, tuple):
                batch_loss = out[0]
            else:
                batch_loss = out
                
            epoch_loss.append(batch_loss)
            batch += 1
            
        loss = np.mean(epoch_loss)
        
        # 性能优化：只在整个 set_batches 循环结束后，统一拿一次最新的模型状态进行深度拷贝
        client_encoder_state = deepcopy(self.client_encoder.state_dict())
        
        return loss, client_encoder_state


    def valid_split(self, server_model, embed_state):
        valid_loss = []
        self.client_encoder.eval()
        server_model.eval()
        self.client_head.eval()
        self.client_encoder.load_state_dict(embed_state)
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(self.valid_loaders):
                batch_x = batch_x.float().to(self.args.device)
                batch_y = batch_y.float().to(self.args.device)

                b, t, n = batch_x.shape
                mask = torch.ones((b, t, n)).to(self.args.device)
                # client_embed
                # self.client_encoder.load_state_dict(embed_state)
                embeds, mean, std, channels, _ = self.client_encoder(self.info, batch_x, mask)
                # server_encoder
                x_enc = server_model(self.info, embeds)
                # client_head
                outputs = self.client_head(x_enc, mean, std, channels)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, self.args.max_backcast_len:
                                  self.args.max_backcast_len+self.args.pred_len, f_dim:]
                batch_y = batch_y[..., f_dim:]

                loss = self.criterion(outputs, batch_y)
                valid_loss.append(loss.item())

        valid_loss = np.average(valid_loss)
        return valid_loss

    def test_split(self, server_model, path_head):
        for test_loader in self.test_loaders:
            preds = []
            trues = []
            self.client_encoder.eval()#existed encoder
            server_model.eval()#existed server

            if self.client_head is None:
                # 注: 此处 x_enc 维度获取可能需要推迟或使用已知参数，可以传一个假输入试探
                # 为了不改变框架，可以直接根据 args 初始化
                pass

            with torch.no_grad():
                for i, (batch_x, batch_y) in enumerate(test_loader):
                    batch_x = batch_x.float().to(self.args.device)
                    batch_y = batch_y.float().to(self.args.device)

                    b, t, n = batch_x.shape
                    mask = torch.ones((b, t, n)).to(self.args.device)

                    # client_embed
                    embeds, mean, std, channels, _ = self.client_encoder(self.info, batch_x, mask)
                    # server_encoder
                    x_enc = server_model(self.info, embeds)
                    if self.client_head is None:
                        self.client_head = ClientHead(self.args, x_enc.shape[1]).to(self.args.device)
                    self.client_head.load_state_dict(torch.load(path_head))
                    self.client_head.eval()
                    # client_head
                    outputs = self.client_head(x_enc, mean, std, channels)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, self.args.max_backcast_len:
                                      self.args.max_backcast_len+batch_y.shape[1], f_dim:]
                    batch_y = batch_y[..., f_dim:]

                    outputs = outputs.detach().cpu().numpy()
                    batch_y = batch_y.detach().cpu().numpy()
                    preds.append(outputs)
                    trues.append(batch_y)

            preds = np.concatenate(preds, 0)
            trues = np.concatenate(trues, 0)

            mae, mse, rmse, mape, mspe = metric(preds, trues)
            self.args.logger.info('Setting: {}, MSE: {:.6f}, MAE: {:.6f}'.format(self.data_id+str(batch_y.shape[1]), mse, mae))

            f = open(os.path.join(self.args.checkpoint, 'result_s' + str(self.args.seed) + '.txt'), 'a')
            f.write(self.data_id + '\n')
            f.write('MSE: {}, MAE: {}'.format(mse, mae))
            f.write('\n')
            f.write('\n')
            f.close()

    def update_lr(self):
        self.scheduler_c1.step()
        self.args.logger.info('Update client_encoder learning rate to {}'.format(self.scheduler_c1.get_last_lr()[0]))
        self.scheduler_s.step()
        self.args.logger.info('Update server_learning rate to {}'.format(self.scheduler_s.get_last_lr()[0]))
        self.scheduler_c2.step()
        self.args.logger.info('Update client_head learning rate to {}'.format(self.scheduler_c2.get_last_lr()[0]))
