import pytorch_lightning as pl
from torch import nn
import torch
import torchmetrics
from transformers import EsmTokenizer,EsmModel
import torch.nn.functional as F
from esm.models.esmc import ESMC
import os

class TextCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, num_classes,maxpool):
        super(TextCNN, self).__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=maxpool)
            )
            for kernel_size in kernel_sizes
        ])

    def forward(self, x):
        x = x.permute(0, 2, 1)#batch representation length
        x = [conv(x) for conv in self.convs]
        x = torch.cat(x, dim=2)
        return x
def set_seed(seed=123):
    """
    设置所有相关随机种子以确保结果可重复
    """
    random.seed(seed)  # Python的random模块
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch
    torch.cuda.manual_seed_all(seed)  # PyTorch GPU
    os.environ['PYTHONHASHSEED'] = str(seed)  # Python哈希种子
    torch.backends.cudnn.deterministic = True  # 确保CUDA的结果是确定的
    torch.backends.cudnn.benchmark = False  # 避免CUDA的基准测试影响随机性
    
class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta
    
    def forward(self, pred, target):
        # 确保pred和target维度一致
        pred = pred.view(-1, 1)
        target = target.view(-1, 1)
        
        # 计算差值
        diff = pred - target
        abs_diff = torch.abs(diff)
        condition = abs_diff <= self.delta
        
        # 分段计算损失
        quadratic = 0.5 * diff ** 2
        linear = self.delta * abs_diff - 0.5 * self.delta ** 2
        
        loss = torch.where(condition, quadratic, linear)
        return loss.mean()

class LogCoshLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target):
        loss = torch.log(torch.cosh(pred - target))
        return torch.mean(loss)

class CustomProteinLoss(nn.Module):
    def __init__(self, mse_weight=0.7, mae_weight=0.3):
        super().__init__()
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
        
    def forward(self, pred, target):
        mse_loss = F.mse_loss(pred, target)
        mae_loss = F.l1_loss(pred, target)
        return self.mse_weight * mse_loss + self.mae_weight * mae_loss

class ClassifierNet(pl.LightningModule):
    def __init__(self):#应该是要上面的Transformer里从encoder里出来的用来做分类
        super(ClassifierNet, self).__init__()
        # 定义参数
        num_features = 512  # 特征数量，也是Transformer的d_model参数
        num_classes = 30  # 类别数量 因为是预测结果也是氨基酸 所以是词表大小 为30
        nhead = 8  # Transformer的头的数量
        num_encoder_layers = 3  # Transformer编码器的层数
        num_decoder_layers = 3  # Transformer解码器的层数
        learning_rate = 0.0001  # 学习率
        num_epochs = 100
        # 初始化模型
        seed = 22
        set_seed(seed)
        self.esm = ESMC.from_pretrained("esmc_300m")
        self.esm_antigen = ESMC.from_pretrained("esmc_300m")
        self.layer1 = nn.Sequential(
            nn.Conv1d(960, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.light_layer1 = nn.Sequential(
            nn.Conv1d(960, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.light_layer2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.antigen_layer1 = nn.Sequential(
            nn.Conv1d(960, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.antigen_layer2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = None#nn.Linear(self.out_channels * len(self.kernel_sizes) * (256 - max(self.kernel_sizes) + 1), 1000)
        self.fc2 = nn.Linear(1000,1)
        self.multihead_attention = nn.MultiheadAttention(embed_dim=128, num_heads=nhead)
        self.multihead_attentio_light = nn.MultiheadAttention(embed_dim=128, num_heads=nhead)
        self.multihead_attentio_antigen = nn.MultiheadAttention(embed_dim=128, num_heads=nhead)
        # 回归任务的评价指标
        self.mse = torchmetrics.MeanSquaredError()
        self.mae = torchmetrics.MeanAbsoluteError()
        self.rmse = torchmetrics.MeanSquaredError(squared=False)  # RMSE
        self.r2score = torchmetrics.R2Score()  # R²决定系数
        self.pearson = torchmetrics.PearsonCorrCoef()  # 皮尔森相关系数
        dtype = torch.float32  # 或 torch.bfloat16
        self.to(dtype)

    def on_train_epoch_start(self):
        # 记录当前使用哪个模型，便于日志记录
        current_model = "esm" if self.current_epoch % 2 == 0 else "esm_antigen"
        self.log("current_model", current_model)
        print(f"Epoch {self.current_epoch}: Using {current_model} model")
        
    def one_hot_encode(self,input_string):
        mapping = {
            'A': 0,
            'T': 1,
            'G': 2,
            'C': 3,
            '<eos>': 4,
            '<sep>': 5,
            '<mask>': 6,
            '<pad>': 7,
        }

        # 将整数转换为one-hot编码
        one_hot_encoded = F.one_hot(input_string, num_classes=len(mapping))

        return one_hot_encoded
    
    def pad_or_truncate_tensor(self,tensor):
        target_length = 1024
        padding_value = [0, 0, 0, 0, 0, 0, 0, 0, 1]
        # 如果张量的长度小于目标长度，那么补齐它
        if tensor.size(0) < target_length:
            padding_length = target_length - tensor.size(0)
            padding_tensor = torch.tensor(padding_value).repeat(padding_length, 1).to(device)
            tensor = torch.cat([tensor, padding_tensor], dim=0)
        # 如果张量的长度大于目标长度，那么截断它
        elif tensor.size(0) > target_length:
            tensor = tensor[:target_length]

        return tensor
    def forward(self, x,y,z):
        embeddings_h = None
        embeddings_l = None
        embeddings_g = None
        device = torch.device("cuda:0")
        for i in range(0,len(x)):
            token = x[i].unsqueeze(0)
            #print(token,self.esm.device)
            if embeddings_h is None:               
                embeddings_h = self.esm(token).embeddings
            else:
                abitembeddings_h = self.esm(token).embeddings
                embeddings_h = torch.cat((embeddings_h, abitembeddings_h), dim=0)
        out_heavy = embeddings_h
        out_heavy = out_heavy.permute(0, 2, 1)#batch representation length
        out_heavy = self.layer1(out_heavy)
        out_heavy = self.layer2(out_heavy)
        for i in range(0,len(y)):
            token = y[i].unsqueeze(0)
            #print(token,self.esm.device)
            if embeddings_l is None:               
                embeddings_l = self.esm(token).embeddings
            else:
                abitembeddings_l = self.esm(token).embeddings
                embeddings_l = torch.cat((embeddings_l, abitembeddings_l), dim=0)
        out_light = embeddings_l
        out_light = out_light.permute(0, 2, 1)
        out_light = self.light_layer1(out_light)
        out_light = self.light_layer2(out_light)
        for i in range(0,len(z)):
            token = z[i].unsqueeze(0)
            #print(token,self.esm.device)
            if embeddings_g is None:               
                embeddings_g = self.esm_antigen(token).embeddings
            else:
                abitembeddings_g = self.esm_antigen(token).embeddings
                embeddings_g = torch.cat((embeddings_g, abitembeddings_g), dim=0)
        out_antigen = embeddings_g
        out_antigen = out_antigen.permute(0, 2, 1)
        out_antigen = self.antigen_layer1(out_antigen)
        out_antigen = self.antigen_layer2(out_antigen)
        out_heavy = out_heavy.permute(2, 0, 1)  # Change the shape to (seq_len, batch, embed_dim)
        out_heavy, attn_weights_h = self.multihead_attention(out_heavy, out_heavy, out_heavy)
        out_heavy = out_heavy.permute(1, 2, 0)  # Change the shape back to (batch, embed_dim, seq_len)
        out_light = out_light.permute(2, 0, 1)  # Change the shape to (seq_len, batch, embed_dim)
        out_light, attn_weights_l = self.multihead_attentio_light(out_light, out_light, out_light)
        out_light = out_light.permute(1, 2, 0)  # Change the shape back to (batch, embed_dim, seq_len)
        out_antigen = out_antigen.permute(2, 0, 1)  # Change the shape to (seq_len, batch, embed_dim)
        out_antigen, attn_weights_g = self.multihead_attentio_antigen(out_antigen, out_antigen, out_antigen)
        out_antigen = out_antigen.permute(1, 2, 0)  # Change the shape back to (batch, embed_dim, seq_len)
        out = torch.cat((out_heavy, out_light,out_antigen), dim=2)
        out = out.reshape(out.size(0), -1)
        if self.fc1 is None:
            self.fc1 = nn.Linear(out.size(1), 1000).to(out.device)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.drop_out(out)
        out = self.fc2(out)
        return out

    def training_step(self, batch, batch_idx):
        self.train()
        x, y,z,label = batch
        x_hat = self.forward(x,y,z)
        # 使用示例
        criterion = HuberLoss(delta=1.0)
        '''
        # 或
        criterion = LogCoshLoss()
        # 或
        criterion = CustomProteinLoss(mse_weight=0.7, mae_weight=0.3)
        '''
            # 调整label维度以匹配x_hat
        label = label.view(-1, 1)  # [100] -> [100, 1]
        loss = criterion(x_hat, label)
        # 计算各项评价指标
        mse = self.mse(x_hat, label)
        mae = self.mae(x_hat, label)
        rmse = self.rmse(x_hat, label)
        r2score = self.r2score(x_hat, label)
        pearson = self.pearson(x_hat, label)
        
        # 记录训练过程中的指标
        self.log('train_loss', loss,  on_epoch=True, prog_bar=True)
        self.log('train_mse', mse, on_epoch=True, prog_bar=True)
        self.log('train_mae', mae, on_epoch=True, prog_bar=True)
        self.log('train_rmse', rmse, on_epoch=True, prog_bar=True)
        self.log('train_r2', r2score, on_epoch=True, prog_bar=True)
        self.log('train_pearson', pearson, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_start(self):
        if self.current_epoch % 2 == 0:
            # 偶数轮：冻结 esm_antigen，解冻 esm
            for param in self.esm_antigen.parameters():
                param.requires_grad = False
            for param in self.esm.parameters():
                param.requires_grad = True
            current_model = "esm (esm_antigen frozen)"
            model_id = 0  # 使用数值表示当前模型
        else:
            # 奇数轮：冻结 esm，解冻 esm_antigen
            for param in self.esm.parameters():
                param.requires_grad = False
            for param in self.esm_antigen.parameters():
                param.requires_grad = True
            current_model = "esm_antigen (esm frozen)"
            model_id = 1  # 使用数值表示当前模型
        
        # 使用数值记录当前模型类型
        self.log("current_model_id", float(model_id))
        print(f"Epoch {self.current_epoch}: Training {current_model}")
        
        # 输出模型参数状态以验证
        esm_trainable = sum(p.numel() for p in self.esm.parameters() if p.requires_grad)
        antigen_trainable = sum(p.numel() for p in self.esm_antigen.parameters() if p.requires_grad)
        print(f"ESM 参数: {esm_trainable:,} trainable")
        print(f"ESM_antigen 参数: {antigen_trainable:,} trainable")
    
    def test_step(self, batch, batch_idx):
        self.eval()
        x, y,z,label = batch
        x_hat = self.forward(x,y,z)
        criterion = HuberLoss(delta=1.0)
            # 调整label维度以匹配x_hat
        label = label.view(-1, 1)  # [100] -> [100, 1]
        loss = criterion(x_hat, label)
    
        # 计算各项评价指标
        mse = self.mse(x_hat, label)
        mae = self.mae(x_hat, label)
        rmse = self.rmse(x_hat, label)
        r2score = self.r2score(x_hat, label)
        pearson = self.pearson(x_hat, label)
        
        # 记录训练过程中的指标
        self.log('test_loss', loss, on_epoch=True, prog_bar=True)
        self.log('test_mse', mse, on_epoch=True, prog_bar=True)
        self.log('test_mae', mae,  on_epoch=True, prog_bar=True)
        self.log('test_rmse', rmse,  on_epoch=True, prog_bar=True)
        self.log('test_r2', r2score, on_epoch=True, prog_bar=True)
        self.log('test_pearson', pearson,  on_epoch=True, prog_bar=True)
        return {"test_loss": loss, "test_r2": r2score, "test_pearson": pearson}

    '''
    def validation_step(self, batch, batch_idx):
        self.eval()
        x, y, z, label = batch
        x_hat = self.forward(x, y, z)
        label = label.view(-1, 1)
        loss = LogCoshLoss()(x_hat, label)  # 与训练一致
        pearson = self.pearson(x_hat, label)
        # 记录验证集指标
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_mse', self.mse(x_hat, label), on_epoch=True)
        self.log("val_pearson", pearson, on_epoch=True, prog_bar=True)
        return pearson
    '''
    def validation_step(self, batch, batch_idx):
        self.eval()
        x, y, z, label = batch
        x_hat = self.forward(x, y, z)
        label = label.view(-1, 1)
        loss = LogCoshLoss()(x_hat, label)  # 与训练一致
        pearson = self.pearson(x_hat, label)
        
        # 记录验证集指标
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_mse', self.mse(x_hat, label), on_epoch=True)
        self.log("val_pearson", pearson, on_epoch=True, prog_bar=True)
        
        # 打印预测值与实际值 (添加此部分)
        if batch_idx == 0:  # 只打印第一个batch，避免过多输出
            print("\n===== 验证集预测与实际值比较 =====")
            for i in range(min(5, len(x_hat))):  # 打印前5个样本
                print(f"样本 {i}: 预测值 = {x_hat[i].item():.4f}, 实际值 = {label[i].item():.4f}, 差值 = {(x_hat[i] - label[i]).item():.4f}")
            
            # 计算整个batch的统计信息
            mean_pred = x_hat.mean().item()
            mean_true = label.mean().item()
            print(f"\n批次统计: 平均预测值 = {mean_pred:.4f}, 平均实际值 = {mean_true:.4f}")
            print(f"预测值范围: [{x_hat.min().item():.4f}, {x_hat.max().item():.4f}]")
            print(f"实际值范围: [{label.min().item():.4f}, {label.max().item():.4f}]")
        
        # 返回更多信息用于epoch_end汇总
        return {"val_loss": loss, "val_pearson": pearson, 
                "pred": x_hat.detach(), "true": label.detach()}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        steplr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size  = 1, gamma = 0.8)#每一步都进行学习率的衰减
        return {"optimizer": optimizer , "lr_scheduler": steplr_scheduler}