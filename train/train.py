import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from pytorch_lightning import Trainer, loggers
import random
from model import ClassifierNet, set_seed  # 添加从model.py导入
'''
class VirusDataset(Dataset):
    def __init__(self, X, y):
        self.X = X#torch.tensor(X, dtype=torch.float32)#因为之前scaler.fit_transform(X)过后是array形状
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
'''
class VirusDataset(Dataset):
    def __init__(self, X, Y,Z,label, max_length=256,max_length_gene=1024):
        self.aacid_to_index = {'<cls>': 0,
                                 '<pad>': 1,
                                 '<eos>': 2,
                                 '<unk>': 3,
                                 'L': 4,
                                 'A': 5,
                                 'G': 6,
                                 'V': 7,
                                 'S': 8,
                                 'E': 9,
                                 'R': 10,
                                 'T': 11,
                                 'I': 12,
                                 'D': 13,
                                 'P': 14,
                                 'K': 15,
                                 'Q': 16,
                                 'N': 17,
                                 'F': 18,
                                 'Y': 19,
                                 'M': 20,
                                 'H': 21,
                                 'W': 22,
                                 'C': 23,
                                 'X': 24,
                                 'B': 25,
                                 'U': 26,
                                 'Z': 27,
                                 'O': 28,
                                 '.': 29,
                                 '-': 30,
                                 '<null_1>': 31,
                                 '<mask>': 32}
        self.start_token = '<cls>'
        self.end_token = '<eos>'
        self.pad_token = '<pad>'
        self.X = [self.tokenize_aacid_sequence(seq, max_length) for seq in X]
        self.Y = [self.tokenize_aacid_sequence(seq, max_length) for seq in Y]
        self.Z = [self.tokenize_aacid_sequence(seq, max_length) for seq in Z]
        self.label = label
    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx],self.Z[idx],self.label[idx]

    def tokenize_aacid_sequence(self, sequence, max_length):
        # 将序列截断或填充到max_length
        sequence = sequence.replace(' ','')
        sequence = [self.aacid_to_index[aacid] for aacid in sequence]
        sequence = [self.aacid_to_index[self.start_token]] + sequence + [self.aacid_to_index[self.end_token]]
        sequence = sequence[:max_length] + [self.aacid_to_index[self.pad_token]] * (max_length - len(sequence))

        # 转换为tensor
        sequence = torch.tensor(sequence, dtype=torch.long)

        return sequence

def main():
    # 读取数据
    selected_columns = pd.read_csv(train_tsv, sep='\t')

    # X是每行的3-6列元素，Y是第一列的元素 第一列是基因
    X = selected_columns.iloc[:, 1].values.reshape(-1, 1).tolist()
    for i in range(len(X)):
        X[i] = ' '.join(X[i])
    Y = selected_columns.iloc[:, 2].values.reshape(-1, 1).tolist()
    for i in range(len(Y)):
        Y[i] = ' '.join(Y[i])
    Z = selected_columns.iloc[:, 3].values.reshape(-1, 1).tolist()
    for i in range(len(Z)):
        Z[i] = ' '.join(Z[i])
    label = selected_columns.iloc[:, 4].values.reshape(-1, 1).tolist()
    for i in range(len(label)):
        label[i] =  label[i][0]
    X_train, X_test, y_train, y_test, Z_train, Z_test,label_train, label_test = train_test_split(X, Y,Z,label, test_size=0.2, random_state=42)

    # 创建数据集
    train_dataset = VirusDataset(X_train, y_train,Z_train,label_train)
    test_dataset = VirusDataset(X_test, y_test,Z_test,label_test)

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    # 创建数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True)


    # 定义参数
    num_features = 512  # 特征数量，也是Transformer的d_model参数
    num_classes = 30  # 类别数量 因为是预测结果也是氨基酸 所以是词表大小 为30
    nhead = 8  # Transformer的头的数量
    num_encoder_layers = 3  # Transformer编码器的层数
    num_decoder_layers = 3  # Transformer解码器的层数
    learning_rate = 0.0001  # 学习率
    num_epochs = 30
    # 初始化模型
    seed = random.randint(0, 10000)
    model = ClassifierNet()
    device = torch.device('cuda:0')
    model = model.to(device)

    csv_logger = loggers.CSVLogger('L_20_AF_lc_ESM3CNN+MUTIATTN_conlogs/')
    # 初始化训练器
    trainer = Trainer(max_epochs=num_epochs,logger = csv_logger,accelerator="gpu", devices=[0])

    # 训练模型
    #tokenized_sequence = tokenize_aacid_sequence(sequence)
    trainer.fit(model, train_dataloaders=train_dataloader,val_dataloaders=val_dataloader)

if __init__ == "__main__":
    train_tsv = "/public/home/ligroupprotein/ckx/affinity/data/final_dataset_train_no_du.tsv"
    main(train_tsv)