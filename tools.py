import time
from typing import Union

from torch import Size, Tensor
from torch.utils.data.dataset import T_co
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import os
import math
from torch.utils.data import DataLoader, Dataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TimeNorm(nn.LayerNorm):
    def __init__(self, normlized_shape: Union[int, list[int], Size], eps: float = 1e-5, elementwise_affine: bool = True,
                 device=None, dtype=None):
        super().__init__(normalized_shape=normlized_shape, eps=eps, elementwise_affine=elementwise_affine,
                         device=device, dtype=dtype)

    def forward(self, input: Tensor) -> Tensor:
        input = input.transpose(-1, -2)
        output = F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)
        output = output.transpose(-1, -2)
        return output


class Data_set(Dataset):
    def __init__(self, data: list):
        self.data = data

    def __getitem__(self, index) -> T_co:
        return self.data[index]

    def __len__(self):
        return len(self.data)


def df2slicelist(df: pd.DataFrame, label, cut=5, step=5):
    lst = []
    start = 0
    end = 256 * cut
    while end <= len(df):
        data = df.iloc[start:end, :]
        data = data.to_numpy()
        data = torch.from_numpy(data)
        lst.append((data, label))
        start += 256 * step
        end += 256 * step
    return lst


def train(train_loader: DataLoader, val_loader: DataLoader, epochs, model_name, model, loss_fn, optimizer,
          early_stop=True, n_stop=5):
    global min_val_loss, baseline, n_warn
    model.cuda()
    if early_stop:
        n_stop = n_stop  # 提前停止的监管轮数
        baseline = 1e-6
        min_val_loss = 1e6
        n_warn = 0

    for epoch in range(epochs):
        start_time = time.time()  # 记录开始时间
        print(f'------{epoch + 1} epoch------')
        model.train()  # 将模型设置为训练模式
        total_train_loss = 0  # 记录每轮迭代的总损失
        train_step = 0  # 记录训练的step
        for data in train_loader:
            inputs, labels = data  # 获取输入数据和标签
            inputs = inputs.reshape(-1, 1, 1280, 18)
            inputs, labels = inputs.cuda(), labels.cuda()  # 将数据放在GPU上

            optimizer.zero_grad()  # 将梯度置零，每个批次样本之间是独立的，不能相互影响
            inputs = inputs.to(torch.float32)
            outputs = model(inputs).cuda()
            loss = loss_fn(outputs, labels)

            loss.backward()  # 误差反向传播
            optimizer.step()  # 更新模型参数
            total_train_loss += loss
            train_step += 1
        print(f'train_loss = {total_train_loss / train_step:.4f}')

        model.eval()  # 将模型设置为验证模式
        total_val_loss = 0
        val_step = 0
        with torch.no_grad():  # 梯度置零
            for data in val_loader:
                inputs, labels = data
                inputs = inputs.reshape(-1, 1, 1280, 18)
                inputs, labels = inputs.cuda(), labels.cuda()
                inputs = inputs.to(torch.float32)

                outputs = model(inputs).cuda()
                loss = loss_fn(outputs, labels)
                total_val_loss += loss
                val_step += 1
        print(f'val_loss = {total_val_loss / val_step:.4f}')
        end_time = time.time()
        print(f'time_cost: {end_time - start_time:.2f}s')

        if early_stop:
            val_loss = total_val_loss / val_step
            if min_val_loss - val_loss >= baseline:  # 表明本轮迭代有显著效果
                n_warn = 0  # 警戒次数重置为0
                torch.save(model, f'saved_models/{model_name}_best.pt')  # 保存最优模型
                min_val_loss = val_loss
            else:
                n_warn += 1
                if n_warn == n_stop:
                    model = torch.load(f'saved_models/{model_name}_best.pt')
                    break
    return model


def test(test_loader: DataLoader, model, indicators, cut=5, group=False, threshold=0.5):
    model.cuda()
    model.eval()
    y_pred = []
    y_true = []
    start = time.time()
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs = inputs.reshape(-1, 1, 1280, 18)
            y_true.extend(labels)
            inputs, labels = inputs.cuda(), labels.cuda()
            inputs = inputs.to(torch.float32)
            outputs = model(inputs).cuda()
            outputs = torch.argmax(outputs, dim=1)
            if group and threshold:
                if torch.sum(outputs) >= threshold * len(outputs):
                    y_pred.extend([1] * len(outputs))
                else:
                    y_pred.extend([0] * len(outputs))
            elif group and not threshold:
                y_pred.extend([outputs[-1]] * len(outputs))
            else:
                y_pred.extend(outputs)
    end = time.time()
    res = evaluate(y_true, y_pred, indicators, cut=cut)
    res['pred_time_total'] = end - start
    if group:
        res['pred_time'] = res['pred_time_total'] / (len(test_loader) / test_loader.batch_size)
    else:
        res['pred_time'] = res['pred_time_total'] / len(test_loader)
    return res


def evaluate(y_true, y_pred, indicators, cut):
    res = {}
    conf_matrix = np.zeros((2, 2))
    for i in range(len(y_true)):
        true = y_true[i]
        pred = y_pred[i]
        conf_matrix[true, pred] += 1
    TN = conf_matrix[0, 0]
    FP = conf_matrix[0, 1]
    FN = conf_matrix[1, 0]
    TP = conf_matrix[1, 1]
    for indicator in indicators:
        if indicator == 'Acc':
            res['Acc'] = (TP + TN) / (TP + TN + FP + FN)
        elif indicator == 'Sen':
            res['Sen'] = TP / (TP + FN)
        elif indicator == 'Spe':
            res['Spe'] = TN / (TN + FP)
        elif indicator == 'FPR':
            res['FPR'] = FP / (len(y_true) * cut / 3600)
    return res


class TAlayer(nn.Module):
    def __init__(self, hidden_size=32, bidirectional=True):
        super().__init__()
        if bidirectional:
            num_direction = 2
        else:
            num_direction = 1
        d = hidden_size * num_direction
        self.v = nn.Parameter(torch.randn(d))

    def forward(self, x):
        x = x.permute(1, 0, 2)
        s = torch.tanh(x) @ self.v  # L,1
        alpha = torch.softmax(s, dim=1).unsqueeze(-1)  # L,1
        r = torch.matmul(x.transpose(-1, -2), alpha).squeeze(-1)  # D
        y = torch.tanh(r)  # D
        return y


class h_swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        sigmoid = self.relu(x + 3) / 6
        return x * sigmoid


class CAlayer(nn.Module):
    def __init__(self, channels, reduction=4):  # reduciton为缩减率，减少计算的通道数量
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mid_c = max(8, channels // reduction)

        self.conv1 = nn.Conv2d(channels, mid_c, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mid_c)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mid_c, channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mid_c, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)  # n,c,h,1
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # n,c,1,w -> n,c,w,1

        y = torch.cat([x_h, x_w], dim=2)  # n,c,h+w,1
        y = self.conv1(y)  # n,c/r,h+w,1
        y = self.bn1(y)  # n,c/r,h+w,1
        y = self.act(y)  # n,c/r,h+w,1

        x_h, x_w = torch.split(y, [h, w], dim=2)  # n,c/r,h,1 and n,c,w,1
        x_w = x_w.permute(0, 1, 3, 2)  # n,c/r,1,w

        a_h = self.conv_h(x_h).sigmoid()  # n,c,h,1
        a_w = self.conv_w(x_w).sigmoid()  # n,c,1,w

        out = identity * a_w * a_h

        return out
