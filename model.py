import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.nn import ChebConv
from torch_geometric.nn.norm import BatchNorm
from deap import DeapDataset


class ECLGCNN(nn.Module):
    def __init__(self, K, T, num_cells):
        """
        :param K: 切比雪夫阶数
        :param num_cells: LSTM 隐藏层单元数
        :param T: GCNN 单元数
        """
        super(ECLGCNN, self).__init__()
        self.K = K
        self.T = T

        self.conv = ChebConv(5, 5, self.K, normalization='sym', bias=False)
        self.batch_norm = BatchNorm(5)
        self.sigmoid1 = nn.Sigmoid()
        self.lstm = nn.LSTM(32 * 5, num_cells)
        self.linear = nn.Linear(num_cells * self.T, 4)
        self.sigmoid2 = nn.Sigmoid()

    def forward(self, x):
        """
        前向传播
        :param x: Batch object that contains 6 graphs
        """
        # GCNN layer
        y = self.conv(x.x, x.edge_index, x.edge_attr, x.batch)
        y = self.batch_norm(y)
        y = self.sigmoid1(y)

        # LSTM layer
        y = torch.reshape(y, (self.T, 1, -1))
        y, (h, c) = self.lstm(y)

        # Dense layer
        y = torch.flatten(y)
        y = self.linear(y)
        y = self.sigmoid2(y)
        y = torch.reshape(y, [2, 2])
        return y


def train(model, device, train_data, train_labels, max_step, e, lr, alpha):
    """
    训练模型
    :param model: 模型
    :param device: 设备
    :param train_data: 训练数据
    :param train_labels: 训练数据的标签
    :param max_step: 最大循环次数
    :param e: 误差阈值
    :param lr: 学习率
    :param alpha: 正则化参数
    :return:
    """
    print('training...')
    writer = SummaryWriter('../logs')

    model.to(device)
    loss_func = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=alpha)

    model.train()
    step = 0
    flag = True
    while flag:
        for i, data in enumerate(train_data):
            output = model(data.to(device))
            output = torch.reshape(output, [1, 2, -1])
            label = torch.reshape(train_labels[i], [1,  -1])
            loss = loss_func(output, label.to(device))

            writer.add_scalar('train_loss', loss.item(), step)
            if step % 100 == 0:
                print(f'训练次数: {step}, Loss: {loss.item()}')

            if loss < e:
                flag = False
                break

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            if step >= max_step:
                flag = False
                break

    writer.close()
    print('training successfully ended.')
    return model


def validate(model, device, val_data, val_labels):
    model.to(device)
    model.eval()

    TP = np.array([0, 0])
    TN = np.array([0, 0])
    FP = np.array([0, 0])
    FN = np.array([0, 0])

    with torch.no_grad():
        for i, data in enumerate(val_data):
            output = model(data.to(device))
            result = torch.argmax(output, dim=1)

            for j in range(2):
                if result[j] == 0 and result[j] == val_labels[i][j]:
                    TP[j] += 1
                if result[j] == 0 and result[j] != val_labels[i][j]:
                    FP[j] += 1
                if result[j] == 1 and result[j] == val_labels[i][j]:
                    TN[j] += 1
                if result[j] == 1 and result[j] != val_labels[i][j]:
                    FN[j] += 1

    acc = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f_score = 2 * precision * recall / (precision + recall)
    return acc, f_score


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = DeapDataset('./data')
    subject_data, label = dataset[0]

    batch = subject_data[0]
    batch.to(device)
    print(batch)
    print(label[0])

    model = ECLGCNN(K=2, T=6, num_cells=30)
    model.to(device)

    loss_func = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=0.1)

    output = model(batch)
    print(output)
    print(output.shape)

    loss = loss_func(output, label[0].to(device))
    print(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    result = torch.argmax(output, dim=1)
    print(result)
