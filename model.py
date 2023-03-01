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

        # self.convs = nn.ModuleList()
        # self.batch_norms = nn.ModuleList()
        #
        # for i in range(T):
        #     self.convs.append(ChebConv(5, 5, self.K, normalization='sym'))
        #     self.batch_norms.append(BatchNorm(5))

        self.conv = ChebConv(5, 5, self.K, normalization='sym')
        self.batch_norm = BatchNorm(5)
        self.sigmoid1 = nn.Sigmoid()
        self.lstm = nn.LSTM(32 * 5, num_cells, batch_first=True)
        self.sigmoid2 = nn.Sigmoid()
        self.linear = nn.Linear(num_cells * self.T, 4)
        self.sigmoid3 = nn.Sigmoid()

    def forward(self, x):
        """
        前向传播
        :param x: list of Batch objects
        """
        batch_size = len(x)

        # GCNN layer
        y_list = []
        for data in x:
            yi = self.conv(data.x, data.edge_index, data.edge_attr, data.batch)
            yi = self.batch_norm(yi)
            y_list.append(yi)

        y = torch.vstack(y_list)
        y = self.sigmoid1(y)

        # LSTM layer
        y = torch.reshape(y, (batch_size, self.T, -1))
        y, (h, c) = self.lstm(y)
        y = self.sigmoid2(y)

        # Dense layer
        y = torch.reshape(y, (batch_size, -1))
        y = self.linear(y)
        y = self.sigmoid3(y)
        y = torch.reshape(y, (batch_size, 2, 2))
        return y


def train(model, device, train_data, batch_size, max_step, e, lr, alpha):
    """
    训练模型
    :param model: 模型
    :param device: 设备
    :param train_data: 训练数据
    :param batch_size: 批次大小
    :param max_step: maximum number of iterations
    :param e: stop iteration threshold
    :param lr: learning rate
    :param alpha: regularization coefficient
    :return: trained model
    """
    print('training...')
    writer = SummaryWriter('../logs')

    model.to(device)
    loss_func = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=alpha)

    for data in train_data:
        data.to(device)

    model.train()
    step = 0
    flag = True
    while flag:
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i: i+batch_size]
            label = []
            for data in batch:
                label.append(data.y)
            label = torch.vstack(label)

            output = model(batch)
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


def validate(model, device, val_data):
    print('validating...')
    model.to(device)
    model.eval()

    TP = np.array([0, 0])
    TN = np.array([0, 0])
    FP = np.array([0, 0])
    FN = np.array([0, 0])

    with torch.no_grad():
        for data in val_data:
            output = model([data.to(device)])
            result = torch.argmax(output, dim=-1).flatten()
            label = data.y

            for j in range(2):
                if result[j] == 0 and result[j] == label[j]:
                    TP[j] += 1
                if result[j] == 0 and result[j] != label[j]:
                    FP[j] += 1
                if result[j] == 1 and result[j] == label[j]:
                    TN[j] += 1
                if result[j] == 1 and result[j] != label[j]:
                    FN[j] += 1

    acc = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    precision = np.where(np.isnan(precision), 0, precision)
    recall = TP / (TP + FN)
    f_score = 2 * precision * recall / (precision + recall)

    print(f'acc: {acc}')
    print(f'precision: {precision}')
    print(f'recall: {recall}')
    print(f'F_score: {f_score}')

    return acc, f_score


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = DeapDataset('./data')
    subject_data = dataset[0]

    batch = subject_data[0]
    batch.to(device)
    label = torch.reshape(batch.y, (1, 2))
    print(batch)
    print(label)

    model = ECLGCNN(K=2, T=6, num_cells=30)
    model.to(device)

    loss_func = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=0.1)

    output = model([batch], 1, device)
    print(output)
    print(output.shape)

    loss = loss_func(output, label)
    print(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    result = torch.argmax(output, dim=-1)
    print(result)
