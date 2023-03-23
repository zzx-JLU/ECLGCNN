import torch
from torch import nn
from torch_geometric.data import Batch
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
        self.dropout1 = nn.Dropout(0.2)
        self.lstm = nn.LSTM(32 * 5, num_cells, batch_first=True)
        self.sigmoid2 = nn.Sigmoid()
        self.dropout2 = nn.Dropout(0.2)
        self.linear = nn.Linear(num_cells * self.T, 2)
        self.sigmoid3 = nn.Sigmoid()

        for param in self.parameters():
            if len(param.shape) < 2:
                nn.init.xavier_uniform_(param.unsqueeze(0))
            else:
                nn.init.xavier_uniform_(param)

    def forward(self, x):
        """
        前向传播
        :param x: Batch object
        """
        batch_size = x.num_graphs

        # GCNN layer
        # y_list = []
        # for data in x:
        #     yi = self.conv(data.x, data.edge_index, data.edge_attr, data.batch)
        #     yi = self.batch_norm(yi)
        #     y_list.append(yi)
        y = self.conv(x.x, x.edge_index, x.edge_attr, x.batch)
        y = self.batch_norm(y)
        y = self.sigmoid1(y)
        # y = self.dropout1(y)

        # LSTM layer
        y = torch.reshape(y, (batch_size, self.T, -1))
        y, (h, c) = self.lstm(y)
        y = self.sigmoid2(y)
        # y = self.dropout2(y)

        # Dense layer
        y = torch.reshape(y, (batch_size, -1))
        y = self.linear(y)
        y = self.sigmoid3(y)
        return y


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = DeapDataset('./data')
    subject_data, labels = dataset[0]

    batch = Batch.from_data_list(subject_data[0:20]).to(device)
    label = labels[0:20].to(device)

    model = ECLGCNN(K=2, T=6, num_cells=30)
    model.to(device)

    loss_func = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=0.1)

    output = model(batch)
    print(output)
    print(output.shape)

    loss = loss_func(output, label[:, 0])
    print(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    result = torch.argmax(output, dim=-1)
    print(result)
