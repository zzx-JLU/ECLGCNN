import _pickle as cPickle
import numpy as np
import torch
import os
from scipy.signal import stft
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from torch_geometric.data import Batch
import torch.nn.functional as F


fs = 128  # 采样频率


def data_calibrate(data):
    """
    数据标定
    :param data: 原始数据
    :return: 标定后的数据
    """
    baseline_time = 3  # 基线数据的时间长度
    # 将 3s 基线时间与 60s 数据分开
    baseline_data, normal_data = np.split(data, [baseline_time * fs], axis=-1)
    # 将基线数据重复 20 次，补成 60s
    baseline_data = np.concatenate([baseline_data] * 20, axis=-1)
    # 用 60s 数据减去基线数据，去除噪声
    return normal_data - baseline_data


def data_divide(data, label):
    """
    数据分割
    :param data: 标定后的数据
    :param label: 标签
    :return: 分割后的数据和标签
    """
    window_size = 6  # 窗口大小
    step = 3  # 窗口滑动的步长
    num = (60 - window_size) // step + 1  # 分割成的段数

    divided_data = []
    for i in range(0, num * step, step):
        segment = data[:, :, i * fs: (i + window_size) * fs]
        divided_data.append(segment)
    divided_data = np.vstack(divided_data)

    divided_label = np.vstack([label] * num)

    return divided_data, divided_label


def set_label(labels):
    """
    打标签
    :param labels: 标签
    :return: 处理后的标签
    """
    return torch.tensor(np.where(labels < 5, 0, 1), dtype=torch.long)  # 小于 5 的元素改为 0，大于等于 5 的改为 1


def feature_extract(data_array):
    """
    提取特征
    :param data_array: 标定、分割过后的数据，每位受试者有 760 条数据
    :return: 特征立方体
    """
    f, t, zxx = stft(data_array, fs=128, window='hann', nperseg=128, noverlap=0, nfft=256, scaling='psd')

    power = np.power(np.abs(zxx), 2)

    fStart = [1, 4, 8, 14, 31]  # 起始频率
    fEnd = [3, 7, 13, 30, 50]  # 终止频率

    de_time = []
    for i in range(1, 7):
        bands = []
        for j in range(len(fStart)):
            index1 = np.where(f == fStart[j])[0][0]
            index2 = np.where(f == fEnd[j])[0][0]
            psd = np.sum(power[:, :, index1:index2, i], axis=2) / (fEnd[j] - fStart[j] + 1)
            de = np.log2(psd)
            bands.append(de)
        de_bands = np.stack(bands, axis=-1)
        de_time.append(de_bands)

    de_features = np.stack(de_time, axis=1)
    return de_features


def gaussian(dist, theta):
    """
    计算高斯核函数
    :param dist: 欧氏距离
    :return: 高斯核函数值
    """
    return torch.exp(- (dist ** 2) / (theta ** 2 * 2))


def get_edge(x, k=5, theta=1, tao=5):
    """
    计算图的边集和边权重
    :param x: 二维张量，每行是一个结点
    :param k: kNN 的 k
    :param theta: 高斯核函数中的参数
    :param tao: 距离有效的阈值
    :return: 边顶点和边权重
    """
    node_num = x.shape[0]
    edge_index = []
    weights = []

    for i in range(node_num):
        # 计算结点 i 与其他结点的距离
        dists = torch.empty(node_num)
        for j in range(node_num):
            dist = F.pairwise_distance(x[i], x[j], p=2, eps=0)
            dists[j] = dist

        # 根据 kNN 策略，选择距离前 k 小的结点
        # 由于结点到自己的距离为 0，排序后第一个元素为 i 自己，因此选择 1-k 的元素
        index = torch.argsort(dists)
        selected_index = index[1:k+1]
        for j in selected_index:
            if ([i, j] not in edge_index) and (dists[j] <= tao):
                edge_index.append([i, j])
                edge_index.append([j, i])
                # 使用高斯核函数计算边权重
                weight = gaussian(dists[j], theta)
                weights.append(weight)
                weights.append(weight)

    return edge_index, weights


def to_graph(data):
    """
    将数据转化为图结构
    :param data: 数据
    :return: Data 类型的图数据
    """
    graph_list = []
    for i in range(data.shape[0]):
        video_graph = []
        for time_data in data[i]:
            edge_index, edge_attr = get_edge(torch.tensor(time_data))
            graph = Data(x=torch.tensor(time_data, dtype=torch.float32),
                         edge_index=torch.tensor(edge_index, dtype=torch.int64).t().contiguous(),
                         edge_attr=torch.tensor(edge_attr, dtype=torch.float32))
            video_graph.append(graph)
        batch = Batch.from_data_list(video_graph)
        graph_list.append(batch)
    return graph_list


def data_processing(data, labels):
    """
    数据处理
    :param data: 数据
    :param labels: 标签
    :return: 处理后的数据和标签
    """
    # 数据标定
    data = data_calibrate(data[:, :32])
    # 数据分割
    data, labels = data_divide(data, labels)
    # 打标签
    labels = set_label(labels)
    # 特征提取
    data = feature_extract(data)
    # 转换为图结构
    graph_list = to_graph(data)

    return graph_list, labels


class DeapDataset(Dataset):
    def __init__(self, root):
        super().__init__(root)

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        file_names = []
        for file_name in self.raw_file_names:
            root, ext = os.path.splitext(file_name)
            file_names.append(root + '.pt')
        return file_names

    def download(self):
        pass

    def process(self):
        for i in range(len(self.raw_paths)):
            with open(self.raw_paths[i], 'rb') as f:
                x = cPickle.load(f, encoding='iso-8859-1')
            processed_data = data_processing(x['data'], x['labels'])
            torch.save(processed_data, self.processed_paths[i])

    def get(self, index):
        data = torch.load(self.processed_paths[index])
        return data

    def len(self):
        return len(self.raw_file_names)


if __name__ == '__main__':
    DEAPData = DeapDataset('./data')

    subject_data, labels = DEAPData[0]
    print(len(subject_data))

    for graph in subject_data:
        print(graph)
        print(graph.edge_index)
        for data in graph.to_data_list():
            print(data.edge_index)
        break

    batch = Batch.from_data_list(subject_data[0:20])
    print(batch)
    print(batch.edge_index)
    print(batch.num_graphs)
