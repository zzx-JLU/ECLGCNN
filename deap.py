import _pickle as cPickle
import numpy as np
import scipy.stats
import torch
import os
from scipy import signal
import math
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from torch_geometric.data import Batch
import torch.nn.functional as F
from scipy.signal import stft


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
    label1, label2 = np.split(labels, [2], axis=1)
    return torch.tensor(np.where(label1 < 5, 0, 1), dtype=torch.long)  # 小于 5 的元素改为 0，大于等于 5 的改为 1


def bandpass_filter(data):
    """
    带通滤波器，提取 5 种频带的信号
    :param data: 要过滤的信号
    :return: 5 种频带的信号
    """
    fStart = [1, 4, 8, 14, 31]  # 起始频率
    fEnd = [3, 7, 13, 30, 50]  # 终止频率
    results = []
    for band_index, band in enumerate(fStart):
        b, a = signal.butter(4, [fStart[band_index] / fs, fEnd[band_index] / fs], 'bandpass', fs=fs)  # 配置滤波器，4 表示滤波器的阶数
        result = signal.filtfilt(b, a, data)
        results.append(result)

    return results


def compute_DE(data):
    """
    计算微分熵
    :param data: 待计算的信号
    :return: 微分熵
    """
    variance = np.var(data, ddof=1)  # 计算方差
    return math.log(2 * math.pi * math.e * variance, 2) / 2  # 微分熵求取公式


# def DE(data, stft_para):
#     """
#     compute DE and PSD
#     --------
#     input:  data [n*m]          n electrodes, m time points
#             stft_para.stftn     frequency domain sampling rate
#             stft_para.fStart    start frequency of each frequency band
#             stft_para.fEnd      end frequency of each frequency band
#             stft_para.window    window length of each sample point(seconds)
#             stft_para.fs        original frequency
#     output: DE [n*l*k]        n electrodes, l windows, k frequency bands
#     """
#     # initialize the parameters
#     STFTN = stft_para['stftn']
#     fStart = stft_para['fStart']
#     fEnd = stft_para['fEnd']
#     fs = stft_para['fs']
#     window = stft_para['window']
#
#     fStartNum = np.zeros([len(fStart)], dtype=int)
#     fEndNum = np.zeros([len(fEnd)], dtype=int)
#     for i in range(0, len(fStart)):
#         fStartNum[i] = int(fStart[i] / fs * STFTN)
#         fEndNum[i] = int(fEnd[i] / fs * STFTN)
#
#     n = data.shape[0]
#     de = np.zeros([n, len(fStart)])
#     # Hanning window
#     Hlength = window * fs
#     # Hwindow = hanning(Hlength)
#     Hwindow = np.array([0.5 - 0.5 * np.cos(2 * np.pi * n / (Hlength + 1)) for n in range(1, Hlength + 1)])
#
#     dataNow = data[0:n]
#     for j in range(0, n):
#         temp = dataNow[j]
#         Hdata = temp * Hwindow
#         FFTdata = fft(Hdata, STFTN)
#         magFFTdata = abs(FFTdata[0:int(STFTN / 2)])
#         for p in range(0, len(fStart)):
#             E = 0
#             for p0 in range(fStartNum[p] - 1, fEndNum[p]):
#                 E = E + magFFTdata[p0] * magFFTdata[p0]
#             E = E / (fEndNum[p] - fStartNum[p] + 1)
#             de[j][p] = math.log(100 * E, 2)
#
#     return de


def feature_extract(data_array):
    """
    提取特征
    :param data_array: 标定、分割过后的数据，每位受试者有 760 条数据
    :return: 特征立方体
    """
    # stft_para = {
    #     'stftn': fs,
    #     'fStart': [1, 4, 8, 14, 31],
    #     'fEnd': [3, 7, 13, 30, 50],
    #     'window': 1,
    #     'fs': fs
    # }
    data_X = []
    for data in data_array:
        # 对每个通道提取 5 种频带
        channels = []
        for channel in data:
            bands = bandpass_filter(channel)
            channels.append(bands)

        item_data = []
        for index in range(0, data.shape[-1], fs):
            channel_data = []
            for channel in channels:
                band_data = []
                for band in channel:
                    # de = compute_DE(band[index: index + fs])  # 计算微分熵
                    de = scipy.stats.differential_entropy(band[index: index + fs])
                    band_data.append(de)
                channel_data.append(band_data)
            item_data.append(channel_data)
        data_X.append(item_data)

    return np.array(data_X)


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


def data_processing(data, label):
    """
    数据处理
    :param data: 数据
    :param label: 标签
    :return: 处理后的数据和标签
    """
    # 数据标定
    data = data_calibrate(data[:, :32])
    # 数据分割
    data, label = data_divide(data, label)
    # 打标签
    label = set_label(label)
    # 特征提取
    data = feature_extract(data)
    # 转换为图结构
    graph = to_graph(data)

    return graph, label


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
            data, label = data_processing(x['data'], x['labels'])
            torch.save([data, label], self.processed_paths[i])

    def get(self, index):
        data = torch.load(self.processed_paths[index])
        return data

    def len(self):
        return len(self.raw_file_names)


if __name__ == '__main__':
    DEAPData = DeapDataset('./data')

    subject_data, label = DEAPData[0]
    print(len(subject_data))
    print(len(label))

    for data, label in DEAPData:
        print(len(data))
        print(len(label))
