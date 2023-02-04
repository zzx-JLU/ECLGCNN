from deap import *
from scipy.signal import stft


if __name__ == '__main__':
    with open('./data/raw/s01.dat', 'rb') as f:
        x = cPickle.load(f, encoding='iso-8859-1')
    data = x['data']
    label = x['labels']

    # 数据标定
    data = data_calibrate(data[:, :32])
    # 数据分割
    data, label = data_divide(data, label)
    print(data.shape)

    f, t, zxx = stft(data, fs=128, window='hann', nperseg=128, noverlap=0, nfft=256)
    print(zxx.shape)

    # index1 = np.where(f == 1)[0][0]
    # index2 = np.where(f == 3)[0][0]
    # DELTA = np.sum(zxx[:, :, index1:index2], axis=2)
    # print(DELTA.shape)
    #
    # variance1 = np.var(DELTA, axis=-1, ddof=1)
    # print(variance1.shape)
    #
    # de1 = np.log2(2 * math.pi * math.e * variance1) / 2
    # print(de1.shape)
    #
    # index1 = np.where(f == 4)[0][0]
    # index2 = np.where(f == 7)[0][0]
    # theta = np.sum(zxx[:, :, index1:index2], axis=2)
    # print(theta.shape)
    #
    # variance2 = np.var(theta, axis=-1, ddof=1)
    # print(variance2.shape)
    #
    # de2 = np.log2(2 * math.pi * math.e * variance2) / 2
    # print(de2.shape)
    #
    # de_data = np.stack([de1, de2], axis=-1)
    # print(de_data.shape)
