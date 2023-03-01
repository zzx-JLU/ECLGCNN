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
    print(f'data.shape: {data.shape}')

    f, t, zxx = stft(data, fs=128, window='hann', nperseg=128, noverlap=0, nfft=256, scaling='psd')
    print(f'zxx.shape: {zxx.shape}')

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
    print(f'de_features.shape: {de_features.shape}')
