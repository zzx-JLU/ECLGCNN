import matplotlib.pyplot as plt
import os

from sklearn.model_selection import KFold
import numpy as np
import torch
from torch import nn

from deap import DeapDataset
from model import ECLGCNN


# parameters for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_fn = nn.CrossEntropyLoss().to(device)
batch_size = 32
max_iteration = 5000  # maximum number of iterations
e = 0.1  # stop iteration threshold
lr = 0.003  # learning rate
alpha = 0.0008  # regularization coefficient
k_fold = 5
model_dir = './model/'
img_dir = './imgs/'

# parameters of model
K = 2
T = 6
num_cells = 30


def train(model, device, train_data, train_labels, loss_fn, optimizer):
    print('training...')

    for data in train_data:
        data.to(device)

    model.to(device)
    model.train()

    step = 0
    flag = True
    while flag:
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i: i + batch_size]
            label = train_labels[i: i + batch_size].to(device)

            output = model(batch)
            loss = loss_fn(output, label)

            if step % 100 == 0:
                print(f'setp: {step}, Loss: {loss.item()}')

            if loss < e:
                flag = False
                break

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            if step >= max_iteration:
                flag = False
                break

    print('training successfully ended.')
    return model


def validate(model, device, val_data, val_labels):
    print('validating...')
    model.to(device)
    model.eval()

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    with torch.no_grad():
        for i, data in enumerate(val_data):
            output = model([data.to(device)])
            result = torch.argmax(output, dim=-1).flatten().item()
            label = val_labels[i]

            if result == 0 and result == label:
                TP += 1
            if result == 0 and result != label:
                FP += 1
            if result == 1 and result == label:
                TN += 1
            if result == 1 and result != label:
                FN += 1

    acc = (TP + TN) / (TP + TN + FP + FN)

    if TP + FP == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)

    recall = TP / (TP + FN)

    if precision + recall == 0:
        f_score = 0
    else:
        f_score = 2 * precision * recall / (precision + recall)

    print(f'acc: {acc}')
    print(f'precision: {precision}')
    print(f'recall: {recall}')
    print(f'F_score: {f_score}')

    return acc, precision, recall, f_score


def binary_sampling(data, label):
    """
    对二分类数据集进行重采样，保证数据平衡
    """
    num = len(data)  # number of samples

    index_0 = torch.nonzero(label == 0, as_tuple=True)[0]
    num_0 = len(index_0)

    index_1 = torch.nonzero(label == 1, as_tuple=True)[0]
    num_1 = len(index_1)
    print([num_0, num_1])

    if abs(num_0 - num_1) < num * 0.25:
        return data, label
    else:
        selected_index = []
        shorter_index, shorter_num = (index_0, num_0) if num_0 < num_1 else (index_1, num_1)
        longer_index, longer_num = (index_0, num_0) if num_0 > num_1 else (index_1, num_1)
        for i in range(longer_num):
            selected_index.append(longer_index[i])
            selected_index.append(shorter_index[i % shorter_num])

        selected_data = []
        for i in selected_index:
            selected_data.append(data[i])
        selected_label = label[selected_index]
        return selected_data, selected_label


def subject_cross_validation(data, labels, seed):
    splits = KFold(n_splits=k_fold, shuffle=True, random_state=seed)

    acc_list = []
    f_score_list = []
    fold_models = []

    for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(data)))):
        print(f'******fold {fold+1}******')
        train_data = []
        for i in train_idx:
            train_data.append(data[i])
        train_labels = labels[train_idx]

        val_data = []
        for i in val_idx:
            val_data.append(data[i])
        val_labels = labels[val_idx]

        train_data, train_labels = binary_sampling(train_data, train_labels)

        model = ECLGCNN(K=K, T=T, num_cells=num_cells)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=alpha)

        trained_model = train(model, device, train_data, train_labels, loss_fn, optimizer)
        fold_models.append(trained_model)

        validate(trained_model, device, train_data, train_labels)
        acc, precision, recall, f_score = validate(trained_model, device, val_data, val_labels)
        acc_list.append(acc)
        f_score_list.append(f_score)

    avg_acc = np.array(acc_list).mean()
    avg_f_score = np.array(f_score_list).mean()
    return fold_models, avg_acc, avg_f_score


def output_figure(path, acc, f_score, title):
    fig = plt.figure(figsize=(10, 5))
    axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    x = np.arange(1, 33, dtype=int)

    axes.plot(x, acc, 'rs-')  # 红色，正方形点，实线
    axes.plot(x, f_score, 'bo--')  # 蓝色，圆点，虚线
    axes.legend(labels=('acc', 'F-score'), loc='lower right')

    axes.set_ylim(0, 1)
    axes.set_yticks(np.arange(0, 1.1, 0.1))
    axes.set_yticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

    axes.set_xticks(x)
    axes.set_xticklabels(x)

    axes.set_title(title)
    axes.grid(True)

    fig.savefig(path)
    fig.show()


def subject_dependent_exp():
    dataset = DeapDataset('./data')

    valence_acc_arrays = []
    valence_f_score_arrays = []
    arousal_acc_arrays = []
    arousal_f_score_arrays = []

    for seed in range(0, 3):
        valence_acc = []
        valence_f_score = []
        arousal_acc = []
        arousal_f_score = []

        step = 1
        for data, label in dataset:
            print(f'-------------subject: {step}-------------')
            print('==========valence==========')
            valence_label = label[:, 0]
            model_path = model_dir + f'seed{seed}/model_{step}_valence.pt'
            if os.path.exists(model_path):
                print('model already exists, loading...')
                trained_models, acc, f_score = torch.load(model_path)
                print('model loaded.')
            else:
                trained_models, acc, f_score = subject_cross_validation(data, valence_label, seed)
                torch.save([trained_models, acc, f_score], model_path)
                print('model saved.')

            valence_acc.append(acc)
            valence_f_score.append(f_score)
            print(f'avg_acc: {acc}, avg_f_score: {f_score}')

            print('==========arousal==========')
            arousal_label = label[:, 1]
            model_path = model_dir + f'seed{seed}/model_{step}_arousal.pt'
            if os.path.exists(model_path):
                print('model already exists, loading...')
                trained_models, acc, f_score = torch.load(model_path)
                print('model loaded.')
            else:
                trained_models, acc, f_score = subject_cross_validation(data, arousal_label, seed)
                torch.save([trained_models, acc, f_score], model_path)
                print('model saved.')

            arousal_acc.append(acc)
            arousal_f_score.append(f_score)
            print(f'avg_acc: {acc}, avg_f_score: {f_score}')

            step += 1

        valence_acc_array = np.array(valence_acc)
        valence_f_score_array = np.array(valence_f_score)
        valence_path = f'./imgs/seed{seed}/dependent_valence.png'
        output_figure(valence_path, valence_acc_array, valence_f_score_array, 'valence')

        arousal_acc_array = np.array(arousal_acc)
        arousal_f_score_array = np.array(arousal_f_score)
        arousal_path = f'./imgs/seed{seed}/dependent_arousal.png'
        output_figure(arousal_path, arousal_acc_array, arousal_f_score_array, 'arousal')

        valence_acc_arrays.append(valence_acc_array)
        valence_f_score_arrays.append(valence_f_score_array)
        arousal_acc_arrays.append(arousal_acc_array)
        arousal_f_score_arrays.append(arousal_f_score_array)

        print('---------RESULT---------')
        print('valence')
        print(f'  acc: {valence_acc_array}')
        print(f'  average acc: {valence_acc_array.mean()}')
        print(f'  f-score: {valence_f_score_array}')
        print(f'  average f-score: {valence_f_score_array.mean()}')
        print('arousal')
        print(f'  acc: {arousal_acc_array}')
        print(f'  average acc: {arousal_acc_array.mean()}')
        print(f'  f-score: {arousal_f_score_array}')
        print(f'  average f-score: {arousal_f_score_array.mean()}')

    avg_valence_acc = sum(valence_acc_arrays) / len(valence_acc_arrays)
    avg_valence_f_score = sum(valence_f_score_arrays) / len(valence_f_score_arrays)
    avg_arousal_acc = sum(arousal_acc_arrays) / len(arousal_acc_arrays)
    avg_arousal_f_score = sum(arousal_f_score_arrays) / len(arousal_f_score_arrays)

    output_figure('./imgs/dependent_valence_avg.png', avg_valence_acc, avg_valence_f_score, 'valence_avg')
    output_figure('./imgs/dependent_arousal_avg.png', avg_arousal_acc, avg_arousal_f_score, 'arousal_avg')


if __name__ == '__main__':
    subject_dependent_exp()
