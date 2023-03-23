import matplotlib.pyplot as plt
import os
import math

from sklearn.model_selection import KFold
import numpy as np
import torch
from torch import nn
from torch_geometric.data import Batch

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

    model.to(device)
    model.train()

    step = 0
    flag = True
    while flag:
        for i in range(0, len(train_data), batch_size):
            batch = Batch.from_data_list(train_data[i: i + batch_size]).to(device)
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


def validate(model, device, val_data, val_labels, loss_fn):
    print('validating...')
    model.to(device)
    model.eval()

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    with torch.no_grad():
        for i, data in enumerate(val_data):
            batch = Batch.from_data_list([data]).to(device)
            output = model(batch)
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

    if abs(num_0 - num_1) < num * 0.4:
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


def subject_cross_validation(data, labels):
    splits = KFold(n_splits=k_fold, shuffle=True, random_state=2)

    acc_list = []
    f_score_list = []
    fold_models = []

    for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(data)))):
        print(f'******fold {fold}******')
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

        validate(trained_model, device, train_data, train_labels, loss_fn)
        acc, precision, recall, f_score = validate(trained_model, device, val_data, val_labels, loss_fn)
        acc_list.append(acc)
        f_score_list.append(f_score)

    avg_acc = np.array(acc_list).mean()
    avg_f_score = np.array(f_score_list).mean()
    return fold_models, avg_acc, avg_f_score


def subject_dependent_exp():
    dataset = DeapDataset('./data')

    valence_acc = []
    valence_f_score = []
    arousal_acc = []
    arousal_f_score = []

    step = 1
    for data, label in dataset:
        print(f'-------------subject: {step}-------------')
        print('==========valence==========')
        valence_label = label[:, 0]
        model_path = model_dir + f'model_{step}_valence.pt'
        if os.path.exists(model_path):
            print('model already exists, loading...')
            trained_models, acc, f_score = torch.load(model_path)
            print('model loaded.')
        else:
            trained_models, acc, f_score = subject_cross_validation(data, valence_label)
            torch.save([trained_models, acc, f_score], model_path)
            print('model saved.')

        valence_acc.append(acc)
        valence_f_score.append(f_score)
        print(f'avg_acc: {acc}, avg_f_score: {f_score}')

        print('==========arousal==========')
        arousal_label = label[:, 1]
        model_path = model_dir + f'model_{step}_arousal.pt'
        if os.path.exists(model_path):
            print('model already exists, loading...')
            trained_models, acc, f_score = torch.load(model_path)
            print('model loaded.')
        else:
            trained_models, acc, f_score = subject_cross_validation(data, arousal_label)
            torch.save([trained_models, acc, f_score], model_path)
            print('model saved.')

        arousal_acc.append(acc)
        arousal_f_score.append(f_score)
        print(f'avg_acc: {acc}, avg_f_score: {f_score}')

        step += 1

    valence_acc_array = np.array(valence_acc)
    valence_f_score_array = np.array(valence_f_score)
    fig1_path = './imgs/dependent_valence.png'
    if not os.path.exists(fig1_path):
        fig1 = plt.figure(figsize=(10, 5))
        axes1 = fig1.add_axes([0.1, 0.1, 0.8, 0.8])
        axes1.plot(valence_acc_array, 'rs-')  # 红色，正方形点，实线
        axes1.plot(valence_f_score_array, 'bo--')  # 蓝色，圆点，虚线
        axes1.legend(labels=('acc', 'f-score'), loc='lower right')
        axes1.set_title('valence')
        fig1.savefig(fig1_path)
        fig1.show()

    arousal_acc_array = np.array(arousal_acc)
    arousal_f_score_array = np.array(arousal_f_score)
    fig2_path = './imgs/dependent_arousal.png'
    if not os.path.exists(fig2_path):
        fig2 = plt.figure(figsize=(10, 5))
        axes2 = fig2.add_axes([0.1, 0.1, 0.8, 0.8])
        axes2.plot(arousal_acc_array, 'rs-')  # 红色，正方形点，实线
        axes2.plot(arousal_f_score_array, 'bo--')  # 蓝色，圆点，虚线
        axes2.legend(labels=('acc', 'f-score'), loc='lower right')
        axes2.set_title('arousal')
        fig2.savefig(fig2_path)
        fig2.show()

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


if __name__ == '__main__':
    # subject_dependent_exp()
    valence_acc1 = np.array([0.90921053, 0.89210526, 0.95526316, 0.84868421, 0.93421053, 0.93684211,
                             0.88157895, 0.90263158, 0.86447368, 0.92368421, 0.80263158, 0.88026316,
                             0.88552632, 0.90263158, 0.93157895, 0.925,      0.80789474, 0.86315789,
                             0.90921053, 0.90394737, 0.8,        0.85657895, 0.87763158, 0.85921053,
                             0.93421053, 0.78421053, 0.94210526, 0.95,       0.93947368, 0.92631579,
                             0.90394737, 0.86578947])
    valence_f_score1 = np.array([0.91449233, 0.85641315, 0.94812191, 0.88061513, 0.91580049, 0.87105462,
                                 0.78322498, 0.88898958, 0.85191167, 0.92344264, 0.77754519, 0.85583819,
                                 0.89725599, 0.89514857, 0.93002493, 0.94108316, 0.69874529, 0.77847794,
                                 0.88949198, 0.88923062, 0.70243996, 0.75826129, 0.78837344, 0.87691933,
                                 0.92853425, 0.48826931, 0.88015721, 0.93180997, 0.92685127, 0.88378154,
                                 0.86022866, 0.86942919])
    arousal_acc1 = np.array([0.93157895, 0.89473684, 0.95394737, 0.92368421, 0.85657895, 0.9,
                             0.93026316, 0.92368421, 0.89473684, 0.93421053, 0.89868421, 0.96710526,
                             0.96578947, 0.90263158, 0.93421053, 0.89736842, 0.86447368, 0.925,
                             0.92631579, 0.93289474, 0.93157895, 0.93289474, 0.86184211, 0.95394737,
                             0.94342105, 0.86315789, 0.88289474, 0.90789474, 0.91052632, 0.91315789,
                             0.87631579, 0.93157895])
    arousal_f_score1 = np.array([0.91840426, 0.86758026, 0.9738418,  0.93612624, 0.87070282, 0.91186608,
                                 0.90457591, 0.90524752, 0.86096741, 0.92561098, 0.91774994, 0.90162661,
                                 0.87683076, 0.84907676, 0.92987184, 0.89957365, 0.8157947,  0.8915495,
                                 0.89491609, 0.85405514, 0.85036519, 0.91060754, 0.90799942, 0.86636537,
                                 0.88194108, 0.87775394, 0.79826178, 0.90905563, 0.86940658, 0.91428995,
                                 0.87472342, 0.89121444])

    valence_acc2 = np.array([0.93157895, 0.88421053, 0.95131579, 0.9,        0.92368421, 0.95526316,
                             0.89342105, 0.83552632, 0.81447368, 0.93157895, 0.80921053, 0.90263158,
                             0.88552632, 0.91052632, 0.93421053, 0.91973684, 0.89473684, 0.87763158,
                             0.925,      0.91184211, 0.75131579, 0.94078947, 0.94078947, 0.90131579,
                             0.93815789, 0.87894737, 0.92894737, 0.93157895, 0.95263158, 0.94210526,
                             0.84473684, 0.88026316])
    valence_f_score2 = np.array([0.93512076, 0.84081951, 0.94371373, 0.91987446, 0.90701049, 0.90965094,
                                 0.80025129, 0.71307256, 0.69636137, 0.92976884, 0.78653074, 0.88584349,
                                 0.89439237, 0.9069842,  0.93243576, 0.93732946, 0.88145589, 0.81800387,
                                 0.91417265, 0.89224197, 0.54251381, 0.93963087, 0.89931039, 0.90374753,
                                 0.93467898, 0.78842955, 0.85149855, 0.90259008, 0.94426043, 0.90887668,
                                 0.6865293,  0.88311942])
    arousal_acc2 = np.array([0.91578947, 0.90131579, 0.96447368, 0.92631579, 0.83289474, 0.88157895,
                             0.89078947, 0.88421053, 0.91184211, 0.91184211, 0.86315789, 0.92105263,
                             0.95131579, 0.91184211, 0.91052632, 0.88947368, 0.85789474, 0.95263158,
                             0.91973684, 0.95,       0.97236842, 0.93026316, 0.91842105, 0.96973684,
                             0.91578947, 0.88026316, 0.85789474, 0.91842105, 0.94473684, 0.91710526,
                             0.80526316, 0.93552632])
    arousal_f_score2 = np.array([0.89533084, 0.8753039,  0.977664,   0.93921037, 0.86562732, 0.89670313,
                                 0.8497698,  0.85821872, 0.88078427, 0.90361759, 0.88581914, 0.77697657,
                                 0.83902148, 0.86394839, 0.91414794, 0.89285319, 0.79572986, 0.93653459,
                                 0.88245299, 0.8839378,  0.92742329, 0.90376527, 0.93918273, 0.91408823,
                                 0.85900807, 0.8932705,  0.74983237, 0.91621908, 0.91226041, 0.91850539,
                                 0.84042916, 0.89720401])

    valence_acc3 = np.array([0.84868421, 0.89210526, 0.94736842, 0.86842105, 0.90526316, 0.95526316,
                             0.88552632, 0.91315789, 0.86842105, 0.93157895, 0.83421053, 0.90789474,
                             0.88815789, 0.91578947, 0.91315789, 0.92236842, 0.72105263, 0.85526316,
                             0.91184211, 0.91447368, 0.81315789, 0.93552632, 0.90394737, 0.875,
                             0.90394737, 0.84736842, 0.89868421, 0.93552632, 0.95131579, 0.88552632,
                             0.82236842, 0.86052632])
    valence_f_score3 = np.array([0.88399749, 0.84798371, 0.93557556, 0.88917741, 0.86665592, 0.90883302,
                                 0.77691855, 0.90400716, 0.85084346, 0.93132285, 0.77715051, 0.90224516,
                                 0.9013443,  0.91393649, 0.9092079,  0.93924921, 0.3653716,  0.66135539,
                                 0.89549444, 0.89712402, 0.69840136, 0.93531146, 0.84591832, 0.88475305,
                                 0.89425836, 0.67366963, 0.8429411,  0.91270161, 0.94450015, 0.7093804,
                                 0.64520369, 0.86117296])
    arousal_acc3 = np.array([0.92894737, 0.87763158, 0.90921053, 0.89210526, 0.87105263, 0.88684211,
                             0.90394737, 0.93026316, 0.88684211, 0.92631579, 0.88289474, 0.92631579,
                             0.93026316, 0.84473684, 0.85263158, 0.92105263, 0.79342105, 0.92631579,
                             0.94605263, 0.94210526, 0.94605263, 0.92631579, 0.83815789, 0.95131579,
                             0.94605263, 0.87894737, 0.90131579, 0.92236842, 0.90263158, 0.92631579,
                             0.80921053, 0.93684211])
    arousal_f_score3 = np.array([0.90997274, 0.83679766, 0.94194903, 0.90815803, 0.87312231, 0.89794373,
                                 0.87730401, 0.91097803, 0.84438376, 0.91764413, 0.90631266, 0.7820423,
                                 0.76389347, 0.64969631, 0.7369481,  0.92365912, 0.59455226, 0.89676139,
                                 0.9110309,  0.86525984, 0.86478027, 0.88946084, 0.89081292, 0.86106641,
                                 0.88752629, 0.89616554, 0.8320706,  0.92098759, 0.86407158, 0.91812783,
                                 0.70949673, 0.8969725])

    valence_avg_acc = (valence_acc1 + valence_acc2 + valence_acc3) / 3
    valence_avg_f = (valence_f_score1 + valence_f_score2 + valence_f_score3) / 3
    arousal_avg_acc = (arousal_acc1 + arousal_acc2 + arousal_acc3) / 3
    arousal_avg_f = (arousal_f_score1 + arousal_f_score2 + arousal_f_score3) / 3

    print(f'average acc of valence: {np.average(valence_avg_acc)}')
    print(f'average acc of arousal: {np.average(arousal_avg_acc)}')

    fig1 = plt.figure(figsize=(10, 5))
    axes1 = fig1.add_axes([0.1, 0.1, 0.8, 0.8])
    axes1.plot(valence_avg_acc, 'rs-')  # 红色，正方形点，实线
    axes1.plot(valence_avg_f, 'bo--')  # 蓝色，圆点，虚线
    axes1.legend(labels=('acc', 'f-score'), loc='lower right')
    axes1.set_title('valence')
    fig1.savefig('./imgs/dependent_valence_average.png')
    fig1.show()

    fig2 = plt.figure(figsize=(10, 5))
    axes2 = fig2.add_axes([0.1, 0.1, 0.8, 0.8])
    axes2.plot(arousal_avg_acc, 'rs-')  # 红色，正方形点，实线
    axes2.plot(arousal_avg_f, 'bo--')  # 蓝色，圆点，虚线
    axes2.legend(labels=('acc', 'f-score'), loc='lower right')
    axes2.set_title('arousal')
    fig2.savefig('./imgs/dependent_arousal_average.png')
    fig2.show()
